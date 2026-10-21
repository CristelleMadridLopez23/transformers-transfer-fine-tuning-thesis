"""Select pooled frozen features and fit independent Ridge, RF and small MLP heads.

All model/pooling selection uses identical composition folds in development.
Test labels are never used for training, normalization or checkpoint selection.
Full runs require successful preflight and a Slurm allocation.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from utils import (LOGGER, audit_fingerprint, cli, load_config, load_embeddings,
                   require_gate, sha256, write_json)


class SmallMLP(nn.Module):
    """Standalone d→hidden→1 GELU/dropout regressor; contains no Transformer."""

    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        """Construct network from integer dimensions and dropout probability; return None."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map B×d floating inputs to B scalar predictions; return Tensor."""
        return self.net(inputs).squeeze(-1)


def fit_mlp(x: np.ndarray, y: np.ndarray, groups: np.ndarray, params: dict[str, Any],
            settings: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit MLP on training arrays/groups with internal grouped early stopping; return payload.

    Early stopping has its own grouped subset within the supplied training fold.
    A second fit uses all supplied training rows for the selected number of epochs.
    Scalers are separately fitted for inner training and all-training refit.
    """
    if len(np.unique(groups)) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("MLP needs finite values and at least two training compositions")
    split = GroupShuffleSplit(n_splits=1, test_size=settings["inner_validation_fraction"], random_state=seed)
    inner_train, inner_val = next(split.split(x, y, groups))
    curves = []
    def run(train_indices: np.ndarray, validation_indices: np.ndarray | None,
            epochs: int) -> dict[str, Any]:
        """Fit on integer training indices; optionally early-stop; return state/scalers/epochs."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        xt = torch.tensor(x_scaler.fit_transform(x[train_indices]), dtype=torch.float32)
        yt = torch.tensor(y_scaler.fit_transform(y[train_indices, None]).ravel(), dtype=torch.float32)
        network = SmallMLP(x.shape[1], settings["hidden_dim"], settings["dropout"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=params["learning_rate"],
                                     weight_decay=params["weight_decay"])
        best, best_epoch, bad, best_state = float("inf"), 0, 0, None
        for epoch in range(epochs):
            network.train()
            for indices in np.array_split(rng.permutation(len(xt)), max(1, int(np.ceil(len(xt) / settings["batch_size"])))):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(network(xt[indices]), yt[indices])
                loss.backward()
                optimizer.step()
            if validation_indices is None:
                continue
            network.eval()
            with torch.inference_mode():
                training_loss = float(nn.functional.mse_loss(network(xt), yt))
                xv = torch.tensor(x_scaler.transform(x[validation_indices]), dtype=torch.float32)
                yv = torch.tensor(y_scaler.transform(y[validation_indices, None]).ravel(), dtype=torch.float32)
                validation_loss = float(nn.functional.mse_loss(network(xv), yv))
            curves.append({"epoch": epoch + 1, "train_mse_standardized": training_loss,
                           "validation_mse_standardized": validation_loss})
            if not np.isfinite(training_loss + validation_loss):
                raise RuntimeError("Nonfinite MLP training loss")
            if validation_loss < best - 1e-8:
                best, best_epoch, bad = validation_loss, epoch + 1, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                bad += 1
                if bad >= settings["patience"]:
                    break
        return {"state": best_state if validation_indices is not None else network.state_dict(),
                "epochs": best_epoch if validation_indices is not None else epochs,
                "x_scaler": x_scaler, "y_scaler": y_scaler}
    early = run(inner_train, inner_val, settings["max_epochs"])
    final = run(np.arange(len(x)), None, early["epochs"])
    return {"kind": "mlp", **final, "early_stopping_state": early["state"],
            "early_x_scaler": early["x_scaler"], "early_y_scaler": early["y_scaler"],
            "dimension": x.shape[1], "hidden": settings["hidden_dim"], "dropout": settings["dropout"],
            "curves": curves, "params": params}


def fit_predictor(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  params: dict[str, Any], config: dict[str, Any]) -> Any:
    """Fit named independent predictor on training-only arrays; return sklearn or MLP payload."""
    if name == "mlp":
        return fit_mlp(x, y, groups, params, config["models"]["mlp"], config["seed"])
    if name == "ridge":
        model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge(**params)),
                                          transformer=StandardScaler())
    elif name == "random_forest":
        model = RandomForestRegressor(**params, random_state=config["seed"],
                                      n_jobs=config["models"]["random_forest"]["n_jobs"])
    else:
        raise ValueError(f"Unknown predictor: {name}")
    model.fit(x, y)
    return model


def predict(model: Any, x: np.ndarray) -> np.ndarray:
    """Predict from sklearn estimator or saved MLP payload and finite 2D x; return eV/atom."""
    if isinstance(model, dict) and model.get("kind") == "mlp":
        network = SmallMLP(model["dimension"], model["hidden"], model["dropout"])
        network.load_state_dict(model["state"], strict=True)
        network.eval()
        with torch.inference_mode():
            raw = network(torch.tensor(model["x_scaler"].transform(x), dtype=torch.float32)).numpy()
        result = model["y_scaler"].inverse_transform(raw[:, None]).ravel()
    else:
        result = model.predict(x)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite supervised predictions")
    return result


def grids(config: dict[str, Any], smoke: bool) -> dict[str, list[dict[str, Any]]]:
    """Create controlled hyperparameter dictionaries from config; return grids per model."""
    models = config["models"]
    result = {
        "ridge": list(ParameterGrid({"alpha": models["ridge"]["alpha"]})),
        "random_forest": list(ParameterGrid({key: models["random_forest"][key] for key in
                                              ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]})),
        "mlp": list(ParameterGrid({key: models["mlp"][key] for key in ["learning_rate", "weight_decay"]}))}
    if smoke:
        result = {name: values[:1] for name, values in result.items()}
        result["random_forest"][0]["n_estimators"] = 5
    return result


def train(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Select common pooling through development MLP CV; fit all heads, saving local artifacts."""
    if not smoke:
        require_gate(config)
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Full training must run inside Slurm")
    frame, arrays = load_embeddings(config, smoke)
    development = frame.split.eq("development").to_numpy()
    # No test energy is passed to any fitting or selection routine.
    frame = frame.loc[development].reset_index(drop=True)
    arrays = {key: value[development] for key, value in arrays.items()}
    if smoke and frame.reduced_formula.nunique() < 3:
        raise RuntimeError("Smoke extraction needs >=3 development compositions for grouped training")
    settings = copy.deepcopy(config)
    if smoke:
        settings["models"]["mlp"].update(max_epochs=3, patience=2)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    candidates = grids(settings, smoke)
    enabled = settings["models"]["enabled"]
    if "mlp" not in enabled:
        raise ValueError("MLP must be enabled as the predeclared primary pooling-selection model")
    output = config["paths"]["reports"] / "audit" / "training_smoke" if smoke else config["paths"]["experiment"] / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_manifest.json").exists():
        from utils import read_json
        previous = read_json(output / "training_manifest.json")
        if smoke and previous.get("fingerprint") == audit_fingerprint(config):
            if all(sha256(config["root"] / record["path"]) == record["sha256"]
                   for record in previous["models"].values()):
                LOGGER.info("reusing_verified_development_training_smoke")
                return previous
        raise RuntimeError("Completed training already exists; inspect it instead of overwriting")
    started = time.monotonic()
    scores = []
    target = frame.energy_per_atom_ev.to_numpy()
    groups = frame.reduced_formula.to_numpy()
    for pooling, x in arrays.items():
        for name in enabled:
            for index, params in enumerate(candidates[name]):
                for fold in sorted(frame.fold.unique()):
                    val = frame.fold.eq(fold).to_numpy()
                    if not val.any() or val.all():
                        raise ValueError("Empty training/validation composition fold")
                    estimator = fit_predictor(name, x[~val], target[~val], groups[~val], params, settings)
                    scores.append({"pooling": pooling, "model": name, "candidate": index,
                                   "fold": int(fold), "mae_ev_atom": float(mean_absolute_error(target[val], predict(estimator, x[val]))),
                                   "train_mae_ev_atom": float(mean_absolute_error(target[~val], predict(estimator, x[~val]))),
                                   "validation_samples": int(val.sum())})
    scores_frame = pd.DataFrame(scores)
    grouped = scores_frame.groupby(["pooling", "model", "candidate"]).mae_ev_atom.mean()
    primary = grouped.xs("mlp", level="model")
    common_pooling = primary.idxmin()[0]
    best = {}
    artifacts = {}
    selected_candidates = {}
    for name in enabled:
        index = int(grouped.loc[(common_pooling, name)].idxmin())
        selected_candidates[name] = index
        params = candidates[name][index]
        estimator = fit_predictor(name, arrays[common_pooling], target, groups, params, settings)
        model_path = output / f"{name}.joblib"
        joblib.dump(estimator, model_path)
        artifacts[name] = {"path": str(model_path.relative_to(config["root"])), "sha256": sha256(model_path)}
        best[name] = params
        if name == "mlp":
            pd.DataFrame(estimator["curves"]).to_csv(output / "mlp_training_curves.csv", index=False)
            torch.save(estimator["early_stopping_state"], output / "mlp_best_inner_validation.pt")
    scores_frame["selected_for_final"] = scores_frame.apply(
        lambda row: row.pooling == common_pooling and row.candidate == selected_candidates[row.model], axis=1)
    scores_frame.to_csv(output / "cross_validation_results.csv", index=False)
    best["common_pooling"] = common_pooling
    best["pooling_selection"] = "small MLP development grouped CV MAE; shared by all three predictors"
    write_json(output / "best_hyperparameters.json", best)
    manifest = {"fingerprint": audit_fingerprint(config), "common_pooling": common_pooling,
                "models": artifacts, "development_samples": len(frame), "test_labels_used": False,
                "encoder_trainable_parameters": 0, "seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "note": "CV is model selection; final independent test evaluates all predeclared models. No model winner selected by test."}
    write_json(output / "training_manifest.json", manifest)
    LOGGER.info("training_complete common_pooling=%s development_samples=%d", common_pooling, len(frame))
    return manifest


def main() -> None:
    """Run CLI training, with --smoke limiting development-only checks."""
    parser = cli(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), args.smoke)


if __name__ == "__main__":
    main()
