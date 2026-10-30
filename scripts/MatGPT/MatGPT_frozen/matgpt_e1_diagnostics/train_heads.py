#!/usr/bin/env python3
"""Entrena cabezas MatGPT congeladas para E1.1 o comparaciones controladas E1.2."""

from __future__ import annotations

import argparse
import logging
import platform
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, E1_EMBED, E1_REPORT, DiagnosticError, energy_metrics, load_inputs, log, logger_for, sha256_file, with_prediction, json_write

LOGGER = logger_for("matgpt_e1_diagnostics")


class Head(nn.Module):
    """MLP pequeña cuya entrada son estados ocultos congelados de dimensión 768."""

    def __init__(self, hidden_dim: int) -> None:
        """Crea una cabeza Linear-GELU-Dropout-Linear.

        Args:
            hidden_dim: Ancho oculto permitido (8 o 32 unidades).

        Returns:
            None; inicializa los parámetros entrenables de la cabeza.
        """
        super().__init__()
        if hidden_dim not in {8, 32}:
            raise DiagnosticError("Solo se permiten MLP8 y MLP32 para este diagnóstico.")
        self.layers = nn.Sequential(nn.Linear(768, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predice una energía normalizada por estructura.

        Args:
            values: Tensor de embeddings con forma `[batch, 768]`.

        Returns:
            Tensor de valores escalares con forma `[batch]`.
        """
        return self.layers(values).squeeze(-1)


def seed_all(seed: int) -> None:
    """Fija generadores pseudoaleatorios para una ejecución.

    Args:
        seed: Semilla no negativa compartida por Python, NumPy y PyTorch.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def composition_batches(frame: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator, count: int) -> list[np.ndarray]:
    """Baraja fórmulas de train y agrupa todos sus polimorfos disponibles.

    Args:
        frame: Dataset con una fila por estructura y columna `composition`.
        indices: Índices globales de estructuras que pertenecen a train.
        rng: Generador NumPy inicializado con la semilla de la ejecución.
        count: Número máximo de composiciones completas por minilote.

    Returns:
        Lista de índices globales, sin dividir polimorfos de una composición.
    """
    groups = [np.asarray(group.index, dtype=int) for _, group in frame.loc[indices].groupby("composition", sort=True)]
    order = rng.permutation(len(groups))
    return [np.concatenate([groups[i] for i in order[start:start + count]]) for start in range(0, len(order), count)]


def pair_loss(prediction: torch.Tensor, energies: torch.Tensor, formulas: list[str], pair_scale: float) -> torch.Tensor:
    """Calcula Huber de diferencias intra-composición, promediada por grupo.

    Args:
        prediction: Energías predichas físicas con gradiente activo.
        energies: Energías DFT físicas del mismo minilote.
        formulas: Composición por fila, alineada con los tensores.
        pair_scale: Escala RMS train-only en eV/átomo, con mínimo positivo.

    Returns:
        Tensor escalar diferenciable; pares de composiciones distintas nunca se comparan.
    """
    group_losses: list[torch.Tensor] = []
    for formula in sorted(set(formulas)):
        positions = [i for i, value in enumerate(formulas) if value == formula]
        if len(positions) < 2:
            continue
        pred_group, true_group = prediction[positions], energies[positions]
        i, j = torch.triu_indices(len(positions), len(positions), offset=1, device=prediction.device)
        residual = ((pred_group[i] - pred_group[j]) - (true_group[i] - true_group[j])) / pair_scale
        group_losses.append(nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0))
    if not group_losses:
        return prediction.sum() * 0.0
    return torch.stack(group_losses).mean()


def objective(prediction: torch.Tensor, energy: torch.Tensor, formulas: list[str], target_scale: float, pair_scale: float, weight: float) -> torch.Tensor:
    """Suma términos Huber absoluto y pairwise con escalas train-only.

    Args:
        prediction: Energía predicha en eV/átomo.
        energy: Etiqueta DFT en eV/átomo.
        formulas: Composición de cada fila del lote.
        target_scale: Desviación estándar de energía train-only.
        pair_scale: RMS de diferencias DFT intracomposición train-only.
        weight: Lambda no negativo que pondera el término pairwise.

    Returns:
        Tensor escalar diferenciable para backpropagation.
    """
    absolute = nn.functional.smooth_l1_loss((prediction - energy) / target_scale, torch.zeros_like(energy), beta=1.0)
    return absolute + weight * pair_loss(prediction, energy, formulas, pair_scale)


def run_mlp(data: pd.DataFrame, vectors: np.ndarray, output: Path, name: str, width: int, weight: float,
            seed: int, max_epochs: int, patience: int, compositions_per_batch: int, device: torch.device,
            dataset_hash: str, embedding_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Entrena una MLP con lotes agrupados por composición y solo valida en val.

    Args:
        data: Dataset histórico con energía física y los tres splits.
        vectors: Embeddings congelados alineados por estructura.
        output: Directorio donde se guarda el punto de control seleccionado.
        name: Identificador único de configuración y pooling.
        width: Ancho oculto, 8 o 32.
        weight: Lambda no negativo de la pérdida pairwise.
        seed: Semilla de inicialización y mezcla de fórmulas.
        max_epochs: Máximo de épocas de entrenamiento.
        patience: Épocas sin mejora de validation antes de detenerse.
        compositions_per_batch: Cantidad máxima de fórmulas agrupadas por paso.
        device: Dispositivo PyTorch validado por la CLI.
        dataset_hash: Hash del dataset usado para este entrenamiento.
        embedding_hash: Hash del caché usado para este entrenamiento.

    Returns:
        Historial por época, filas train/val y predicciones individuales train/val.
    """
    seed_all(seed)
    train_ids = np.flatnonzero(data.split.astype(str).eq("train").to_numpy())
    val_ids = np.flatnonzero(data.split.astype(str).eq("val").to_numpy())
    y = data.energy_dft_ev_atom.to_numpy(dtype=np.float32)
    mean = float(y[train_ids].mean())
    target_scale = max(float(y[train_ids].std()), 1e-6)
    pair_values: list[float] = []
    train_data = data.iloc[train_ids]
    for _, group in train_data.groupby("composition", sort=True):
        values = group.energy_dft_ev_atom.to_numpy(float)
        pair_values.extend((values[i] - values[j] for i in range(len(values)) for j in range(i + 1, len(values))))
    pair_scale = max(float(np.sqrt(np.mean(np.square(pair_values)))) if pair_values else 0.0, 0.001)
    x_t = torch.from_numpy(vectors.copy()).to(device)
    y_t = torch.from_numpy(y).to(device)
    model = Head(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(seed)
    best_value, best_epoch, best_state, waited = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []
    formulas_all = data.composition.astype(str).tolist()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_ids in composition_batches(data, train_ids, rng, compositions_per_batch):
            batch_formulas = [formulas_all[i] for i in batch_ids]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_t[batch_ids]) * target_scale + mean
            loss = objective(prediction, y_t[batch_ids], batch_formulas, target_scale, pair_scale, weight)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_prediction = model(x_t[val_ids]) * target_scale + mean
            val_objective = float(objective(val_prediction, y_t[val_ids], [formulas_all[i] for i in val_ids], target_scale, pair_scale, weight).cpu())
            train_prediction = model(x_t[train_ids]).cpu().numpy() * target_scale + mean
            val_prediction_np = val_prediction.cpu().numpy()
        val_frame = data.iloc[val_ids].copy()
        val_frame["energy_pred_ev_atom"] = val_prediction_np
        val_stats, _, _ = energy_metrics(val_frame)
        train_frame = data.iloc[train_ids].copy()
        train_frame["energy_pred_ev_atom"] = train_prediction
        train_stats, _, _ = energy_metrics(train_frame)
        history.append({"configuration": name, "seed": seed, "epoch": epoch,
                        "train_objective_normalized": float(np.mean(train_losses)), "validation_objective_normalized": val_objective,
                        "train_mae_ev_atom": train_stats["energy_mae_ev_atom"], "validation_mae_ev_atom": val_stats["energy_mae_ev_atom"],
                        "train_relative_mae_mev_atom": train_stats["relative_energy_mae_mev_atom"],
                        "validation_relative_mae_mev_atom": val_stats["relative_energy_mae_mev_atom"]})
        if val_objective < best_value - 1e-12:
            best_value, best_epoch, waited = val_objective, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if waited >= patience:
            break
    if best_state is None:
        raise DiagnosticError(f"No se seleccionó punto de control de validación: {name}, semilla {seed}.")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        selected_train = model(x_t[train_ids]).cpu().numpy() * target_scale + mean
        selected_val = model(x_t[val_ids]).cpu().numpy() * target_scale + mean
    payload = {"state_dict": best_state, "hidden_dim": width, "target_mean_train_ev_atom": mean,
               "target_scale_train_ev_atom": target_scale, "pair_scale_train_ev_atom": pair_scale,
               "pairwise_lambda": weight, "seed": seed, "best_epoch": best_epoch,
               "configuration": name, "dataset_sha256": dataset_hash, "embedding_cache_sha256": embedding_hash,
               "backbone_frozen": True, "backbone_sha256": "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
               "training_config": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                   "max_epochs": max_epochs, "early_stopping_patience": patience,
                   "batch_sampler": "grouped_by_composition", "compositions_per_batch": compositions_per_batch,
                   "loss": "Huber absoluta normalizada + lambda * Huber de pares intra-composición normalizados",
                   "pairwise_lambda": weight, "seed": seed, "device": str(device)}}
    torch.save(payload, output / "checkpoints" / f"{name}_seed{seed}.pt")
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for split, ids, values in (("train", train_ids, selected_train), ("val", val_ids, selected_val)):
        local = with_prediction(data.iloc[ids].copy(), values)
        local["experiment"] = output.name
        local["configuration"] = name
        local["pooling"] = "last_token" if "last_token" in name else "masked_mean"
        local["head_type"] = f"mlp{width}"
        local["loss_type"] = "absolute_huber" if weight == 0 else "absolute_plus_pairwise_huber"
        local["pairwise_lambda"] = weight
        local["seed"] = seed
        prediction_rows.append(local)
        stats, per_formula, _ = energy_metrics(local)
        metric_rows.append({"experiment": output.name, "configuration": name, "head_type": f"mlp{width}",
                            "pooling": "last_token" if "last" in name else "masked_mean", "loss_type": "absolute_huber" if weight == 0 else "absolute_plus_pairwise_huber",
                            "pairwise_lambda": weight, "seed": seed, "split": split, "best_epoch": best_epoch,
                            **stats, "n_families": int(local.polymorph_family.nunique()),
                            "mae_by_family": local.assign(abs_error=(local.energy_pred_ev_atom-local.energy_dft_ev_atom).abs()).groupby("polymorph_family").abs_error.mean().to_dict(),
                            "mean_dft_range_mev_atom": float(per_formula.dft_range_mev_atom.mean()),
                            "mean_predicted_range_mev_atom": float(per_formula.predicted_range_mev_atom.mean())})
    return history, metric_rows, pd.concat(prediction_rows, ignore_index=True)


def run_ridge(data: pd.DataFrame, vectors: np.ndarray, output: Path, alphas: list[float], pooling: str = "last_token", experiment: str = "e1_1", dataset_hash: str | None = None, embedding_hash: str | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Evalúa Ridge y el baseline constante, ajustando escala solo en train.

    Args:
        data: Dataset con energías y partición histórica.
        vectors: Matriz de embeddings alineada con el dataset.
        output: Directorio de checkpoints y predicciones.
        alphas: Valores Ridge positivos a evaluar.
        pooling: Nombre del caché de entrada.
        experiment: Fase que determina si se incluye el baseline constante.
        dataset_hash: SHA-256 del dataset de origen.
        embedding_hash: SHA-256 del caché de vectores.

    Returns:
        Filas de métricas y predicciones individuales train/validation.
    """
    train = data.split.astype(str).eq("train").to_numpy()
    val = data.split.astype(str).eq("val").to_numpy()
    y = data.energy_dft_ev_atom.to_numpy(float)
    scaler = StandardScaler().fit(vectors[train])
    xtrain, xval = scaler.transform(vectors[train]), scaler.transform(vectors[val])
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    if experiment == "e1_1":
        train_mean = float(y[train].mean())
        for split, mask in (("train", train), ("val", val)):
            local = with_prediction(data.loc[mask].copy(), np.full(int(mask.sum()), train_mean, dtype=float))
            name = f"{experiment}_{pooling}_constant_train_mean"
            predictions.append(local.assign(configuration=name, pooling=pooling, head_type="constant", loss_type="constant_train_mean",
                                             pairwise_lambda=0.0, seed="not_applicable", experiment=output.name))
            stats, formulas, _ = energy_metrics(local)
            rows.append({"experiment": output.name, "configuration": name, "head_type": "constant", "pooling": pooling,
                         "loss_type": "constant_train_mean", "pairwise_lambda": 0.0, "seed": "not_applicable", "split": split,
                         "best_epoch": None, **stats, "mae_by_family": local.assign(abs_error=(local.energy_pred_ev_atom-local.energy_dft_ev_atom).abs()).groupby("polymorph_family").abs_error.mean().to_dict(),
                         "mean_dft_range_mev_atom": float(formulas.dft_range_mev_atom.mean()),
                         "mean_predicted_range_mev_atom": float(formulas.predicted_range_mev_atom.mean())})
    for alpha in alphas:
        # SVD evita formar XᵀX, que puede amplificar el mal condicionamiento
        # de embeddings estandarizados en un conjunto pequeño (n < p).
        model = Ridge(alpha=alpha, solver="svd").fit(xtrain, y[train])
        name = f"{experiment}_{pooling}_ridge_a{alpha:g}"
        for split, mask, x in (("train", train, xtrain), ("val", val, xval)):
            local = data.loc[mask].copy()
            local = with_prediction(local, model.predict(x))
            stats, formulas, _ = energy_metrics(local)
            predictions.append(local.assign(configuration=name, pooling=pooling, head_type="ridge",
                                             loss_type="squared_error", pairwise_lambda=0.0, seed="not_applicable", experiment=output.name))
            rows.append({"experiment": output.name, "configuration": name, "head_type": "ridge",
                         "pooling": pooling, "loss_type": "squared_error", "ridge_alpha": alpha,
                         "seed": "not_applicable", "split": split, "best_epoch": None, **stats,
                         "mae_by_family": local.assign(abs_error=(local.energy_pred_ev_atom-local.energy_dft_ev_atom).abs()).groupby("polymorph_family").abs_error.mean().to_dict(),
                         "mean_dft_range_mev_atom": float(formulas.dft_range_mev_atom.mean()),
                         "mean_predicted_range_mev_atom": float(formulas.predicted_range_mev_atom.mean())})
        import joblib
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler, "alpha": alpha, "pooling": pooling,
                     "dataset_sha256": dataset_hash, "embedding_cache_sha256": embedding_hash,
                     "backbone_sha256": "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
                    "training_config": {"model": "sklearn.linear_model.Ridge", "solver": "svd", "alpha": alpha,
                         "feature_scaling": "StandardScaler fit on train only", "target_scaling": "none"}},
                    output / "checkpoints" / f"{name}.joblib")
    return rows, pd.concat(predictions, ignore_index=True)


def main() -> None:
    """Valida argumentos y ejecuta una cuadrícula E1.1 o comparación E1.2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("e1_1", "e1_2"), default="e1_1")
    parser.add_argument("--dataset", type=Path, default=E1_REPORT / "dataset.csv")
    parser.add_argument("--embeddings", type=Path, default=E1_EMBED / "matgpt_uncond_last_token.npz")
    parser.add_argument("--metadata", type=Path, default=E1_EMBED / "embedding_metadata.csv")
    parser.add_argument("--audit", type=Path, default=E1_EMBED / "embedding_audit.json")
    parser.add_argument("--pooling", choices=("last_token", "masked_mean"), default="last_token")
    parser.add_argument("--head", choices=("ridge", "mlp8", "mlp32"), default=None, help="En E1.2, cabeza fijada tras revisar validation.")
    parser.add_argument("--pairwise-lambda", type=float, default=None, help="En E1.2, lambda fijado antes de comparar poolings; Ridge requiere 0.")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Alpha Ridge seleccionado usando validation.")
    parser.add_argument("--ridge-alphas", default="0.1,1,10,100")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--compositions-per-batch", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.max_epochs < 1 or args.patience < 1 or args.compositions_per_batch < 1:
            raise DiagnosticError("Épocas, paciencia y composiciones por lote deben ser positivos.")
        data, vectors, audit = load_inputs(args.dataset, args.embeddings, args.metadata, args.audit)
        out = args.out_dir or ALT / "reports" / ("matgpt_e1_1" if args.experiment == "e1_1" else f"matgpt_e1_2_{args.pooling}")
        if out.exists() and any(out.iterdir()):
            raise DiagnosticError(f"Ya hay resultados en {out}; conserva el histórico y especifica --out-dir nuevo.")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise DiagnosticError("Se solicitó CUDA, pero no está disponible.")
        if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise DiagnosticError("Se solicitó MPS, pero no está disponible.")
        device = torch.device(device_name)
        output = out.resolve()
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("checkpoints").mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name("config.yaml"), output / "config.yaml")
        metric_rows: list[dict[str, Any]] = []
        pred_frames: list[pd.DataFrame] = []
        histories: list[dict[str, Any]] = []
        if args.experiment == "e1_1":
            configs = [(width, weight) for width in (8, 32) for weight in (0.0, 0.1, 0.3, 1.0)]
            alphas = [float(x) for x in args.ridge_alphas.split(",")]
            if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
                raise DiagnosticError("Ridge alpha debe ser una lista no vacía de números finitos positivos.")
            ridge_rows, ridge_predictions = run_ridge(data, vectors, output, alphas, args.pooling, args.experiment, sha256_file(args.dataset), sha256_file(args.embeddings))
            metric_rows.extend(ridge_rows)
            pred_frames.append(ridge_predictions)
        else:
            if args.head is None or args.pairwise_lambda is None or args.pairwise_lambda < 0:
                raise DiagnosticError("E1.2 requiere --head y --pairwise-lambda preseleccionados con validation.")
            if args.head == "ridge" and (args.pairwise_lambda != 0 or args.ridge_alpha is None or args.ridge_alpha <= 0):
                raise DiagnosticError("Ridge en E1.2 requiere lambda=0 y --ridge-alpha positivo, preseleccionado con validation.")
            if args.pooling == "last_token" and audit.get("pooling", "").lower().find("end") < 0:
                raise DiagnosticError("El caché seleccionado no documenta last-token pooling.")
            if args.pooling == "masked_mean" and "mean" not in audit.get("pooling", "").lower():
                raise DiagnosticError("El caché seleccionado no documenta masked mean pooling.")
            configs = [] if args.head == "ridge" else [(int(args.head.removeprefix("mlp")), float(args.pairwise_lambda))]
            if args.head == "ridge":
                ridge_rows, ridge_predictions = run_ridge(data, vectors, output, [float(args.ridge_alpha)], args.pooling, args.experiment, sha256_file(args.dataset), sha256_file(args.embeddings))
                metric_rows.extend(ridge_rows)
                pred_frames.append(ridge_predictions)
        for width, weight in configs:
            for seed in [int(value) for value in args.seeds.split(",")]:
                name = f"{args.experiment}_{args.pooling}_mlp{width}_lambda{weight:g}"
                history, rows, predictions = run_mlp(data, vectors, output, name, width, weight, seed,
                    args.max_epochs, args.patience, args.compositions_per_batch, device,
                    sha256_file(args.dataset), sha256_file(args.embeddings))
                metric_rows.extend(rows)
                pred_frames.append(predictions)
                histories.extend(history)
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(output / "metrics.csv", index=False)
        aggregate_keys = [key for key in ("configuration", "head_type", "pooling", "loss_type", "pairwise_lambda", "ridge_alpha", "split") if key in metrics]
        aggregate_metrics = [key for key in ("energy_mae_ev_atom", "energy_rmse_ev_atom", "relative_energy_mae_mev_atom",
            "ground_state_accuracy", "pairwise_ordering_accuracy", "mean_dft_range_mev_atom", "mean_predicted_range_mev_atom",
            "mean_range_ratio", "mean_absolute_range_error_mev_atom") if key in metrics]
        metrics.groupby(aggregate_keys, dropna=False)[aggregate_metrics].agg(["mean", "std"]).to_csv(output / "metrics_by_configuration.csv")
        family_rows = []
        for metric_row in metric_rows:
            for family, value in metric_row.get("mae_by_family", {}).items():
                family_rows.append({key: metric_row.get(key) for key in ("experiment", "configuration", "head_type", "pooling", "loss_type", "pairwise_lambda", "seed", "split")} |
                                   {"polymorph_family": family, "mae_ev_atom": value})
        pd.DataFrame(family_rows).to_csv(output / "mae_by_family.csv", index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(output / "predictions_train_val.csv", index=False)
        pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
        manifest = {"experiment": args.experiment, "pooling": args.pooling, "selection_uses_test": False,
                    "test_labels_read_for_metrics": False, "device": str(device), "dataset_sha256": sha256_file(args.dataset),
                    "embedding_sha256": sha256_file(args.embeddings), "backbone_sha256": audit["checkpoint_sha256"],
                    "split_counts": data.groupby("split").agg(rows=("structure_id", "size"), compositions=("composition", "nunique")).to_dict("index"),
                    "seeds": [int(value) for value in args.seeds.split(",")],
                    "loss_scales": "target std and RMS within-composition DFT pair differences calculated from train only; floor 0.001 eV/atom",
                    "early_stopping": "minimum validation normalized training objective; validation labels only",
                    "pareto_selection": "review absolute MAE, relative-energy MAE, ground-state accuracy, pairwise accuracy, and predicted ranges jointly; no scalar composite score",
                    "embedding_audit": str(args.audit),
                    "configured_hyperparameters": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                        "max_epochs": args.max_epochs, "early_stopping_patience": args.patience,
                        "compositions_per_batch": args.compositions_per_batch, "ridge_alphas": args.ridge_alphas,
                        "mlp_hidden_units": [8, 32] if args.experiment == "e1_1" else ([int(args.head.removeprefix("mlp"))] if args.head and args.head.startswith("mlp") else []),
                        "pairwise_lambda_grid": [0.0, 0.1, 0.3, 1.0] if args.experiment == "e1_1" else [args.pairwise_lambda],
                        "dropout": 0.1, "seeds": [int(value) for value in args.seeds.split(",")]},
                    "software_versions": {"python": platform.python_version(), "pytorch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                    "script_sha256": sha256_file(Path(__file__)),
                    "common_helpers_sha256": sha256_file(Path(__file__).with_name("common.py")),
                    "config_sha256": sha256_file(Path(__file__).with_name("config.yaml"))}
        json_write(output / "run_manifest.json", manifest)
        log(LOGGER, logging.INFO, "training_complete", experiment=args.experiment, configs=len(metrics.configuration.unique()), rows=len(metrics), device=str(device), test_evaluated=False)
    except (DiagnosticError, OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "training_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
