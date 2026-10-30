#!/usr/bin/env python3
"""Train the E1 linear probe and MLP on frozen, cached MatGPT embeddings."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.common import ALT, DATASET, E1Error, REPORTS, configure_logging, sha256_file, verify_backbone_files, write_json

LOGGER = logging.getLogger("matgpt_e1")


class MatGPTRegressor(nn.Module):
    """Small MLP trained over frozen MatGPT embeddings."""

    def __init__(self, hidden_dim: int, hidden_units: int = 32, dropout: float = 0.1) -> None:
        """Initialize the specified Linear-GELU-Dropout-Linear head.

        Args:
            hidden_dim: Embedding dimension from checkpoint architecture.
            hidden_units: Hidden layer width.
            dropout: Dropout probability.
        """
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, hidden_units), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_units, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict normalized energy for each embedding.

        Args:
            embeddings: Tensor shaped ``[batch, hidden_dim]``.

        Returns:
            One scalar per row.
        """
        return self.regressor(embeddings).squeeze(-1)


class LinearProbe(nn.Module):
    """Linear baseline with a one-dimensional output per structure."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the linear mapping.

        Args:
            input_dim: Frozen embedding width.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict a normalized scalar energy.

        Args:
            embeddings: Tensor shaped ``[batch, input_dim]``.

        Returns:
            Tensor shaped ``[batch]``.
        """
        return self.linear(embeddings).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds.

    Args:
        seed: Reproducibility seed.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Calculate energy errors in eV/atom and meV/atom.

    Args:
        actual: Ground truth energies.
        predicted: Predicted energies in eV/atom.

    Returns:
        Metric dictionary with R2 omitted when undefined.
    """
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = None if len(actual) < 2 or denominator <= 0 else float(1.0 - np.sum(residual ** 2) / denominator)
    return {"mae_ev_per_atom": mae, "rmse_ev_per_atom": rmse, "mae_mev_per_atom": mae * 1000.0, "r2": r2}


def fit_head(
    kind: str, x: torch.Tensor, y: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
    config: dict[str, Any], device: torch.device, report_dir: Path, hidden_dim: int,
    y_mean: float, y_std: float, dataset_frame: pd.DataFrame, hashes: dict[str, str],
    dataset_hash: str, embedding_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train a head with validation-MAE early stopping, without test access.

    Args:
        kind: Either ``mlp`` or ``linear``.
        x: Training embeddings.
        y: Normalized training target.
        x_val: Validation embeddings.
        y_val: Normalized validation target.
        config: Training and head configuration.
        device: PyTorch training device.
        report_dir: Checkpoint destination.
        hidden_dim: Embedding width.
        y_mean: Train target mean in eV/atom.
        y_std: Train target standard deviation in eV/atom.
        dataset_frame: Metadata corresponding to embeddings.
        hashes: Verified backbone file hashes.
        dataset_hash: SHA-256 of the prepared dataset used for this run.
        embedding_hash: SHA-256 of the embedding cache used for this run.

    Returns:
        Best checkpoint payload and epoch history.
    """
    train_cfg = config["training"]
    if kind == "mlp":
        model: nn.Module = MatGPTRegressor(hidden_dim, int(config["regressor"]["hidden_dim"]), float(config["regressor"]["dropout"]))
        max_epochs = int(train_cfg["max_epochs"])
        patience = int(train_cfg["early_stopping_patience"])
    else:
        model = LinearProbe(hidden_dim)
        max_epochs = int(train_cfg["linear_probe_max_epochs"])
        patience = int(train_cfg["linear_probe_patience"])
    model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) == 0:
        raise E1Error(f"Cabeza {kind} sin parámetros entrenables.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(TensorDataset(x, y), batch_size=int(train_cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(int(train_cfg["seed"])))
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    wait = 0
    history: list[dict[str, Any]] = []
    train_targets_ev = y.cpu().numpy() * y_std + y_mean
    val_targets_ev = y_val.cpu().numpy() * y_std + y_mean
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            train_pred = model(x.to(device)).detach().cpu().numpy() * y_std + y_mean
            val_pred = model(x_val.to(device)).detach().cpu().numpy() * y_std + y_mean
            val_loss = float(loss_fn(model(x_val.to(device)), y_val.to(device)).detach().cpu())
        train_metric = metrics(train_targets_ev, train_pred)
        val_metric = metrics(val_targets_ev, val_pred)
        record = {"head": kind, "epoch": epoch, "train_loss_smooth_l1_normalized": float(np.mean(batch_losses)),
                  "val_loss_smooth_l1_normalized": val_loss,
                  "train_mae_ev_per_atom": train_metric["mae_ev_per_atom"], "val_mae_ev_per_atom": val_metric["mae_ev_per_atom"],
                  "train_rmse_ev_per_atom": train_metric["rmse_ev_per_atom"], "val_rmse_ev_per_atom": val_metric["rmse_ev_per_atom"]}
        history.append(record)
        if float(val_metric["mae_ev_per_atom"]) < best_mae - 1e-12:
            best_mae = float(val_metric["mae_ev_per_atom"])
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is None:
        raise E1Error(f"No se obtuvo checkpoint de validación para {kind}.")
    payload = {
        "head": kind, "state_dict": best_state, "input_dim": hidden_dim,
        "target_mean_train_ev_per_atom": y_mean, "target_std_train_ev_per_atom": y_std,
        "best_epoch": best_epoch, "best_val_mae_ev_per_atom": best_mae,
        "training_config": config, "train_ids": dataset_frame.loc[dataset_frame.split == "train", "structure_id"].astype(str).tolist(),
        "validation_ids": dataset_frame.loc[dataset_frame.split == "val", "structure_id"].astype(str).tolist(),
        "test_ids": dataset_frame.loc[dataset_frame.split == "test", "structure_id"].astype(str).tolist(),
        "backbone_checkpoint_sha256": hashes["checkpoint_sha256"], "backbone_model_info_sha256": hashes["model_info_sha256"],
        "dataset_sha256": dataset_hash, "embedding_cache_sha256": embedding_hash,
    }
    torch.save(payload, report_dir / f"best_{kind}.pt")
    LOGGER.info("head_trained", extra={"head": kind, "best_epoch": best_epoch, "best_val_mae_ev_per_atom": best_mae})
    return payload, history


def run(dataset_path: Path, embeddings_path: Path, metadata_path: Path, report_dir: Path, device_name: str) -> dict[str, Any]:
    """Train mean and linear baselines plus the MLP using train/validation only.

    Args:
        dataset_path: Prepared dataset CSV.
        embeddings_path: NPZ containing the frozen embeddings.
        metadata_path: CSV mapping embeddings to structures.
        report_dir: E1 output directory.
        device_name: cpu, cuda, mps, or auto.

    Returns:
        Summary of training and validation results.
    """
    for path in (dataset_path, embeddings_path, metadata_path):
        if not path.is_file():
            raise E1Error(f"Falta archivo de entrada E1: {path}")
    if any((report_dir / name).exists() for name in ("best_mlp.pt", "best_linear.pt", "training_history.csv", "training_summary.json")):
        raise E1Error(f"Ya existen resultados de entrenamiento en {report_dir}; usa un directorio nuevo.")
    config_path = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = verify_backbone_files()
    embedding_audit_path = embeddings_path.with_name("embedding_audit.json")
    if not embedding_audit_path.is_file():
        raise E1Error("Falta embedding_audit.json; no se puede validar el cache.")
    embedding_audit = __import__("json").loads(embedding_audit_path.read_text(encoding="utf-8"))
    if embedding_audit.get("checkpoint_sha256") != hashes["checkpoint_sha256"] or embedding_audit.get("dataset_sha256") != sha256_file(dataset_path):
        raise E1Error("Hashes del cache no coinciden con checkpoint/dataset actuales.")
    if embedding_audit.get("cache_sha256") != sha256_file(embeddings_path) or embedding_audit.get("metadata_sha256") != sha256_file(metadata_path):
        raise E1Error("El cache fue modificado después de la extracción.")
    script_dir = Path(__file__).resolve().parent
    dependency_hashes = {
        "extractor_script_sha256": sha256_file(script_dir / "extract_embeddings.py"),
        "common_helpers_sha256": sha256_file(script_dir / "common.py"),
        "official_model_source_sha256": sha256_file(ALT / "models" / "MatGPT-source" / "mat-concat" / "utils" / "models.py"),
    }
    if any(embedding_audit.get(key) != value for key, value in dependency_hashes.items()):
        raise E1Error("El código extractor, helpers o arquitectura cambió desde la creación del cache.")
    data = pd.read_csv(dataset_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if data["structure_id"].astype(str).tolist() != metadata["structure_id"].astype(str).tolist():
        raise E1Error("El orden de IDs del cache no corresponde exactamente al dataset.")
    with np.load(embeddings_path, allow_pickle=False) as cache:
        x_np = cache["embeddings"].astype(np.float32, copy=True)
    if x_np.shape != (len(data), int(embedding_audit["dimension"])) or not np.isfinite(x_np).all():
        raise E1Error("Cache de embeddings con forma incorrecta o valores no finitos.")
    y_np = data["energy_per_atom_ev"].to_numpy(dtype=np.float64)
    train_mask = data["split"].astype(str).eq("train").to_numpy()
    val_mask = data["split"].astype(str).eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise E1Error("Train y validation deben tener muestras.")
    y_mean = float(y_np[train_mask].mean())
    y_std = float(y_np[train_mask].std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 0:
        raise E1Error("La desviación estándar train-only del target no es válida.")
    seed = int(config["training"]["seed"])
    set_seed(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise E1Error("CUDA fue solicitado pero no está disponible.")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise E1Error("MPS fue solicitado pero no está disponible.")
    device = torch.device(device_name)
    x = torch.from_numpy(x_np[train_mask].copy())
    x_val = torch.from_numpy(x_np[val_mask].copy())
    y = torch.from_numpy(((y_np[train_mask] - y_mean) / y_std).astype(np.float32))
    y_val = torch.from_numpy(((y_np[val_mask] - y_mean) / y_std).astype(np.float32))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    dataset_hash = sha256_file(dataset_path)
    embedding_hash = sha256_file(embeddings_path)
    mean_value = y_mean
    rows = []
    for split, mask in (("train", train_mask), ("val", val_mask)):
        result = metrics(y_np[mask], np.full(int(mask.sum()), mean_value))
        rows.append({"head": "train_mean", "split": split, **result})
    mlp_payload, mlp_history = fit_head("mlp", x, y, x_val, y_val, config, device, report_dir, x_np.shape[1], y_mean, y_std, data, hashes, dataset_hash, embedding_hash)
    linear_payload, linear_history = fit_head("linear", x, y, x_val, y_val, config, device, report_dir, x_np.shape[1], y_mean, y_std, data, hashes, dataset_hash, embedding_hash)
    for head, payload, history in (("mlp", mlp_payload, mlp_history), ("linear", linear_payload, linear_history)):
        best_record = history[payload["best_epoch"] - 1]
        for split in ("train", "val"):
            rows.append({"head": head, "split": split,
                         "mae_ev_per_atom": best_record[f"{split}_mae_ev_per_atom"],
                         "rmse_ev_per_atom": best_record[f"{split}_rmse_ev_per_atom"],
                         "mae_mev_per_atom": best_record[f"{split}_mae_ev_per_atom"] * 1000,
                         "r2": None})
    pd.DataFrame(mlp_history + linear_history).to_csv(report_dir / "training_history.csv", index=False)
    pd.DataFrame(rows).to_csv(report_dir / "baseline_metrics_train_val.csv", index=False)
    summary = {"status": "TRAINED_VALIDATION_ONLY", "device": str(device), "embedding_dim": int(x_np.shape[1]),
               "train_target_mean_ev_per_atom": y_mean, "train_target_std_ev_per_atom": y_std,
               "trainable_mlp_parameters": int(x_np.shape[1] * int(config["regressor"]["hidden_dim"]) + int(config["regressor"]["hidden_dim"]) + int(config["regressor"]["hidden_dim"]) + 1),
               "mlp_best_epoch": mlp_payload["best_epoch"], "mlp_best_val_mae_ev_per_atom": mlp_payload["best_val_mae_ev_per_atom"],
               "linear_best_epoch": linear_payload["best_epoch"], "linear_best_val_mae_ev_per_atom": linear_payload["best_val_mae_ev_per_atom"],
               "trainable_backbone_parameters": 0, "test_evaluated": False,
               "checkpoint_sha256": hashes["checkpoint_sha256"], "dataset_sha256": sha256_file(dataset_path),
               "embedding_cache_sha256": sha256_file(embeddings_path)}
    write_json(report_dir / "training_summary.json", summary)
    LOGGER.info("training_complete", extra={"mlp_best_epoch": mlp_payload["best_epoch"], "device": str(device), "test_evaluated": False})
    return summary


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments.

    Returns:
        CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--embeddings", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "matgpt_uncond_last_token.npz")
    parser.add_argument("--metadata", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "embedding_metadata.csv")
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    """Run training and report errors in structured logs."""
    logger = configure_logging()
    args = parse_args()
    try:
        run(args.dataset, args.embeddings, args.metadata, args.report_dir, args.device)
    except (E1Error, OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("training_failed", extra={"error": str(exc)})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
