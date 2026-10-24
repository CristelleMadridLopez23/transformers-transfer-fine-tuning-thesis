#!/usr/bin/env python3
"""Entrena configuraciones R2/R3 con MatGPT congelado y folds por composición."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA,
    FAMILIES, FOLDS, REPORT_DIR, SEEDS, ExperimentError, load_dev, load_folds, log, sha256_file,
    split_indices, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.losses import two_component_smooth_l1
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.metrics import triplet_metrics
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.two_head_predictor import TwoHeadPredictor, trainable_parameter_count

SPECS: dict[str, dict[str, Any]] = {
    "R2-SMALL4": {"width": 4, "dropout": 0.1, "weight_decay": 1e-4, "patience": 20, "family": "R2"},
    "R2-REF8": {"width": 8, "dropout": 0.1, "weight_decay": 1e-4, "patience": 20, "family": "R2-R3-BASE"},
    "R3-DROP30": {"width": 8, "dropout": 0.3, "weight_decay": 1e-4, "patience": 20, "family": "R3"},
    "R3-WD1E2": {"width": 8, "dropout": 0.1, "weight_decay": 1e-2, "patience": 20, "family": "R3"},
    "R3-EARLY8": {"width": 8, "dropout": 0.1, "weight_decay": 1e-4, "patience": 8, "family": "R3"},
}


def triplet_indices(data: pd.DataFrame, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve índices [composición,3] en orden de familia y energías.

    Args:
        data: Tabla de desarrollo ordenada.
        row_indices: Índices del subconjunto.
    Returns:
        IDs de fila y targets eV/átomo en orden canónico.
    """
    rank = {family: i for i, family in enumerate(FAMILIES)}
    subset = data.iloc[row_indices].assign(_rank=data.iloc[row_indices].polymorph_family.map(rank))
    subset = subset.sort_values(["composition_id", "_rank"])
    ids = subset.index.to_numpy().reshape(-1, 3)
    targets = data.iloc[ids.reshape(-1)].E_DFT_eV_atom.to_numpy(np.float32).reshape(-1, 3)
    return ids, targets


def run_one(model_name: str, fold: int, seed: int, max_epochs: int = 200, device_name: str = "cpu") -> dict[str, Any]:
    """Train one heads-only run, save inner-selected checkpoint and single outer evaluation."""
    if model_name not in SPECS or fold not in range(5) or seed not in SEEDS or max_epochs < 1:
        raise ExperimentError("Especificación/fold/semilla/épocas inválida.")
    data, folds = load_dev(), load_folds(load_dev())
    cache_path = REPORT_DIR / "features_cache" / "layer_features.npz"
    if not cache_path.is_file():
        raise ExperimentError("Falta el cache congelado R4.")
    cache = np.load(cache_path, allow_pickle=False)
    if cache["structure_id"].astype(str).tolist() != data.structure_id.tolist():
        raise ExperimentError("Los IDs del cache y dataset no están alineados.")
    h_end = np.array(cache["FINAL_END"], dtype=np.float32, copy=True)
    h_mean = np.array(cache["FINAL_MEAN"], dtype=np.float32, copy=True)
    cache.close()
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ExperimentError("El checkpoint E0 cambió antes del entrenamiento.")
    inner_train, inner_val, outer = split_indices(data, folds, fold)
    out = REPORT_DIR / ("R2_head_capacity" if model_name.startswith("R2") else "R3_regularization")
    run_dir = out / "runs" / model_name / f"fold_{fold}_seed_{seed}"
    pred_path, status_path = run_dir / "predictions.csv", run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("status") == "PASS" and pred_path.is_file() and (run_dir / "head.pt").is_file():
            return {"status": "PASS", "reused": True, "path": str(run_dir)}
        raise ExperimentError(f"Corrida incompleta; se preserva para auditoría: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = SPECS[model_name]
    run_manifest = {"status": "PREPARED", "model": model_name, "fold": fold, "seed": seed, **config,
        "dataset_sha256": sha256_file(DATASET), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "fold_manifest_sha256": sha256_file(FOLDS), "feature_cache_sha256": sha256_file(cache_path),
        "trainer_sha256_at_run_start": sha256_file(Path(__file__)), "config_sha256": sha256_file(Path(__file__).with_name("config.yaml")),
        "inner_train_compositions": sorted(data.iloc[inner_train].composition_id.unique().tolist()),
        "inner_validation_compositions": sorted(data.iloc[inner_val].composition_id.unique().tolist()),
        "outer_validation_compositions": sorted(data.iloc[outer].composition_id.unique().tolist()),
        "test_used": False, "selection_metric": "inner-validation sum of normalized SmoothL1(mu)+SmoothL1(delta), same H2 rule"}
    write_json(run_dir / "run_manifest.json", run_manifest)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name)
    x_end = torch.tensor(np.array(h_end, copy=True), dtype=torch.float32, device=device)
    x_mean = torch.tensor(np.array(h_mean, copy=True), dtype=torch.float32, device=device)
    if torch.is_inference(x_end) or torch.is_inference(x_mean) or x_end.requires_grad or x_mean.requires_grad:
        raise ExperimentError("Features deben ser tensores ordinarios sin gradiente.")
    train_groups, y_train = triplet_indices(data, inner_train)
    val_groups, y_val = triplet_indices(data, inner_val)
    outer_groups, _ = triplet_indices(data, outer)
    mu_train = y_train.mean(axis=1)
    delta_train = y_train - mu_train[:, None]
    mu_mean = float(mu_train.mean())
    scale_mu = max(float(mu_train.std()), 1e-6)
    scale_delta = max(float(delta_train.std()), 1e-6)
    model = TwoHeadPredictor(width=int(config["width"]), dropout=float(config["dropout"])).to(device)
    backbone_before = hashlib.sha256(b"".join(t.numpy().tobytes() for t in (x_end.detach().cpu(), x_mean.detach().cpu()))).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_score, best_epoch, best_state, waited = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []

    def evaluate(group_ids: np.ndarray, targets: np.ndarray, epoch: int) -> tuple[float, dict[str, float]]:
        """Evalúa todas las composiciones inner-val con pesos de la época actual."""
        model.eval()
        total, comp_losses, struct_losses, mae_abs, mae_mu, mae_delta = [], [], [], [], [], []
        with torch.no_grad():
            for ids, truth_np in zip(group_ids, targets):
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                hcomp = x_mean[ids_t].mean(dim=0, keepdim=True)
                hstruct = x_end[ids_t].unsqueeze(0)
                mu_raw, delta_raw, _ = model(hcomp, hstruct)
                y_t = torch.tensor(truth_np, dtype=torch.float32, device=device)
                mu_true, delta_true = y_t.mean().reshape(1), (y_t - y_t.mean()).reshape(1, 3)
                lp, ld, objective = two_component_smooth_l1(mu_raw * scale_mu + mu_mean,
                    delta_raw * scale_delta, mu_true, delta_true, scale_mu, scale_delta, beta=1.0)
                total.append(float(objective)); comp_losses.append(float(lp)); struct_losses.append(float(ld))
                energy = (mu_raw * scale_mu + mu_mean) + delta_raw * scale_delta
                mae_abs.extend(torch.abs(energy - y_t).cpu().tolist())
                mae_mu.append(float(torch.abs((mu_raw.squeeze() * scale_mu + mu_mean) - mu_true.squeeze()).item()))
                mae_delta.extend(torch.abs(delta_raw.squeeze(0) * scale_delta - delta_true.squeeze(0)).cpu().tolist())
        return float(np.mean(total)), {"inner_val_loss_mu": float(np.mean(comp_losses)),
            "inner_val_loss_delta": float(np.mean(struct_losses)), "inner_val_MAE_absolute_mev_atom": float(np.mean(mae_abs) * 1000),
            "inner_val_MAE_mu_mev_atom": float(np.mean(mae_mu) * 1000),
            "inner_val_MAE_delta_mev_atom": float(np.mean(mae_delta) * 1000)}

    t0 = time.perf_counter()
    for epoch in range(1, min(max_epochs, 200) + 1):
        model.train()
        order = rng.permutation(len(train_groups))
        batch_losses = []
        grad_mu_norms, grad_delta_norms = [], []
        for start in range(0, len(order), 8):
            selected = order[start:start + 8]
            ids = train_groups[selected]
            y_batch = torch.tensor(y_train[selected], dtype=torch.float32, device=device)
            flat = torch.tensor(ids.reshape(-1), dtype=torch.long, device=device)
            h_comp = x_mean[flat].reshape(len(selected), 3, 768).mean(dim=1)
            h_struct = x_end[flat].reshape(len(selected), 3, 768)
            mu_raw, delta_raw, energy = model(h_comp, h_struct)
            true_mu = y_batch.mean(dim=1)
            true_delta = y_batch - true_mu[:, None]
            loss_mu, loss_delta, loss = two_component_smooth_l1(mu_raw * scale_mu + mu_mean,
                delta_raw * scale_delta, true_mu, true_delta, scale_mu, scale_delta)
            if not torch.isfinite(loss):
                raise ExperimentError(f"Loss NaN/Inf {model_name}/fold{fold}/seed{seed}/epoch{epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_mu_norms.append(float(sum(p.grad.norm().item() for p in model.composition.parameters() if p.grad is not None)))
            grad_delta_norms.append(float(sum(p.grad.norm().item() for p in model.residual.parameters() if p.grad is not None)))
            if min(grad_mu_norms[-1], grad_delta_norms[-1]) <= 0:
                raise ExperimentError("Gradiente ausente en una cabeza.")
            optimizer.step()
            batch_losses.append((float(loss_mu.detach()), float(loss_delta.detach())))
        val_score, val_stats = evaluate(val_groups, y_val, epoch)
        history.append({"model": model_name, "fold": fold, "seed": seed, "epoch": epoch,
            "train_loss_mu": float(np.mean(batch_losses, axis=0)[0]), "train_loss_delta": float(np.mean(batch_losses, axis=0)[1]),
            "inner_validation_objective": val_score, **val_stats, "scale_mu_ev_atom": scale_mu,
            "scale_delta_ev_atom": scale_delta, "target_mean_mu_ev_atom": mu_mean,
            "grad_norm_mu": float(np.mean(grad_mu_norms)), "grad_norm_delta": float(np.mean(grad_delta_norms)),
            "learning_rate": 0.001, "weight_decay": config["weight_decay"], "dropout": config["dropout"]})
        if val_score < best_score - 1e-12:
            best_score, best_epoch, waited = val_score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if waited >= int(config["patience"]):
            break
    if best_state is None:
        raise ExperimentError("No hubo checkpoint seleccionado por inner-validation.")
    model.load_state_dict(best_state, strict=True)
    torch.save({"model": model_name, "width": config["width"], "dropout": config["dropout"],
        "weight_decay": config["weight_decay"], "fold": fold, "seed": seed, "state_dict": best_state,
        "best_epoch": best_epoch, "best_inner_validation_objective": best_score,
        "scale_mu_ev_atom": scale_mu, "scale_delta_ev_atom": scale_delta, "target_mean_mu_ev_atom": mu_mean,
        "trainable_head_parameters": trainable_parameter_count(model), "backbone_updated": False}, run_dir / "head.pt")
    model.eval()
    pred_rows = []
    with torch.no_grad():
        for ids in outer_groups:
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            mu_raw, delta_raw, _ = model(x_mean[ids_t].mean(dim=0, keepdim=True), x_end[ids_t].unsqueeze(0))
            mu_value = float(mu_raw.squeeze().item() * scale_mu + mu_mean)
            delta_values = delta_raw.squeeze(0).cpu().numpy().astype(float) * scale_delta
            for local, ix in enumerate(ids):
                row = data.iloc[ix]
                pred_rows.append({"structure_id": row.structure_id, "composition_id": row.composition_id,
                    "polymorph_family": row.polymorph_family, "fold": fold, "seed": seed, "model": model_name,
                    "E_DFT_eV_atom": float(row.E_DFT_eV_atom), "E_pred_eV_atom": mu_value + delta_values[local],
                    "E_pred_mu_eV_atom": mu_value, "delta_pred_centered_eV_atom": delta_values[local],
                    "head_width": config["width"], "dropout": config["dropout"], "weight_decay": config["weight_decay"],
                    "best_epoch": best_epoch, "selected_inner_objective": best_score, "historical_split": row.split})
    predictions = pd.DataFrame(pred_rows)
    if len(predictions) != 42 or predictions.structure_id.nunique() != 42 or not np.isfinite(predictions.E_pred_eV_atom).all():
        raise ExperimentError("OOF externo de cabeza inválido.")
    # Materialized feature arrays are immutable inputs; verify their digest as a no-update sentinel.
    after = hashlib.sha256(b"".join(t.numpy().tobytes() for t in (x_end.detach().cpu(), x_mean.detach().cpu()))).hexdigest()
    if backbone_before != after:
        raise ExperimentError("Features/backbone cambiaron durante el entrenamiento.")
    predictions.to_csv(pred_path, index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    elapsed = time.perf_counter() - t0
    run_manifest.update({"status": "PASS", "best_epoch": best_epoch, "epochs_run": len(history),
        "training_seconds": elapsed, "trainable_parameters": trainable_parameter_count(model),
        "feature_input_sha256_before_after_match": True, "prediction_rows": len(predictions),
        "head_sha256": sha256_file(run_dir / "head.pt"), "checkpoint_sha256_after": sha256_file(CHECKPOINT)})
    write_json(run_dir / "run_manifest.json", run_manifest)
    write_json(status_path, {"status": "PASS", "best_epoch": best_epoch, "epochs_run": len(history),
        "training_seconds": elapsed, "test_evaluated": False})
    return {"status": "PASS", "reused": False, "path": str(run_dir), "rows": len(predictions), "seconds": elapsed}


def run_configs(models: tuple[str, ...], device: str = "cpu", folds_arg: tuple[int, ...] = tuple(range(5)),
                seeds_arg: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Ejecuta o reanuda todas las corridas solicitadas sin repetir éxitos."""
    results = []
    for model_name in models:
        if model_name not in SPECS:
            raise ExperimentError(f"Modelo no definido: {model_name}")
        for fold in folds_arg:
            for seed in seeds_arg:
                result = run_one(model_name, fold, seed, device_name=device)
                results.append({"model": model_name, "fold": fold, "seed": seed, **result})
                log("head_run_complete", model=model_name, fold=fold, seed=seed,
                    rows=result.get("rows", 42), seconds=result.get("seconds", 0), test_used=False)
    for model_name in models:
        root = REPORT_DIR / ("R2_head_capacity" if model_name.startswith("R2") else "R3_regularization")
        parts = sorted((root / "runs" / model_name).glob("fold_*_seed_*/predictions.csv"))
        if not parts:
            continue
        model_dir = root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.read_csv(path) for path in parts], ignore_index=True).to_csv(model_dir / "oof_predictions.csv", index=False)
        histories = sorted((root / "runs" / model_name).glob("fold_*_seed_*/history.csv"))
        pd.concat([pd.read_csv(path) for path in histories], ignore_index=True).to_csv(model_dir / "training_history.csv", index=False)
        manifest = {"status": "PASS" if len(parts) == 15 else "PARTIAL", "model": model_name,
            "completed_runs": len(parts), "expected_runs": 15, "folds": list(folds_arg),
            "seeds": list(seeds_arg), "prediction_rows": int(sum(len(pd.read_csv(path)) for path in parts)),
            "test_used": False, "trainer_sha256": sha256_file(Path(__file__))}
        write_json(model_dir / "run_manifest.json", manifest)
    return {"status": "PASS", "runs_requested": len(results), "results": results}


def main() -> None:
    """Valida argumentos y ejecuta las configuraciones R2/R3 seleccionadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    try:
        result = run_configs(tuple(args.models), args.device, tuple(args.folds), tuple(args.seeds))
        log("head_configurations_complete", status=result["status"], runs=result["runs_requested"], test_used=False)
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError) as exc:
        log("head_training_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
