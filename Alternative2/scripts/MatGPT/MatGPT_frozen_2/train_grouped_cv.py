#!/usr/bin/env python3
"""Entrena H0/H1/H2 sobre vectores congelados y folds exactos de E1.3."""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import ALT, CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA, FAMILIES, FOLDS, MODELS, REPORT_DIR, SEEDS, ExperimentError, group_inner_indices, load_development, load_fold_manifest, log, sha256_file, verify_provenance, write_json


class MLP8(nn.Module):
    """Cabeza MLP8 histórica con GELU y dropout 0.1."""
    def __init__(self) -> None:
        """Crea una MLP 768→8→1."""
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve una salida escalar por vector."""
        return self.layers(x).squeeze(-1)


class TwoHeads(nn.Module):
    """Cabeza de media composicional más cabeza residual centrada."""
    def __init__(self) -> None:
        """Inicializa dos MLP8 independientes."""
        super().__init__()
        self.composition = MLP8()
        self.residual = MLP8()

    def forward(self, xcomp: torch.Tensor, xstruct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula mu, residuos que suman cero y energía reconstruida."""
        mu = self.composition(xcomp)
        raw_residual = self.residual(xstruct)
        delta = raw_residual - raw_residual.mean(dim=1, keepdim=True)
        return mu, delta, mu[:, None] + delta


def groups_for(data: pd.DataFrame, indices: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Devuelve composiciones completas como grupos de índices globales."""
    part = data.iloc[indices]
    return [(str(name), np.asarray(indices[part.index.get_indexer(group.index)], dtype=int))
            for name, group in part.groupby("composition_id", sort=True)]


def make_batches(data: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Baraja composiciones completas y crea minibatches de 8 grupos."""
    groups = [np.asarray(group.index, dtype=int) for _, group in data.iloc[indices].groupby("composition_id", sort=True)]
    order = rng.permutation(len(groups))
    return [np.stack([groups[ix] for ix in order[start:start + 8]]) for start in range(0, len(order), 8)]


def tensor_groups(data: pd.DataFrame, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construye matriz [grupos, 3] con orden familiar canónico y etiquetas."""
    d = data.iloc[ids].copy()
    fam_rank = {family: index for index, family in enumerate(FAMILIES)}
    d["_family_rank"] = d.polymorph_family.map(fam_rank)
    d = d.sort_values(["composition_id", "_family_rank"])
    group_ids = d.index.to_numpy().reshape(-1, 3)
    if any(set(data.iloc[ix].polymorph_family) != set(FAMILIES) for ix in group_ids):
        raise ExperimentError("Batches requieren tripletes completos de familias.")
    return group_ids, data.iloc[group_ids.reshape(-1)].E_DFT_eV_atom.to_numpy(np.float32).reshape(-1, 3)


def pair_scale(y: np.ndarray, group_ids: np.ndarray, data: pd.DataFrame) -> float:
    """Calcula RMS de diferencias DFT entre pares, con piso E1.3 1 meV/átomo."""
    diffs = []
    for ids in group_ids:
        vals = data.iloc[ids].E_DFT_eV_atom.to_numpy(float)
        diffs.extend(vals[i] - vals[j] for i in range(3) for j in range(i + 1, 3))
    return max(float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0, 0.001)


def run_one(data: pd.DataFrame, features: dict[str, np.ndarray], folds: pd.DataFrame, model_name: str,
            fold: int, seed: int, run_dir: Path, max_epochs: int = 200) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Ajusta una cabeza en inner-train, selecciona con inner-val y predice outer una vez."""
    start_time = time.perf_counter()
    train_ids, val_ids, outer_ids = group_inner_indices(data, folds, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "status.json").exists():
        status = __import__("json").loads((run_dir / "status.json").read_text())
        if status.get("status") == "PASS" and (run_dir / "predictions.csv").is_file() and (run_dir / "head.pt").is_file():
            return pd.read_csv(run_dir / "predictions.csv"), pd.read_csv(run_dir / "history.csv").to_dict("records")
        raise ExperimentError(f"Corrida existente incompleta; conservar y revisar: {run_dir}")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ExperimentError("El checkpoint cambió antes del training run.")
    run_provenance = {"status": "PREPARED", "model": model_name, "fold": fold, "seed": seed,
        "dataset_sha256": sha256_file(DATASET), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "fold_manifest_sha256": sha256_file(FOLDS),
        "feature_cache_sha256": sha256_file(REPORT_DIR / "features" / "frozen_features_development.npz"),
        "training_code_sha256": sha256_file(Path(__file__)), "test_used": False,
        "train_groups": sorted(data.iloc[train_ids].composition_id.unique().tolist()),
        "inner_validation_groups": sorted(data.iloc[val_ids].composition_id.unique().tolist()),
        "outer_validation_groups": sorted(data.iloc[outer_ids].composition_id.unique().tolist())}
    write_json(run_dir / "run_provenance.json", run_provenance)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # npz se carga a memoria ordinaria: no son inference tensors.
    end_np = np.array(features["h_end"], dtype=np.float32, copy=True)
    mean_np = np.array(features["h_mean"], dtype=np.float32, copy=True)
    x_end, x_mean = torch.tensor(end_np), torch.tensor(mean_np)
    if torch.is_inference(x_end) or torch.is_inference(x_mean):
        raise ExperimentError("Features no deben conservar modo inference.")
    y_np = data.E_DFT_eV_atom.to_numpy(np.float32)
    y = torch.tensor(np.array(y_np, copy=True))
    train_groups, y_train_groups = tensor_groups(data, train_ids)
    val_groups, y_val_groups = tensor_groups(data, val_ids)
    outer_groups, _ = tensor_groups(data, outer_ids)
    train_values = y_np[train_ids]
    if model_name == "H0":
        target_mean = float(train_values.mean())
        target_scale = max(float(train_values.std()), 1e-6)
        scale_comp = target_scale
        scale_struct = float("nan")
        pscale = pair_scale(y_np, train_groups, data)
        model: nn.Module = MLP8()
        parameters = list(model.parameters())
    else:
        mu_train = y_train_groups.mean(axis=1)
        delta_train = y_train_groups - mu_train[:, None]
        target_mean = float(mu_train.mean())
        scale_comp = max(float(mu_train.std()), 1e-6)
        scale_struct = max(float(delta_train.std()), 1e-6)
        target_scale = 1.0
        pscale = float("nan")
        model = TwoHeads()
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, waited = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []

    def evaluate_groups(group_ids: np.ndarray, targets: np.ndarray, train_phase: bool) -> tuple[float, dict[str, float]]:
        """Evalúa las pérdidas mean/struct o E1.3 compuestas sobre tripletes completos."""
        terms: list[tuple[float, float]] = []
        for ids, target in zip(group_ids, targets):
            ix = torch.tensor(ids, dtype=torch.long)
            y_trip = torch.tensor(target, dtype=torch.float32)
            if model_name == "H0":
                prediction = model(x_end[ix]) * target_scale + target_mean
                abs_loss = nn.functional.smooth_l1_loss((prediction - y_trip) / target_scale,
                                                        torch.zeros_like(y_trip), beta=1.0, reduction="mean")
                pair_losses = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        residual = ((prediction[i] - prediction[j]) - (y_trip[i] - y_trip[j])) / pscale
                        pair_losses.append(nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0))
                pair_loss = torch.stack(pair_losses).mean()
                terms.append((float(abs_loss.detach()), float(pair_loss.detach())))
            else:
                if model_name == "H1":
                    comp_x, struct_x = x_end[ix].mean(dim=0, keepdim=True), x_end[ix].unsqueeze(0)
                else:
                    comp_x, struct_x = x_mean[ix].mean(dim=0, keepdim=True), x_end[ix].unsqueeze(0)
                mu_raw, delta_raw, _ = model(comp_x, struct_x)
                mu_pred = mu_raw.squeeze(0) * scale_comp + target_mean
                delta_pred = delta_raw.squeeze(0) * scale_struct
                mu_true = y_trip.mean()
                delta_true = y_trip - mu_true
                lcomp = nn.functional.smooth_l1_loss((mu_pred - mu_true) / scale_comp, torch.zeros_like(mu_pred), beta=1.0)
                lstruct = nn.functional.smooth_l1_loss((delta_pred - delta_true) / scale_struct, torch.zeros_like(delta_true), beta=1.0)
                terms.append((float(lcomp.detach()), float(lstruct.detach())))
        means = np.asarray(terms).mean(axis=0)
        return float(means.sum()), {"loss_comp_or_absolute": float(means[0]), "loss_struct_or_pairwise": float(means[1])}

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[tuple[float, float]] = []
        group_order = rng.permutation(len(train_groups))
        for offset in range(0, len(group_order), 8):
            sel = group_order[offset:offset + 8]
            batch_ids = train_groups[sel]
            batch_targets = y_train_groups[sel]
            if model_name == "H0":
                flat = torch.tensor(batch_ids.reshape(-1), dtype=torch.long)
                pred = model(x_end[flat]) * target_scale + target_mean
                truth = y[flat]
                absolute = nn.functional.smooth_l1_loss((pred - truth) / target_scale, torch.zeros_like(truth), beta=1.0)
                pair_terms = []
                pred3, truth3 = pred.reshape(-1, 3), truth.reshape(-1, 3)
                for p3, t3 in zip(pred3, truth3):
                    vals = [nn.functional.smooth_l1_loss(((p3[i]-p3[j])-(t3[i]-t3[j]))/pscale,
                        torch.zeros((), dtype=p3.dtype), beta=1.0) for i in range(3) for j in range(i+1, 3)]
                    pair_terms.append(torch.stack(vals).mean())
                pairwise = torch.stack(pair_terms).mean()
                loss = absolute + pairwise
                batch_losses.append((float(absolute.detach()), float(pairwise.detach())))
            else:
                # Sort groups already have canonical family order; all three remain in each batch.
                flat = torch.tensor(batch_ids.reshape(-1), dtype=torch.long)
                xstruct = x_end[flat].reshape(len(sel), 3, 768)
                xcomp = (xstruct.mean(dim=1) if model_name == "H1" else x_mean[flat].reshape(len(sel), 3, 768).mean(dim=1))
                mu_raw, delta_raw, _ = model(xcomp, xstruct)
                true3 = torch.tensor(batch_targets, dtype=torch.float32)
                mu_true = true3.mean(dim=1)
                delta_true = true3 - mu_true[:, None]
                mu_pred = mu_raw * scale_comp + target_mean
                delta_pred = delta_raw * scale_struct
                lcomp = nn.functional.smooth_l1_loss((mu_pred-mu_true)/scale_comp, torch.zeros_like(mu_true), beta=1.0)
                lstruct = nn.functional.smooth_l1_loss((delta_pred-delta_true)/scale_struct, torch.zeros_like(delta_true), beta=1.0)
                loss = lcomp + lstruct
                batch_losses.append((float(lcomp.detach()), float(lstruct.detach())))
            if not torch.isfinite(loss):
                raise ExperimentError(f"Loss NaN/Inf {model_name}/fold{fold}/seed{seed}/epoch{epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_comp = 0.0; grad_struct = 0.0
            if model_name != "H0":
                grad_comp = float(sum(p.grad.detach().norm().item() for p in model.composition.parameters() if p.grad is not None))
                grad_struct = float(sum(p.grad.detach().norm().item() for p in model.residual.parameters() if p.grad is not None))
                if grad_comp == 0 or grad_struct == 0:
                    raise ExperimentError("Gradiente ausente en una de las dos cabezas.")
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_score, val_parts = evaluate_groups(val_groups, y_val_groups, False)
        train_parts = np.mean(np.asarray(batch_losses), axis=0)
        history.append({"model": model_name, "fold": fold, "seed": seed, "epoch": epoch,
            "train_loss_first": float(train_parts[0]), "train_loss_second": float(train_parts[1]),
            "inner_validation_objective": val_score, "inner_validation_first": val_parts["loss_comp_or_absolute"],
            "inner_validation_second": val_parts["loss_struct_or_pairwise"], "learning_rate": 0.001,
            "scale_comp_ev_atom": scale_comp, "scale_struct_ev_atom": scale_struct,
            "target_mean_ev_atom": target_mean, "pair_scale_ev_atom": pscale,
            "grad_norm_composition": grad_comp if model_name != "H0" else np.nan,
            "grad_norm_residual": grad_struct if model_name != "H0" else np.nan})
        if val_score < best - 1e-12:
            best, best_epoch, waited = val_score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if waited >= 20:
            break
    if best_state is None:
        raise ExperimentError("Early stopping no seleccionó estado.")
    model.load_state_dict(best_state, strict=True)
    torch.save({"model": model_name, "fold": fold, "seed": seed, "state_dict": best_state,
        "best_epoch": best_epoch, "best_inner_validation_objective": best,
        "scales": {"target_mean_ev_atom": target_mean, "target_scale_ev_atom": target_scale,
                   "scale_comp_ev_atom": scale_comp, "scale_struct_ev_atom": scale_struct, "pair_scale_ev_atom": pscale},
        "backbone_trainable_parameters": 0}, run_dir / "head.pt")
    model.eval()
    out_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for ids in outer_groups:
            ix = torch.tensor(ids, dtype=torch.long)
            if model_name == "H0":
                predictions = (model(x_end[ix]) * target_scale + target_mean).cpu().numpy()
                mu_out, deltas_out = np.nan, np.full(3, np.nan)
            else:
                xstruct = x_end[ix].unsqueeze(0)
                comp_x = xstruct.mean(dim=1) if model_name == "H1" else x_mean[ix].mean(dim=0, keepdim=True)
                mu_raw, delta_raw, total = model(comp_x, xstruct)
                mu_out = float(mu_raw.squeeze().item() * scale_comp + target_mean)
                deltas_out = delta_raw.squeeze(0).cpu().numpy() * scale_struct
                predictions = total.squeeze(0).cpu().numpy() * 1.0
                # forward returns raw head-space outputs: unnormalize each component explicitly.
                predictions = mu_out + deltas_out
            for ix_row, pred, delta in zip(ids, predictions, deltas_out):
                row = data.iloc[ix_row]
                out_rows.append({"structure_id": row.structure_id, "composition_id": row.composition_id,
                    "polymorph_family": row.polymorph_family, "fold": fold, "seed": seed, "model": model_name,
                    "E_DFT_eV_atom": float(row.E_DFT_eV_atom), "E_pred_eV_atom": float(pred),
                    "E_pred_mu_eV_atom": float(mu_out) if np.isfinite(mu_out) else np.nan,
                    "delta_pred_centered_eV_atom": float(delta) if np.isfinite(delta) else np.nan,
                    "best_epoch": best_epoch, "historical_split": row.split})
    pred_frame = pd.DataFrame(out_rows)
    if len(pred_frame) != 42 or not np.isfinite(pred_frame.E_pred_eV_atom).all():
        raise ExperimentError("Predicciones outer inválidas.")
    pred_frame.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    elapsed = time.perf_counter() - start_time
    run_provenance.update({"status": "PASS", "best_epoch": best_epoch, "epochs_run": len(history),
        "training_seconds": elapsed, "prediction_rows": len(pred_frame), "trainable_head_parameters": sum(p.numel() for p in parameters)})
    write_json(run_dir / "run_provenance.json", run_provenance)
    write_json(run_dir / "status.json", {"status": "PASS", "best_epoch": best_epoch, "test_evaluated": False, "training_seconds": elapsed})
    return pred_frame, history


def run_all(out_dir: Path, max_epochs: int = 200, selection: tuple[str, ...] = MODELS,
            folds_to_run: tuple[int, ...] = tuple(range(5)), seeds_to_run: tuple[int, ...] = SEEDS) -> None:
    """Completa o reanuda modelos/folds/semillas preservando ejecuciones exitosas."""
    audit = verify_provenance()
    data = load_development()
    folds = load_fold_manifest(data)
    feature_path = out_dir / "features" / "frozen_features_development.npz"
    meta_path = feature_path.with_name("feature_metadata.csv")
    manifest_path = feature_path.with_name("embeddings_manifest.json")
    if not all(path.is_file() for path in (feature_path, meta_path, manifest_path)):
        raise ExperimentError("Extrae primero los embeddings con extract_frozen_features.py.")
    manifest = __import__("json").loads(manifest_path.read_text())
    if manifest.get("checkpoint_sha256") != audit["checkpoint_sha256"] or manifest.get("dataset_sha256") != sha256_file(DATASET):
        raise ExperimentError("Caché de features corresponde a otras fuentes.")
    meta = pd.read_csv(meta_path)
    if meta.structure_id.astype(str).tolist() != data.structure_id.astype(str).tolist():
        raise ExperimentError("Orden o IDs de caché no coinciden con development.")
    with np.load(feature_path, allow_pickle=False) as cached:
        if cached["structure_id"].astype(str).tolist() != data.structure_id.astype(str).tolist():
            raise ExperimentError("IDs internos del cache difieren del dataset.")
        features = {"h_end": np.array(cached["h_end"], dtype=np.float32, copy=True),
                    "h_mean": np.array(cached["h_mean"], dtype=np.float32, copy=True)}
    if features["h_end"].shape != (210, 768) or features["h_mean"].shape != (210, 768):
        raise ExperimentError("Feature cache no tiene forma [210,768].")
    records = []
    for name in selection:
        for fold in folds_to_run:
            for seed in seeds_to_run:
                run_dir = out_dir / "runs" / name / f"fold_{fold}_seed_{seed}"
                pred, _ = run_one(data, features, folds, name, fold, seed, run_dir, max_epochs=max_epochs)
                records.append(pred)
                log(logging.INFO, "head_run_complete", model=name, fold=fold, seed=seed, rows=len(pred), test=False)
                # Flush OOF progress after each run so a stopped job can resume safely.
                all_parts = sorted((out_dir / "runs").glob("*/fold_*_seed_*/predictions.csv"))
                pd.concat([pd.read_csv(path) for path in all_parts], ignore_index=True).to_csv(out_dir / "oof_predictions_partial.csv", index=False)
    all_parts = sorted((out_dir / "runs").glob("*/fold_*_seed_*/predictions.csv"))
    all_predictions = pd.concat([pd.read_csv(path) for path in all_parts], ignore_index=True)
    all_predictions.to_csv(out_dir / "oof_predictions.csv", index=False)
    history_parts = sorted((out_dir / "runs").glob("*/fold_*_seed_*/history.csv"))
    pd.concat([pd.read_csv(path) for path in history_parts], ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    manifest_out = {"status": "PASS" if len(all_parts) == 45 else "PARTIAL", "completed_runs": len(all_parts),
        "expected_runs": 45, "models": list(selection), "folds": list(folds_to_run), "seeds": list(seeds_to_run),
        "predictions": len(all_predictions), "checkpoint_sha256": audit["checkpoint_sha256"],
        "test_used": False, "source_code_sha256": sha256_file(Path(__file__))}
    write_json(out_dir / "training_manifest.json", manifest_out)


def main() -> None:
    """Valida filtros CLI y ejecuta la CV local sobre cabezas pequeñas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    try:
        if args.max_epochs < 1:
            parser.error("--max-epochs debe ser positivo")
        run_all(args.out_dir, args.max_epochs, tuple(args.models), tuple(args.folds), tuple(args.seeds))
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(logging.ERROR, "grouped_training_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
