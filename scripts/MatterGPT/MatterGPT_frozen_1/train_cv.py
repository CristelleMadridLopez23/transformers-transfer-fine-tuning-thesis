"""Train frozen-feature probes in the historical composition-grouped folds."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss

from common import REPORT_DIR, SEEDS, environment, logger, sha256, stable_ridge_predict, write_json
from heads import MLP8, TwoHead

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FEATURES = ("BLOCK03_MEAN", "BLOCK03_LAST", "BLOCK06_MEAN", "BLOCK06_LAST", "BLOCK09_MEAN", "BLOCK09_LAST", "BLOCK12_MEAN", "BLOCK12_LAST", "FINAL_MEAN", "FINAL_END")


def _seed(seed: int) -> None:
    """Set deterministic Python/NumPy/PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _composition_indices(meta: pd.DataFrame, names: list[str]) -> list[np.ndarray]:
    """Return physical row-index triplets for requested composition names."""
    result = []
    for name in names:
        idx = np.flatnonzero(meta.composition_id.to_numpy() == name)
        if len(idx) != 3:
            raise ValueError(f"{name}: se requieren tres polimorfos")
        result.append(idx)
    return result


def _inner_groups(meta: pd.DataFrame, train_idx: np.ndarray, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic composition holdout; no structure-level splitting."""
    groups = sorted(meta.iloc[train_idx].composition_id.unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(0.2 * len(groups))))
    return sorted(groups[n_val:]), sorted(groups[:n_val])


def _fit_m0(features: np.ndarray, y: np.ndarray, meta: pd.DataFrame, outer_train: np.ndarray,
            outer_val: np.ndarray, seed: int, run_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit an absolute MLP8 with inner-train normalization and early stopping."""
    group_train, group_inner = _inner_groups(meta, outer_train, 13_072_026 + seed)
    inner_train = np.flatnonzero(meta.composition_id.isin(group_train).to_numpy())
    inner_val = np.flatnonzero(meta.composition_id.isin(group_inner).to_numpy())
    y_mean, y_std = float(y[inner_train].mean()), float(y[inner_train].std() or 1.0)
    x = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model = MLP8(features.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    smooth = SmoothL1Loss(beta=1.0)
    groups = _composition_indices(meta, group_train)
    rng = np.random.default_rng(seed)
    best, best_epoch, stale, hist = float("inf"), 0, 0, []
    best_state = None
    for epoch in range(1, 201):
        model.train(); order = rng.permutation(len(groups)); train_loss = []
        for start in range(0, len(order), 8):
            batch_groups = [groups[i] for i in order[start:start+8]]
            idx = np.concatenate(batch_groups)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[idx]); abs_loss = smooth(pred, target[idx])
            pair_losses = [smooth(pred[j:j+3][:, None] - pred[j:j+3][None, :], target[idx][j:j+3][:, None] - target[idx][j:j+3][None, :]) for j in range(0, len(idx), 3)]
            loss = abs_loss + torch.stack(pair_losses).mean()
            loss.backward(); optimizer.step(); train_loss.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): val = float(torch.mean(torch.abs(model(x[inner_val]) - target[inner_val])).item())
        hist.append({"epoch": epoch, "train_objective": float(np.mean(train_loss)), "inner_val_normalized_mae": val})
        if val < best - 1e-8:
            best, best_epoch, stale = val, epoch, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= 20: break
    if best_state is None: raise RuntimeError("M0 no guardó un checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred = model(x[outer_val]).numpy() * y_std + y_mean
    run_dir.mkdir(parents=True, exist_ok=True); torch.save(best_state, run_dir / "head.pt")
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    write_json(run_dir / "training.json", {"best_epoch": best_epoch, "inner_val_mae_normalized": best, "target_mean": y_mean, "target_scale": y_std, "seed": seed})
    return pred, hist


def _fit_m2(comp_feat: np.ndarray, struct_feat: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
            outer_train: np.ndarray, outer_val: np.ndarray, seed: int, run_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit composition mean and centered residual heads without outer leakage."""
    group_train, group_inner = _inner_groups(meta, outer_train, 13_072_026 + seed)
    tr_triplets, va_triplets = _composition_indices(meta, group_train), _composition_indices(meta, group_inner)
    # Use stable per-composition row lookup, independent of CSV ordering.
    y_group = np.stack([y[idx] for idx in _composition_indices(meta, sorted(meta.composition_id.unique().tolist()))])
    names = sorted(meta.composition_id.unique().tolist()); lookup = {name: i for i, name in enumerate(names)}
    tr_gidx = np.array([lookup[name] for name in group_train]); va_gidx = np.array([lookup[name] for name in group_inner])
    means = y_group.mean(axis=1); deltas = y_group - means[:, None]
    train_means = means[[lookup[g] for g in group_train]]
    mu_center = float(train_means.mean())
    mu_scale = float(train_means.std() or 1.0)
    delta_scale = float(deltas[[lookup[g] for g in group_train]].std() or 1.0)
    comp_group = np.stack([comp_feat[idx].mean(axis=0) for idx in _composition_indices(meta, names)])
    struct_group = np.stack([struct_feat[idx] for idx in _composition_indices(meta, names)])
    xmu = torch.tensor(comp_group, dtype=torch.float32); xstruct = torch.tensor(struct_group, dtype=torch.float32)
    tmu = torch.tensor((means - mu_center) / mu_scale, dtype=torch.float32)
    tdelta = torch.tensor(deltas / delta_scale, dtype=torch.float32)
    model = TwoHead(comp_feat.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    smooth = SmoothL1Loss(beta=1.0); rng = np.random.default_rng(seed)
    best, best_epoch, stale, hist, best_state = float("inf"), 0, 0, [], None
    for epoch in range(1, 201):
        model.train(); order = rng.permutation(tr_gidx); losses=[]; comps=[]; structs=[]
        for start in range(0, len(order), 8):
            batch = order[start:start+8]; optimizer.zero_grad(set_to_none=True)
            mu_pred, delta_pred, _ = model(xmu[batch], xstruct[batch])
            lmu = smooth((mu_pred - tmu[batch]) / 1.0, torch.zeros_like(mu_pred))
            ldelta = smooth((delta_pred - tdelta[batch]) / 1.0, torch.zeros_like(delta_pred))
            loss = lmu + ldelta; loss.backward(); optimizer.step()
            losses.append(float(loss.detach())); comps.append(float(lmu.detach())); structs.append(float(ldelta.detach()))
        model.eval()
        with torch.no_grad():
            _, _, pred_inner = model(xmu[va_gidx], xstruct[va_gidx])
            true_inner = torch.tensor(y_group[va_gidx], dtype=torch.float32)
            mu_val = pred_inner.mean(dim=1, keepdim=True) * mu_scale + mu_center
            delta_val = pred_inner - pred_inner.mean(dim=1, keepdim=True)
            pred_energy_val = mu_val + delta_val * delta_scale
            val = float(torch.abs(pred_energy_val - true_inner).mean())
        hist.append({"epoch": epoch, "train_total": float(np.mean(losses)), "train_comp": float(np.mean(comps)), "train_struct": float(np.mean(structs)), "inner_val_absolute_mae_ev_atom": val})
        if val < best - 1e-8:
            best, best_epoch, stale = val, epoch, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= 20: break
    if best_state is None: raise RuntimeError("M2 no guardó un checkpoint")
    model.load_state_dict(best_state); model.eval()
    val_names = meta.iloc[outer_val].composition_id.unique().tolist()
    with torch.no_grad():
        mu_out, delta_out, _ = model(xmu[[lookup[g] for g in val_names]], xstruct[[lookup[g] for g in val_names]])
        pred_trip = (mu_out[:, None].numpy() * mu_scale + mu_center + delta_out.numpy() * delta_scale)
    # Fit on all outer-train after inner-only checkpoint selection, as specified.
    # Current checkpoint weights remain selected by inner validation; no outer labels enter.
    pred_lookup = {g: pred_trip[i] for i, g in enumerate(val_names)}
    pred_map = {(g, fam): float(pred_lookup[g][i]) for g in val_names for i, fam in enumerate(meta[meta.composition_id == g].polymorph_family.tolist())}
    pred = np.array([pred_map[(meta.iloc[idx].composition_id, meta.iloc[idx].polymorph_family)] for idx in outer_val])
    run_dir.mkdir(parents=True, exist_ok=True); torch.save(best_state, run_dir / "head.pt")
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    write_json(run_dir / "training.json", {"best_epoch": best_epoch, "inner_val_absolute_mae_ev_atom": best, "mean_center_ev_atom": mu_center, "mean_scale_ev_atom": mu_scale, "residual_scale_ev_atom": delta_scale, "seed": seed})
    return pred, hist


def run(models: list[str]) -> dict[str, Any]:
    """Run requested candidates over all shared outer folds and seeds."""
    cache_path = REPORT_DIR / "features/development_features.npz"
    meta_path = REPORT_DIR / "features/development_metadata.csv"
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Ejecute audit.py --phase extract primero")
    archive = np.load(cache_path); meta = pd.read_csv(meta_path); y = meta.energy_dft_ev_atom.to_numpy(float)
    out_rows: list[dict[str, Any]] = []; history_rows: list[dict[str, Any]] = []; ridge_rows: list[dict[str, Any]] = []
    model_features = {name: archive[name] for name in archive.files if name in FEATURES}
    for fold in sorted(meta.outer_fold.unique()):
        outer_val = np.flatnonzero(meta.outer_fold.to_numpy() == fold); outer_train = np.flatnonzero(meta.outer_fold.to_numpy() != fold)
        if set(meta.iloc[outer_train].composition_id) & set(meta.iloc[outer_val].composition_id): raise RuntimeError("Composición cruza outer fold")
        for seed in SEEDS:
            _seed(seed)
            if "M0" in models:
                pred, hist = _fit_m0(archive["FINAL_END"], y, meta, outer_train, outer_val, seed, REPORT_DIR / f"cv/runs/M0/fold_{fold}_seed_{seed}")
                history_rows += [{"model": "M0", "fold": fold, "seed": seed, **r} for r in hist]
                out_rows += [{**meta.iloc[idx].to_dict(), "model": "M0", "seed": seed, "prediction_ev_atom": float(p)} for idx, p in zip(outer_val, pred)]
            if "M2" in models:
                pred, hist = _fit_m2(archive["FINAL_MEAN"], archive["FINAL_END"], y, meta, outer_train, outer_val, seed, REPORT_DIR / f"cv/runs/M2/fold_{fold}_seed_{seed}")
                history_rows += [{"model": "M2", "fold": fold, "seed": seed, **r} for r in hist]
                out_rows += [{**meta.iloc[idx].to_dict(), "model": "M2", "seed": seed, "prediction_ev_atom": float(p)} for idx, p in zip(outer_val, pred)]
            if "M4" in models:
                inner_train_groups, inner_val_groups = _inner_groups(meta, outer_train, 13_072_026 + seed)
                inner_train = outer_train[meta.iloc[outer_train].composition_id.isin(inner_train_groups).to_numpy()]
                inner_val = outer_train[meta.iloc[outer_train].composition_id.isin(inner_val_groups).to_numpy()]
                best = None
                for feature, matrix in model_features.items():
                    for alpha in ALPHAS:
                        scaler = StandardScaler().fit(matrix[inner_train]); estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-8).fit(scaler.transform(matrix[inner_train]), y[inner_train])
                        val = stable_ridge_predict(estimator, scaler.transform(matrix[inner_val])); score = float(np.mean(np.abs(val - y[inner_val])))
                        candidate = (score, feature, alpha)
                        if best is None or candidate < best: best = candidate
                assert best is not None
                _, feature, alpha = best; matrix = model_features[feature]
                scaler = StandardScaler().fit(matrix[outer_train]); estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-8).fit(scaler.transform(matrix[outer_train]), y[outer_train])
                pred = stable_ridge_predict(estimator, scaler.transform(matrix[outer_val]))
                run_dir = REPORT_DIR / f"cv/runs/M4/fold_{fold}_seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
                import joblib
                joblib.dump({"scaler": scaler, "ridge": estimator, "feature": feature, "alpha": alpha}, run_dir / "ridge.joblib")
                ridge_rows.append({"fold": fold, "seed": seed, "feature": feature, "alpha": alpha, "inner_val_mae_ev_atom": best[0], "train_compositions": len(set(meta.iloc[outer_train].composition_id)), "ridge_sha256": sha256(run_dir / "ridge.joblib")})
                out_rows += [{**meta.iloc[idx].to_dict(), "model": "M4", "seed": seed, "prediction_ev_atom": float(p)} for idx, p in zip(outer_val, pred)]
    predictions = pd.DataFrame(out_rows)
    predictions.to_csv(REPORT_DIR / "cv/oof_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(REPORT_DIR / "cv/training_history.csv", index=False)
    pd.DataFrame(ridge_rows).to_csv(REPORT_DIR / "cv/ridge_selection.csv", index=False)
    script_dir=Path(__file__).resolve().parent
    scripts={name:sha256(script_dir/name) for name in ("train_cv.py","heads.py","common.py")}
    scripts["energy_metrics.py"]=sha256(script_dir.parents[1]/"energy_metrics.py")
    write_json(REPORT_DIR / "cv/cv_manifest.json", {"status": "PASS", "models": models, "folds": sorted(meta.outer_fold.unique().tolist()), "seeds": SEEDS, "runs": int(len(predictions.groupby(["model", "outer_fold", "seed"]))), "backbone_checkpoint_sha256": sha256(next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file()) / "models/MatterGPT/pytorch_model.pt"), "feature_cache_sha256": sha256(cache_path), "dataset_sha256": sha256(next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file()) / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"), "ridge_solver": "lsqr, tol=1e-8 for numerical stability with 768 features and fewer training samples; same Ridge objective", "scripts_sha256":scripts, "environment": environment()})
    return {"status": "PASS", "predictions": len(predictions), "runs": int(len(predictions.groupby(["model", "outer_fold", "seed"]))) }


def main() -> None:
    """Run grouped cross-validation for the selected predefined probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("M0", "M2", "M4"), default=["M0", "M2", "M4"])
    args = parser.parse_args()
    log = logger("mattergpt_cv")
    try: log.info("mattergpt_cv_complete %s", run(args.models))
    except Exception as exc: log.exception("mattergpt_cv_failed error=%s", exc); raise


if __name__ == "__main__":
    main()
