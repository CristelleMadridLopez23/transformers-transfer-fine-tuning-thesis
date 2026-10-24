#!/usr/bin/env python3
"""R1 y R4: Ridge sobre embeddings MatGPT, con alpha y escalas train-only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (ALPHAS, FAMILIES, REPORT_DIR, ExperimentError,
    load_dev, load_folds, log, sha256_file, split_indices, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.data_and_folds import ordered_triplet


def groups(frame: pd.DataFrame, indices: np.ndarray) -> list[pd.DataFrame]:
    """Devuelve tripletes completos ordenados por fórmula y familia."""
    subset = frame.iloc[indices]
    return [ordered_triplet(g) for _, g in subset.groupby("composition_id", sort=True)]


def feature_for_group(group: pd.DataFrame, features: np.ndarray, id_to_row: dict[str, int]) -> np.ndarray:
    """Promedia vectores de los tres candidatos, de forma simétrica."""
    return features[[id_to_row[sid] for sid in group.structure_id]].mean(axis=0)


def fit_select(features: np.ndarray, data: pd.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray,
               variant: str, target_kind: str, alpha_grid: tuple[float, ...] = ALPHAS) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Selecciona alpha exclusivamente por MAE inner-val y devuelve scaler/modelo.

    Args:
        features: Matriz [210,768] asociada por orden a data.structure_id.
        data: Dataset development-only.
        train_ids: Filas inner-train.
        val_ids: Filas inner-validation.
        variant: Nombre de pooling/capa.
        target_kind: absolute, mu o delta.
    Returns:
        Alpha ganador, objetos fitted (scaler/modelo) y auditoría de todos los alphas.
    """
    id_to_row = {sid: i for i, sid in enumerate(data.structure_id)}
    train_groups, val_groups = groups(data, train_ids), groups(data, val_ids)
    if target_kind == "absolute":
        x_train = features[train_ids]
        y_train = data.iloc[train_ids].E_DFT_eV_atom.to_numpy(float)
        x_val = features[val_ids]
        y_val = data.iloc[val_ids].E_DFT_eV_atom.to_numpy(float)
    elif target_kind == "mu":
        x_train = np.stack([feature_for_group(g, features, id_to_row) for g in train_groups])
        y_train = np.asarray([g.E_DFT_eV_atom.mean() for g in train_groups])
        x_val = np.stack([feature_for_group(g, features, id_to_row) for g in val_groups])
        y_val = np.asarray([g.E_DFT_eV_atom.mean() for g in val_groups])
    elif target_kind == "delta":
        x_train = features[train_ids]
        y_train = data.iloc[train_ids].E_DFT_eV_atom.to_numpy(float)
        train_lookup = {int(ix): float(g.E_DFT_eV_atom.mean()) for g in train_groups for ix in g.index}
        y_train = np.asarray([value - train_lookup[int(ix)] for ix, value in zip(train_ids, y_train)])
        x_val = features[val_ids]
        raw_val = data.iloc[val_ids].E_DFT_eV_atom.to_numpy(float)
        val_lookup = {int(ix): float(g.E_DFT_eV_atom.mean()) for g in val_groups for ix in g.index}
        y_val = np.asarray([value - val_lookup[int(ix)] for ix, value in zip(val_ids, raw_val)])
    else:
        raise ExperimentError(f"Target Ridge desconocido: {target_kind}")
    # Promote cached float32 features before centering/scaling so small alpha values do not
    # solve a nearly singular system in single precision.
    x_train64 = np.asarray(x_train, dtype=np.float64)
    x_val64 = np.asarray(x_val, dtype=np.float64)
    scaler = StandardScaler().fit(x_train64)
    x_train_scaled, x_val_scaled = scaler.transform(x_train64), scaler.transform(x_val64)
    records, fitted = [], {}
    best_alpha, best_score = None, float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-8).fit(x_train_scaled, y_train)
        pred = model.predict(x_val_scaled)
        if not np.isfinite(pred).all():
            raise ExperimentError(f"Ridge no finito para {variant}/{target_kind}/alpha={alpha}.")
        if target_kind == "delta":
            pred_frame = pd.DataFrame({"composition_id": data.iloc[val_ids].composition_id.to_numpy(), "pred": pred})
            pred = pred - pred_frame.groupby("composition_id").pred.transform("mean").to_numpy()
        score = float(np.mean(np.abs(pred - y_val)) * 1000.0)
        records.append({"feature_variant": variant, "target_kind": target_kind, "alpha": float(alpha),
            "inner_validation_MAE_mev_atom": score, "selection_partition": "inner_validation", "scaler_fit_partition": "inner_train"})
        if score < best_score:
            best_alpha, best_score = float(alpha), score
    assert best_alpha is not None
    final = Ridge(alpha=best_alpha, solver="lsqr", tol=1e-8).fit(x_train_scaled, y_train)
    fitted = {"scaler": scaler, "model": final, "alpha": best_alpha, "train_n": int(len(y_train)),
        "inner_val_n": int(len(y_val)), "n_coefficients": int(final.coef_.size),
        "coefficient_l2_norm": float(np.linalg.norm(final.coef_)), "feature_variant": variant, "target_kind": target_kind,
        "solver": "lsqr", "solver_tolerance": 1e-8}
    return best_alpha, fitted, records


def predict_target(fitted: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Predice valores en el espacio eV/átomo con scaler ajustado en train."""
    return np.asarray(fitted["model"].predict(fitted["scaler"].transform(features)), dtype=float)


def predict_outer(fitted: dict[str, Any], data: pd.DataFrame, outer_ids: np.ndarray,
                  features: np.ndarray, target_kind: str) -> np.ndarray:
    """Calcula predicción outer; residual se centra dentro de cada triplete."""
    if target_kind == "mu":
        id_to_row = {sid: i for i, sid in enumerate(data.structure_id)}
        out_groups = groups(data, outer_ids)
        mu_by_comp = {str(g.composition_id.iloc[0]): float(predict_target(fitted,
            np.stack([feature_for_group(g, features, id_to_row)]))[0]) for g in out_groups}
        return data.iloc[outer_ids].composition_id.map(mu_by_comp).to_numpy(float)
    result = predict_target(fitted, features[outer_ids])
    if target_kind == "delta":
        frame = data.iloc[outer_ids].reset_index(drop=True)
        result = result - pd.Series(result).groupby(frame.composition_id).transform("mean").to_numpy()
    return result


def feature_variants(cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Obtiene las diez vistas, ocho capas crudas y dos controles post-ln_f."""
    names = [f"L{layer:02d}_{pool}" for layer in (3, 6, 9, 12) for pool in ("END", "MEAN")]
    names += ["FINAL_END", "FINAL_MEAN"]
    return {name: np.array(cache[name], dtype=np.float32, copy=True) for name in names}


def run_ridges() -> dict[str, Any]:
    """Entrena R1-A/B/C/D y R4 layer/pooling probes con selección anidada."""
    data, folds = load_dev(), load_folds(load_dev())
    cache_path = REPORT_DIR / "features_cache" / "layer_features.npz"
    if not cache_path.is_file():
        raise ExperimentError("Primero ejecutar freeze_and_extract.py.")
    cache = np.load(cache_path, allow_pickle=False)
    if cache["structure_id"].astype(str).tolist() != data.structure_id.tolist():
        raise ExperimentError("Orden de IDs de feature cache cambió.")
    variants = feature_variants(cache)
    cache.close()
    all_preds, selections = [], []
    output_specs = [("R1-A-ABS", "FINAL_END", "absolute"), ("R1-B-MU", "FINAL_END", "mu"),
                    ("R1-C-DELTA", "FINAL_END", "delta")]
    for fold in range(5):
        train_ids, val_ids, outer_ids = split_indices(data, folds, fold)
        # H2-comparable two-component readout combines separately selected μ and centered δ probes.
        for model_name, feature_name, target_kind in output_specs:
            feature = variants[feature_name]
            alpha, fitted, recs = fit_select(feature, data, train_ids, val_ids, feature_name, target_kind)
            for rec in recs:
                selections.append({"fold": fold, "model": model_name, **rec, "selected": rec["alpha"] == alpha})
            pred = predict_outer(fitted, data, outer_ids, feature, target_kind)
            if target_kind == "absolute":
                energy = pred
                mu = np.full(len(outer_ids), np.nan)
                delta = np.full(len(outer_ids), np.nan)
            elif target_kind == "mu":
                energy = pred
                mu = pred
                delta = np.full(len(outer_ids), np.nan)
            else:
                energy = pred
                mu = np.full(len(outer_ids), np.nan)
                delta = pred
            all_preds.extend(_rows(data, outer_ids, model_name, fold, energy, mu, delta, feature_name,
                                   target_kind, fitted["alpha"], seed=-1))
        # R1-D uses independently selected mean and residual alpha, both with end pooling.
        fitted_parts = {}
        for target_kind in ("mu", "delta"):
            alpha, fitted, recs = fit_select(variants["FINAL_END"], data, train_ids, val_ids, "FINAL_END", target_kind)
            fitted_parts[target_kind] = fitted
            selections.extend({"fold": fold, "model": "R1-D-COMBINED", **rec, "selected": rec["alpha"] == alpha} for rec in recs)
        mu_pred = predict_outer(fitted_parts["mu"], data, outer_ids, variants["FINAL_END"], "mu")
        delta_pred = predict_outer(fitted_parts["delta"], data, outer_ids, variants["FINAL_END"], "delta")
        all_preds.extend(_rows(data, outer_ids, "R1-D-COMBINED", fold, mu_pred + delta_pred, mu_pred, delta_pred,
            "FINAL_END", "mu_plus_delta", {"mu": fitted_parts["mu"]["alpha"], "delta": fitted_parts["delta"]["alpha"]}, seed=-1))
        # R4: inner-val chooses representation separately for absolute, mean and residual objectives.
        selected: dict[str, tuple[str, dict[str, Any], float]] = {}
        for target_kind in ("absolute", "mu", "delta"):
            scored = []
            for feature_name, feature in variants.items():
                alpha, fitted, recs = fit_select(feature, data, train_ids, val_ids, feature_name, target_kind)
                best_score = min(r["inner_validation_MAE_mev_atom"] for r in recs)
                for rec in recs:
                    selections.append({"fold": fold, "model": f"R4-{target_kind.upper()}-{feature_name}", **rec, "selected": rec["alpha"] == alpha})
                scored.append((best_score, feature_name, fitted))
            best_score, feature_name, fitted = min(scored, key=lambda x: (x[0], x[1]))
            selected[target_kind] = (feature_name, fitted, best_score)
            selections.append({"fold": fold, "model": f"R4-SELECTED-{target_kind.upper()}", "feature_variant": feature_name,
                "target_kind": target_kind, "alpha": fitted["alpha"], "inner_validation_MAE_mev_atom": best_score,
                "selection_partition": "inner_validation", "selected": True})
        for target_kind, (feature_name, fitted, _) in selected.items():
            feature = variants[feature_name]
            pred = predict_outer(fitted, data, outer_ids, feature, target_kind)
            all_preds.extend(_rows(data, outer_ids, f"R4-SELECTED-{target_kind.upper()}", fold, pred,
                pred if target_kind == "mu" else np.full(len(outer_ids), np.nan),
                pred if target_kind == "delta" else np.full(len(outer_ids), np.nan),
                feature_name, target_kind, fitted["alpha"], seed=-1))
        mu_name, mu_fit, _ = selected["mu"]
        delta_name, delta_fit, _ = selected["delta"]
        mu_pred = predict_outer(mu_fit, data, outer_ids, variants[mu_name], "mu")
        delta_pred = predict_outer(delta_fit, data, outer_ids, variants[delta_name], "delta")
        all_preds.extend(_rows(data, outer_ids, "R4-SELECTED-2COMP", fold, mu_pred + delta_pred, mu_pred, delta_pred,
            f"mu:{mu_name};delta:{delta_name}", "mu_plus_delta", {"mu": mu_fit["alpha"], "delta": delta_fit["alpha"]}, seed=-1))
        # Fixed post-ln_f controls, never selected on outer data.
        for feature_name in ("FINAL_END", "FINAL_MEAN"):
            fits = {}
            for target_kind in ("mu", "delta"):
                alpha, fitted, recs = fit_select(variants[feature_name], data, train_ids, val_ids, feature_name, target_kind)
                fits[target_kind] = fitted
                selections.extend({"fold": fold, "model": f"R4-FIXED-{feature_name}", **rec,
                    "selected": rec["alpha"] == alpha} for rec in recs)
            mu_pred = predict_outer(fits["mu"], data, outer_ids, variants[feature_name], "mu")
            delta_pred = predict_outer(fits["delta"], data, outer_ids, variants[feature_name], "delta")
            all_preds.extend(_rows(data, outer_ids, f"R4-FIXED-{feature_name}", fold, mu_pred + delta_pred, mu_pred, delta_pred,
                feature_name, "mu_plus_delta", {"mu": fits["mu"]["alpha"], "delta": fits["delta"]["alpha"]}, seed=-1))
    pred_frame = pd.DataFrame(all_preds)
    for experiment in ("R1", "R4"):
        out = REPORT_DIR / ("R1_linear_probes" if experiment == "R1" else "R4_layers_pooling")
        out.mkdir(parents=True, exist_ok=True)
        subset = pred_frame.loc[pred_frame.model.str.startswith(experiment)].copy()
        if (out / "oof_predictions.csv").exists():
            raise ExperimentError(f"Se preserva salida Ridge existente y no se sobrescribe: {out}")
        subset.to_csv(out / "oof_predictions.csv", index=False)
        pd.DataFrame([record for record in selections if str(record.get("model", "")).startswith(experiment)]).to_csv(
            out / "validation_selection.csv", index=False)
        write_json(out / "run_manifest.json", {"status": "PASS", "experiment": experiment,
            "models": sorted(subset.model.unique().tolist()),
            "selection": "inner-validation only; StandardScaler fitted on inner-train",
            "folds": 5, "alphas": list(ALPHAS), "test_used": False,
            "ridge_deterministic_not_seed_duplicated": True, "code_sha256": sha256_file(Path(__file__)),
            "prediction_rows": len(subset)})
    log("ridge_probes_complete", rows=len(pred_frame), selection_rows=len(selections), test_used=False)
    return {"status": "PASS", "rows": len(pred_frame), "selection_rows": len(selections)}


def _rows(data: pd.DataFrame, ids: np.ndarray, model: str, fold: int, energy: np.ndarray, mu: np.ndarray,
          delta: np.ndarray, layer_pooling: str, target: str, alpha: Any, seed: int) -> list[dict[str, Any]]:
    """Serializa predicciones con trazabilidad de target y feature."""
    rows = []
    alpha_mu = alpha.get("mu") if isinstance(alpha, dict) else alpha
    alpha_delta = alpha.get("delta") if isinstance(alpha, dict) else alpha
    for j, ix in enumerate(ids):
        row = data.iloc[ix]
        rows.append({"structure_id": row.structure_id, "composition_id": row.composition_id,
            "polymorph_family": row.polymorph_family, "fold": fold, "seed": seed, "model": model,
            "E_DFT_eV_atom": float(row.E_DFT_eV_atom), "E_pred_eV_atom": float(energy[j]),
            "E_pred_mu_eV_atom": float(mu[j]) if np.isfinite(mu[j]) else np.nan,
            "delta_pred_centered_eV_atom": float(delta[j]) if np.isfinite(delta[j]) else np.nan,
            "layer_pooling": layer_pooling, "target_kind": target, "alpha_mu": alpha_mu, "alpha_delta": alpha_delta,
            "historical_split": row.split, "selection_rule": "inner-validation MAE for target; scaler fit on inner-train"})
    return rows


def main() -> None:
    """Ejecuta las sondas Ridge fijadas por configuración."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run_ridges()
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError) as exc:
        log("ridge_probe_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
