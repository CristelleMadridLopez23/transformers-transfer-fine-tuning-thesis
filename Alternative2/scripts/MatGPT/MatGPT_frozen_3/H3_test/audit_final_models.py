#!/usr/bin/env python3
"""Audita H2/R4 y reconstruye Ridge solo con development antes de ver test.

R4 guarda predicciones OOF y selecciones, no estimadores. Este script replica
su scaler/Ridge interno en cada fold y exige reproducir el OOF archivado.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import (CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA, EXPECTED_COMMIT, EXPECTED_DATASET_SHA,
                    FEATURES, FEATURE_MANIFEST, FAMILIES, FOLDS as FOLD_PATH, H2_FEATURES, H2_OOF,
                    H2_RUNS, H3_MANIFEST, H3_REPORT, MODEL_INFO, REPORT_DIR,
                    R4_OOF, R4_ROOT, R4_SELECTION, SOURCE_MANIFEST, SOURCE_ROOT,
                    H3TestError, read_dev_targets_safely, read_test_inputs_safely,
                    sha256_file, write_json)


def _audit_reconstruct_r4() -> list[dict[str, Any]]:
    """Reconstruye cinco Ridge absolutos y coteja sus OOF históricos.

    Returns:
        Metadatos por fold de Ridge, scaler y error de reproducción outer.
    """
    dev = read_dev_targets_safely()
    ids = dev.structure_id.tolist()
    index = {sid: row for row, sid in enumerate(ids)}
    with np.load(FEATURES, allow_pickle=False) as cache:
        feature_ids = cache["structure_id"].astype(str).tolist()
        if feature_ids != ids:
            raise H3TestError("El cache R4 development no alinea con IDs/orden históricos.")
        x = np.array(cache["L03_MEAN"], dtype=np.float64, copy=True)
    if x.shape != (210, 768) or not np.isfinite(x).all():
        raise H3TestError("L03_MEAN development inválido.")
    fold_table = pd.read_csv(FOLD_PATH)
    fold_table = fold_table.loc[fold_table.historical_split.isin(["train", "val"])].copy()
    if set(fold_table.structure_id.astype(str)) != set(ids):
        raise H3TestError("Fold manifest development no corresponde a las 210 estructuras.")
    selection = pd.read_csv(R4_SELECTION)
    selected = selection.loc[selection.model.eq("R4-SELECTED-ABSOLUTE")].copy()
    oof = pd.read_csv(R4_OOF)
    oof = oof.loc[oof.model.eq("R4-SELECTED-ABSOLUTE")].copy()
    if len(selected) != 5 or len(oof) != 210 or oof.structure_id.duplicated().any():
        raise H3TestError("No hay cinco selecciones R4 y 210 predicciones OOF únicas.")
    output_dir = REPORT_DIR / "locked_models" / "ridge_by_fold"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    targets = dev.E_DFT_eV_atom.to_numpy(dtype=np.float64)
    for fold in range(5):
        fold_rows = fold_table.loc[fold_table.outer_fold.eq(fold)].set_index("structure_id")
        train_ids = np.flatnonzero(dev.structure_id.map(fold_rows.inner_role).eq("inner_train").to_numpy())
        val_ids = np.flatnonzero(dev.structure_id.map(fold_rows.inner_role).eq("inner_validation").to_numpy())
        outer_ids = np.flatnonzero(dev.structure_id.map(fold_rows.outer_role).eq("outer_validation").to_numpy())
        if (len(train_ids), len(val_ids), len(outer_ids)) != (135, 33, 42):
            raise H3TestError(f"Conteos R4 fold {fold} distintos al protocolo histórico.")
        row = selected.loc[selected.fold.eq(fold)]
        if len(row) != 1 or row.iloc[0].feature_variant != "L03_MEAN":
            raise H3TestError(f"Selección R4 histórica cambió en fold {fold}.")
        alpha = float(row.iloc[0].alpha)
        expected_score = float(row.iloc[0].inner_validation_MAE_mev_atom)
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=alpha, solver="lsqr", tol=1e-8))])
        pipeline.fit(x[train_ids], targets[train_ids])
        val_prediction = pipeline.predict(x[val_ids])
        if not np.isfinite(val_prediction).all():
            raise H3TestError(f"R4 inner-validation produce NaN/Inf fold={fold}.")
        val_mae = float(np.mean(np.abs(val_prediction - targets[val_ids])) * 1000)
        # Saved validation value is an integrity check of the exact selected fit.
        if not np.isclose(val_mae, expected_score, rtol=0, atol=1e-5):
            raise H3TestError(f"R4 inner-validation no reproduce fold={fold}: {val_mae} vs {expected_score} meV/atom.")
        outer_prediction = pipeline.predict(x[outer_ids])
        if not np.isfinite(outer_prediction).all():
            raise H3TestError(f"R4 outer-validation produce NaN/Inf fold={fold}.")
        ids_outer = dev.iloc[outer_ids].structure_id.astype(str).tolist()
        archived = oof.loc[oof.fold.eq(fold)].set_index("structure_id").reindex(ids_outer)
        if archived.E_pred_eV_atom.isna().any():
            raise H3TestError(f"R4 OOF IDs ausentes fold {fold}.")
        max_error = float(np.max(np.abs(outer_prediction - archived.E_pred_eV_atom.to_numpy(float))))
        if max_error > 1e-6:
            raise H3TestError(f"R4 outer OOF no reproducible fold={fold}; max error={max_error} eV/atom.")
        model_path = output_dir / f"ridge_fold_{fold}.joblib"
        joblib.dump(pipeline, model_path)
        scaler = pipeline.named_steps["scaler"]
        ridge = pipeline.named_steps["ridge"]
        metadata = {"status": "PASS", "fold": fold, "alpha_historical_selected": alpha,
                    "feature_variant": "L03_MEAN", "training_rows": len(train_ids),
                    "inner_validation_rows": len(val_ids), "outer_validation_rows": len(outer_ids),
                    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                    "ridge_coefficients_scaled_space": ridge.coef_.tolist(),
                    "intercept": float(ridge.intercept_), "inner_validation_MAE_reproduced_mev_atom": val_mae,
                    "inner_validation_MAE_archived_mev_atom": expected_score,
                    "outer_oof_max_abs_difference_ev_atom": max_error,
                    "model_sha256": sha256_file(model_path), "test_used": False}
        write_json(model_path.with_suffix(".json"), metadata)
        records.append(metadata)
    return records


def audit() -> dict[str, Any]:
    """Audita procedencia, test disyunto y disponibilidad de quince H2 heads.

    Returns:
        JSON-ready preflight metadata. No test DFT target is loaded.
    """
    for path in (H3_MANIFEST, H3_REPORT, DATASET, FOLD_PATH, FEATURES, H2_FEATURES,
                 H2_OOF, R4_OOF, R4_SELECTION, CHECKPOINT, MODEL_INFO, SOURCE_MANIFEST):
        if not path.is_file():
            raise H3TestError(f"Falta fuente de auditoría: {path}")
    h3_manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if h3_manifest.get("status") != "PASS" or h3_manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise H3TestError("H3 development no certifica el checkpoint esperado.")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise H3TestError("SHA-256 del checkpoint MatGPT no coincide.")
    if source.get("resolved_commit") != EXPECTED_COMMIT:
        raise H3TestError("La procedencia local de MatGPT no es el commit histórico fijado.")
    if h3_manifest["input_hashes"]["dataset"]["sha256"] != EXPECTED_DATASET_SHA:
        raise H3TestError("El hash histórico del dataset no coincide con el preflight H3.")
    test_inputs = read_test_inputs_safely()
    dev = read_dev_targets_safely()
    test_ids = set(test_inputs.structure_id)
    dev_ids = set(dev.structure_id)
    if test_ids & dev_ids:
        raise H3TestError("IDs test se solapan con development.")
    test_formulas = set(test_inputs.composition_id)
    dev_formulas = set(dev.composition_id)
    if test_formulas & dev_formulas:
        raise H3TestError("Composiciones test se solapan con development.")
    folds = pd.read_csv(FOLD_PATH)
    if test_ids & set(folds.structure_id.astype(str)):
        raise H3TestError("El fold manifest de desarrollo contiene IDs del test reservado.")
    h2_models: list[dict[str, Any]] = []
    for fold in range(5):
        for seed in (42, 123, 2026):
            run_dir = H2_RUNS / f"fold_{fold}_seed_{seed}"
            head = run_dir / "head.pt"
            run_manifest_path = run_dir / "run_provenance.json"
            status_path = run_dir / "status.json"
            if not all(p.is_file() for p in (head, run_manifest_path, status_path)):
                raise H3TestError(f"Falta H2 checkpoint/procedencia fold={fold}, seed={seed}.")
            run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if run.get("status") != "PASS" or run.get("test_used") is not False or status.get("test_evaluated") is not False:
                raise H3TestError(f"H2 fold={fold}/seed={seed} no acredita entrenamiento sin test.")
            if run.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA or run.get("fold_manifest_sha256") != sha256_file(FOLD_PATH):
                raise H3TestError(f"Procedencia H2 inconsistente fold={fold}/seed={seed}.")
            trained_groups = set(run.get("train_groups", [])) | set(run.get("inner_validation_groups", []))
            if trained_groups & test_formulas:
                raise H3TestError(f"Fuga de composición test en checkpoint H2 fold={fold}/seed={seed}.")
            h2_models.append({"fold": fold, "seed": seed, "path": str(head.relative_to(Path.cwd())),
                              "sha256": sha256_file(head), "best_epoch": run["best_epoch"],
                              "checkpoint_metadata_sha256": sha256_file(run_manifest_path)})
    if len(h2_models) != 15:
        raise H3TestError("Se requieren los quince checkpoints H2 originales.")
    r4_files = list(R4_ROOT.rglob("*.joblib")) + list(R4_ROOT.rglob("*.pkl"))
    ridge_models = _audit_reconstruct_r4()
    audit_path = REPORT_DIR / "provenance/model_audit.json"
    result = {"status": "PASS", "stage": "audit_before_test_labels", "test_labels_loaded": False,
              "checkpoint_sha256": sha256_file(CHECKPOINT), "model_info_sha256": sha256_file(MODEL_INFO),
              "source_commit": source["resolved_commit"], "dataset_sha256_previously_audited": EXPECTED_DATASET_SHA,
              "test_structures": len(test_inputs), "test_compositions": test_inputs.composition_id.nunique(),
              "test_structure_ids": sorted(test_ids), "test_compositions_list": sorted(test_formulas),
              "development_structures": len(dev), "development_compositions": dev.composition_id.nunique(),
              "test_ids_disjoint_from_development_and_fold_manifest": True,
              "R4_serialized_estimators_found": len(r4_files),
              "R4_reconstruction": "Ridge + StandardScaler, exact archived L03_MEAN/alpha per fold, fit historical inner_train only; verified against historical inner validation and outer OOF before any test prediction.",
              "R4_models": ridge_models, "H2_models": h2_models,
              "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST),
              "feature_cache_sha256": sha256_file(FEATURES), "fold_manifest_sha256": sha256_file(FOLD_PATH),
              "R4_validation_selection_sha256": sha256_file(R4_SELECTION),
              "R4_oof_sha256": sha256_file(R4_OOF), "H2_oof_sha256": sha256_file(H2_OOF),
              "test_label_columns_accessed": [], "status_checks": "PASS"}
    write_json(audit_path, result)
    return result


def main() -> None:
    """Ejecuta auditoría de modelos R4/H2 sin leer targets del test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = audit()
        print(json.dumps({"status": result["status"], "R4_models_reconstructed": len(result["R4_models"]),
                          "H2_checkpoints": len(result["H2_models"]), "test_labels_loaded": False}, ensure_ascii=False))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3-TEST audit FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
