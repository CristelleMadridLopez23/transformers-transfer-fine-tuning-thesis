#!/usr/bin/env python3
"""Guarda entradas del test y fija el protocolo antes de cargar etiquetas DFT."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import (CHECKPOINT, EXPECTED_CHECKPOINT_SHA, EXPECTED_COMMIT, FAMILIES,
                    FOLDS, MODEL_INFO, REPORT_DIR, SCRIPT_DIR, SOURCE_MANIFEST, H3TestError,
                    read_test_inputs_safely, sha256_file, write_json)


def _hash_scripts() -> dict[str, str]:
    """Calcula hashes de todos los scripts H3-TEST antes de crear el lock."""
    return {str(path.relative_to(Path.cwd())): sha256_file(path)
            for path in sorted(SCRIPT_DIR.rglob("*.py")) if "__pycache__" not in path.parts}


def prepare() -> dict[str, Any]:
    """Persiste inputs únicamente y crea el manifiesto inmutable pre-etiquetas.

    Returns:
        El contenido del lock que bloquea modelos, ensemble y métricas.
    """
    audit_path = REPORT_DIR / "provenance/model_audit.json"
    input_path = REPORT_DIR / "predictions/test_inputs.csv"
    lock_path = REPORT_DIR / "provenance/test_evaluation_lock.json"
    previous_lock_sha = None
    if lock_path.exists():
        feature_path = REPORT_DIR / "predictions/test_features.npz"
        feature_manifest_path = REPORT_DIR / "predictions/test_features_manifest.json"
        prediction_path = REPORT_DIR / "predictions/test_predictions_prelabels.csv"
        if prediction_path.exists():
            raise H3TestError("Ya existe inferencia; no se puede reemplazar el lock de protocolo.")
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if previous.get("status") != "LOCKED_BEFORE_TEST_LABELS" or previous.get("test_label_columns_read") != []:
            raise H3TestError("El lock existente no es un lock previo limpio sin acceso a etiquetas.")
        previous_lock_sha = sha256_file(lock_path)
        archived = REPORT_DIR / f"provenance/test_evaluation_lock_superseded_{len(list((REPORT_DIR / 'provenance').glob('test_evaluation_lock_superseded_*.json'))) + 1}.json"
        if archived.exists():
            raise H3TestError("Ya existe archivo del lock anterior; se preserva y no se sobrescribe.")
        shutil.copy2(lock_path, archived)
        if feature_path.exists() or feature_manifest_path.exists():
            if not (feature_path.is_file() and feature_manifest_path.is_file()):
                raise H3TestError("Cache de embeddings incompleto; no se puede emitir lock sucesor.")
            feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
            if (feature_manifest.get("status") != "PASS" or feature_manifest.get("test_labels_loaded") is not False
                    or feature_manifest.get("input_csv_sha256") != previous["test_inputs"]["sha256"]
                    or sha256_file(feature_path) != feature_manifest.get("feature_archive_sha256")):
                raise H3TestError("Cache de embeddings no corresponde al protocolo previo sin etiquetas.")
    if not audit_path.is_file():
        raise H3TestError("Ejecuta primero audit_final_models.py y resuelve cualquier bloqueo.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("test_labels_loaded") is not False:
        raise H3TestError("La auditoría de modelos no habilita la fase de lock.")
    frame = read_test_inputs_safely()
    if {"energy_total_ev", "energy_per_atom_ev", "num_atoms"} & set(frame.columns):
        raise H3TestError("El archivo de entradas contiene columnas de etiqueta prohibidas.")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.exists():
        existing = pd.read_csv(input_path, dtype={"structure_id": str, "composition_id": str})
        try:
            pd.testing.assert_frame_equal(existing, frame, check_dtype=False, check_like=False)
        except AssertionError as exc:
            raise H3TestError("Las entradas existentes no son idénticas a las revalidadas sin etiquetas.") from exc
    else:
        frame.to_csv(input_path, index=False)
    if not set(frame.polymorph_family).issubset(FAMILIES):
        raise H3TestError("Familia polimórfica desconocida.")
    ridge_models = []
    for fold, audit_model in enumerate(audit["R4_models"]):
        path = REPORT_DIR / "locked_models/ridge_by_fold" / f"ridge_fold_{fold}.joblib"
        if not path.is_file() or sha256_file(path) != audit_model["model_sha256"]:
            raise H3TestError(f"Ridge de fold {fold} cambió desde la auditoría.")
        ridge_models.append({"fold": fold, "path": str(path.relative_to(Path.cwd())),
                             "sha256": sha256_file(path), "alpha": audit_model["alpha_historical_selected"],
                             "feature": "L03_MEAN", "training_rows": audit_model["training_rows"]})
    h2_models = audit["H2_models"]
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    config_path = SCRIPT_DIR / "config.yaml"
    lock: dict[str, Any] = {
        "status": "LOCKED_BEFORE_TEST_LABELS", "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_primary": "H3-A", "model_comparison": "H2",
        "checkpoint": {"path": str(CHECKPOINT.relative_to(Path.cwd())), "sha256": sha256_file(CHECKPOINT),
                       "expected_sha256": EXPECTED_CHECKPOINT_SHA},
        "matgpt_source_commit": EXPECTED_COMMIT, "project_git_head": commit,
        "tokenizer_vocabulary": {"path": str(MODEL_INFO.relative_to(Path.cwd())), "sha256": sha256_file(MODEL_INFO),
                                 "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
                                 "resolved_commit": source.get("resolved_commit")},
        "dataset_hash_from_pretest_audit": audit["dataset_sha256_previously_audited"],
        "fold_manifest": {"path": str(FOLDS.relative_to(Path.cwd())), "sha256": sha256_file(FOLDS),
                           "test_ids_in_manifest": 0},
        "test_inputs": {"path": str(input_path.relative_to(Path.cwd())), "sha256": sha256_file(input_path),
                        "structures": len(frame), "compositions": int(frame.composition_id.nunique()),
                        "columns": frame.columns.tolist(), "no_label_columns": True},
        "features": {"R4": "L03_MEAN: raw decoder block 3 mean over SLICES tokens, pre-ln_f",
                     "H2_composition": "FINAL_MEAN: post-ln_f masked mean over SLICES tokens",
                     "H2_residual": "FINAL_END: post-ln_f hidden state at true <end> token"},
        "R4_historical_models": ridge_models,
        "H2_historical_models": h2_models,
        "ensemble_rule": {
            "H2": "mean of energies from the same 5 folds x 3 seeds",
            "H3-A": "mean over folds/seeds of (per-fold R4 mean of its 3 energy predictions + H2 energy centered within each composition)",
            "no_fold_or_seed_selection_on_test": True},
        "metrics_locked": ["absolute MAE/RMSE/bias/R2 prediction/R2 OLS", "composition-mean MAE",
                           "centered residual MAE", "DFT-GS-anchored and own-min delta MAE",
                           "GS with DFT gap >1 meV/atom", "pair ordering with DFT ties excluded at 1 meV/atom",
                           "composition-level paired bootstrap and exact binomial intervals"],
        "tie_rules": {"GS_DFT_gap_exclusion_mev_atom": 1.0, "predicted_GS_abstention_tolerance_mev_atom": 1.0,
                      "pairwise_DFT_tie_tolerance_mev_atom": 1.0, "pairwise_predicted_ties_are_incorrect": True},
        "bootstrap": {"replicates": 20000, "seed": 13072026, "sampling_unit": "composition triplet"},
        "config_sha256": sha256_file(config_path), "script_sha256_at_lock": _hash_scripts(),
        "reconstruction_audit_sha256": sha256_file(audit_path),
        "supersedes_previous_lock_sha256": previous_lock_sha,
        "lock_revision_reason": "Corrección de ejecución/tolerancia de precisión numérica antes de leer etiquetas; mismos datos, pesos, predicciones y métricas bloqueadas.",
        "test_label_columns_read": [], "strategy_locked": True,
    }
    write_json(lock_path, lock)
    return lock


def main() -> None:
    """Crea un lock único antes de predicción y carga de targets reservados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        lock = prepare()
        print(json.dumps({"status": lock["status"], "structures": lock["test_inputs"]["structures"],
                          "models_locked": len(lock["H2_historical_models"]) + len(lock["R4_historical_models"]),
                          "test_labels_read": False}, ensure_ascii=False))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3-TEST lock FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
