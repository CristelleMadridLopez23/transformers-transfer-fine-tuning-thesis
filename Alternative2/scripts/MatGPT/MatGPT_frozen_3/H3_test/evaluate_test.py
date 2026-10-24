#!/usr/bin/env python3
"""Abre labels test solo después de fijar y verificar sus predicciones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import (DATASET, FAMILIES, REPORT_DIR, SCRIPT_DIR, H3TestError,
                    read_test_inputs_safely, read_test_labels_after_lock,
                    sha256_file, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.metrics import bootstrap_paired, summarize_model


def evaluate() -> dict[str, Any]:
    """Verifica hash pre-etiquetas, carga 24 labels y calcula H2/H3-A.

    Returns:
        Resumen de métricas e invariantes para ambos ensembles.
    """
    lock_path = REPORT_DIR / "provenance/test_evaluation_lock.json"
    input_path = REPORT_DIR / "predictions/test_inputs.csv"
    prediction_path = REPORT_DIR / "predictions/test_predictions_prelabels.csv"
    ensemble_path = REPORT_DIR / "predictions/test_ensemble_predictions_prelabels.csv"
    pred_manifest_path = REPORT_DIR / "predictions/prelabel_prediction_manifest.json"
    if not all(path.is_file() for path in (lock_path, input_path, prediction_path, ensemble_path, pred_manifest_path)):
        raise H3TestError("Faltan el lock, predictions pre-label o su manifiesto.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pred_manifest = json.loads(pred_manifest_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST_LABELS" or lock.get("test_label_columns_read") != []:
        raise H3TestError("Lock inválido o ya contiene acceso a targets.")
    if pred_manifest.get("status") != "PASS" or pred_manifest.get("test_labels_loaded") is not False:
        raise H3TestError("Predicciones no certifican estado pre-etiquetas PASS.")
    current_scripts = {str(path.relative_to(Path.cwd())): sha256_file(path)
                       for path in sorted(SCRIPT_DIR.rglob("*.py")) if "__pycache__" not in path.parts}
    if current_scripts != lock.get("script_sha256_at_lock"):
        raise H3TestError("Los scripts cambiaron después del lock; no se permite abrir etiquetas con otro protocolo.")
    if sha256_file(input_path) != lock["test_inputs"]["sha256"]:
        raise H3TestError("Las entradas test cambiaron después del lock.")
    if sha256_file(prediction_path) != pred_manifest["test_predictions_prelabels_sha256"]:
        raise H3TestError("El CSV prelabels cambió después de congelar sus predicciones.")
    if sha256_file(ensemble_path) != pred_manifest["ensemble_predictions_prelabels_sha256"]:
        raise H3TestError("El ensemble prelabels cambió después de congelar sus predicciones.")
    per_run = pd.read_csv(prediction_path)
    ensemble = pd.read_csv(ensemble_path)
    dft_names = {"E_DFT_eV_atom", "E_DFT_total_eV", "energy_total_ev", "energy_per_atom_ev", "num_atoms_DFT"}
    if dft_names & set(per_run.columns) or dft_names & set(ensemble.columns):
        raise H3TestError("Las predicciones prelabels contienen información de referencia DFT.")
    inputs = read_test_inputs_safely(DATASET)
    # Verify source bytes after lock but before parsing any test target columns.
    dataset_sha_after_lock = sha256_file(DATASET)
    if dataset_sha_after_lock != lock["dataset_hash_from_pretest_audit"]:
        raise H3TestError("El archivo fuente cambió respecto al hash pre-auditado del dataset.")
    if inputs.structure_id.tolist() != pd.read_csv(input_path).structure_id.astype(str).tolist():
        raise H3TestError("La tabla de entradas no conserva los IDs bloqueados.")

    # First and only point at which the reserved DFT labels are read.
    labels = read_test_labels_after_lock(DATASET)
    actual = inputs.merge(labels, on="structure_id", how="left", validate="one_to_one")
    if actual.E_DFT_eV_atom.isna().any() or len(actual) != 24:
        raise H3TestError("No se alinearon exactamente las 24 etiquetas test.")
    if set(actual.composition_id) != set(inputs.composition_id) or actual.groupby("composition_id").size().ne(3).any():
        raise H3TestError("Etiquetas/entradas test no conservan composición y triplete.")
    evaluated = actual.merge(ensemble, on=["structure_id", "composition_id", "polymorph_family"],
                             how="left", validate="one_to_one")
    pred_columns = ["E_pred_H2_eV_atom", "E_pred_H3A_eV_atom"]
    if evaluated[pred_columns].isna().any().any() or not np.isfinite(evaluated[pred_columns].to_numpy(float)).all():
        raise H3TestError("Faltan predicciones para evaluar.")
    if len(per_run) != 360 or per_run.duplicated(["structure_id", "fold", "seed"]).any():
        raise H3TestError("La tabla fold×semilla tiene que conservar 360 predicciones únicas.")
    if set(per_run.structure_id.astype(str)) != set(actual.structure_id.astype(str)):
        raise H3TestError("La población del CSV prelabels no coincide con las 24 estructuras.")

    summary_rows, component_rows, pairs_by_model = [], [], {}
    for model, column in (("H2", "E_pred_H2_eV_atom"), ("H3-A", "E_pred_H3A_eV_atom")):
        summary, composition_metrics, pairs = summarize_model(evaluated, column)
        summary["model"] = model
        summary_rows.append(summary)
        composition_metrics.insert(0, "model", model)
        component_rows.append(composition_metrics)
        pairs.insert(0, "model", model)
        pairs_by_model[model] = pairs
    metrics = pd.DataFrame(summary_rows)
    comps = pd.concat(component_rows, ignore_index=True)
    pair_frame = pd.concat(pairs_by_model.values(), ignore_index=True)
    # Main invariant: H3-A adds only a triplet-constant Ridge mean, so all
    # relative energy and ranking metrics must match H2 exactly (numerically).
    invariant_metrics = ["MAE_delta_DFT_GS_anchored_mev_atom_macro", "MAE_delta_own_min_mev_atom_macro",
                         "GS_accuracy", "pairwise_accuracy", "SignedPairwiseMAE_mev_atom"]
    h2, h3 = metrics.set_index("model").loc["H2"], metrics.set_index("model").loc["H3-A"]
    invariant_differences = {key: float(h3[key] - h2[key]) for key in invariant_metrics}
    if any(abs(value) > 1e-10 for value in invariant_differences.values()):
        raise H3TestError(f"H3-A/H2 ranking metrics violate the locked identity: {invariant_differences}")
    h2_comp = comps.loc[comps.model.eq("H2")].set_index("composition_id")
    h3_comp = comps.loc[comps.model.eq("H3-A")].set_index("composition_id")
    if h2_comp.DFT_GS_family.to_dict() != h3_comp.DFT_GS_family.to_dict() or \
            h2_comp.predicted_GS_family.to_dict() != h3_comp.predicted_GS_family.to_dict() or \
            h2_comp.pairwise_correct_count.to_dict() != h3_comp.pairwise_correct_count.to_dict():
        raise H3TestError("Identidad de GS/pairwise por fórmula H3-A vs H2 falló.")
    bootstrap = bootstrap_paired(evaluated, "E_pred_H3A_eV_atom", "E_pred_H2_eV_atom", 20000, 13072026)
    metrics_dir = REPORT_DIR / "metrics"
    predictions_dir = REPORT_DIR / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Explicit ID-level table is the first artifact containing test labels.
    evaluated.to_csv(predictions_dir / "test_predictions_with_dft.csv", index=False)
    metrics.to_csv(metrics_dir / "test_metrics_summary.csv", index=False)
    comps.to_csv(metrics_dir / "test_metrics_by_composition.csv", index=False)
    pair_frame.to_csv(metrics_dir / "test_pairwise_results.csv", index=False)
    write_json(metrics_dir / "test_bootstrap_and_intervals.json", bootstrap)
    integrity = {"status": "PASS", "test_labels_loaded_after_lock": True,
        "lock_sha256": sha256_file(lock_path), "prelabel_predictions_sha256": sha256_file(prediction_path),
        "prelabel_ensemble_sha256": sha256_file(ensemble_path), "dataset_sha256_checked_after_lock": dataset_sha_after_lock,
        "N_structures": len(evaluated), "N_compositions": int(evaluated.composition_id.nunique()),
        "all_triplets_complete": True, "test_used_for_selection": False,
        "H3A_H2_relative_and_ranking_metric_differences": invariant_differences,
        "MSE_decomposition_max_abs_error": float(comps.MSE_identity_abs_error.max()),
        "bootstrap_replicates": 20000, "test_predictions_regenerated_after_labels": False}
    write_json(metrics_dir / "evaluation_integrity.json", integrity)
    return {"metrics": metrics.to_dict(orient="records"), "integrity": integrity,
            "bootstrap": bootstrap, "by_composition": comps}


def main() -> None:
    """Ejecuta evaluación bloqueada una sola vez; no ajusta parámetros."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = evaluate()
        print(json.dumps({"status": result["integrity"]["status"],
                          "N_structures": result["integrity"]["N_structures"],
                          "N_compositions": result["integrity"]["N_compositions"],
                          "metrics": result["metrics"], "test_used_for_selection": False},
                         ensure_ascii=False, default=float))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3-TEST evaluation FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
