#!/usr/bin/env python3
"""Predice H2 y H3-A con cinco Ridge y quince MLP8 congeladas, sin etiquetas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import (DATASET, FAMILIES, H2_FEATURES, H2_OOF, REPORT_DIR,
                    H3TestError, read_dev_targets_safely, sha256_file, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.load_frozen_ensemble import audit_loads, load_h2, load_ridge


def _triplet_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordena familias en orden fijo perovskita, hexagonal y aguja."""
    rank = {family: i for i, family in enumerate(FAMILIES)}
    result = frame.assign(_family_order=frame.polymorph_family.map(rank)).sort_values(
        ["composition_id", "_family_order"], kind="stable")
    if len(result) % 3 or result.groupby("composition_id").size().ne(3).any():
        raise H3TestError("Predicción requiere tripletes físicos completos.")
    return result.drop(columns="_family_order")


def _run_h2(model: Any, payload: dict[str, Any], h_mean: np.ndarray,
            h_end: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve μ, residuo centrado y energía H2 por triplete.

    Args:
        model: Cabezas H2 cargadas strict=True y en eval.
        payload: Escalas del checkpoint original.
        h_mean: FINAL_MEAN en orden canónico [3,768].
        h_end: FINAL_END en orden canónico [3,768].
        device: CPU para estas cabezas pequeñas.
    Returns:
        Media escalar, tres residuos centrados y tres energías eV/átomo.
    """
    x_mean = torch.tensor(np.array(h_mean, copy=True), dtype=torch.float32, device=device).unsqueeze(0)
    x_end = torch.tensor(np.array(h_end, copy=True), dtype=torch.float32, device=device).unsqueeze(0)
    if torch.is_inference(x_mean) or torch.is_inference(x_end) or x_mean.requires_grad or x_end.requires_grad:
        raise H3TestError("Features deben ser tensores ordinarios sin autograd.")
    scales = payload["scales"]
    with torch.no_grad():
        mu_raw, delta_raw, _ = model(x_mean.mean(dim=1), x_end)
        mu = mu_raw.squeeze(0) * float(scales["scale_comp_ev_atom"]) + float(scales["target_mean_ev_atom"])
        delta = delta_raw.squeeze(0) * float(scales["scale_struct_ev_atom"])
    mu_np = np.full(3, float(mu.item()), dtype=np.float64)
    delta_np = delta.cpu().numpy().astype(np.float64)
    # Convert the two model heads first, then reconstruct in float64. This
    # avoids float32 rounding in (mu + delta) and preserves H3-A's exact
    # relative-energy identity by construction.
    energy_np = mu_np + delta_np
    if abs(float(delta_np.sum())) > 1e-7 or not np.isfinite(energy_np).all():
        raise H3TestError("La cabeza H2 produjo valores no finitos o residuos no centrados.")
    return mu_np, delta_np, energy_np


def predict(device: str = "cpu") -> dict[str, Any]:
    """Ejecuta replay OOF y predicciones test H2/H3-A pre-etiquetas.

    Args:
        device: Dispositivo para MLP8 pequeñas.
    Returns:
        Manifiesto de predicción e identidades algebraicas.
    """
    if device != "cpu":
        raise H3TestError("El protocolo bloqueado fija inferencia de cabezas en CPU.")
    lock_path = REPORT_DIR / "provenance/test_evaluation_lock.json"
    inputs_path = REPORT_DIR / "predictions/test_inputs.csv"
    features_path = REPORT_DIR / "predictions/test_features.npz"
    feature_manifest_path = features_path.with_name("test_features_manifest.json")
    prediction_path = REPORT_DIR / "predictions/test_predictions_prelabels.csv"
    ensemble_path = REPORT_DIR / "predictions/test_ensemble_predictions_prelabels.csv"
    manifest_path = REPORT_DIR / "predictions/prelabel_prediction_manifest.json"
    if not all(path.is_file() for path in (lock_path, inputs_path, features_path, feature_manifest_path)):
        raise H3TestError("Lock, entradas o embeddings test incompletos.")
    if any(path.exists() for path in (prediction_path, ensemble_path, manifest_path)):
        raise H3TestError("Predicciones pre-etiquetas ya existen; no se regeneran.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(inputs_path) != lock["test_inputs"]["sha256"]:
        raise H3TestError("Las entradas no coinciden con el lock.")
    if feature_manifest.get("test_labels_loaded") is not False or feature_manifest.get("status") != "PASS":
        raise H3TestError("Cache de features no certifica extracción sin test labels.")
    if sha256_file(features_path) != feature_manifest["feature_archive_sha256"]:
        raise H3TestError("El cache de test cambió desde su manifiesto.")
    load_audit = audit_loads(lock, device)
    inputs = pd.read_csv(inputs_path, dtype={"structure_id": str, "composition_id": str})
    inputs = _triplet_order(inputs).reset_index(drop=True)
    with np.load(features_path, allow_pickle=False) as archive:
        feature_ids = archive["structure_id"].astype(str).tolist()
        if feature_ids != inputs.structure_id.tolist():
            raise H3TestError("Orden de IDs de embeddings distinto al archivo de entradas bloqueado.")
        l03 = np.array(archive["L03_MEAN"], dtype=np.float32, copy=True)
        final_end = np.array(archive["FINAL_END"], dtype=np.float32, copy=True)
        final_mean = np.array(archive["FINAL_MEAN"], dtype=np.float32, copy=True)
    if any(array.shape != (24, 768) or not np.isfinite(array).all() for array in (l03, final_end, final_mean)):
        raise H3TestError("Features test con forma/valores inválidos.")
    test_index = {sid: ix for ix, sid in enumerate(inputs.structure_id)}

    # Verify historical H2 checkpoints against their actual outer OOF on the
    # exact development feature cache before applying them to test.
    dev = read_dev_targets_safely()
    with np.load(H2_FEATURES, allow_pickle=False) as archive:
        dev_feature_ids = archive["structure_id"].astype(str).tolist()
        if dev_feature_ids != dev.structure_id.tolist():
            raise H3TestError("Cache histórico H2 development no alinea.")
        h2_end_dev = np.array(archive["h_end"], dtype=np.float32, copy=True)
        h2_mean_dev = np.array(archive["h_mean"], dtype=np.float32, copy=True)
    folds = pd.read_csv(lock["fold_manifest"]["path"])
    if sha256_file(Path(lock["fold_manifest"]["path"])) != lock["fold_manifest"]["sha256"]:
        raise H3TestError("Fold manifest cambió desde el lock.")
    archived_h2 = pd.read_csv(H2_OOF)
    archived_h2 = archived_h2.loc[archived_h2.model.eq("H2")].copy()
    replay_rows = []
    prediction_rows = []
    for fold in range(5):
        ridge = load_ridge(fold, lock)
        # R4's selected absolute model was trained on the historical inner-train
        # structures and verified against this fold's archived OOF.
        fold_r4 = ridge.predict(l03.astype(np.float64))
        if not np.isfinite(fold_r4).all():
            raise H3TestError(f"R4 fold {fold} genera NaN/Inf sobre las entradas bloqueadas de test.")
        r4_by_id = dict(zip(inputs.structure_id, fold_r4))
        r4_mu = {str(formula): float(np.mean([r4_by_id[sid] for sid in part.structure_id]))
                 for formula, part in inputs.groupby("composition_id", sort=False)}
        fold_table = folds.loc[folds.outer_fold.eq(fold)].set_index("structure_id")
        outer_ids = set(fold_table.loc[fold_table.outer_role.eq("outer_validation")].index.astype(str))
        dev_rows = dev.loc[dev.structure_id.isin(outer_ids)].copy()
        dev_rows = _triplet_order(dev_rows).reset_index(drop=True)
        dev_idx = {sid: ix for ix, sid in enumerate(dev.structure_id)}
        for seed in (42, 123, 2026):
            model, payload = load_h2(fold, seed, lock, device)
            for formula, group in dev_rows.groupby("composition_id", sort=False):
                row_ids = [dev_idx[sid] for sid in group.structure_id]
                mu_h2, delta_h2, energy_h2 = _run_h2(model, payload, h2_mean_dev[row_ids], h2_end_dev[row_ids], device)
                for local_i, source in enumerate(group.itertuples(index=False)):
                    archived = archived_h2.loc[(archived_h2.fold.eq(fold)) & (archived_h2.seed.eq(seed)) &
                                                archived_h2.structure_id.astype(str).eq(str(source.structure_id))]
                    if len(archived) != 1:
                        raise H3TestError(f"H2 OOF histórico no contiene una fila única: {source.structure_id}.")
                    difference = abs(float(energy_h2[local_i]) - float(archived.iloc[0].E_pred_eV_atom))
                    if difference > 1e-6:
                        raise H3TestError(f"H2 checkpoint no reproduce OOF para {source.structure_id}: {difference} eV/átomo.")
                    replay_rows.append({"fold": fold, "seed": seed, "structure_id": str(source.structure_id),
                                        "max_abs_difference_ev_atom": difference})

            for formula, group in inputs.groupby("composition_id", sort=False):
                group = _triplet_order(group).reset_index(drop=True)
                idx = [test_index[sid] for sid in group.structure_id]
                mu_h2, delta_h2, energy_h2 = _run_h2(model, payload, final_mean[idx], final_end[idx], device)
                mean_r4 = r4_mu[str(formula)]
                energy_h3 = mean_r4 + delta_h2
                # The centered H2 residual is computed in float32; its sum can
                # leave sub-micro-eV/atom roundoff after the historical scaling.
                if abs(float(energy_h3.mean()) - mean_r4) > 1e-6:
                    raise H3TestError(f"H3-A compositional mean does not equal R4 on {formula}.")
                if np.max(np.abs((energy_h3[:, None] - energy_h3[None, :]) -
                                 (energy_h2[:, None] - energy_h2[None, :]))) > 1e-10:
                    raise H3TestError(f"H3-A/H2 pairwise differences differ on {formula}.")
                if int(np.argmin(energy_h3)) != int(np.argmin(energy_h2)):
                    raise H3TestError(f"H3-A/H2 GS differs by construction on {formula}.")
                for local_i, source in enumerate(group.itertuples(index=False)):
                    prediction_rows.append({"structure_id": str(source.structure_id), "composition_id": str(formula),
                        "polymorph_family": source.polymorph_family, "fold": fold, "seed": seed,
                        "E_pred_R4_absolute_eV_atom": float(r4_by_id[str(source.structure_id)]),
                        "E_pred_mu_R4_eV_atom": mean_r4, "E_pred_mu_H2_eV_atom": float(mu_h2[local_i]),
                        "delta_pred_H2_centered_eV_atom": float(delta_h2[local_i]),
                        "E_pred_H2_eV_atom": float(energy_h2[local_i]),
                        "E_pred_H3A_eV_atom": float(energy_h3[local_i])})
    per_run = pd.DataFrame(prediction_rows)
    if len(per_run) != 24 * 5 * 3 or per_run.duplicated(["structure_id", "fold", "seed"]).any():
        raise H3TestError("Se esperaban 360 predicciones únicas por estructura/fold/semilla.")
    ensemble = per_run.groupby(["structure_id", "composition_id", "polymorph_family"], as_index=False).agg(
        E_pred_R4_absolute_eV_atom=("E_pred_R4_absolute_eV_atom", "mean"),
        E_pred_mu_R4_eV_atom=("E_pred_mu_R4_eV_atom", "mean"),
        E_pred_mu_H2_eV_atom=("E_pred_mu_H2_eV_atom", "mean"),
        delta_pred_H2_centered_eV_atom=("delta_pred_H2_centered_eV_atom", "mean"),
        E_pred_H2_eV_atom=("E_pred_H2_eV_atom", "mean"),
        E_pred_H3A_eV_atom=("E_pred_H3A_eV_atom", "mean"))
    for formula, group in ensemble.groupby("composition_id"):
        if np.max(np.abs(group.E_pred_H3A_eV_atom.to_numpy() -
                         (group.E_pred_mu_R4_eV_atom.to_numpy() + group.delta_pred_H2_centered_eV_atom.to_numpy()))) > 1e-10:
            raise H3TestError(f"Ensemble H3-A no reconstruye mu+delta en {formula}.")
        if int(np.argmin(group.E_pred_H3A_eV_atom.to_numpy())) != int(np.argmin(group.E_pred_H2_eV_atom.to_numpy())):
            raise H3TestError(f"Ensemble H3-A/H2 GS no coincide en {formula}.")
    output_dir = prediction_path.parent
    per_run.to_csv(prediction_path, index=False)
    ensemble.to_csv(ensemble_path, index=False)
    pd.DataFrame(replay_rows).to_csv(output_dir / "H2_outer_oof_reproduction.csv", index=False)
    max_replay = max(row["max_abs_difference_ev_atom"] for row in replay_rows)
    if max_replay > 1e-6:
        raise H3TestError("La reproducción H2 excede tolerancia de 1e-6 eV/átomo.")
    pred_manifest = {"status": "PASS", "test_labels_loaded": False,
        "lock_sha256": sha256_file(lock_path), "test_predictions_prelabels_sha256": sha256_file(prediction_path),
        "ensemble_predictions_prelabels_sha256": sha256_file(ensemble_path),
        "rows_fold_seed": len(per_run), "rows_ensemble": len(ensemble),
        "models_used": {"Ridge": 5, "H2_MLP8": 15}, "H2_OOF_replay_rows": len(replay_rows),
        "H2_OOF_replay_max_abs_error_ev_atom": max_replay,
        "algebraic_checks": {"H3A_delta_centered": True, "ridge_mean_same_for_triplet": True,
                             "H3A_H2_pairwise_delta_identical": True, "H3A_H2_GS_identical": True},
        "no_dft_label_columns": True, "input_has_no_geometry": True,
        "prediction_script_sha256": sha256_file(Path(__file__)), "feature_archive_sha256": sha256_file(features_path),
        "model_load_audit_sha256": sha256_file(REPORT_DIR / "provenance/model_load_audit.json")}
    write_json(manifest_path, pred_manifest)
    replay_path = REPORT_DIR / "provenance/H2_outer_oof_reproduction.json"
    write_json(replay_path, {"status": "PASS", "rows": len(replay_rows),
                            "max_abs_error_ev_atom": max_replay, "tolerance_ev_atom": 1e-6,
                            "test_labels_loaded": False})
    return pred_manifest


def main() -> None:
    """Predice y congela H2/H3-A antes de permitir la fase de evaluación DFT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    try:
        result = predict(args.device)
        print(json.dumps({"status": result["status"], "rows": result["rows_fold_seed"],
                          "ensemble_rows": result["rows_ensemble"], "H2_replay_max_error": result["H2_OOF_replay_max_abs_error_ev_atom"],
                          "test_labels_loaded": False}, ensure_ascii=False))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3-TEST prediction FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
