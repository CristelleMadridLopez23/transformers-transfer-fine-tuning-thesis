"""Freeze the MatterGPT test protocol and write predictions before DFT labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from common import CONDITION, MODEL_DIR, REPORT_DIR, SEEDS, TEST_INPUTS, logger, sha256, stable_ridge_predict, write_json
from heads import TwoHead


def run() -> dict[str, Any]:
    """Ensemble all five folds and three seeds on label-free test features."""
    cv_path = REPORT_DIR / "cv/cv_manifest.json"
    invariant_path = REPORT_DIR / "h3a/invariance.json"
    if not cv_path.is_file() or not invariant_path.is_file() or json.loads(invariant_path.read_text()).get("status") != "PASS":
        raise RuntimeError("CV y M3-A deben estar completos y pasar invariancia antes del lock")
    features_path = REPORT_DIR / "test/test_features_prelabel.npz"
    metadata_path = REPORT_DIR / "test/test_metadata_prelabel.csv"
    if not features_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Falta cache de test sin etiquetas")
    archive = np.load(features_path); meta = pd.read_csv(metadata_path)
    source_inputs = pd.read_csv(TEST_INPUTS)
    if list(meta.structure_id.astype(str)) != list(source_inputs.structure_id.astype(str)):
        raise RuntimeError("El cache test no conserva el orden/IDs de test_inputs")
    out = []
    for fold in range(5):
        for seed in SEEDS:
            m2dir = REPORT_DIR / f"cv/runs/M2/fold_{fold}_seed_{seed}"
            transform = json.loads((m2dir / "training.json").read_text())
            head = TwoHead(archive["FINAL_END"].shape[1]); state = torch.load(m2dir / "head.pt", map_location="cpu", weights_only=True); head.load_state_dict(state); head.eval()
            m4obj = joblib.load(REPORT_DIR / f"cv/runs/M4/fold_{fold}_seed_{seed}/ridge.joblib")
            pred_m2 = np.full(len(meta), np.nan); pred_m4 = np.full(len(meta), np.nan)
            for composition in meta.composition_id.unique():
                idx = np.flatnonzero(meta.composition_id.to_numpy() == composition)
                if len(idx) != 3: raise RuntimeError(f"Test triplet incompleto: {composition}")
                idx = idx[np.argsort(meta.iloc[idx].polymorph_family.map({"dist_perovskite":0,"hexagonal":1,"needle":2}).to_numpy())]
                comp = archive["FINAL_MEAN"][idx].mean(axis=0, keepdims=True)
                struct = archive["FINAL_END"][idx][None, ...]
                with torch.inference_mode(): mu, delta, _ = head(torch.tensor(comp, dtype=torch.float32), torch.tensor(struct, dtype=torch.float32))
                values = mu.numpy()[:, None] * transform["mean_scale_ev_atom"] + transform["mean_center_ev_atom"] + delta.numpy() * transform["residual_scale_ev_atom"]
                pred_m2[idx] = values[0]
                pred_m4[idx] = stable_ridge_predict(m4obj["ridge"], m4obj["scaler"].transform(archive[m4obj["feature"]][idx]))
            # M3-A substitutes only each fold model's absolute composition mean.
            pred_m3a = pred_m4.copy()
            for composition in meta.composition_id.unique():
                idx = np.flatnonzero(meta.composition_id.to_numpy() == composition)
                pred_m3a[idx] = pred_m4[idx].mean() + pred_m2[idx] - pred_m2[idx].mean()
            for idx, row in meta.iterrows():
                out.append({"structure_id": row.structure_id, "composition_id": row.composition_id, "polymorph_family": row.polymorph_family,
                            "fold": fold, "seed": seed, "M2_prediction_ev_atom": float(pred_m2[idx]), "M4_prediction_ev_atom": float(pred_m4[idx]), "M3A_prediction_ev_atom": float(pred_m3a[idx])})
    pred = pd.DataFrame(out)
    if not np.isfinite(pred[["M2_prediction_ev_atom", "M4_prediction_ev_atom", "M3A_prediction_ev_atom"]].to_numpy()).all():
        raise RuntimeError("Predicciones prelabel no finitas")
    # Persist individual run predictions and ensemble before the test target reader exists.
    pred.to_csv(REPORT_DIR / "test/test_predictions_by_fold_seed_prelabels.csv", index=False)
    ensemble = pred.groupby(["structure_id", "composition_id", "polymorph_family"], as_index=False)[["M2_prediction_ev_atom", "M4_prediction_ev_atom", "M3A_prediction_ev_atom"]].mean()
    ensemble.to_csv(REPORT_DIR / "test/predictions_prelabels.csv", index=False)
    # Exact predicted relative invariance under the ensemble.
    max_error = 0.0
    for _comp, group in ensemble.groupby("composition_id"):
        a = group.sort_values("polymorph_family").M2_prediction_ev_atom.to_numpy()
        b = group.sort_values("polymorph_family").M3A_prediction_ev_atom.to_numpy()
        max_error = max(max_error, float(np.max(np.abs((a[:, None]-a[None,:])-(b[:,None]-b[None,:])))))
    if max_error > 1e-10: raise RuntimeError("M3-A no conserva las diferencias relativas en test")
    scripts = sorted(Path(__file__).resolve().parent.glob("*.py"))
    lock = {
        "status": "LOCKED_BEFORE_TEST_LABELS", "test_labels_read": False, "condition": list(CONDITION),
        "primary_model": "M3-A = per-fold M4 composition mean + centered M2 residual",
        "ensemble": "arithmetic mean over all 5 folds x 3 seeds for each structure; no selection",
        "folds": list(range(5)), "seeds": list(SEEDS), "ties": {"dft_gap_mev_atom": 1.0, "predicted_gs_abstention_mev_atom": 1.0, "pairwise_dft_tie_mev_atom": 1.0},
        "checkpoint_sha256": sha256(MODEL_DIR / "pytorch_model.pt"), "test_inputs_sha256": sha256(TEST_INPUTS),
        "test_feature_sha256": sha256(features_path), "predictions_prelabels_sha256": sha256(REPORT_DIR / "test/predictions_prelabels.csv"),
        "fold_manifest_sha256": sha256(Path(__file__).resolve().parents[4] / "Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"),
        "config_sha256": sha256(Path(__file__).resolve().parent / "config.yaml"),
        "scripts_sha256": {p.name: sha256(p) for p in scripts}, "m3a_relative_invariance_max_error_ev_atom": max_error,
        "limit": "Project history has already exposed MatGPT H3-A test results; MatterGPT protocol was fixed without inspecting those DFT targets."
    }
    write_json(REPORT_DIR / "test/test_lock.json", lock)
    return {"status": lock["status"], "rows": len(pred), "ensemble_rows": len(ensemble), "max_relative_error": max_error}


def main() -> None:
    """Lock decisions and generate prelabel predictions."""
    log = logger("mattergpt_test_lock")
    try: log.info("test_lock_complete %s", run())
    except Exception as exc: log.exception("test_lock_failed error=%s", exc); raise


if __name__ == "__main__":
    main()
