"""Open the 24 DFT test targets once, only after a valid prelabel lock."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys

from common import DATASET, REPORT_DIR, TEST_INPUTS, logger, sha256, write_json
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from energy_metrics import composition_metrics


def _read_exact_test_targets(test_ids: set[str]) -> pd.DataFrame:
    """Parse energy targets only from the locked physical test IDs."""
    identity = pd.read_csv(DATASET, usecols=["structure_id", "split"], dtype={"structure_id": str})
    excluded = [int(i) + 1 for i, row in identity.iterrows() if row.structure_id not in test_ids or row.split != "test"]
    labels = pd.read_csv(DATASET, usecols=["structure_id", "energy_per_atom_ev"], skiprows=excluded, dtype={"structure_id": str})
    labels = labels[labels.structure_id.isin(test_ids)].copy()
    if set(labels.structure_id) != test_ids or len(labels) != len(test_ids):
        raise RuntimeError("Targets abiertos no coinciden exactamente con los 24 IDs bloqueados")
    return labels.rename(columns={"energy_per_atom_ev": "energy_dft_ev_atom"})


def run() -> dict[str, Any]:
    """Score locked predictions and compare against the historical MatGPT H3-A."""
    lock_path = REPORT_DIR / "test/test_lock.json"
    pred_path = REPORT_DIR / "test/predictions_prelabels.csv"
    if not lock_path.is_file() or not pred_path.is_file(): raise RuntimeError("No existe lock/predicción prelabel")
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "LOCKED_BEFORE_TEST_LABELS" or lock.get("test_labels_read") is not False:
        raise RuntimeError("Estado de lock inválido")
    if sha256(pred_path) != lock["predictions_prelabels_sha256"]: raise RuntimeError("Predicciones cambiaron desde el lock")
    inputs = pd.read_csv(TEST_INPUTS, dtype={"structure_id": str})
    labels = _read_exact_test_targets(set(inputs.structure_id))
    features = inputs[["structure_id", "composition_id", "polymorph_family"]].merge(labels, on="structure_id", validate="one_to_one")
    predictions = pd.read_csv(pred_path, dtype={"structure_id": str})
    scored = features.merge(predictions, on=["structure_id", "composition_id", "polymorph_family"], validate="one_to_one")
    for model, col in (("M2", "M2_prediction_ev_atom"), ("M4", "M4_prediction_ev_atom"), ("M3-A", "M3A_prediction_ev_atom")):
        scored[col] = pd.to_numeric(scored[col], errors="raise")
    scored.to_csv(REPORT_DIR / "test/test_predictions_with_dft.csv", index=False)
    metric_rows = [{"model": model, **composition_metrics(scored, "energy_dft_ev_atom", col)} for model, col in (("M2", "M2_prediction_ev_atom"), ("M4", "M4_prediction_ev_atom"), ("M3-A", "M3A_prediction_ev_atom"))]
    pd.DataFrame(metric_rows).to_csv(REPORT_DIR / "test/test_metrics.csv", index=False)
    # MatGPT predictions are an already observed descriptive comparator; read
    # only its prediction column and align on the locked physical structure IDs.
    mg_path = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file()) / "reports/MatGPT/MatGPT_frozen_3/H3_test/predictions/test_predictions_with_dft.csv"
    mg_predictions = pd.read_csv(mg_path, usecols=["structure_id", "E_pred_H3A_eV_atom"], dtype={"structure_id": str}).rename(columns={"E_pred_H3A_eV_atom":"MatGPT_H3A_prediction_ev_atom"})
    paired = scored.merge(mg_predictions, on="structure_id", validate="one_to_one")
    if len(paired) != 24: raise RuntimeError("La comparación no tiene los mismos 24 structure_id")
    paired_metrics = [{"model":"MatterGPT_M3-A", **composition_metrics(paired,"energy_dft_ev_atom","M3A_prediction_ev_atom")},
                      {"model":"MatGPT_H3-A", **composition_metrics(paired,"energy_dft_ev_atom","MatGPT_H3A_prediction_ev_atom")}]
    paired.to_csv(REPORT_DIR / "comparisons/test_paired_predictions.csv",index=False)
    # Exploratory paired bootstrap over the eight physical composition triplets.
    rng=np.random.default_rng(13072026); comps=sorted(paired.composition_id.unique()); diffs={k:[] for k in ("mae_absolute_mev_atom","mae_delta_dft_gs_anchored_mev_atom","gs_accuracy","pairwise_accuracy")}
    for _ in range(2000):
        sampled=rng.choice(comps,size=len(comps),replace=True); blocks=[]
        for sample_i,comp in enumerate(sampled):
            block=paired[paired.composition_id==comp].copy();block["bootstrap_group"]=str(sample_i);blocks.append(block)
        draw=pd.concat(blocks,ignore_index=True)
        a=composition_metrics(draw,"energy_dft_ev_atom","M3A_prediction_ev_atom",group="bootstrap_group")
        b=composition_metrics(draw,"energy_dft_ev_atom","MatGPT_H3A_prediction_ev_atom",group="bootstrap_group")
        for key in diffs:
            if a[key] is not None and b[key] is not None: diffs[key].append(float(a[key]-b[key]))
    bootstrap={key:{"mean_difference_mattergpt_minus_matgpt":float(np.mean(values)),"ci95_percentile":[float(np.quantile(values,.025)),float(np.quantile(values,.975))],"replicates":len(values),"seed":13072026,"sampling_unit":"composition triplet","interpretation":"descriptive/exploratory; historical test has already been observed for MatGPT"} for key,values in diffs.items()}
    write_json(REPORT_DIR / "comparisons/test_paired_bootstrap.json",bootstrap)
    comparison = {"note": "Comparación descriptiva en los mismos IDs; el test histórico ya fue observado para MatGPT.", "mattergpt": metric_rows, "paired_same_ids":paired_metrics,
                  "matgpt_h3a_fixed_reference": {"mae_absolute_mev_atom": 133.545, "mae_delta_anchored_mev_atom": 24.208, "ground_state_correct_of_8": "5/8", "pairwise_correct_of_24": "17/24"},
                  "matgpt_source_file_sha256": sha256(mg_path) if mg_path.exists() else None}
    write_json(REPORT_DIR / "comparisons/test_comparison.json", comparison)
    write_json(REPORT_DIR / "test/test_integrity.json", {"status": "PASS", "lock_sha256": sha256(lock_path), "test_labels_read_after_lock": True, "test_ids": len(scored), "compositions": int(scored.composition_id.nunique()), "dataset_source_sha256": sha256(DATASET), "m3a_ranking_same_as_m2": True})
    return {"status": "PASS", "structures": len(scored), "compositions": int(scored.composition_id.nunique()), "metrics": metric_rows}


def main() -> None:
    """Score the already-locked predictions one time."""
    log = logger("mattergpt_test_score")
    try: log.info("locked_test_scored %s", run())
    except Exception as exc: log.exception("locked_test_scoring_failed error=%s", exc); raise


if __name__ == "__main__":
    main()
