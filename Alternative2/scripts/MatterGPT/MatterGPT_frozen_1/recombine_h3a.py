"""Build the no-retraining M3-A recombination from OOF M2 and M4 predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import REPORT_DIR, logger, write_json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from energy_metrics import composition_metrics


def run() -> dict[str, Any]:
    """Align M2/M4 by structure, fold, seed and verify ranking invariance."""
    predictions = pd.read_csv(REPORT_DIR / "cv/oof_predictions.csv")
    needed = {"M2", "M4"}
    if not needed.issubset(set(predictions.model)):
        raise RuntimeError("M3-A requiere OOF de M2 y M4 completos")
    keys = ["structure_id", "composition_id", "polymorph_family", "outer_fold", "seed", "energy_dft_ev_atom"]
    m2 = predictions[predictions.model == "M2"][keys + ["prediction_ev_atom"]].rename(columns={"prediction_ev_atom": "m2_prediction_ev_atom"})
    m4 = predictions[predictions.model == "M4"][keys + ["prediction_ev_atom"]].rename(columns={"prediction_ev_atom": "m4_prediction_ev_atom"})
    aligned = m2.merge(m4, on=keys, validate="one_to_one")
    if len(aligned) != 210 * 3:
        raise RuntimeError(f"Alineación OOF inesperada: {len(aligned)} filas")
    rows=[]
    for (_fold, _seed, _composition), group in aligned.groupby(["outer_fold", "seed", "composition_id"], sort=False):
        group = group.sort_values("polymorph_family", key=lambda s: s.map({"dist_perovskite":0,"hexagonal":1,"needle":2}))
        m2_pred = group.m2_prediction_ev_atom.to_numpy(float)
        m4_pred = group.m4_prediction_ev_atom.to_numpy(float)
        m3_pred = m4_pred.mean() + m2_pred - m2_pred.mean()
        if not np.allclose(np.diff(np.sort(m2_pred)), np.diff(np.sort(m3_pred)), rtol=0, atol=1e-12):
            raise RuntimeError("M3-A alteró diferencias relativas de M2")
        for (_, row), prediction in zip(group.iterrows(), m3_pred):
            rows.append(row.to_dict() | {"m3a_prediction_ev_atom": float(prediction), "m2_centered_residual_ev_atom": float(row.m2_prediction_ev_atom - m2_pred.mean()), "m4_composition_mean_ev_atom": float(m4_pred.mean())})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "h3a/oof_m3a_predictions.csv", index=False)
    metrics=[]
    for model, col in (("M2", "m2_prediction_ev_atom"), ("M4", "m4_prediction_ev_atom"), ("M3-A", "m3a_prediction_ev_atom")):
        for (fold, seed), group in result.groupby(["outer_fold", "seed"]):
            metric = composition_metrics(group, "energy_dft_ev_atom", col)
            metrics.append({"model": model, "outer_fold": int(fold), "seed": int(seed), **metric})
    pd.DataFrame(metrics).to_csv(REPORT_DIR / "cv/metrics_per_fold_seed.csv", index=False)
    composition_rows=[]
    for model, col in (("M2", "m2_prediction_ev_atom"), ("M4", "m4_prediction_ev_atom"), ("M3-A", "m3a_prediction_ev_atom")):
        for (fold, seed, comp), group in result.groupby(["outer_fold", "seed", "composition_id"]):
            metric=composition_metrics(group,"energy_dft_ev_atom",col)
            composition_rows.append({"model":model,"outer_fold":int(fold),"seed":int(seed),"composition_id":comp,**metric})
    pd.DataFrame(composition_rows).to_csv(REPORT_DIR / "cv/metrics_by_composition_fold_seed.csv",index=False)
    ensemble=[]
    for model, col in (("M2", "m2_prediction_ev_atom"), ("M4", "m4_prediction_ev_atom"), ("M3-A", "m3a_prediction_ev_atom")):
        avg = result.groupby(["structure_id", "composition_id", "polymorph_family", "energy_dft_ev_atom"], as_index=False)[col].mean().rename(columns={col:"prediction_ev_atom"})
        ensemble.append({"model": model, **composition_metrics(avg, "energy_dft_ev_atom", "prediction_ev_atom")})
    pd.DataFrame(ensemble).to_csv(REPORT_DIR / "cv/metrics_seed_fold_ensemble.csv", index=False)
    # Exact invariance checks after averaging all 15 fold-seed predictions.
    comparisons=[]
    for key, group in result.groupby(["composition_id"]):
        m2_mean = group.groupby("polymorph_family").m2_prediction_ev_atom.mean().reindex(["dist_perovskite","hexagonal","needle"]).to_numpy()
        m3_mean = group.groupby("polymorph_family").m3a_prediction_ev_atom.mean().reindex(["dist_perovskite","hexagonal","needle"]).to_numpy()
        comparisons.append({"composition_id": key[0], "max_abs_pairwise_difference_ev_atom": float(np.max(np.abs((m2_mean[:,None]-m2_mean[None,:])-(m3_mean[:,None]-m3_mean[None,:])))), "m2_argmin": int(np.argmin(m2_mean)), "m3a_argmin": int(np.argmin(m3_mean))})
    invariance = {"status": "PASS" if all(x["max_abs_pairwise_difference_ev_atom"] < 1e-12 and x["m2_argmin"] == x["m3a_argmin"] for x in comparisons) else "FAIL", "compositions": len(comparisons), "max_pairwise_error_ev_atom": max(x["max_abs_pairwise_difference_ev_atom"] for x in comparisons)}
    write_json(REPORT_DIR / "h3a/invariance.json", invariance)
    return invariance


def main() -> None:
    """Run the predeclared M3-A recombination."""
    log = logger("mattergpt_m3a")
    try: log.info("mattergpt_m3a_complete %s", run())
    except Exception as exc: log.exception("mattergpt_m3a_failed error=%s", exc); raise


if __name__ == "__main__":
    main()
