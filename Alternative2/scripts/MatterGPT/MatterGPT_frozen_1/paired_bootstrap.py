"""Paired development OOF bootstrap by composition, preserving physical triplets."""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from common import REPORT_DIR, logger, write_json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from energy_metrics import composition_metrics


def run(replicates: int = 2000, seed: int = 13072026) -> dict[str, Any]:
    """Bootstrap model differences using paired resamples of composition triplets."""
    oof = pd.read_csv(REPORT_DIR / "h3a/oof_m3a_predictions.csv")
    ensemble = oof.groupby(["structure_id", "composition_id", "polymorph_family", "energy_dft_ev_atom"], as_index=False)[["m2_prediction_ev_atom", "m4_prediction_ev_atom", "m3a_prediction_ev_atom"]].mean()
    rng = np.random.default_rng(seed); compositions = sorted(ensemble.composition_id.unique())
    distributions = {"M3-A_minus_M2": {k: [] for k in ("mae_absolute_mev_atom", "mae_delta_dft_gs_anchored_mev_atom", "gs_accuracy", "pairwise_accuracy")},
                     "M3-A_minus_M4": {k: [] for k in ("mae_absolute_mev_atom", "mae_delta_dft_gs_anchored_mev_atom", "gs_accuracy", "pairwise_accuracy")}}
    for _ in range(replicates):
        sampled = rng.choice(compositions, size=len(compositions), replace=True)
        chunks = []
        for sample_id, comp in enumerate(sampled):
            chunk = ensemble[ensemble.composition_id == comp].copy(); chunk["bootstrap_group"] = str(sample_id); chunks.append(chunk)
        draw = pd.concat(chunks, ignore_index=True)
        metrics = {model: composition_metrics(draw, "energy_dft_ev_atom", col, group="bootstrap_group") for model, col in (("M2","m2_prediction_ev_atom"),("M4","m4_prediction_ev_atom"),("M3-A","m3a_prediction_ev_atom"))}
        for baseline in ("M2", "M4"):
            key=f"M3-A_minus_{baseline}"
            for metric in distributions[key]:
                a,b=metrics["M3-A"][metric],metrics[baseline][metric]
                distributions[key][metric].append(float(a-b) if a is not None and b is not None else float("nan"))
    report={"status":"PASS","replicates":replicates,"seed":seed,"sampling_unit":"composition triplet; all three polymorphs stay together","intervals":{}}
    for comparison,metrics in distributions.items():
        report["intervals"][comparison]={}
        for metric,values in metrics.items():
            arr=np.asarray(values,dtype=float);arr=arr[np.isfinite(arr)]
            report["intervals"][comparison][metric]={"mean_difference":float(arr.mean()),"ci95_percentile":[float(np.quantile(arr,.025)),float(np.quantile(arr,.975))],"n_valid":int(len(arr)),"interpretation":"exploratory; small number of chemical groups and prior model selection"}
    write_json(REPORT_DIR / "cv/paired_bootstrap_ci.json",report)
    return {"replicates":replicates,"comparisons":list(report["intervals"])}


def main() -> None:
    """Run the prespecified composition bootstrap."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--replicates",type=int,default=2000);parser.add_argument("--seed",type=int,default=13072026);args=parser.parse_args()
    log=logger("mattergpt_bootstrap")
    try:log.info("paired_bootstrap_complete %s",run(args.replicates,args.seed))
    except Exception as exc:log.exception("paired_bootstrap_failed error=%s",exc);raise


if __name__=="__main__":main()
