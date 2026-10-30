"""Derive additional locked-test diagnostics from already scored rows only."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

from common import REPORT_DIR, logger, sha256, write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from energy_metrics import composition_metrics


def run() -> dict[str, object]:
    """Add mean/residual/range diagnostics without re-reading source labels."""
    scored_path=REPORT_DIR/"test/test_predictions_with_dft.csv"
    lock_path=REPORT_DIR/"test/test_lock.json"
    if not scored_path.is_file() or not lock_path.is_file(): raise FileNotFoundError("Faltan predicciones ya puntuadas o lock")
    frame=pd.read_csv(scored_path)
    rows=[]
    for model,col in (("M2","M2_prediction_ev_atom"),("M4","M4_prediction_ev_atom"),("M3-A","M3A_prediction_ev_atom")):
        rows.append({"model":model,**composition_metrics(frame,"energy_dft_ev_atom",col)})
    pd.DataFrame(rows).to_csv(REPORT_DIR/"test/test_metrics_extended.csv",index=False)
    report={"status":"PASS","input_scored_predictions_sha256":sha256(scored_path),"test_lock_sha256":sha256(lock_path),"test_source_labels_reopened":False,"derived_metrics":rows,"script_sha256":sha256(Path(__file__).resolve()),"note":"Post-score metric expansion over the exact locked predictions/targets; no model selection and no new target-file read."}
    write_json(REPORT_DIR/"test/test_metrics_supplement_provenance.json",report)
    return {"status":"PASS","rows":len(rows)}


def main() -> None:
    """Write supplemental metrics from the existing scored table."""
    log=logger("mattergpt_test_metrics_supplement")
    try:log.info("supplemental_test_metrics_complete %s",run())
    except Exception as exc:log.exception("supplemental_test_metrics_failed error=%s",exc);raise


if __name__=="__main__":main()
