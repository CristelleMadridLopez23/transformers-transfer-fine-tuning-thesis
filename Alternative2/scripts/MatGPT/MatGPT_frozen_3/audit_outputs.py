#!/usr/bin/env python3
"""Audita integridad y cobertura de los artefactos finales R1-R4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (CHECKPOINT, EXPECTED_CHECKPOINT_SHA, REPORT_DIR,
    ExperimentError, load_dev, log, sha256_file, write_json)


def run() -> dict[str,Any]:
    """Valida 75 corridas, OOF, exclusión de test y formatos PNG solamente."""
    data=load_dev();dev_ids=set(data.structure_id.astype(str))
    models=("R2-SMALL4","R2-REF8","R3-DROP30","R3-WD1E2","R3-EARLY8")
    run_counts={};per_model={}
    for model in models:
        parent="R2_head_capacity" if model.startswith("R2") else "R3_regularization"
        root=REPORT_DIR/parent/"runs"/model
        statuses=[]
        for fold in range(5):
            for seed in (42,123,2026):
                run_dir=root/f"fold_{fold}_seed_{seed}"
                status_path=run_dir/"status.json"
                if not status_path.is_file():raise ExperimentError(f"Falta corrida: {run_dir}")
                state=json.loads(status_path.read_text())
                if state.get("status")!="PASS":raise ExperimentError(f"Corrida no PASS: {run_dir}")
                prediction=pd.read_csv(run_dir/"predictions.csv")
                if len(prediction)!=42 or prediction.structure_id.astype(str).nunique()!=42:
                    raise ExperimentError(f"La corrida no contiene exactamente 42 IDs outer: {run_dir}")
                if set(prediction.historical_split.astype(str))-{"train","val"}:
                    raise ExperimentError("Un ID de test aparece en predicciones OOF.")
                if not np.isfinite(prediction.E_pred_eV_atom.to_numpy(float)).all():
                    raise ExperimentError(f"Predicciones no finitas: {run_dir}")
                statuses.append(state)
        oof_path=REPORT_DIR/parent/model/"oof_predictions.csv"
        oof=pd.read_csv(oof_path)
        if len(oof)!=630 or set(oof.structure_id.astype(str))!=dev_ids or not oof.groupby("structure_id").seed.nunique().eq(3).all():
            raise ExperimentError(f"OOF incompleto o desalineado: {model}")
        per_model[model]={"status":"PASS","runs":len(statuses),"oof_rows":len(oof),
            "unique_structures":int(oof.structure_id.nunique()),"training_seconds_total":float(sum(s["training_seconds"] for s in statuses)),
            "training_seconds_median_per_run":float(np.median([s["training_seconds"] for s in statuses]))}
        run_counts[model]=len(statuses)
    for subdir,expected_models,expected_rows in (("R1_linear_probes",4,840),("R4_layers_pooling",6,1260)):
        frame=pd.read_csv(REPORT_DIR/subdir/"oof_predictions.csv")
        if len(frame)!=expected_rows or frame.model.nunique()!=expected_models or set(frame.structure_id.astype(str))!=dev_ids:
            raise ExperimentError(f"OOF Ridge inválido en {subdir}.")
        if not np.isfinite(frame.E_pred_eV_atom.to_numpy(float)).all():raise ExperimentError(f"Ridge no finito en {subdir}.")
    ensemble=pd.read_csv(REPORT_DIR/"comparisons"/"ensemble_predictions.csv")
    if len(ensemble)!=18*210 or ensemble.groupby("model").structure_id.nunique().ne(210).any():
        raise ExperimentError("La tabla comparativa no cubre 18 modelos por 210 IDs.")
    if set(ensemble.structure_id.astype(str))-dev_ids:raise ExperimentError("ID fuera de development en ensemble.")
    pngs=sorted((REPORT_DIR/"comparisons"/"figures").glob("*.png"))
    if len(pngs)<13:raise ExperimentError(f"Solo hay {len(pngs)} figuras PNG.")
    forbidden={".pdf",".svg",".eps",".jpg",".jpeg",".html",".pptx"}
    other=[str(p) for p in REPORT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    if other:raise ExperimentError(f"Artefactos gráficos distintos de PNG encontrados: {other[:5]}")
    checkpoint_sha=sha256_file(CHECKPOINT)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ExperimentError("SHA-256 del checkpoint cambió durante R1-R4.")
    result={"status":"PASS","unique_head_runs":sum(run_counts.values()),"expected_head_runs":75,
        "head_runs_by_model":per_model,"ridge_R1_rows":840,"ridge_R4_rows":1260,"ensemble_rows":len(ensemble),
        "development_structures":210,"development_compositions":70,"outer_folds":5,"seeds":[42,123,2026],
        "png_figures":len(pngs),"non_png_figure_formats_found":other,"test_used":False,
        "checkpoint_sha256":checkpoint_sha,"all_predictions_finite":True}
    if result["unique_head_runs"]!=75:raise ExperimentError("No se completaron las 75 corridas heads-only.")
    write_json(REPORT_DIR/"manifests"/"final_audit.json",result)
    return result


def main()->None:
    """Ejecuta la auditoría final sin abrir datos del test histórico."""
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        result=run();log("output_audit_complete",status=result["status"],head_runs=result["unique_head_runs"],pngs=result["png_figures"],test_used=False)
    except (ExperimentError,OSError,ValueError,KeyError,TypeError) as exc:
        log("output_audit_failed",error=str(exc));raise SystemExit(2) from exc


if __name__=="__main__":main()
