#!/usr/bin/env python3
"""Audita integridad de features, 45 corridas y predicciones OOF H0/H1/H2."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, MODELS, REPORT_DIR, SEEDS, ExperimentError, load_development, log, sha256_file, verify_provenance, write_json


def audit(out_dir: Path, phase: str) -> dict[str, Any]:
    """Comprueba controles preflight o resultados sin evaluar test.

    Args:
        out_dir: Carpeta de resultados versionada.
        phase: preflight, results o all.

    Returns:
        Dictamen PASS/PARTIAL y conteos de controles verificados.
    """
    identity=verify_provenance()
    data=load_development()
    checks:dict[str,Any]={"checkpoint_sha256":identity["checkpoint_sha256"],"checkpoint_expected":identity["checkpoint_sha256"]==EXPECTED_CHECKPOINT_SHA,
        "development_rows":len(data),"development_compositions":int(data.composition_id.nunique()),"historical_test_used":False}
    if phase in {"preflight","all"}:
        pre=out_dir/"integrity_checks.json"
        if not pre.is_file():raise ExperimentError("Falta integrity_checks.json de preflight.")
        p=json.loads(pre.read_text(encoding="utf-8"))
        if p.get("status")!="PASS" or p.get("backbone_trainable_parameters")!=0 or p.get("test_used",False):
            raise ExperimentError("Integridad crítica de preflight no PASS.")
        cache=out_dir/"features"/"frozen_features_development.npz"
        meta=out_dir/"features"/"feature_metadata.csv"
        emb=out_dir/"features"/"embeddings_manifest.json"
        if all(x.is_file() for x in (cache,meta,emb)):
            m=json.loads(emb.read_text())
            with np.load(cache,allow_pickle=False) as z:
                ids=z["structure_id"].astype(str).tolist()
                if ids!=data.structure_id.astype(str).tolist() or z["h_end"].shape!=(210,768) or z["h_mean"].shape!=(210,768):
                    raise ExperimentError("Cache IDs/dimensiones no son solo development.")
                if not np.isfinite(z["h_end"]).all() or not np.isfinite(z["h_mean"]).all():raise ExperimentError("Cache contiene NaN/Inf.")
            checks["features_status"]=m.get("status");checks["feature_ids_match_development"]=True
        elif phase=="all":raise ExperimentError("Falta el caché congelado.")
        checks["preflight_status"]="PASS"
    if phase in {"results","all"}:
        manifest_path=out_dir/"training_manifest.json"
        oof_path=out_dir/"oof_predictions.csv"
        if not manifest_path.is_file() or not oof_path.is_file():raise ExperimentError("Faltan manifiesto de training o OOF.")
        manifest=json.loads(manifest_path.read_text())
        oof=pd.read_csv(oof_path)
        if manifest.get("status")!="PASS" or manifest.get("completed_runs")!=45 or manifest.get("test_used",True):
            raise ExperimentError("No están completadas las 45 corridas o se marcó uso de test.")
        if len(oof)!=1890 or set(oof.model.unique())!=set(MODELS):raise ExperimentError("OOF debe tener 1,890 filas H0/H1/H2.")
        dev_ids=set(data.structure_id.astype(str))
        if not set(oof.structure_id.astype(str)).issubset(dev_ids):raise ExperimentError("OOF contiene IDs fuera de development.")
        counts=oof.groupby(["model","fold","seed"]).structure_id.nunique()
        if len(counts)!=45 or not counts.eq(42).all():raise ExperimentError("Cobertura OOF distinta de 42 estructuras por corrida.")
        if set(oof.seed.unique())!=set(SEEDS) or set(oof.fold.unique())!=set(range(5)):raise ExperimentError("Semillas/folds incorrectos.")
        if not np.isfinite(oof.E_pred_eV_atom).all():raise ExperimentError("Predicciones OOF no finitas.")
        for (model,seed),g in oof.groupby(["model","seed"]):
            if len(g)!=210 or g.structure_id.nunique()!=210:raise ExperimentError(f"{model}/{seed} no tiene 210 IDs únicos.")
        for model in ("H1","H2"):
            for (seed,fold,comp),g in oof.loc[oof.model.eq(model)].groupby(["seed","fold","composition_id"]):
                if len(g)!=3 or g.E_pred_mu_eV_atom.nunique()!=1:raise ExperimentError("μ no común a los tres polimorfos o triplete incompleto.")
                if abs(float(g.delta_pred_centered_eV_atom.sum()))>1e-6:raise ExperimentError("Residuo no suma cero.")
                if not np.allclose(g.E_pred_eV_atom,g.E_pred_mu_eV_atom+g.delta_pred_centered_eV_atom,atol=1e-6,rtol=0):raise ExperimentError("Predicción no reconstruye μ+δ.")
        if sha256_file(CHECKPOINT)!=EXPECTED_CHECKPOINT_SHA:raise ExperimentError("El checkpoint original cambió.")
        h0=out_dir/"H0_error_decomposition.csv"
        if h0.is_file():
            d=pd.read_csv(h0)
            error=(d.mse_total_mev2_atom2-d.mse_mean_component_mev2_atom2-d.mse_residual_component_mev2_atom2).abs().max()
            if error>1e-6:raise ExperimentError(f"MSE decomposition error {error}")
            checks["max_mse_identity_error"]=float(error)
        checks.update({"completed_runs":int(manifest["completed_runs"]),"oof_rows":len(oof),"oof_ids_per_model_seed":210,"folds":5,"seeds":list(SEEDS),"H1_H2_centering_and_reconstruction":"PASS"})
        checks["results_status"]="PASS"
    status="PASS" if (phase=="preflight" and checks.get("preflight_status")=="PASS") or (phase=="results" and checks.get("results_status")=="PASS") or (phase=="all" and checks.get("preflight_status")=="PASS" and checks.get("results_status")=="PASS") else "PARTIAL"
    checks["status"]=status
    write_json(out_dir/"audit_cv.json",checks)
    return checks


def main() -> None:
    """Valida CLI y ejecuta auditoría de resultados."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--phase",choices=("preflight","results","all"),default="all");parser.add_argument("--out-dir",type=Path,default=REPORT_DIR);args=parser.parse_args()
    try:
        result=audit(args.out_dir,args.phase);log(logging.INFO,"cv_audit_complete",status=result["status"],phase=args.phase)
    except (ExperimentError,OSError,ValueError,KeyError,RuntimeError) as exc:
        log(logging.ERROR,"cv_audit_failed",phase=args.phase,error=str(exc));raise SystemExit(2) from exc


if __name__=="__main__":main()
