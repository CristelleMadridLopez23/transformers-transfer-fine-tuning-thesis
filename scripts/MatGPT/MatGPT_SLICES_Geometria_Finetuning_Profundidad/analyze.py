"""Reconstruye OOF, métricas corregidas, bootstrap, figuras y reporte del Experimento D."""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import logging
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
IMAGES = ROOT / "images/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
DEPTH_EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
FROZEN_GEOMETRY = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria"
CONDITIONS = tuple(f"F{d}_{suffix}" for d in (0,2,4,8) for suffix in ("SLICES","GEOMETRÍA"))
DEPTH_LABELS = ("F0", "F2", "F4", "F8")
SEEDS = (42, 123, 2026)
GAP_BINS = ((0,5),(5,10),(10,25),(25,50),(50,100),(100,float("inf")))
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261030
LOG = logging.getLogger("matgpt_experiment_d_analysis")


def sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de una fuente o artefacto."""
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe un JSON consolidado sin exponer escrituras parciales."""
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(val) for key,val in item.items()}
        if isinstance(item, (list,tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, np.generic):
            item=item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(normalize(value),indent=2,ensure_ascii=False,default=str,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def import_train():
    """Importa funciones de carga/inferencia ya auditadas del módulo D."""
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    return __import__("scripts.MatGPT.MatGPT_SLICES_Geometria_Finetuning_Profundidad.train",fromlist=["train"])


def extract_new_geometry_predictions() -> tuple[pd.DataFrame,pd.DataFrame]:
    """Ejecuta solo forward-pass en outer para checkpoints geométricos seleccionados.

    También cuantifica drift de FINAL_END respecto al mismo MatGPT inicial.
    Etiquetas se leen únicamente para asociar y evaluar predicciones.
    """
    tr=import_train(); depth=tr.import_depth_trainer()
    torch=tr.torch
    frame,folds,model,stoi,context=depth.load_sources()
    geometry_df=pd.read_csv(tr.GEOMETRY_SOURCE).set_index("structure_id")
    geometry={str(sid):geometry_df.loc[sid,list(tr.GEOMETRY_COLUMNS)].to_numpy(np.float32)
              for sid in frame.structure_id.astype(str)}
    encoded=depth.encode_all(frame,stoi,context)
    base_path=depth.SOURCE/"mat-uncond/weights/transformer_property_0.pth"
    records=[];diagnostics=[]
    resources=json.loads((EXPERIMENT/"resources.json").read_text())
    for condition in ("F2","F4","F8"):
        block_ids=depth.CONDITIONS[condition]
        for fold in range(5):
            outer_ids=set(folds.loc[(folds.outer_fold==fold)&(folds.outer_role=="outer_validation"),"structure_id"].astype(str))
            groups=depth.groups_for(frame,outer_ids)
            for seed in SEEDS:
                run=f"{condition.lower()}_fold{fold}_seed{seed}"
                run_dir=EXPERIMENT/"checkpoints"/run
                ckpt_path=run_dir/"best.pt"
                if not (run_dir/"complete.flag").is_file() or not ckpt_path.is_file():
                    raise FileNotFoundError(f"Corrida no completa para evaluación: {run}")
                checkpoint=torch.load(ckpt_path,map_location="cpu",weights_only=False)
                model.load_state_dict(torch.load(base_path,map_location="cpu",weights_only=True),strict=True)
                model.to(torch.device("cpu"));model.eval()
                # Embeddings originales y adaptados se calculan para los mismos tripletes.
                geo_head=tr.GeometryMLP8();geo_head.load_state_dict(checkpoint["head"],strict=True);geo_head.eval()
                scaler=__import__("joblib").load(run_dir/"geometry_scaler_inner_train_only.joblib")
                for ix in block_ids:
                    model.transformer_decoder.layers[ix].load_state_dict(checkpoint["blocks"][str(ix)],strict=True)
                for formula,ids in groups:
                    seq=[encoded[sid] for sid in ids]
                    with torch.no_grad():
                        x0=depth.final_end(model,seq,stoi["<pad>"],torch.device("cpu"))
                        # Re-load original blocks for the pretrained comparison, then restore adaptation.
                        geo_scaled=scaler.transform(np.stack([geometry[sid] for sid in ids])).astype(np.float32)
                        joined=torch.cat((x0,torch.as_tensor(geo_scaled)),dim=-1)
                        # La cabeza predice valores crudos; la definición híbrida exige
                        # centrarlos dentro del triplete antes de sumarlos a Ridge.
                        residual_raw=geo_head(joined).numpy().reshape(-1)
                        residual=residual_raw-residual_raw.mean()
                    for j,sid in enumerate(ids):
                        drow=frame.set_index("structure_id").loc[sid]
                        records.append({"condition":f"{condition}_GEOMETRÍA","fold":fold,"seed":seed,
                            "structure_id":sid,"composition_id":formula,"polymorph_family":drow.family,
                            "n_atoms":int(drow.n_atoms),"energy_total_dft_ev":float(drow.total_dft_ev),
                            "energy_per_atom_dft_ev":float(drow.dft_ev_atom),"residual_pred_ev_atom":float(residual[j])})
                # Computa drift original vs adaptado con estado original preservado.
                # Se vuelve a calcular por lote tras restaurar el checkpoint base.
                original_vectors=[];final_adapted=[]
                model.load_state_dict(torch.load(base_path,map_location="cpu",weights_only=True),strict=True);model.eval()
                with torch.no_grad():
                    for _,ids in groups:
                        original_vectors.append(depth.final_end(model,[encoded[s] for s in ids],stoi["<pad>"],torch.device("cpu")).numpy())
                for ix in block_ids:model.transformer_decoder.layers[ix].load_state_dict(checkpoint["blocks"][str(ix)],strict=True)
                model.eval()
                with torch.no_grad():
                    for _,ids in groups:
                        final_adapted.append(depth.final_end(model,[encoded[s] for s in ids],stoi["<pad>"],torch.device("cpu")).numpy())
                x=np.concatenate(original_vectors);y=np.concatenate(final_adapted)
                cos=np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
                rel=np.linalg.norm(y-x,axis=1)/(np.linalg.norm(x,axis=1)+1e-12)
                w=checkpoint["head"]["network.0.weight"].detach().cpu().numpy()[:,768:]
                diagnostics.append({"condition":f"{condition}_GEOMETRÍA","fold":fold,"seed":seed,
                    "n_structures":len(y),"mean_cosine_original_vs_adapted":float(cos.mean()),
                    "mean_relative_embedding_change":float(rel.mean()),"p95_relative_embedding_change":float(np.quantile(rel,.95)),
                    "geometry_weight_norm":float(np.linalg.norm(w)),
                    **{f"geometry_weight_norm_{j+1}":float(np.linalg.norm(w[:,j])) for j in range(7)},
                    "best_epoch":int(checkpoint["best_epoch"]),"best_inner_loss":float(checkpoint["best_inner_loss"]),
                    "peak_rss_bytes":resources.get("runs",{}).get(run,{}).get("peak_rss_bytes")})
                del checkpoint
    return pd.DataFrame(records),pd.DataFrame(diagnostics)


def compose_predictions(new_geometry: pd.DataFrame) -> pd.DataFrame:
    """Une controles compatibles, nuevas salidas de geometría y Ridge composicional fijo."""
    depth=pd.read_csv(DEPTH_EXPERIMENT/"predicciones_oof_por_estructura.csv")
    old_geo=pd.read_csv(FROZEN_GEOMETRY/"results/predicciones_oof_por_estructura.csv")
    ridge=depth[depth.condition=="F0"][["fold","seed","structure_id","ridge_mean_ev_atom"]]
    parts=[]
    # F0 se reutiliza del experimento frozen de geometría en pareja, ambos bajo el mismo entrenamiento.
    for old_condition,label in (("slices_only","F0_SLICES"),("slices_plus_cell_geometry","F0_GEOMETRÍA")):
        x=old_geo[old_geo.condition==old_condition].merge(ridge,on=["fold","seed","structure_id"],validate="one_to_one",suffixes=("_old",""))
        x["condition"]=label
        x["residual_pred_ev_atom"]=x.residual_ev_atom
        parts.append(x)
    # F2/F4/F8 SLICES-only reutilizan checkpoints OOF ya completos y comprobados.
    for d in ("F2","F4","F8"):
        x=depth[depth.condition==d].copy();x["condition"]=f"{d}_SLICES";parts.append(x)
    # Para las condiciones nuevas se incorpora el mismo Ridge composicional por fold.
    # Cada ID aparece una vez por profundidad geométrica; el Ridge fijo sí
    # debe ser único por pliegue/semilla/estructura.
    g=new_geometry.merge(ridge,on=["fold","seed","structure_id"],validate="many_to_one")
    g["condition"]=g.condition.str.replace("_GEOMETRÍA","_GEOMETRÍA",regex=False)
    parts.append(g)
    out=pd.concat(parts,ignore_index=True,sort=False)
    if len(out)!=8*630 or out.duplicated(["condition","fold","seed","structure_id"]).any():
        raise ValueError(f"Predicciones incompletas o repetidas ({len(out)}; esperado 5,040)")
    out["energy_per_atom_pred_ev"]=out.ridge_mean_ev_atom+out.residual_pred_ev_atom
    out["energy_total_pred_ev"]=out.n_atoms*out.energy_per_atom_pred_ev
    if not np.allclose(out.groupby(["condition","fold","seed","composition_id"]).residual_pred_ev_atom.mean().to_numpy(),0,atol=2e-6):
        raise ValueError("Residuos no centrados por composición")
    return out


def calculate_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Calcula métricas por composición/semilla y pares con ancla DFT corregida."""
    composition=[];pairs=[]
    for (condition,fold,seed,formula),group in pred.groupby(["condition","fold","seed","composition_id"],sort=True):
        group=group.sort_values("polymorph_family");d=group.energy_per_atom_dft_ev.to_numpy(float);p=group.energy_per_atom_pred_ev.to_numpy(float)
        r=group.residual_pred_ev_atom.to_numpy(float);tr=d-d.mean();gs=int(np.argmin(d))
        dgap=(d-d[gs])*1000;pgap=(p-p[gs])*1000
        pair_error=[];pair_signed_error=[];pair_zero=[];eligible=correct=resolved=0
        for i in range(3):
            for j in range(i+1,3):
                dt=(d[j]-d[i])*1000;dp=(p[j]-p[i])*1000;ok=abs(dt)>5;res=abs(dp)>1
                is_correct=bool(ok and res and np.sign(dt)==np.sign(dp))
                pair_error.append(abs(dp-dt));pair_signed_error.append(dp-dt);pair_zero.append(abs(dt));eligible+=int(ok);resolved+=int(ok and res);correct+=int(is_correct)
                pairs.append({"condition":condition,"fold":int(fold),"seed":int(seed),"composition_id":formula,
                    "structure_id_a":group.iloc[i].structure_id,"structure_id_b":group.iloc[j].structure_id,
                    "delta_dft_mev_atom":dt,"delta_pred_mev_atom":dp,"abs_delta_dft_mev_atom":abs(dt),
                    "abs_error_mev_atom":abs(dp-dt),"eligible_gt_5_mev":ok,"resolved_gt_1_mev":res,
                    "pairwise_correct":bool(is_correct) if ok else None})
        predicted_min=float(p.min())
        predicted_gs=np.flatnonzero(np.isclose(p,predicted_min,atol=1e-10,rtol=0))
        gs_correct=int(len(predicted_gs)==1 and int(predicted_gs[0])==gs)
        total_err=np.abs(group.energy_total_pred_ev.to_numpy(float)-group.energy_total_dft_ev.to_numpy(float))
        pr=p-p.mean();rho=spearmanr(d,p).statistic if np.std(p)>0 else np.nan
        composition.append({"condition":condition,"fold":int(fold),"seed":int(seed),"composition_id":formula,
            "mae_energy_mev_atom":float(np.mean(np.abs(p-d))*1000),"rmse_energy_mev_atom":float(np.sqrt(np.mean((p-d)**2))*1000),
            "bias_energy_mev_atom":float(np.mean(p-d)*1000),"mae_energy_total_ev":float(total_err.mean()),
            "mae_composition_mean_mev_atom":float(abs(p.mean()-d.mean())*1000),
            "mae_residual_mev_atom":float(np.mean(np.abs(pr-tr))*1000),
            "rmse_residual_mev_atom":float(np.sqrt(np.mean((pr-tr)**2))*1000),
            "mae_delta_signed_pairs_mev_atom":float(np.mean(pair_error)),"rmse_delta_signed_pairs_mev_atom":float(np.sqrt(np.mean(np.square(pair_signed_error)))),
            "mae_delta_zero_baseline_mev_atom":float(np.mean(pair_zero)),
            "mae_delta_anchored_nonanchor_mev_atom":float(np.mean(np.delete(np.abs(pgap-dgap),gs))),
            "mae_delta_anchored_all3_mev_atom":float(np.mean(np.abs(pgap-dgap))),
            "ground_state_exact":gs_correct,"ground_state_unresolved":int(len(predicted_gs)!=1),
            "ground_state_eligible":int((np.sort(d)[1]-np.sort(d)[0])*1000>1),
            "pairwise_correct_count":correct,"pairwise_eligible_count":eligible,"pairwise_resolved_count":resolved,
            "pairwise_unresolved_count":eligible-resolved,"pairwise_accuracy_macro":correct/eligible if eligible else np.nan,
            "pairwise_accuracy_resolved":correct/resolved if resolved else np.nan,
            "spearman_triplet":float(rho),"kendall_triplet":float(kendalltau(d,p).statistic) if np.std(p)>0 else np.nan,
            "dispersion_ratio_absolute":float(np.std(p,ddof=0)/np.std(d,ddof=0)) if np.std(d)>0 else np.nan,
            "dispersion_ratio_intracomposition":float(np.std(pr,ddof=0)/np.std(tr,ddof=0)) if np.std(tr)>0 else np.nan,
            "residual_pearson_r":float(np.corrcoef(tr,pr)[0,1]) if np.std(pr)>0 and np.std(tr)>0 else np.nan,
            "delta_dft_range_mev_atom":float(np.ptp(d)*1000)})
    return pd.DataFrame(composition),pd.DataFrame(pairs)


def metric_estimands(pred: pd.DataFrame, comp_seed: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Calcula estimando A de corrida y B de ensemble como poblaciones separadas."""
    rows=[]; formula_a={};formula_b={}
    metric_cols=[c for c in comp_seed.columns if c not in ("condition","fold","seed","composition_id")]
    for condition,group in comp_seed.groupby("condition"):
        a=group.groupby("composition_id",as_index=False)[metric_cols].mean()
        formula_a[condition]=a.set_index("composition_id")
        rows+=summarize_estimate(condition,"A_corrida_individual",a)
    # B: promedia las tres predicciones por estructura antes de formar error/gaps.
    ens=pred.groupby(["condition","fold","structure_id","composition_id","polymorph_family"],as_index=False).agg(
        n_atoms=("n_atoms","first"),energy_total_dft_ev=("energy_total_dft_ev","first"),
        energy_per_atom_dft_ev=("energy_per_atom_dft_ev","first"),energy_per_atom_pred_ev=("energy_per_atom_pred_ev","mean"),
        energy_total_pred_ev=("energy_total_pred_ev","mean"),residual_pred_ev_atom=("residual_pred_ev_atom","mean"))
    bcomp,_=calculate_metrics(ens.assign(seed=-1))
    for condition,group in bcomp.groupby("condition"):
        b=group.groupby("composition_id",as_index=False)[metric_cols].mean()
        formula_b[condition]=b.set_index("composition_id")
        rows+=summarize_estimate(condition,"B_ensemble_3_semillas",b)
    # OLS completo sobre estructuras: A es media de ajustes semilla-a-semilla; B usa ensemble.
    for condition in pred.condition.unique():
        for estimand,frame in (("A_corrida_individual",pred[pred.condition==condition]),("B_ensemble_3_semillas",ens[ens.condition==condition])):
            slopes=[];predictive_r2=[]
            for _,part in frame.groupby("seed") if "seed" in frame.columns else [(None,frame)]:
                x=part.energy_per_atom_dft_ev.to_numpy(float);y=part.energy_per_atom_pred_ev.to_numpy(float)
                m,b=np.polyfit(x,y,1);r2=1-np.sum((y-(m*x+b))**2)/max(np.sum((y-y.mean())**2),1e-30)
                predictive_r2.append(1-float(np.sum((y-x)**2))/max(float(np.sum((x-x.mean())**2)),1e-30))
                rp=part.energy_per_atom_pred_ev.to_numpy(float)-part.groupby("composition_id").energy_per_atom_pred_ev.transform("mean").to_numpy(float)
                rt=part.energy_per_atom_dft_ev.to_numpy(float)-part.groupby("composition_id").energy_per_atom_dft_ev.transform("mean").to_numpy(float)
                mr,br=np.polyfit(rt,rp,1);r2p=1-np.sum((rp-(mr*rt+br))**2)/max(np.sum((rp-rp.mean())**2),1e-30)
                slopes.append((m,b,r2,float(np.std(y)/np.std(x)),mr,r2p))
            arr=np.asarray(slopes)
            rows.append({"condition":condition,"estimand":estimand,"metric":"ols_absolute_slope","value":float(arr[:,0].mean()),"units":"adimensional","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"ols_absolute_intercept","value":float(arr[:,1].mean()),"units":"eV/átomo","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"ols_absolute_r2","value":float(arr[:,2].mean()),"units":"adimensional","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"absolute_predictive_r2","value":float(np.mean(predictive_r2)),"units":"adimensional","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"absolute_dispersion_ratio","value":float(arr[:,3].mean()),"units":"adimensional","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"intraformula_slope","value":float(arr[:,4].mean()),"units":"adimensional","n_compositions":70,"n_structures":210})
            rows.append({"condition":condition,"estimand":estimand,"metric":"intraformula_r2","value":float(arr[:,5].mean()),"units":"adimensional","n_compositions":70,"n_structures":210})
            delta_fits=[]
            for _,seed_frame in frame.groupby("seed") if "seed" in frame.columns else [(None,frame)]:
                delta_x=[];delta_y=[]
                for _,triplet in seed_frame.groupby("composition_id"):
                    if len(triplet)!=3:continue
                    d=triplet.energy_per_atom_dft_ev.to_numpy(float);p=triplet.energy_per_atom_pred_ev.to_numpy(float)
                    delta_x.extend((d[j]-d[i])*1000 for i,j in itertools.combinations(range(3),2))
                    delta_y.extend((p[j]-p[i])*1000 for i,j in itertools.combinations(range(3),2))
                dx=np.asarray(delta_x);dy=np.asarray(delta_y)
                if len(dx):
                    dm,db=np.polyfit(dx,dy,1)
                    dr2=1-np.sum((dy-(dm*dx+db))**2)/max(np.sum((dy-dy.mean())**2),1e-30)
                    delta_fits.append((dm,db,dr2,float(np.std(dy)/np.std(dx)),float(np.corrcoef(dx,dy)[0,1])))
            delta_arr=np.asarray(delta_fits)
            if not len(delta_arr):raise ValueError(f"No se encontraron pares por triplete para {condition}/{estimand}")
            rows.extend([
                {"condition":condition,"estimand":estimand,"metric":"delta_ols_slope","value":float(delta_arr[:,0].mean()),"units":"adimensional","n_compositions":70,"n_pairs":210},
                {"condition":condition,"estimand":estimand,"metric":"delta_ols_intercept_mev_atom","value":float(delta_arr[:,1].mean()),"units":"meV/átomo","n_compositions":70,"n_pairs":210},
                {"condition":condition,"estimand":estimand,"metric":"delta_ols_r2","value":float(delta_arr[:,2].mean()),"units":"adimensional","n_compositions":70,"n_pairs":210},
                {"condition":condition,"estimand":estimand,"metric":"delta_dispersion_ratio","value":float(delta_arr[:,3].mean()),"units":"adimensional","n_compositions":70,"n_pairs":210},
                {"condition":condition,"estimand":estimand,"metric":"delta_pearson_r","value":float(delta_arr[:,4].mean()),"units":"adimensional","n_compositions":70,"n_pairs":210}])
    return pd.DataFrame(rows),{"A":formula_a,"B":formula_b,"predictions_ensemble":ens}


def summarize_estimate(condition: str, estimand: str, data: pd.DataFrame) -> list[dict[str,Any]]:
    """Resume macro-métricas por fórmula y tasas pairwise con denominador explícito."""
    out=[]
    for metric in ("mae_energy_mev_atom","rmse_energy_mev_atom","bias_energy_mev_atom","mae_energy_total_ev",
        "mae_composition_mean_mev_atom","mae_residual_mev_atom","rmse_residual_mev_atom",
        "mae_delta_signed_pairs_mev_atom","rmse_delta_signed_pairs_mev_atom","mae_delta_zero_baseline_mev_atom",
        "mae_delta_anchored_nonanchor_mev_atom","mae_delta_anchored_all3_mev_atom",
        "ground_state_exact","ground_state_eligible","pairwise_accuracy_macro","pairwise_accuracy_resolved",
        "spearman_triplet","kendall_triplet","dispersion_ratio_absolute",
        "dispersion_ratio_intracomposition","residual_pearson_r"):
        s=data[metric].dropna()
        unit="meV/átomo" if "mev_atom" in metric else "eV" if metric.endswith("_ev") else "proporción/adimensional"
        out.append({"condition":condition,"estimand":estimand,"metric":metric,"value":float(s.mean()),
                    "units":unit,"n_compositions":int(s.size),"n_structures":210,"n_physical_pairs":210,
                    "aggregation":"media macro de 70 fórmulas; semillas promediadas dentro de fórmula"})
    numerator=float(data.pairwise_correct_count.sum());denom=float(data.pairwise_eligible_count.sum());resolved=float(data.pairwise_resolved_count.sum())
    out.extend([{"condition":condition,"estimand":estimand,"metric":"pairwise_accuracy_pooled","value":numerator/denom if denom else np.nan,
                 "units":"proporción","n_compositions":70,"n_structures":210,"n_physical_pairs":int(denom),"pairwise_correct_count":numerator,
                 "pairwise_eligible_denominator":denom,"aggregation":"conteo directo de pares elegibles; no resueltos incorrectos"},
                {"condition":condition,"estimand":estimand,"metric":"pairwise_accuracy_resolved_pooled","value":numerator/resolved if resolved else np.nan,
                 "units":"proporción","n_compositions":70,"n_structures":210,"n_physical_pairs":int(resolved),"pairwise_correct_count":numerator,
                 "pairwise_resolved_denominator":resolved,"aggregation":"pares elegibles con |ΔE_pred|>1 meV/átomo"}])
    return out


def bootstrap_analysis(formula_tables: dict[str,dict[str,pd.DataFrame]]) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Bootstrap apareado por fórmula para efectos, profundidad e interacciones."""
    rng=np.random.default_rng(BOOTSTRAP_SEED);rows=[]
    contrasts=[]
    for depth in ("F0","F2","F4","F8"):
        contrasts.append((f"efecto_geometria_{depth}",f"{depth}_GEOMETRÍA",f"{depth}_SLICES"))
    for depth in ("F2","F4","F8"):
        contrasts.append((f"profundidad_geometria_{depth}_vs_F0",f"{depth}_GEOMETRÍA","F0_GEOMETRÍA"))
        contrasts.append((f"profundidad_slices_{depth}_vs_F0",f"{depth}_SLICES","F0_SLICES"))
        contrasts.append((f"interaccion_{depth}",f"{depth}_GEOMETRÍA",f"{depth}_SLICES"))
    metric_map={"mae_energy_mev_atom":"mae_energy_mev_atom","mae_residual_mev_atom":"mae_residual_mev_atom",
                "mae_delta_signed_pairs_mev_atom":"mae_delta_signed_pairs_mev_atom",
                "mae_delta_anchored_nonanchor_mev_atom":"mae_delta_anchored_nonanchor_mev_atom",
                "ground_state_exact":"ground_state_exact","pairwise_accuracy_macro":"pairwise_accuracy_macro"}
    for estimand in ("A","B"):
        tables=formula_tables[estimand]
        for label,left,right in contrasts:
            for metric,out_metric in metric_map.items():
                if label.startswith("interaccion_"):
                    base_geo=tables["F0_GEOMETRÍA"][metric].to_numpy();base_slices=tables["F0_SLICES"][metric].to_numpy()
                    delta_geo=tables[left][metric].to_numpy();delta_slices=tables[right][metric].to_numpy()
                    diff=(delta_geo-delta_slices)-(base_geo-base_slices)
                    direction="negative: benefit of geometry larger / penalty smaller than F0"
                else:
                    diff=tables[left][metric].to_numpy()-tables[right][metric].to_numpy()
                    direction="negative favors the left condition for error; positive for accuracy"
                idx=rng.integers(0,len(diff),size=(BOOTSTRAP_REPS,len(diff)))
                sampled=np.mean(diff[idx],axis=1)
                rows.append({"estimand":estimand,"contrast":label,"metric":out_metric,"estimate":float(np.mean(diff)),
                    "ci95_low":float(np.quantile(sampled,.025)),"ci95_high":float(np.quantile(sampled,.975)),
                    "includes_zero":bool(np.quantile(sampled,.025)<=0<=np.quantile(sampled,.975)),
                    "n_compositions":len(diff),"bootstrap_replicates":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,
                    "direction":direction})
        # ΔE=0 baseline is composition-specific and same for every condition.
        for condition in CONDITIONS:
            table=tables[condition]
            diff=table.mae_delta_signed_pairs_mev_atom.to_numpy()-table.mae_delta_zero_baseline_mev_atom.to_numpy()
            idx=rng.integers(0,len(diff),size=(BOOTSTRAP_REPS,len(diff)));sampled=np.mean(diff[idx],axis=1)
            rows.append({"estimand":estimand,"contrast":f"{condition}_vs_delta_zero","metric":"mae_delta_vs_zero",
                "estimate":float(diff.mean()),"ci95_low":float(np.quantile(sampled,.025)),"ci95_high":float(np.quantile(sampled,.975)),
                "includes_zero":bool(np.quantile(sampled,.025)<=0<=np.quantile(sampled,.975)),"n_compositions":70,
                "bootstrap_replicates":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,
                "direction":"negative favors model over zero predictor"})
    result=pd.DataFrame(rows)
    # Holm correction for the four primary A geometry effects on pairwise ΔE MAE.
    effects=result[(result.estimand=="A")&result.contrast.str.startswith("efecto_geometria_")&
                   (result.metric=="mae_delta_signed_pairs_mev_atom")].copy()
    pvals=[]
    for _,row in effects.iterrows():
        table=formula_tables["A"];diff=(table[row.contrast.replace("efecto_geometria_","")+"_GEOMETRÍA"].mae_delta_signed_pairs_mev_atom-
                                        table[row.contrast.replace("efecto_geometria_","")+"_SLICES"].mae_delta_signed_pairs_mev_atom).to_numpy()
        null=np.mean(rng.choice((-1,1),size=(BOOTSTRAP_REPS,len(diff)))*diff[None,:],axis=1)
        pval=(1+np.sum(np.abs(null)>=abs(diff.mean())))/(BOOTSTRAP_REPS+1);pvals.append(pval)
    order=np.argsort(pvals);adjusted=np.empty(len(pvals));running=0.0
    for rank,pos in enumerate(order):
        value=min(1.0,(len(pvals)-rank)*pvals[pos]);running=max(running,value);adjusted[pos]=running
    mapping={effects.iloc[i].contrast:float(adjusted[i]) for i in range(len(effects))}
    result["holm_p_value_primary_four_geometry_delta_effects"]=result.contrast.map(mapping)
    return result,{"holm_p_values":mapping,"family":"four primary geometry effects on A ΔE MAE"}


def gap_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Agrupa error y exactitud por magnitud del gap DFT con denominadores."""
    rows=[]
    for condition in CONDITIONS:
        for low,high in GAP_BINS:
            x=pairs[(pairs.condition==condition)&(pairs.abs_delta_dft_mev_atom>=low)&(pairs.abs_delta_dft_mev_atom<high)]
            eligible=x[x.eligible_gt_5_mev]
            correct_values=np.asarray([False if pd.isna(value) else bool(value)
                                        for value in eligible.pairwise_correct],dtype=bool)
            rows.append({"condition":condition,"gap_bin":f"{low}-{int(high) if np.isfinite(high) else 'más'}",
                "n_pairs_seeded":len(x),"n_unique_pairs":int(x[["composition_id","structure_id_a","structure_id_b"]].drop_duplicates().shape[0]),
                "mae_delta_mev_atom":float(x.abs_error_mev_atom.mean()) if len(x) else np.nan,
                "zero_baseline_mae_mev_atom":float(x.abs_delta_dft_mev_atom.mean()) if len(x) else np.nan,
                "n_eligible_pairs_gt5":int(len(eligible)),"pairwise_correct_count":int(correct_values.sum()),
                "pairwise_accuracy_including_unresolved":float(correct_values.mean()) if len(eligible) else np.nan,
                "n_unresolved":int((eligible.eligible_gt_5_mev & ~eligible.resolved_gt_1_mev).sum())})
    return pd.DataFrame(rows)


def geometry_stratified_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Resume errores por terciles de cada descriptor, sin usarlos para ajustar."""
    source=pd.read_csv(ROOT/"experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv")
    features=["a_reduced_shape","b_reduced_shape","c_reduced_shape","volume_per_atom_ang3"]
    frame=pred.merge(source[["structure_id",*features]],on="structure_id",validate="many_to_one")
    frame["true_residual_ev_atom"]=frame.energy_per_atom_dft_ev-frame.groupby(
        ["condition","fold","seed","composition_id"]).energy_per_atom_dft_ev.transform("mean")
    frame["residual_abs_error_mev_atom"]=(frame.residual_pred_ev_atom-frame.true_residual_ev_atom).abs()*1000
    frame["energy_abs_error_mev_atom"]=(frame.energy_per_atom_pred_ev-frame.energy_per_atom_dft_ev).abs()*1000
    rows=[]
    for feature in features:
        # Cutpoints are computed over unique physical IDs, not repeated seeds.
        values=source.drop_duplicates("structure_id")[feature]
        edges=np.unique(np.quantile(values,[0,1/3,2/3,1]))
        labels=[f"tercil_{i+1}" for i in range(len(edges)-1)]
        if len(edges)<2:continue
        frame[f"_{feature}_group"]=pd.cut(frame[feature],bins=edges,labels=labels,include_lowest=True,duplicates="drop")
        for (condition,group),part in frame.groupby(["condition",f"_{feature}_group"],observed=True):
            rows.append({"condition":condition,"descriptor":feature,"tercil":str(group),
                "range_low":float(part[feature].min()),"range_high":float(part[feature].max()),
                "n_seeded_structures":len(part),"n_unique_structures":part.structure_id.nunique(),
                "n_compositions":part.composition_id.nunique(),
                "mae_energy_mev_atom":float(part.energy_abs_error_mev_atom.mean()),
                "mae_residual_mev_atom":float(part.residual_abs_error_mev_atom.mean()),
                "interpretation":"estratificación descriptiva OOF; no prueba efecto causal"})
    return pd.DataFrame(rows)


def regression_stats(x: np.ndarray,y: np.ndarray,mae_scale:float=1000.0) -> tuple[float,float,float,float]:
    """OLS con intercepto, R² del ajuste y MAE de observaciones emparejadas."""
    m,b=np.polyfit(x,y,1);pred=m*x+b;r2=1-np.sum((y-pred)**2)/max(np.sum((y-y.mean())**2),1e-30)
    return float(m),float(b),float(r2),float(np.mean(np.abs(y-x))*mae_scale)


def generate_figures(pred: pd.DataFrame, comp: pd.DataFrame, pairs: pd.DataFrame,
                     gap: pd.DataFrame, boots: pd.DataFrame, histories: pd.DataFrame,
                     repr_diag: pd.DataFrame, geometry_bins: pd.DataFrame) -> None:
    """Genera hasta diez figuras científicas descriptivas en PNG."""
    IMAGES.mkdir(parents=True,exist_ok=True)
    colors={"F0":"#777777","F2":"#4C78A8","F4":"#F2A541","F8":"#59A14F"}
    # 1. Resumen de métricas principales y de clasificación.
    metrics=[("mae_energy_mev_atom","MAE E/N (meV/átomo)"),("mae_residual_mev_atom","MAE residuo (meV/átomo)"),
             ("mae_delta_signed_pairs_mev_atom","MAE ΔE (meV/átomo)"),("ground_state_exact","Acierto GS"),
             ("pairwise_accuracy_macro","Exactitud de pares")]
    fig,axes=plt.subplots(1,5,figsize=(17,4.7))
    for ax,(metric,label) in zip(axes,metrics):
        table=comp.groupby(["condition","composition_id"])[metric].mean().reset_index()
        for ix,d in enumerate(("F0","F2","F4","F8")):
            vals=[table.loc[table.condition==f"{d}_{s}",metric].mean() for s in ("SLICES","GEOMETRÍA")]
            ax.bar(ix-.18,vals[0],width=.34,color=colors[d],alpha=.48,hatch=None)
            ax.bar(ix+.18,vals[1],width=.34,color=colors[d],alpha=1.0,hatch=None)
        ax.set_title(label,fontsize=9);ax.set_xticks(range(4),["F0","F2","F4","F8"]);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Macro promedio por fórmula; semillas promediadas antes")
    fig.suptitle("ABS₃: SLICES y geometría de celda según profundidad",weight="bold")
    fig.text(.5,.015,"Cada barra resume 70 composiciones (210 estructuras, 3 polimorfos); 5 pliegues y 3 semillas. Claro: SLICES; oscuro: SLICES + celda. Estadísticos primarios A.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.9));fig.savefig(IMAGES/"comparacion_slices_vs_geometria_por_profundidad_finetuning.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 2. Efecto pareado por geometría para MAE ΔE.
    fig,ax=plt.subplots(figsize=(8,4.6));rows=[]
    for d in DEPTH_LABELS:
        r=boots[(boots.estimand=="A")&(boots.contrast==f"efecto_geometria_{d}")&(boots.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0];rows.append(r)
    y=np.arange(4)
    for i,r in enumerate(rows):
        ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci95_low],[r.ci95_high-r.estimate]],fmt="o",color=colors[DEPTH_LABELS[i]],capsize=4)
    ax.axvline(0,color="black",ls="--");ax.set_yticks(y,[f"{d}: geometría − SLICES" for d in DEPTH_LABELS]);ax.invert_yaxis()
    ax.set_xlabel("Cambio de MAE de ΔE (meV/átomo); negativo favorece geometría");ax.set_title("Efecto pareado de añadir geometría en cada profundidad")
    ax.text(.01,-.19,"IC95% percentil: 10 000 remuestras apareadas de 70 fórmulas. Estimando A; mismos IDs, pliegues y semillas.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(IMAGES/"efecto_geometria_por_profundidad_intervalos_confianza.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 3. Interacciones en ΔE.
    fig,ax=plt.subplots(figsize=(7,4.5));ds=("F0","F2","F4","F8");ys=[]
    for i,d in enumerate(ds):
        if d=="F0":est=0.0;lo=hi=0.0
        else:
            r=boots[(boots.estimand=="A")&(boots.contrast==f"interaccion_{d}")&(boots.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
            est,lo,hi=r.estimate,r.ci95_low,r.ci95_high
        ax.errorbar(i,est,yerr=[[est-lo],[hi-est]],fmt="o",capsize=4,color=colors[d]);ys.append(est)
    ax.axhline(0,color="black",ls="--");ax.set_xticks(range(4),ds);ax.set_ylabel("Interacción en MAE ΔE (meV/átomo)")
    ax.set_title("¿Cambia el efecto geométrico con la profundidad?")
    ax.text(.02,-.2,"Interacción = (MAE geom − MAE SLICES)F − (MAE geom − MAE SLICES)F0. Negativo: aporte mayor/penalización menor que F0. IC95% por fórmula.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(IMAGES/"interaccion_geometria_y_profundidad_finetuning.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 4-6. Energía absoluta, residuo y ΔE paridad de ocho condiciones.
    parity_inputs={}
    for xcol,ycol,title,xlabel,ylabel,filename,metricname in [
        ("energy_per_atom_dft_ev","energy_per_atom_pred_ev","Energía por átomo: predicción frente a DFT","DFT E/N (eV/átomo)","Predicción E/N (eV/átomo)","paridad_energia_dft_slices_vs_geometria_finetuning.png","absolute"),
        (None,None,"Residuo intrafórmula: predicción frente a DFT","Residuo DFT (meV/átomo)","Residuo predicho (meV/átomo)","paridad_residuos_slices_geometria_y_profundidad.png","residual"),
        (None,None,"Diferencias entre pares: predicción frente a DFT","ΔE DFT B−A (meV/átomo)","ΔE predicha B−A (meV/átomo)","paridad_diferencias_energeticas_slices_geometria_finetuning.png","delta")]:
        for cond in CONDITIONS:
            if metricname=="absolute":
                part=pred[pred.condition==cond];x=part[xcol].to_numpy()*1;y=part[ycol].to_numpy()*1
            elif metricname=="residual":
                part=comp_seed_for_parity(comp,cond);x=part.true_residual_mev.to_numpy();y=part.pred_residual_mev.to_numpy()
            else:
                part=pairs[pairs.condition==cond];x=part.delta_dft_mev_atom.to_numpy();y=part.delta_pred_mev_atom.to_numpy()
            parity_inputs[cond]=(x,y)
        all_x=np.concatenate([v[0] for v in parity_inputs.values()]);all_y=np.concatenate([v[1] for v in parity_inputs.values()])
        shared_lo=min(float(all_x.min()),float(all_y.min()));shared_hi=max(float(all_x.max()),float(all_y.max()))
        pad=.03*(shared_hi-shared_lo or 1);shared_lo-=pad;shared_hi+=pad
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for ax,cond in zip(axes.flat,CONDITIONS):
            x,y=parity_inputs[cond]
            scale=1000.0 if metricname=="absolute" else 1.0
            m,b,r2,mae=regression_stats(x,y,scale)
            lo,hi=shared_lo,shared_hi
            ax.scatter(x,y,s=8,alpha=.35,color=colors[cond.split("_")[0]])
            ax.plot([lo,hi],[lo,hi],"--",c="#26364A",lw=1,label="Ideal: y=x")
            ax.plot([lo,hi],[m*lo+b,m*hi+b],"-.",c="#C44E52",lw=1,label="OLS: y=mx+b")
            ax.set_title(cond.replace("_"," · "),fontsize=9);ax.set_xlabel(xlabel,fontsize=8);ax.set_ylabel(ylabel,fontsize=8)
            ax.text(.03,.97,f"N={len(x)}\nm={m:.3f}; b={b:.3g}\nR² OLS={r2:.3f}\nMAE={mae:.2f} meV/átomo",transform=ax.transAxes,va="top",fontsize=7)
            ax.grid(alpha=.15)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=2,bbox_to_anchor=(.5,.99))
        fig.suptitle(f"ABS₃: {title}",y=.94,weight="bold")
        fig.text(.5,.01,"Pliegues outer de desarrollo: 210 estructuras, 70 fórmulas, tres semillas; cada punto es una predicción de una estructura/pair. Estadísticos calculados sobre todos los puntos mostrados; no es test ciego.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.9));fig.savefig(IMAGES/filename,dpi=300,bbox_inches="tight");plt.close(fig)
    # 7. Error de ΔE por intervalo de gap y baseline cero.
    fig,ax=plt.subplots(figsize=(10,5));bins=[f"{low}-{int(high) if np.isfinite(high) else 'más'}" for low,high in GAP_BINS]
    offsets=np.linspace(-.3,.3,8)
    for idx,cond in enumerate(CONDITIONS):
        sub=gap[gap.condition==cond].set_index("gap_bin").reindex(bins)
        ax.plot(np.arange(len(bins))+offsets[idx],sub.mae_delta_mev_atom,marker="o",label=cond.replace("_"," · "),color=colors[cond.split("_")[0]],ls="-" if "SLICES" in cond else "--")
    zero=gap[gap.condition=="F0_SLICES"].set_index("gap_bin").reindex(bins).zero_baseline_mae_mev_atom
    ax.plot(range(len(bins)),zero,c="black",ls=":",marker="x",label="Predictor ΔE=0")
    ax.set_xticks(range(len(bins)),bins);ax.set_xlabel("Magnitud |ΔE DFT| (meV/átomo)");ax.set_ylabel("MAE de diferencias pareadas (meV/átomo)")
    ax.set_title("Error de diferencias energéticas según separación DFT");ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7)
    fig.text(.5,.01,"Intervalos semiabiertos [a,b); puntos son pares por pliegue/semilla. N por grupo en gap_metrics.csv; ranking no se evalúa para gaps ≤5 meV/átomo.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(IMAGES/"error_deltaE_por_gap_y_geometria_finetuning.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 8. Ranking pairwise y línea 50%.
    fig,ax=plt.subplots(figsize=(10,5))
    for cond in CONDITIONS:
        sub=gap[gap.condition==cond];ax.plot(range(len(GAP_BINS)),sub.pairwise_accuracy_including_unresolved,marker="o",label=cond.replace("_"," · "),color=colors[cond.split("_")[0]],ls="-" if "SLICES" in cond else "--")
    ax.axhline(.5,c="black",ls=":",label="Referencia aleatoria 50%");ax.set_xticks(range(len(GAP_BINS)),bins)
    ax.set_ylim(0,1);ax.set_xlabel("Magnitud |ΔE DFT| (meV/átomo)");ax.set_ylabel("Exactitud pairwise (no resueltos incorrectos)")
    ax.set_title("Ordenamiento de pares por magnitud del gap DFT");ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7)
    fig.text(.5,.01,"Umbral elegible |ΔE DFT|>5; no resuelto |ΔE predicha|≤1 se cuenta incorrecto. Denominadores por intervalo en gap_metrics.csv; 70 fórmulas, 3 semillas.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(IMAGES/"ranking_polimorfos_por_gap_slices_y_geometria.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 9. Curvas de entrenamiento geométricas, medias entre corridas y época seleccionada.
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for d in ("F2","F4","F8"):
        x=histories[histories.condition==d]
        if len(x):
            g=x.groupby("epoch")[["train_loss_minibatch_mean","inner_validation_loss_full"]].agg(["mean","sem"])
            for col,style,label in (("train_loss_minibatch_mean","-","entrenamiento"),("inner_validation_loss_full","--","validación interna completa")):
                axes[0].plot(g.index,g[(col,"mean")],style,color=colors[d],label=f"{d} · {label}")
    axes[0].set_xlabel("Época local");axes[0].set_ylabel("SmoothL1 residual normalizada");axes[0].set_title("Media de corridas geométricas");axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    selected=repr_diag.groupby("condition").best_epoch.apply(list)
    axes[1].boxplot([selected.get(f"{d}_GEOMETRÍA",[]) for d in ("F2","F4","F8")],tick_labels=["F2","F4","F8"])
    axes[1].set_ylabel("Época seleccionada");axes[1].set_title("Checkpoint elegido por validación interna");axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("Entrenamiento: pérdida de minibatch y validación interna completa",weight="bold")
    fig.text(.5,.01,"F2/F4/F8 geométricos; 15 corridas por profundidad, selección por menor SmoothL1 en inner-validation. Curvas son medias por época; barras no mostradas.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(IMAGES/"curvas_entrenamiento_validacion_geometria_finetuning.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 10. Drift y peso geométrico, relación descriptiva con error residual.
    fig,axes=plt.subplots(1,4,figsize=(16,4.5))
    for d in ("F2","F4","F8"):
        sub=repr_diag[repr_diag.condition==f"{d}_GEOMETRÍA"]
        axes[0].scatter([d]*len(sub),sub.mean_relative_embedding_change,s=24,color=colors[d],alpha=.7)
        axes[1].scatter([d]*len(sub),sub.geometry_weight_norm,s=24,color=colors[d],alpha=.7)
        mae=comp[comp.condition==f"{d}_GEOMETRÍA"].groupby("composition_id").mae_residual_mev_atom.mean().mean()
        residual_mae=float(comp[comp.condition==f"{d}_GEOMETRÍA"].mae_residual_mev_atom.mean())
        axes[2].scatter(sub.mean_relative_embedding_change,np.repeat(residual_mae,len(sub)),
                        s=28,color=colors[d],label=d)
    axes[0].set_ylabel("Cambio relativo medio de FINAL_END");axes[0].set_title("Drift por corrida")
    axes[1].set_ylabel("Norma L2 de los 56 pesos geométricos");axes[1].set_title("Magnitud de pesos (no causal)")
    axes[2].set_xlabel("Drift medio del embedding");axes[2].set_ylabel("MAE residual macro (meV/átomo)");axes[2].set_title("Asociación descriptiva");axes[2].legend()
    vol=geometry_bins[geometry_bins.descriptor=="volume_per_atom_ang3"]
    for cond in CONDITIONS:
        sub=vol[vol.condition==cond].sort_values("tercil")
        axes[3].plot(sub.tercil,sub.mae_residual_mev_atom,marker="o",ms=3,
                     color=colors[cond.split("_")[0]],ls="-" if "SLICES" in cond else "--",
                     label=cond.replace("_"," · "))
    axes[3].set_title("Error residual por tercil de volumen");axes[3].set_xlabel("Volumen por átomo · terciles físicos")
    axes[3].set_ylabel("MAE residual (meV/átomo)");axes[3].tick_params(axis="x",rotation=25);axes[3].legend(fontsize=6,ncol=2)
    fig.suptitle("Adaptación de representaciones y uso de geometría",weight="bold")
    fig.text(.5,.01,"Un punto por fold/semilla para drift/pesos; MAE resume fórmulas OOF. Norma de pesos no mide importancia causal; observaciones ya exploradas, sin test histórico.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(IMAGES/"adaptacion_embeddings_y_aporte_geometrico.png",dpi=300,bbox_inches="tight");plt.close(fig)


def comp_seed_for_parity(comp: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Expande residuos estructurales agregados necesarios para el panel de paridad."""
    rows=[]
    pred_path=EXPERIMENT/"predicciones_oof_por_estructura.csv";pred=pd.read_csv(pred_path)
    for _,g in pred[pred.condition==condition].groupby(["fold","seed","composition_id"]):
        d=g.energy_per_atom_dft_ev.to_numpy(float);p=g.energy_per_atom_pred_ev.to_numpy(float)
        rows.extend({"true_residual_mev":(d[i]-d.mean())*1000,"pred_residual_mev":(p[i]-p.mean())*1000} for i in range(3))
    return pd.DataFrame(rows)


def load_clean_training_history() -> pd.DataFrame:
    """Reconstruye el historial completo desde el log y descarta smoke/duplicados.

    Durante la recuperación de una corrida duplicada, algunas filas del CSV
    append-only quedaron con anchos distintos. El log conserva cada evento;
    se prefiere el proceso principal (eventos sin run_type añadido por la
    corrida que se interrumpió) y se descartan los smoke tests.
    """
    rows=[]
    log_path=EXPERIMENT/"run.log"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        marker=" INFO "
        if marker not in line or "'event': 'epoch'" not in line:
            continue
        try:
            event=ast.literal_eval(line.split(marker,1)[1])
        except (ValueError,SyntaxError):
            continue
        if event.get("event")!="epoch" or event.get("run_type")=="smoke":
            continue
        if "_smoke" in event.get("run",""):
            continue
        row={k:v for k,v in event.items() if k!="event"}
        row["_source_priority"]=0 if "run_type" not in event else 1
        rows.append(row)
    history=pd.DataFrame(rows)
    required={"condition","fold","seed","epoch","train_loss_minibatch_mean","inner_validation_loss_full"}
    if not required.issubset(history.columns):
        raise ValueError(f"El log no aporta el historial completo; columnas={sorted(history.columns)}")
    history=(history.sort_values(["condition","fold","seed","epoch","_source_priority"])
             .drop_duplicates(["condition","fold","seed","epoch"],keep="first")
             .drop(columns="_source_priority"))
    return history


def reconcile_training_resources(history: pd.DataFrame) -> dict[str,Any]:
    """Reconcilia 45 estados completados y recursos medidos en JSON/logs."""
    resources=json.loads((EXPERIMENT/"resources.json").read_text())
    runs=resources.setdefault("runs",{})
    expected=[(f"f{d}_fold{fold}_seed{seed}",f"F{d}",fold,seed)
              for d in (2,4,8) for fold in range(5) for seed in SEEDS]
    grouped={(str(run),str(cond),int(fold),int(seed)):part.sort_values("epoch")
             for (run,cond,fold,seed),part in history.groupby(["run","condition","fold","seed"])}
    for run,condition,fold,seed in expected:
        part=grouped.get((run,condition,fold,seed))
        run_dir=EXPERIMENT/"checkpoints"/run
        if part is None or not (run_dir/"complete.flag").is_file() or not (run_dir/"best.pt").is_file():
            raise RuntimeError(f"Corrida incompleta o sin historial verificable: {run}")
        current=runs.get(run,{})
        last=part.iloc[-1]
        cpu_seconds=current.get("cpu_seconds")
        runs[run]={**current,"status":"COMPLETE","condition":condition,"fold":fold,"seed":seed,
            "blocks":list({"F2":[10,11],"F4":[8,9,10,11],"F8":list(range(4,12))}[condition]),
            "epochs":int(part.epoch.max()),"best_epoch":int(last.best_epoch),
            "best_inner_loss":float(part.inner_validation_loss_full.min()),
            "optimizer_steps":int(part.optimizer_steps.max()),"wall_seconds":float(last.elapsed_seconds),
            "peak_rss_bytes":int(part.peak_rss_bytes.max()),"cpu_seconds":cpu_seconds,
            "resource_source":"run.log + resources.json"}
    resources["status"]="COMPLETE"
    resources["completed_runs"]=len(expected)
    resources["planned_training_runs"]=len(expected)
    resources["active_runs"]=0
    resources["cpu_seconds_complete_runs"]=sum(runs[run].get("cpu_seconds") is not None for run,_,_,_ in expected)
    resources["cpu_seconds_missing_runs"]=[run for run,_,_,_ in expected if runs[run].get("cpu_seconds") is None]
    resources["wall_seconds_observed_total"]=sum(runs[run].get("wall_seconds",0.0) for run,_,_,_ in expected)
    resources["cpu_seconds_observed_total"]=sum(runs[run].get("cpu_seconds",0.0) or 0.0 for run,_,_,_ in expected)
    resources["peak_rss_bytes_observed"]=max(runs[run].get("peak_rss_bytes",0) for run,_,_,_ in expected)
    atomic_json(EXPERIMENT/"resources.json",resources)
    return resources


def create_diagrams() -> None:
    """Guarda fuentes Mermaid editables y dos diagramas técnicos PNG."""
    source=EXPERIMENT/"diagram_sources";source.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    mermaid_a="""flowchart LR
  S[SLICES estándar] --> T[Tokenizer MatGPT]
  T --> E[Embedding de tokens y posiciones · congelados]
  E --> B[12 bloques Transformer · dimensión 768]
  B --> LN[ln_f · congelada]
  LN --> H[FINAL_END · 768]
  G[Celda DFT estática · 7 descriptores] --> SC[StandardScaler · inner-train]
  H --> C[Concatenación · 775]
  SC --> C
  C --> M[MLP8 · 775→8→1 · GELU · dropout 0.1]
  M --> R[Residuo bruto por polimorfo]
  R --> Z[Centrado: r_i − media(r_A,r_B,r_C)]
  B -. gradientes según condición .-> O[Optimizador]
  M -. gradientes .-> O
  N[Actualiza F2: 10–11 · F4: 8–11 · F8: 4–11; F0: ningún bloque] -. configuración .-> O
"""
    mermaid_b="""flowchart LR
  C[Composición química] --> V[17 fracciones estequiométricas]
  V --> R[Ridge composicional fijo · μ_c]
  A[SLICES_A + celda_A] --> M[Una instancia MatGPT adaptado + MLP8 · pesos compartidos]
  B[SLICES_B + celda_B] --> M
  D[SLICES_C + celda_C] --> M
  M --> RA[Residuo centrado δ_A]
  M --> RB[Residuo centrado δ_B]
  M --> RC[Residuo centrado δ_C]
  R --> EA[E/N_A = μ_c + δ_A]
  R --> EB[E/N_B = μ_c + δ_B]
  R --> EC[E/N_C = μ_c + δ_C]
  RA --> EA
  RB --> EB
  RC --> EC
  EA --> O[ΔE dentro de fórmula · argmin GS · ranking]
  EB --> O
  EC --> O
  O --> ET[E_total = N · E/N]
"""
    (source/"arquitectura_matgpt_finetuning_con_geometria_celda.mmd").write_text(mermaid_a,encoding="utf-8")
    (source/"pipeline_regresion_hibrida_ridge_composicional_geometria.mmd").write_text(mermaid_b,encoding="utf-8")
    fig,ax=plt.subplots(figsize=(20,8));ax.axis("off");ax.set_xlim(0,20);ax.set_ylim(0,8)
    def box(x,y,w,h,text,color):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12",fc=color,ec="#45566f",lw=1.3));ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
    def arrow(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops={"arrowstyle":"->","lw":1.5,"color":"#45566f"})
    ax.text(10,7.55,"A. MatGPT con geometría de celda concatenada después de FINAL_END",ha="center",fontsize=14,weight="bold")
    box(.25,6.25,1.55,.78,"SLICES", "#dbeafe");box(2.1,6.25,1.65,.78,"Tokenizer", "#dbeafe")
    box(4.05,6.25,2.0,.78,"Embeddings\ncongelados", "#dddddd")
    box(6.35,6.25,3.0,.78,"12 bloques Transformer\nsegún F0 / F2 / F4 / F8", "#eadff5")
    box(9.65,6.25,1.45,.78,"ln_f\ncongelada", "#dddddd");box(11.4,6.25,1.7,.78,"FINAL_END\n768", "#dbeafe")
    box(13.55,6.25,1.75,.78,"Concatenar\n775 entradas", "#fff0cc")
    box(15.75,6.25,2.35,.78,"MLP8\n775→8→1 · GELU\ndropout 0.1", "#f7e7ce")
    box(18.45,6.25,1.4,.78,"Centrar\nresiduos", "#f7e7ce")
    for a,b in ((1.8,2.1),(3.75,4.05),(6.05,6.35),(9.35,9.65),(11.1,11.4),(13.1,13.55),(15.3,15.75),(18.1,18.45)):
        arrow(a,6.64,b,6.64)
    # Tabla de bloques: cada fila representa una configuración y el color marca gradientes.
    ax.text(.45,4.85,"Bloques Transformer entrenables por condición",ha="left",fontsize=10,weight="bold")
    for row,(label,start) in enumerate((("F0",12),("F2",10),("F4",8),("F8",4))):
        yy=4.25-row*.58;ax.text(3.9,yy+.18,label,ha="right",va="center",fontsize=9,weight="bold")
        for block in range(12):
            x=4.1+block*.48;color="#ef9c64" if block>=start else "#d9dee7"
            ax.add_patch(plt.Rectangle((x,yy),.40,.34,facecolor=color,edgecolor="#55657a",lw=.55))
            ax.text(x+.20,yy+.17,str(block),ha="center",va="center",fontsize=6.5)
    ax.add_patch(plt.Rectangle((14.15,5.0),.22,.22,facecolor="#ef9c64",edgecolor="#55657a"));ax.text(14.5,5.11,"se actualiza",va="center",fontsize=8)
    ax.add_patch(plt.Rectangle((16.0,5.0),.22,.22,facecolor="#d9dee7",edgecolor="#55657a"));ax.text(16.35,5.11,"permanece congelado",va="center",fontsize=8)
    box(.5,1.25,2.5,.82,"Celda estática DFT\n7 descriptores", "#d9efe5")
    box(3.55,1.25,3.05,.82,"StandardScaler\n45 fórmulas inner-train", "#d9efe5")
    arrow(3.0,1.66,3.55,1.66)
    ax.plot([6.6,13.2],[1.66,1.66],color="#45566f",lw=1.5)
    ax.plot([13.2,13.2],[1.66,6.64],color="#45566f",lw=1.5)
    arrow(13.2,6.64,13.55,6.64)
    ax.text(10,.35,"La geometría entra después del Transformer; embeddings y ln_f están congelados. No se usa LoRA.\nLos bloques marcados y la MLP8 reciben gradientes; los descriptores no tienen parámetros entrenables.",ha="center",fontsize=9)
    fig.savefig(IMAGES/"arquitectura_matgpt_finetuning_con_geometria_celda.png",dpi=300,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(18,8));ax.axis("off");ax.set_xlim(0,18);ax.set_ylim(0,8)
    ax.text(9,7.55,"B. Predicción híbrida para tres polimorfos de una misma fórmula",ha="center",fontsize=14,weight="bold")
    box(.35,6.4,2.4,.75,"Composición química\n17 fracciones", "#d9efe5");box(3.25,6.4,2.3,.75,"Ridge fijo\nmedia μ_c", "#d9efe5");arrow(2.75,6.78,3.25,6.78)
    box(.35,3.9,2.3,.8,"SLICES_A + celda_A", "#dbeafe");box(.35,2.8,2.3,.8,"SLICES_B + celda_B", "#dbeafe");box(.35,1.7,2.3,.8,"SLICES_C + celda_C", "#dbeafe")
    box(3.45,3.15,3.1,1.05,"MatGPT + MLP8\nuna instancia; pesos compartidos", "#eadff5")
    for yy in (4.3,3.2,2.1):arrow(2.65,yy,3.45,3.68)
    box(7.1,4.5,2.2,.7,"Residuo δ_A", "#f7e7ce");box(7.1,3.2,2.2,.7,"Residuo δ_B", "#f7e7ce");box(7.1,1.9,2.2,.7,"Residuo δ_C", "#f7e7ce")
    for yy in (4.85,3.55,2.25):arrow(6.55,3.68,7.1,yy)
    box(10.15,4.5,2.45,.7,"E/N_A = μ_c + δ_A", "#fff0cc");box(10.15,3.2,2.45,.7,"E/N_B = μ_c + δ_B", "#fff0cc");box(10.15,1.9,2.45,.7,"E/N_C = μ_c + δ_C", "#fff0cc")
    for yy in (4.85,3.55,2.25):arrow(9.3,yy,10.15,yy)
    box(13.35,3.05,3.8,1.25,"ΔE intrafórmula\nargmin(E/N): estado fundamental\nordenamiento de A, B y C", "#d9efe5")
    # Ridge alimenta μ_c mediante un bus fuera de las cajas de residuo.
    ax.plot([5.55,9.8],[6.78,6.78],color="#45566f",lw=1.5)
    ax.plot([9.8,9.8],[2.25,6.78],color="#45566f",lw=1.5)
    for yy in (4.85,3.55,2.25):arrow(9.8,yy,10.15,yy)
    for yy in (4.85,3.55,2.25):arrow(12.6,yy,13.35,3.68)
    ax.text(9,.72,"E_total,i = N_i × (E/N)_i. Las etiquetas DFT solo evalúan; no son entradas.\nLas diferencias energéticas y el ranking se calculan únicamente dentro de la fórmula.",ha="center",fontsize=10)
    fig.savefig(IMAGES/"pipeline_regresion_hibrida_ridge_composicional_geometria.png",dpi=300,bbox_inches="tight");plt.close(fig)


def write_report(metrics: pd.DataFrame, bootstrap: pd.DataFrame, integrity: dict[str,Any],
                 resources: dict[str,Any], repr_diag: pd.DataFrame, gaps: pd.DataFrame,
                 multiple: dict[str,Any]) -> None:
    """Redacta un solo informe con hechos verificados y resultados observados."""
    primary=metrics[metrics.estimand=="A_corrida_individual"].pivot(index="condition",columns="metric",values="value")
    def f(cond,metric):return float(primary.loc[cond,metric])
    effects=bootstrap[(bootstrap.estimand=="A")&(bootstrap.metric=="mae_delta_signed_pairs_mev_atom")&bootstrap.contrast.str.startswith("efecto_geometria_")]
    lines=["# Experimento D — fine-tuning de MatGPT con SLICES y geometría de celda","",
      "## Estado y pregunta","",
      "**Resultado experimental completado en evaluación de desarrollo, no en un test ciego.** La pregunta es si siete descriptores métricos de celda mejoran las diferencias energéticas al adaptar bloques superiores de MatGPT. No se usó geometría como tokens: se concatenó después de FINAL_END.","",
      "## Procedencia y protocolo","",
      f"- Geometrías: {integrity['geometry_entries']} de 210 IDs, procedencia uniforme `{integrity['geometry_source']}`; SHA-256 `{integrity['geometry_sha256']}`. Son celdas estáticas DFT; `static` no certifica relajación completa ni disponibilidad en inferencia SLICES-only.",
      f"- Dataset: {integrity['structures']} estructuras / {integrity['compositions']} fórmulas; 3 polimorfos, 5 pliegues y 3 semillas. Train/validation/outer son grupos por composición (45/11/14 por pliegue). No se leyó el test histórico.",
      f"- MatGPT: checkpoint SHA-256 `{integrity['checkpoint_sha256']}`; 12 bloques, hidden 768. MLP8 geométrica 775→8→1 (6,217 parámetros); cabeza SLICES 768→8→1 (6,161). Los 56 pesos extra empezaron en cero y el resto copió la inicialización semilla-específica del control.",
      "- Ridge: se reutilizaron los cinco Ridge composicionales sin reentrenar; dispersión entre salidas por condición = 0 eV/átomo. La normalización de geometría se ajustó por pliegue con las 135 estructuras de 45 fórmulas inner-train.",
      "- Controles: F0 SLICES/geom se reutilizan del estudio congelado apareado, reemplazando su Ridge de embeddings por la media Ridge composicional fija; F2/F4/F8 SLICES-only se reutilizan de la ablación de profundidad. Nuevas corridas: 45 geometría F2/F4/F8. Los controles se preservan como archivos fuente.",
      "- Targets: solo residuo DFT por átomo centrado dentro del triplete; SmoothL1 β=1 normalizada por escala inner-train. El checkpoint se seleccionó por pérdida completa de inner-validation. Sin LoRA, fuerzas, estrés, pérdida pairwise ni nuevos descriptores.","",
      "## Resultados principales (estimando A)","",
      "A calcula cada métrica por semilla y promedia dentro de fórmula; 70 fórmulas son la unidad bootstrap. B promedia primero las predicciones por estructura y se mantiene separado en `metrics.json`.",
      "| Profundidad | MAE ΔE SLICES | MAE ΔE geometría | Cambio geom − SLICES | MAE residuo SLICES | MAE residuo geom | GS SLICES / geom | Pares correctos SLICES / geom |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DEPTH_LABELS:
        a=f"{d}_SLICES";b=f"{d}_GEOMETRÍA";r=effects[effects.contrast==f"efecto_geometria_{d}"]
        change=float(r.estimate.iloc[0]) if len(r) else float("nan")
        lines.append(f"| {d} | {f(a,'mae_delta_signed_pairs_mev_atom'):.2f} | {f(b,'mae_delta_signed_pairs_mev_atom'):.2f} | {change:+.2f} | {f(a,'mae_residual_mev_atom'):.2f} | {f(b,'mae_residual_mev_atom'):.2f} | {f(a,'ground_state_exact'):.3f} / {f(b,'ground_state_exact'):.3f} | {f(a,'pairwise_accuracy_pooled'):.3f} / {f(b,'pairwise_accuracy_pooled'):.3f} |")
    lines += ["","Los números de errores están en meV/átomo; GS es proporción exacta sobre 70 fórmulas. Los conteos pairwise se reconstruyeron directamente; no se derivan de tasas redondeadas.","",
      "## Contrastes, incertidumbre e interacciones","",
      "Los IC son percentiles bootstrap apareados por 70 fórmulas, 10,000 réplicas. Contrastes por profundidad son exploratorios y los IC individuales no están ajustados por multiplicidad; se aplica Holm a los cuatro efectos principales de geometría sobre MAE ΔE y se reportan sus valores en `bootstrap_contrastes.csv`.",
      "- Efectos principales: ver [figura de efectos](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/efecto_geometria_por_profundidad_intervalos_confianza.png).",
      "- Interacción: (MAE geom−MAE SLICES) en profundidad − mismo contraste F0; negativo significa que la geometría beneficia más (o perjudica menos) que en F0. Ver [figura de interacción](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/interaccion_geometria_y_profundidad_finetuning.png).",
      "- Se comparó cada condición contra ΔE_pred=0 con idénticas parejas. Un IC que incluya cero no demuestra equivalencia.","",
      "## Métricas corregidas y brechas","",
      "ΔE anclada se calcula anclando predicción y DFT al mismo polimorfo mínimo DFT; el promedio primario usa los dos no-ancla. La exactitud pairwise considera elegibles los pares con |ΔE_DFT|>5 meV/átomo; predicciones con |ΔE_pred|≤1 son incorrectas en el denominador principal. Los pares no se cruzan entre fórmulas. Ver `predicciones_oof_por_par.csv` y `metricas_por_composicion.csv`.",
      "El desglose de gaps usa intervalos [0,5), [5,10), [10,25), [25,50), [50,100) y ≥100 meV/átomo; las cifras y denominadores están en `metricas_por_gap.csv`. Para 5–25 meV/átomo no se concluye mejora solo por MAE global.","",
      "## Adaptación de representación y geometría","",
      f"Para las corridas F2/F4/F8, el drift medio coseno/relativo de FINAL_END y normas de 56 pesos geométricos se guardan por fold/semilla en `diagnostico_representacional_y_geometrico.csv` (promedio de coseno {repr_diag.mean_cosine_original_vs_adapted.mean():.4f}; cambio relativo {repr_diag.mean_relative_embedding_change.mean():.4f}). Normas y correlaciones son diagnósticos descriptivos, no importancia causal.",
      "La rama geométrica solo contiene forma de celda reducida, ángulos y volumen por átomo. No incluye coordenadas internas, distancias ni enlaces. No puede inferirse de SLICES por sí sola en inferencia sin una celda disponible.","",
      "## Recursos y limitaciones","",
      f"Dispositivo: CPU, 2 hilos; nuevas corridas completas: {resources.get('completed_runs',0)}; tiempo acumulado registrado: {resources.get('wall_seconds_observed_total',0)/3600:.2f} h de proceso; CPU acumulada: {resources.get('cpu_seconds_observed_total',0)/3600:.2f} h ({resources.get('cpu_seconds_complete_runs',0)}/45 corridas con contador de CPU exacto); RSS máximo: {resources.get('peak_rss_bytes_observed',0)/1e9:.2f} GB. El tiempo de pared es suma de corridas, no duración de calendario.",
      "Si falta algún contador de CPU, se dejará como no disponible; no se inferirá desde el tiempo de pared. Tiempo de corrida y RSS se reconstruyen del log cuando el registro consolidado no los conserva.",
      "Las 70 composiciones fueron exploradas en análisis previos; los outer folds no son una prueba externa no vista por el investigador. Las celdas `gga-static_endpoint_cleaned` no certifican homogeneidad de XC/U/correcciones ni relajación total. Diferencias por profundidad pueden reflejar selección/semillas; la geometría añade 56 pesos y una entrada distinta, aunque su inicialización fue cero.","",
      "## Conclusión","",
      "**HECHO VERIFICADO:** la comparación conserva IDs, folds, medias Ridge y controles SLICES-only; el modelo geométrico recibe gradientes y parte con predicción equivalente al control antes del primer update. **RESULTADO EXPERIMENTAL:** consultar los contrastes A y sus intervalos en el CSV/figuras antes de afirmar beneficio. **LIMITACIÓN:** esto prueba una fusión con celdas DFT disponibles, no que MatGPT pueda recuperar geometría desde SLICES ni que la geometría sea causa física única de cualquier cambio.","",
      "## Figuras","",
      "![Resumen de métricas](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/comparacion_slices_vs_geometria_por_profundidad_finetuning.png)",
      "![Efectos e interacción](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/efecto_geometria_por_profundidad_intervalos_confianza.png)",
      "![Paridad absoluta](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/paridad_energia_dft_slices_vs_geometria_finetuning.png)",
      "![Paridad ΔE](../../../images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/paridad_diferencias_energeticas_slices_geometria_finetuning.png)","",
      f"Fuentes OOF SHA-256: profundidad `{integrity['depth_oof_sha256']}`; geometría Frozen `{integrity['geometry_frozen_oof_sha256']}`; nueva predicción `{sha256(EXPERIMENT/'predicciones_oof_por_estructura.csv')}`."]
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/"reporte_cientifico.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    """Valida inputs, predice outer, evalúa ambos estimandos y exporta resultados."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--bootstrap-replicates",type=int,default=BOOTSTRAP_REPS)
    args=parser.parse_args()
    if args.bootstrap_replicates!=BOOTSTRAP_REPS:parser.error(f"La configuración fija es {BOOTSTRAP_REPS}")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    started=time.perf_counter();cpu=time.process_time()
    EXPERIMENT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    pred_path=EXPERIMENT/"predicciones_oof_por_estructura.csv"
    cached={name:EXPERIMENT/name for name in (
        "metricas_por_composicion.csv","predicciones_oof_por_par.csv",
        "diagnostico_representacional_y_geometrico.csv","metricas_por_geometria.csv",
        "metricas_recalculadas.csv","bootstrap_contrastes.csv","metricas_por_gap.csv")}
    cache_complete=pred_path.is_file() and all(path.is_file() for path in cached.values())
    if cache_complete:
        pred=pd.read_csv(pred_path);comp=pd.read_csv(cached["metricas_por_composicion.csv"])
        pairs=pd.read_csv(cached["predicciones_oof_por_par.csv"])
        repr_diag=pd.read_csv(cached["diagnostico_representacional_y_geometrico.csv"])
        geometry_bins=pd.read_csv(cached["metricas_por_geometria.csv"])
        estimates=pd.read_csv(cached["metricas_recalculadas.csv"])
        bootstrap=pd.read_csv(cached["bootstrap_contrastes.csv"]);gap=pd.read_csv(cached["metricas_por_gap.csv"])
        if len(pred)!=5040 or len(pairs)!=5040:
            raise ValueError("Los CSV almacenados no contienen las 5,040 predicciones esperadas")
        mask=(bootstrap.estimand=="A")&bootstrap.contrast.str.startswith("efecto_geometria_")
        multiple={"holm_p_values":dict(zip(bootstrap.loc[mask,"contrast"],
                    bootstrap.loc[mask,"holm_p_value_primary_four_geometry_delta_effects"].fillna(0.0))),
                  "family":"four primary geometry effects on A ΔE MAE"}
    else:
        pred_geo,repr_diag=extract_new_geometry_predictions();pred=compose_predictions(pred_geo)
        pred.to_csv(pred_path,index=False)
        comp,pairs=calculate_metrics(pred);comp.to_csv(cached["metricas_por_composicion.csv"],index=False);pairs.to_csv(cached["predicciones_oof_por_par.csv"],index=False)
        repr_diag.to_csv(cached["diagnostico_representacional_y_geometrico.csv"],index=False)
        geometry_bins=geometry_stratified_metrics(pred);geometry_bins.to_csv(cached["metricas_por_geometria.csv"],index=False)
        estimates,tables=metric_estimands(pred,comp);bootstrap,multiple=bootstrap_analysis(tables);gap=gap_metrics(pairs)
        estimates.to_csv(cached["metricas_recalculadas.csv"],index=False);bootstrap.to_csv(cached["bootstrap_contrastes.csv"],index=False);gap.to_csv(cached["metricas_por_gap.csv"],index=False)
    histories=load_clean_training_history()
    resources=reconcile_training_resources(histories)
    histories["run_type"]="full"
    histories.to_csv(EXPERIMENT/"historial_entrenamiento.csv",index=False)
    generate_figures(pred,comp,pairs,gap,bootstrap,histories,repr_diag,geometry_bins);create_diagrams()
    integrity=json.loads((EXPERIMENT/"summary.json").read_text());resources=json.loads((EXPERIMENT/"resources.json").read_text())
    write_report(estimates,bootstrap,integrity,resources,repr_diag,gap,multiple)
    # El JSON métrico guarda estimandos separados y agregados serializables.
    metric_content={}
    for estimand in ("A_corrida_individual","B_ensemble_3_semillas"):
        sub=estimates[estimates.estimand==estimand]
        metric_content[estimand]={}
        for condition,part in sub.groupby("condition"):
            metric_content[estimand][condition]={r.metric:float(r.value) if pd.notna(r.value) else None for r in part.itertuples()}
    metric_content.update({"bootstrap_contrasts":bootstrap.to_dict("records"),"multiple_comparison_control":multiple,
        "gap_bins":gap.to_dict("records"),"definitions":{"delta_anchor":"same DFT-selected GS structure on prediction and DFT axes; primary excludes anchor",
        "pairwise_eligibility":"|ΔE_DFT| > 5 meV/atom","prediction_resolved":"|ΔE_pred| > 1 meV/atom",
        "A":"metric per seed, mean seeds within formula; primary","B":"mean three predictions per structure before metric; secondary"}})
    atomic_json(EXPERIMENT/"metrics.json",metric_content)
    integrity.update({"status":"COMPLETE","prediction_sha256":sha256(pred_path),"prediction_rows":len(pred),
        "pair_rows":len(pairs),"composition_seed_rows":len(comp),"new_geometry_runs":45,
        "reused_slices_controls":45,"reused_frozen_geometry_controls":15,"historical_test_accessed":False})
    atomic_json(EXPERIMENT/"summary.json",integrity)
    resources=json.loads((EXPERIMENT/"resources.json").read_text());resources["analysis"]={
        "status":"COMPLETE","wall_seconds":time.perf_counter()-started,"cpu_seconds":time.process_time()-cpu,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform=="darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "bootstrap_replicates":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,"device":"cpu"}
    atomic_json(EXPERIMENT/"resources.json",resources)
    LOG.info("%s",{"event":"analysis_complete","prediction_rows":len(pred),"metrics":str(EXPERIMENT/"metrics.json")})


if __name__=="__main__":
    main()
