"""Auditoría OOF de suficiencia representacional para MatGPT/MatterGPT ABS3.

Reutiliza embeddings congelados, geometría, cohortes y particiones históricas.
Los probes se ajustan exclusivamente dentro de cada fold; no modifica backbones.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
EXP = ROOT / "experiments/abs3_representation_sufficiency_audit"
IMG = ROOT / "images/abs3_representation_sufficiency_audit"
REPORT = ROOT / "reports/ABS3/abs3_representation_sufficiency_audit"
DATA = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
FOLDS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
EMBED = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71/embeddings"
M2_SOURCE = ROOT / "experiments/mattergpt_abs3_frozen_ridge_mlp8/predicciones_oof_por_estructura.csv"
RIDGE_SOURCE = ROOT / "experiments/abs3_geometry_transformers_pairwise_ridge"
GEOM = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
FAMILY_ORDER = ["dist_perovskite", "hexagonal", "needle"]
ELEMENTS = ["Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_MULTIPLIERS = [0.25, 1.0, 4.0]
KRR_LAMBDAS = [0.001, 0.01, 0.1, 1.0]
BOOT_N = 10_000
BOOT_SEED = 20261103
LOG = logging.getLogger("abs3.representation_audit")
T0, C0 = time.monotonic(), time.process_time()


def sha256(path: Path) -> str:
    """Devuelve SHA-256 de un archivo leído por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    """Escribe JSON de forma atómica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    """Escribe CSV de forma atómica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix(path.suffix + ".tmp"), index=False)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any], dict[tuple[int, str], float]]:
    """Carga cohorte, particiones, embeddings y medias Ridge M2 históricas.

    Returns:
        Dataset ordenado, folds, embeddings MAT/MATTER, manifiesto y mu M2 por fold/fórmula.
    """
    d = pd.read_csv(DATA, dtype={"structure_id": str, "composition_id": str}, low_memory=False)
    f = pd.read_csv(FOLDS, dtype={"structure_id": str, "composition_group": str})
    f = f[f.historical_split.isin(["train", "val"])].copy()
    if len(d) != 210 or d.structure_id.nunique() != 210 or d.composition_id.nunique() != 70:
        raise ValueError("Cohorte ABS3 no coincide con 210 estructuras/70 composiciones")
    if d.groupby("composition_id").size().ne(3).any() or d.structure_id.duplicated().any():
        raise ValueError("Triplete incompleto o structure_id duplicado")
    if set(d.structure_id) != set(f.structure_id) or set(d.composition_id) != set(f.composition_group):
        raise ValueError("Dataset y manifiesto histórico de folds no coinciden por IDs")
    if d[GEOM].isna().any().any() or not np.isfinite(d[GEOM].to_numpy(float)).all():
        raise ValueError("Geometría con datos ausentes/no finitos")
    d = d.sort_values(["composition_id", "family"], key=lambda s: s.map({v:i for i,v in enumerate(FAMILY_ORDER)}) if s.name == "family" else s, kind="stable").reset_index(drop=True)
    mpath = EMBED / "embedding_manifest.json"
    manifest = json.loads(mpath.read_text())
    features: dict[str, np.ndarray] = {}
    for name, archive, key in [("MAT", "matgpt_original_final_end.npz", "matgpt"), ("MATTER", "mattergpt_original_final_end.npz", "mattergpt")]:
        rec = manifest[key]
        p = EMBED / archive
        if sha256(p) != rec["sha256"]:
            raise ValueError(f"Hash de embedding {name} no coincide con manifiesto")
        with np.load(p, allow_pickle=False) as z:
            ids, vals = z["structure_id"].astype(str), z["embeddings"].astype(np.float64)
        if vals.shape != (210, 768) or not np.isfinite(vals).all() or set(ids) != set(d.structure_id):
            raise ValueError(f"Caché inválida para {name}")
        ix = {sid:i for i,sid in enumerate(ids)}
        features[name] = vals[[ix[s] for s in d.structure_id]]
    m2 = pd.read_csv(M2_SOURCE, dtype={"structure_id": str, "composition_id": str})
    mu_col = next((c for c in ["mu_ridge_m2_ev_atom", "mu_ridge_ev_atom", "ridge_mu_ev_atom", "mu_composition_ev_atom", "ridge_mean_ev_atom"] if c in m2.columns), None)
    if mu_col is None:
        likely = [c for c in m2.columns if "mu" in c.lower() and ("ridge" in c.lower() or "m2" in c.lower())]
        if len(likely) != 1:
            raise ValueError(f"No se identifica columna mu Ridge M2; columnas={list(m2.columns)}")
        mu_col = likely[0]
    mu: dict[tuple[int,str],float] = {}
    formula_col = "formula" if "formula" in m2.columns else "composition_id"
    for (fold, formula), g in m2.groupby(["fold", formula_col]):
        vals = g[mu_col].astype(float).unique()
        if len(vals) > 1 and not np.allclose(vals, vals[0], rtol=0, atol=1e-12):
            raise ValueError(f"mu M2 no es constante para {formula}, fold {fold}")
        mu[(int(fold),str(formula))] = float(np.mean(vals))
    return d, f, features, manifest, mu


def fold_rows(folds: pd.DataFrame, fold: int, d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Retorna índices de estructuras para inner train/val/outer por IDs."""
    q = folds[folds.outer_fold.eq(fold)].drop_duplicates("structure_id").set_index("structure_id")
    roles = q.loc[d.structure_id].reset_index()
    return {name: np.flatnonzero(roles.inner_role.eq(role).to_numpy() if name != "outer" else roles.outer_role.eq("outer_validation").to_numpy())
            for name, role in [("train","inner_train"),("val","inner_validation"),("outer","outer_validation")]}


def pair_indices(d: pd.DataFrame, comp_ids: set[str]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Construye los tres pares orientados canónicos por composición."""
    ii, jj, yy = [], [], []
    for _, idx in d[d.composition_id.isin(comp_ids)].groupby("composition_id", sort=False).groups.items():
        ix = list(idx)
        if len(ix) != 3:
            raise ValueError("Se esperaba triplete de tres polimorfos")
        for a,b in [(0,1),(0,2),(1,2)]:
            ii.append(ix[a]); jj.append(ix[b]); yy.append(1000*(float(d.loc[ix[b],"energy_dft_ev_atom"])-float(d.loc[ix[a],"energy_dft_ev_atom"])))
    return np.asarray(ii,int),np.asarray(jj,int),np.asarray(yy,float)


def rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Kernel RBF numéricamente estable."""
    return np.exp(-gamma * cdist(a,b,"sqeuclidean"))


def pair_kernel(k: np.ndarray, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """Construye kernel inducido para diferencias de puntuación escalar."""
    return k[jj[:,None],ll[None,:]]-k[jj[:,None],kk[None,:]]-k[ii[:,None],ll[None,:]]+k[ii[:,None],kk[None,:]]


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """MAE de vectores pareados."""
    return float(np.mean(np.abs(y-p))) if len(y) else float("nan")


def metric_block(d: pd.DataFrame, idx: np.ndarray, pred_e: np.ndarray, pred_de: dict[tuple[int,int],float]) -> dict[str,Any]:
    """Métricas de energía y ranking a nivel de estructuras/composiciones outer."""
    y = d.loc[idx,"energy_dft_ev_atom"].to_numpy(float)*1000
    p = pred_e[idx]*1000
    rows=[]; by_formula=[]; correct=eligible=unresolved=0; gs_ok=gs_n=0; trip_ok=trip_n=0
    for formula,g in d.loc[idx].groupby("composition_id",sort=False):
        ids=list(g.index); e=g.energy_dft_ev_atom.to_numpy(float)*1000
        pe=np.asarray([pred_e[d.index.get_loc(s)]*1000 for s in ids])
        pairs=[]
        for a,b in [(0,1),(0,2),(1,2)]:
            de=e[b]-e[a]; dp=pred_de[(ids[a],ids[b])]
            pairs.append((de,dp)); rows.append(abs(de-dp))
            if abs(de)>5:
                eligible+=1; unresolved+=int(abs(dp)<=1); correct+=int(abs(dp)>1 and np.sign(de)==np.sign(dp))
        by_formula.append(float(np.mean([abs(a-b) for a,b in pairs])))
        order=np.argsort(e,kind="stable")
        if e[order[1]]-e[order[0]]>1:
            gs_n+=1; gs_ok+=int(np.argmin(pe)==order[0] and (np.sort(pe)[1]-np.sort(pe)[0])>1)
        if all(abs(z[0])>1 for z in pairs):
            trip_n+=1; trip_ok+=int(np.array_equal(np.argsort(e,kind="stable"),np.argsort(pe,kind="stable")))
    all_pairs=[]
    for (i,j),dp in pred_de.items():
        if i in set(idx) and j in set(idx): all_pairs.append((1000*(d.loc[j,"energy_dft_ev_atom"]-d.loc[i,"energy_dft_ev_atom"]),dp))
    yy=np.asarray([a for a,b in all_pairs]); pp=np.asarray([b for a,b in all_pairs])
    slope,inter=np.polyfit(yy,pp,1) if len(yy)>1 else (float("nan"),float("nan"))
    return {"structures":len(idx),"energy_mae_mev_atom":float(np.mean(abs(p-y))),
            "energy_rmse_global_mev_atom":float(np.sqrt(np.mean((p-y)**2))),"energy_bias_mev_atom":float(np.mean(p-y)),
            "deltaE_mae_estimand_A_mev_atom":float(np.mean(by_formula)),"deltaE_rmse_global_mev_atom":float(np.sqrt(np.mean((pp-yy)**2))),
            "pairwise_correct":correct,"pairwise_eligible":eligible,"pairwise_accuracy":correct/eligible if eligible else float("nan"),
            "unresolved_eligible":unresolved,"groundstate_correct":gs_ok,"groundstate_eligible":gs_n,"groundstate_accuracy":gs_ok/gs_n if gs_n else float("nan"),
            "full_triplet_correct":trip_ok,"full_triplet_eligible":trip_n,"full_triplet_accuracy":trip_ok/trip_n if trip_n else float("nan"),
            "ols_slope":float(slope),"ols_intercept_mev_atom":float(inter),"deltaE_predictive_r2":float(1-np.sum((pp-yy)**2)/np.sum((yy-yy.mean())**2))}


def fit_predict_probe(kind: str, xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray,
                      yval: np.ndarray, xouter: np.ndarray, youter: np.ndarray,
                      train_groups: np.ndarray, val_groups: np.ndarray,
                      structure_train: np.ndarray, structure_val: np.ndarray, structure_outer: np.ndarray,
                      pair_train: tuple[np.ndarray,np.ndarray], pair_val: tuple[np.ndarray,np.ndarray],
                      pair_outer: tuple[np.ndarray,np.ndarray]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    """Selecciona Ridge/PLS/KRR solo en validación y predice val/outer.

    Features son diferencias de estructuras; train_groups/val_groups retienen la unidad fórmula.
    """
    best=None
    if kind in ("RIDGE","PLS"):
        comp_grid=[None] if kind=="RIDGE" else [2,4,8,16]
        for ncomp in comp_grid:
            rot=None
            if ncomp:
                ncomp=min(int(ncomp),xtrain.shape[0],xtrain.shape[1])
                try:
                    pls=PLSRegression(n_components=ncomp,scale=False,max_iter=500,tol=1e-7)
                    pls.fit(xtrain,ytrain)
                    rot=pls.x_rotations_
                except Exception:
                    continue
            xt=xtrain if rot is None else xtrain@rot
            xv=xval if rot is None else xval@rot
            xo=xouter if rot is None else xouter@rot
            for alpha in ALPHAS:
                model=Ridge(alpha=alpha,fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000).fit(xt,ytrain)
                pv=model.predict(xv); score=mae(yval,pv)
                key=(score, -alpha, (ncomp or 0))
                if best is None or key<best[0]: best=(key,alpha,ncomp,rot,pv,xo,model)
        if best is None: raise ValueError("No fue posible ajustar Ridge/PLS")
        _,alpha,ncomp,rot,pv,xo,model=best
        po=model.predict(xo)
        # Volver a coeficientes en espacio de input permite una puntuación escalar por estructura.
        beta=model.coef_ if rot is None else rot@model.coef_
        return pv,po,{"alpha":float(alpha),"n_components":ncomp,"beta":np.asarray(beta).reshape(-1)}
    if kind=="KRR":
        med=np.median(pdist(structure_train,"sqeuclidean")); base=1/max(float(med),1e-12)
        for mult in GAMMA_MULTIPLIERS:
            gamma=base*mult
            kt=rbf(structure_train,structure_train,gamma)
            kv=rbf(structure_val,structure_train,gamma)
            ko=rbf(structure_outer,structure_train,gamma)
            it,jt=pair_train; iv,jv=pair_val; io,jo=pair_outer
            K=pair_kernel(kt,it,jt,it,jt)
            Kval=pair_kernel(kv,iv,jv,it,jt)
            Kout=pair_kernel(ko,io,jo,it,jt)
            for lam in KRR_LAMBDAS:
                alpha=np.linalg.solve(K+lam*np.eye(len(K)),ytrain)
                pv=Kval@alpha; score=mae(yval,pv)
                key=(score,-lam,-gamma)
                if best is None or key<best[0]: best=(key,gamma,lam,pv,Kout,alpha,structure_train,it,jt)
        if best is None: raise ValueError("No KRR válido")
        _,gamma,lam,pv,Kout,alpha,xref,it,jt=best
        po=Kout@alpha
        return pv,po,{"gamma":float(gamma),"lambda":float(lam),"dual_alpha":alpha,"x_reference":xref,"pair_i":it,"pair_j":jt}
    raise ValueError(f"Probe desconocido: {kind}")


def pair_indices_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Índices de pares secuenciales; cada tres filas deben representar un triplete."""
    if len(x)%3: raise ValueError("Se requieren tripletes completos para pairwise KRR")
    ii=[];jj=[]
    for c in range(len(x)//3):
        ii.extend([3*c,3*c,3*c+1]);jj.extend([3*c+1,3*c+2,3*c+2])
    return np.asarray(ii,int),np.asarray(jj,int),np.asarray(y,float)


def build_pairs_for_indices(d: pd.DataFrame, feat: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Construye X diferencial y etiquetas, ordenadas por fórmula y family canónica."""
    sub=d.loc[idx].copy().sort_values(["composition_id","family"],key=lambda s:s.map({v:i for i,v in enumerate(FAMILY_ORDER)}) if s.name=="family" else s,kind="stable")
    ids=sub.index.to_numpy()
    # Los índices devueltos por pair_indices son globales por el orden del dataframe, filtrado; crear lookup.
    lookup={old:i for i,old in enumerate(ids)}
    # Reconstruir pares localmente con el mismo orden de grupos del sub-dataframe.
    li=[];lj=[];ly=[]
    for _,g in sub.groupby("composition_id",sort=False):
        q=list(g.index)
        for a,b in [(0,1),(0,2),(1,2)]:
            li.append(lookup[q[a]]);lj.append(lookup[q[b]])
            ly.append(1000*(float(d.loc[q[b],"energy_dft_ev_atom"])-float(d.loc[q[a],"energy_dft_ev_atom"])))
    li=np.asarray(li,int);lj=np.asarray(lj,int);ly=np.asarray(ly,float)
    return feat[ids[lj]]-feat[ids[li]],ly,ids,li,lj


def structural_analysis(d: pd.DataFrame, feats: dict[str,np.ndarray]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Resume distancias de tripletes y varianza entre/dentro de composición."""
    rows=[];var=[]
    for name,h in feats.items():
        for formula,g in d.groupby("composition_id",sort=False):
            ix=g.index.to_list(); v=h[ix]
            for a,b in [(0,1),(0,2),(1,2)]:
                u,w=v[a],v[b]; norm=np.linalg.norm(u-w)
                cos=float(np.dot(u,w)/(np.linalg.norm(u)*np.linalg.norm(w)+1e-15))
                rows.append({"extractor":name,"formula":formula,"polymorph_i":g.iloc[a].family,"polymorph_j":g.iloc[b].family,
                    "cosine_similarity":cos,"l2":norm,"normalized_l2":norm/(np.linalg.norm(u)+np.linalg.norm(w)+1e-15),"near_identical":norm<1e-10})
        grand=h.mean(axis=0);means=np.stack([h[g.index].mean(axis=0) for _,g in d.groupby("composition_id",sort=False)])
        between=3*np.sum((means-grand)**2);total=np.sum((h-grand)**2);within=total-between
        var.append({"extractor":name,"between_composition_ss":float(between),"within_composition_ss":float(within),
                    "between_fraction":float(between/total),"within_fraction":float(within/total),
                    "within_to_between_ratio":float(within/max(between,1e-15)),"global_mean_norm":float(np.linalg.norm(grand)),
                    "mean_embedding_norm":float(np.mean(np.linalg.norm(h,axis=1)))})
    return pd.DataFrame(rows),pd.DataFrame(var)


def main() -> None:
    """Ejecuta probes OOF, controles y curvas con recursos CPU limitados."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    EXP.mkdir(parents=True,exist_ok=True); IMG.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    with threadpool_limits(limits=1):
        d,folds,features,emb_manifest,mu=load_inputs()
        dist,var=structural_analysis(d,features)
        atomic_csv(EXP/"structural_distances.csv",dist);atomic_csv(EXP/"intracomposition_variance.csv",var)
        # B1/B2/B3 main OOF. Keep probes and Transformers separate.
        predictions=[]; selections=[]; pair_outputs=[]; metric_outputs=[]
        for fold in range(5):
            ix=fold_rows(folds,fold,d); ids={k:set(d.iloc[v].composition_id.astype(str)) for k,v in ix.items()}
            # Geometry-only and embedding-only/hybrid feature families, train-only scalers.
            groups={k:np.flatnonzero(d.composition_id.isin(ids[k]).to_numpy()) for k in ids}
            groups["train"]=ix["train"];groups["val"]=ix["val"];groups["outer"]=ix["outer"]
            for extractor in ["GEO","MAT","MATTER"]:
                raw= d[GEOM].to_numpy(float) if extractor=="GEO" else features[extractor]
                scaler=StandardScaler().fit(raw[ix["train"]]); x=scaler.transform(raw)
                models={"Ridge":(x,"RIDGE"),"PLS_Ridge":(x,"PLS"),"KRR_RBF":(x,"KRR")}
                for model_name,(xx,kind) in models.items():
                    Xtr,ytr,ids_tr,ti,tj=build_pairs_for_indices(d,xx,ix["train"])
                    Xv,yv,ids_v,vi,vj=build_pairs_for_indices(d,xx,ix["val"])
                    Xo,yo,ids_o,oi,oj=build_pairs_for_indices(d,xx,ix["outer"])
                    pv,po,params=fit_predict_probe(kind,Xtr,ytr,Xv,yv,Xo,yo,
                        d.loc[ids_tr,"composition_id"].to_numpy(),d.loc[ids_v,"composition_id"].to_numpy(),
                        xx[ids_tr],xx[ids_v],xx[ids_o],(ti,tj),(vi,vj),(oi,oj))
                    selections.append({"fold":fold,"extractor":extractor,"model":model_name,**{k:v for k,v in params.items() if np.isscalar(v)}})
                    # Map pairwise predictions to structure scores; both pair models are conservative.
                    if kind in ("RIDGE","PLS"):
                        score=xx[ids_o]@params["beta"]
                    else:
                        # f(x) from the dual difference representation, using train pairs.
                        kt=rbf(params["x_reference"],params["x_reference"],params["gamma"])
                        alpha=params["dual_alpha"]; pi=params["pair_i"];pj=params["pair_j"]
                        ko=rbf(xx[ids_o],params["x_reference"],params["gamma"])
                        score=(ko[:,pj]-ko[:,pi])@alpha
                    score=score-np.array([score[np.where(d.loc[ids_o,"composition_id"].to_numpy()==f)[0]].mean() for f in d.loc[ids_o,"composition_id"].to_numpy()])
                    pred_e=np.asarray([mu[(fold,str(d.loc[s,"composition_id"]))]+score[j]/1000 for j,s in enumerate(ids_o)])
                    pred_de={(int(ids_o[oi[p]]),int(ids_o[oj[p]])):float(po[p]) for p in range(len(po))}
                    full_pred_e=np.full(len(d),np.nan);full_pred_e[ids_o]=pred_e
                    met=metric_block(d,ids_o,full_pred_e,pred_de)
                    metric_outputs.append({"condition":extractor,"probe":model_name,"fold":fold,**met})
                    for p,(a,b) in enumerate(zip(oi,oj)):
                        pair_outputs.append({"condition":extractor,"probe":model_name,"fold":fold,
                            "formula":d.loc[ids_o[a],"composition_id"],"structure_i":d.loc[ids_o[a],"structure_id"],
                            "structure_j":d.loc[ids_o[b],"structure_id"],"delta_dft_mev_atom":float(yo[p]),
                            "delta_pred_mev_atom":float(po[p]),"absolute_error_mev_atom":float(abs(yo[p]-po[p]))})
                    for j,s in enumerate(ids_o):
                        predictions.append({"condition":extractor,"probe":model_name,"fold":fold,"formula":d.loc[s,"composition_id"],
                            "structure_id":d.loc[s,"structure_id"],"polymorph":d.loc[s,"family"],"energy_dft_ev_atom":float(d.loc[s,"energy_dft_ev_atom"]),
                            "energy_pred_ev_atom":float(pred_e[j]),"score_centered_mev_atom":float(score[j])})
                    LOG.info("fold %d %s/%s outer ΔE MAE %.2f",fold,extractor,model_name,met["deltaE_mae_estimand_A_mev_atom"])
        pred=pd.DataFrame(predictions);atomic_csv(EXP/"predicciones_oof_por_estructura.csv",pred)
        atomic_csv(EXP/"predicciones_oof_por_par.csv",pd.DataFrame(pair_outputs))
        atomic_csv(EXP/"energy_probe_results.csv",pd.DataFrame(selections))
        atomic_csv(EXP/"metrics_per_fold.csv",pd.DataFrame(metric_outputs))
        # Note: further probe/curve/control stages are added in the report only when their artifacts exist.
        preflight={"status":"PASS","dataset_sha256":sha256(DATA),"fold_manifest_sha256":sha256(FOLDS),
            "embedding_manifest_sha256":sha256(EMBED/"embedding_manifest.json"),"embedding_shapes":{k:list(v.shape) for k,v in features.items()},
            "dataset":{"structures":len(d),"compositions":d.composition_id.nunique(),"triplets_complete":bool(d.groupby("composition_id").size().eq(3).all())},
            "geometry_source":"gga-static_endpoint_cleaned","geometry_features":GEOM,"folds":5,"partition_sizes":[45,11,14],
            "test_accessed":False,"backbones_modified":False}
        atomic_json(EXP/"preflight.json",preflight)
        atomic_json(EXP/"embedding_manifest.json",emb_manifest)
        atomic_json(EXP/"checkpoint_manifest.json",{"MAT":{"sha256":emb_manifest["matgpt"]["checkpoint_sha256"],"status":"verified from historical embedding manifest"},
            "MATTER":{"sha256":emb_manifest["mattergpt"]["checkpoint_sha256"],"property_vector":[1.0,2.0],"status":"verified from historical embedding manifest"}})
        atomic_json(EXP/"config.yaml",{"dataset":str(DATA.relative_to(ROOT)),"folds":str(FOLDS.relative_to(ROOT)),"extractors":"original frozen FINAL_END 768d",
            "geometry":GEOM,"pair_sign":"E_j-E_i, meV/atom","pairwise_thresholds":{"eligible_gt":5,"unresolved_le":1},
            "regressors":{"ridge_alpha":ALPHAS,"pls_components":[2,4,8,16],"krr_gamma_multipliers":GAMMA_MULTIPLIERS,"krr_lambda":KRR_LAMBDAS},
            "test_accessed":False,"outer_used_for_selection":False})
        # Structural distance, diversity, Energy probing summaries.
        result=[]
        for (condition,probe,fold),g in pred.groupby(["condition","probe","fold"]):
            idx=np.flatnonzero(d.outer_fold.eq(fold).to_numpy()) if "outer_fold" in d else None
        summary={"status":"PARTIAL_EVIDENCE","structural_distance_rows":len(dist),"oof_prediction_rows":len(pred),
                 "completed_stages":["preflight","structural_distances","energy_pairwise_ridge_pls_krr"],
                 "pending_stages":["geometry_decoding","few_shot_learning_curves","100_permutation_controls","conditional_additive_kernel_value","bootstrap_and_full_report"],
                 "historical_test_accessed":False}
        atomic_json(EXP/"summary.json",summary)
        resources={"status":"PARTIAL_EVIDENCE","wall_seconds":time.monotonic()-T0,"process_cpu_seconds":time.process_time()-C0,
            "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"device":"cpu","cpu_threads":1,
            "cuda":False,"mps":False,"test_accessed":False,"python_version":platform.python_version(),"numpy_version":np.__version__,"pandas_version":pd.__version__}
        atomic_json(EXP/"resources.json",resources)
        LOG.info("Auditoría parcial ejecutada: %s",json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
