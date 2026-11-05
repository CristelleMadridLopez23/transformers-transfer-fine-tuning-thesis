"""Evalúa probes congelados por capa en ABS₃ con selección interna sin leakage.

La selección FINAL/BEST se hace fold a fold usando únicamente inner-validation;
outer se consulta una vez para predecir y formar métricas OOF. Incluye decodificación
geométrica, kernel aditivo, controles de permutación y figuras científicas.
"""
from __future__ import annotations

import hashlib
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/abs3_transformer_layerwise_audit"
IMG = ROOT / "images/abs3_transformer_layerwise_audit"
REPORT = ROOT / "reports/ABS3/abs3_transformer_layerwise_audit"
DATA = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
FOLDS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
OLD_AUDIT = ROOT / "experiments/abs3_representation_sufficiency_audit"
PREVIOUS = ROOT / "experiments/abs3_geometry_transformers_pairwise_ridge"
EMBED = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71/embeddings"
M2_SOURCE = ROOT / "experiments/mattergpt_abs3_frozen_ridge_mlp8/predicciones_oof_por_estructura.csv"
GEOM = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KRR_GAMMA_MULTIPLIERS = [0.25, 1.0, 4.0]
KRR_LAMBDAS = [0.001, 0.01, 0.1, 1.0]
ETA_GRID = [0.0, 0.1, 1.0, 10.0]
PAIR_SEED = 20261103
BOOT_N = 10000
BOOT_SEED = 20261103
LOG = logging.getLogger("abs3.layerwise.audit")


def atomic_json(path: Path, obj: Any) -> None:
    """Escribe JSON atómico con validación de valores finitos.

    Args:
        path: Destino.
        obj: Objeto serializable.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Escribe CSV mediante archivo temporal.

    Args:
        path: Destino.
        frame: Tabla de salida.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256(path: Path) -> str:
    """Devuelve SHA-256 de un archivo.

    Args:
        path: Archivo.
    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layer_cache(model: str, expected_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Carga y valida caché layerwise por IDs, dimensiones y valores finitos.

    Args:
        model: Identificador MAT o MATTER.
        expected_ids: IDs canónicos del dataset.
    Returns:
        Diccionario de capa a matriz [210,768].
    """
    path = EXP / ("embeddings_matgpt_layers.npz" if model == "MAT" else "embeddings_mattergpt_layers.npz")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["structure_id"].astype(str)
        if not np.array_equal(ids, expected_ids):
            raise ValueError(f"IDs de caché {model} no coinciden con dataset")
        layers = {key: archive[key].astype(np.float64) for key in archive.files if key != "structure_id"}
    if not layers or any(x.shape != (210, 768) or not np.isfinite(x).all() for x in layers.values()):
        raise ValueError(f"Caché layerwise {model} con shape/valores inválidos")
    return layers


def load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], float]]:
    """Carga cohort, folds y μ Ridge M2 histórico verificado.

    Args:
        None.
    Returns:
        Dataset, asignaciones históricas y media composicional por fold.
    """
    data = pd.read_csv(DATA, dtype={"structure_id": str, "composition_id": str}, low_memory=False)
    folds = pd.read_csv(FOLDS, dtype={"structure_id": str, "composition_group": str})
    folds = folds[folds.historical_split.isin(["train", "val"])].copy()
    data = data.sort_values(["composition_id", "family"], kind="stable").reset_index(drop=True)
    if len(data) != 210 or data.composition_id.nunique() != 70 or data.groupby("composition_id").size().ne(3).any():
        raise ValueError("Cohorte ABS₃ no contiene 70 tripletes completos")
    if set(data.structure_id) != set(folds.structure_id):
        raise ValueError("Folds históricos y dataset no coinciden por structure_id")
    ridge = pd.read_csv(M2_SOURCE, dtype={"structure_id": str, "formula": str})
    mu_col = next((c for c in ["mu_ridge_m2_ev_atom", "mu_ridge_ev_atom", "ridge_mu_ev_atom", "mu_composition_ev_atom", "ridge_mean_ev_atom"] if c in ridge), None)
    if mu_col is None:
        raise ValueError("No se identifica μ del Ridge M2 histórico")
    formula_col = "formula" if "formula" in ridge else "composition_id"
    means: dict[tuple[int, str], float] = {}
    for (fold, formula), group in ridge.groupby(["fold", formula_col]):
        values = group[mu_col].astype(float).unique()
        if len(values) > 1 and not np.allclose(values, values[0], rtol=0, atol=1e-12):
            raise ValueError(f"Ridge M2 no comparte μ en triplete {formula}")
        means[int(fold), str(formula)] = float(np.mean(values))
    return data, folds, means


def fold_roles(folds: pd.DataFrame, fold: int, data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Obtiene train, validation y outer por asignación estable de estructura.

    Args:
        folds: Manifiesto histórico.
        fold: Fold externo.
        data: Cohorte canónica.
    Returns:
        Índices de fila para las tres particiones.
    """
    part = folds[folds.outer_fold.eq(fold)].drop_duplicates("structure_id").set_index("structure_id").loc[data.structure_id]
    roles = {"train": "inner_train", "val": "inner_validation", "outer": "outer_validation"}
    return {key: np.flatnonzero((part.outer_role.eq(value) if key == "outer" else part.inner_role.eq(value)).to_numpy()) for key, value in roles.items()}


def local_pairs(data: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Crea los tres pares canónicos con targets en meV/átomo.

    Args:
        data: Dataset completo.
        indices: Índices de estructuras de la partición.
    Returns:
        Índices locales i/j, target ΔE y fórmulas en orden de pares.
    """
    subset = data.loc[indices].sort_values(["composition_id", "family"], kind="stable")
    old_ids = subset.index.to_list(); loc = {old: i for i, old in enumerate(old_ids)}
    ii=[]; jj=[]; yy=[]; formulas=[]
    for formula, group in subset.groupby("composition_id", sort=False):
        ids=list(group.index)
        if len(ids)!=3: raise ValueError(f"Triplete incompleto: {formula}")
        for a,b in [(0,1),(0,2),(1,2)]:
            ii.append(loc[ids[a]]);jj.append(loc[ids[b]])
            yy.append(1000*(float(data.loc[ids[b],"energy_dft_ev_atom"])-float(data.loc[ids[a],"energy_dft_ev_atom"])))
            formulas.append(str(formula))
    return np.asarray(ii,int),np.asarray(jj,int),np.asarray(yy,float),formulas


def pair_rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """RBF por estructuras.

    Args:
        a: Matriz izquierda.
        b: Matriz derecha.
        gamma: Parámetro RBF positivo.
    Returns:
        Matriz kernel densa.
    """
    return np.exp(-gamma*cdist(a,b,"sqeuclidean"))


def induced_pair_kernel(k: np.ndarray, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """Kernel inducido para diferencia de scores, antisimétrico y consistente.

    Args:
        k: Kernel entre estructuras.
        ii/jj: Índices de pares de filas.
        kk/ll: Índices de pares de columnas.
    Returns:
        Kernel pairwise.
    """
    return k[jj[:,None],ll[None,:]]-k[jj[:,None],kk[None,:]]-k[ii[:,None],ll[None,:]]+k[ii[:,None],kk[None,:]]


def pair_mae(y: np.ndarray, p: np.ndarray, formulas: list[str]) -> float:
    """Calcula estimando A: tres errores por fórmula y macro-promedio.

    Args:
        y: ΔE DFT.
        p: ΔE predicha.
        formulas: Fórmula asociada con cada par.
    Returns:
        MAE macro por composición, meV/átomo.
    """
    frame=pd.DataFrame({"formula":formulas,"error":np.abs(y-p)})
    return float(frame.groupby("formula").error.mean().mean())


def rank_stats(y: np.ndarray, p: np.ndarray, formulas: list[str]) -> dict[str, float]:
    """Calcula exactitud pairwise y full-triplet respetando agrupación.

    Args:
        y: Diferencias DFT.
        p: Diferencias predichas.
        formulas: Fórmula por par.
    Returns:
        Conteos y métricas de ranking.
    """
    eligible=np.abs(y)>5;resolved=np.abs(p)>1
    correct=eligible&resolved&(np.sign(y)==np.sign(p))
    triplets=[];ground=[]
    for formula in dict.fromkeys(formulas):
        ix=np.flatnonzero(np.asarray(formulas)==formula)
        if len(ix)!=3:continue
        d=y[ix];q=p[ix]
        triplets.append((bool(np.all(np.abs(d)>1)),bool(np.array_equal(np.argsort([0,d[0],d[1]]),np.argsort([0,q[0],q[1]]))) if False else False))
    return {"pairwise_correct":int(correct.sum()),"pairwise_eligible":int(eligible.sum()),
        "pairwise_accuracy":float(correct.sum()/eligible.sum()) if eligible.any() else float("nan"),
        "pairwise_resolved_accuracy":float(correct[eligible&resolved].sum()/max(int((eligible&resolved).sum()),1)),
        "unresolved_eligible":int((eligible&~resolved).sum())}


def metrics_for_predictions(data: pd.DataFrame, indices: np.ndarray, scores: np.ndarray,
                            means: dict[tuple[int,str],float], fold: int) -> tuple[dict[str,Any], pd.DataFrame, pd.DataFrame]:
    """Reconstruye energía, pares y métricas a partir de scores estructurales.

    Args:
        data: Cohorte completa.
        indices: Estructuras del outer fold.
        scores: Puntuación meV por estructura.
        means: μ del Ridge M2 por fold/fórmula.
        fold: Pliegue evaluado.
    Returns:
        Métricas agregadas, tabla estructura y tabla de pares.
    """
    sub=data.loc[indices].copy().sort_values(["composition_id","family"],kind="stable")
    score_by={int(i):float(v) for i,v in zip(sub.index,scores)}
    sub["score_mev_atom"]=[score_by[int(i)] for i in sub.index]
    sub["score_centered_mev_atom"]=sub.groupby("composition_id").score_mev_atom.transform(lambda x:x-x.mean())
    sub["mu_ridge_m2_ev_atom"]=[means[(fold,str(f))] for f in sub.composition_id]
    sub["energy_pred_ev_atom"]=sub.mu_ridge_m2_ev_atom+sub.score_centered_mev_atom/1000
    sub["energy_error_mev_atom"]=(sub.energy_pred_ev_atom-sub.energy_dft_ev_atom)*1000
    pair_rows=[];composition_rows=[];correct=eligible=unresolved=gs_ok=gs_n=trip_ok=trip_n=0
    for formula,g in sub.groupby("composition_id",sort=False):
        g=g.sort_values("family",kind="stable");y=g.energy_dft_ev_atom.to_numpy(float)*1000;p=g.energy_pred_ev_atom.to_numpy(float)*1000
        local=[]
        for a,b in [(0,1),(0,2),(1,2)]:
            dy=float(y[b]-y[a]);dp=float(p[b]-p[a]);err=abs(dp-dy);local.append(err)
            ok=abs(dy)>5;res=abs(dp)>1;eligible+=int(ok);unresolved+=int(ok and not res);correct+=int(ok and res and np.sign(dy)==np.sign(dp))
            pair_rows.append({"model":"PENDING","condition":"PENDING","fold":fold,"formula":str(formula),
                "structure_i":str(g.iloc[a].structure_id),"structure_j":str(g.iloc[b].structure_id),
                "polymorph_i":str(g.iloc[a].family),"polymorph_j":str(g.iloc[b].family),
                "delta_dft_mev_atom":dy,"delta_pred_mev_atom":dp,"absolute_error_mev_atom":err,
                "eligible_pair":ok,"prediction_resolved":res,"pair_correct":bool(ok and res and np.sign(dy)==np.sign(dp))})
        order_y=np.argsort(y,kind="stable");order_p=np.argsort(p,kind="stable")
        gs_elig=bool(y[order_y[1]]-y[order_y[0]]>1);gs_n+=int(gs_elig)
        pred_no_tie=bool(np.sort(p)[1]-np.sort(p)[0]>1)
        gs_good=bool(gs_elig and order_y[0]==order_p[0] and pred_no_tie);gs_ok+=int(gs_good)
        trip_elig=bool(all(abs(q)>1 for q in [y[1]-y[0],y[2]-y[0],y[2]-y[1]]));trip_n+=int(trip_elig)
        trip_good=bool(trip_elig and np.array_equal(order_y,order_p));trip_ok+=int(trip_good)
        composition_rows.append({"model":"PENDING","condition":"PENDING","fold":fold,"formula":str(formula),
            "energy_mae_mev_atom":float(np.mean(np.abs(sub.loc[g.index,"energy_error_mev_atom"]))),
            "deltaE_mae_estimand_A_mev_atom":float(np.mean(local)),"pairwise_correct":sum(int(r["pair_correct"]) for r in pair_rows[-3:]),
            "pairwise_eligible":sum(int(r["eligible_pair"]) for r in pair_rows[-3:]),"groundstate_correct":int(gs_good),
            "groundstate_eligible":int(gs_elig),"full_triplet_correct":int(trip_good),"full_triplet_eligible":int(trip_elig)})
    pair_frame=pd.DataFrame(pair_rows);comp_frame=pd.DataFrame(composition_rows)
    y=pair_frame.delta_dft_mev_atom.to_numpy(float);p=pair_frame.delta_pred_mev_atom.to_numpy(float)
    slope,inter=np.polyfit(y,p,1);ols_r2=1-np.sum((p-(slope*y+inter))**2)/np.sum((y-y.mean())**2)
    summary={"structures":len(sub),"compositions":sub.composition_id.nunique(),"pairs":len(pair_frame),
        "energy_mae_mev_atom":float(comp_frame.energy_mae_mev_atom.mean()),
        "energy_rmse_global_mev_atom":float(np.sqrt(np.mean(sub.energy_error_mev_atom.to_numpy()**2))),
        "deltaE_mae_estimand_A_mev_atom":float(comp_frame.deltaE_mae_estimand_A_mev_atom.mean()),
        "deltaE_rmse_global_mev_atom":float(np.sqrt(np.mean((p-y)**2)),),
        "pairwise_correct":correct,"pairwise_eligible":eligible,"pairwise_accuracy":correct/eligible if eligible else np.nan,
        "pairwise_resolved_accuracy":correct/max(eligible-unresolved,1),"unresolved_eligible":unresolved,
        "groundstate_correct":gs_ok,"groundstate_eligible":gs_n,"groundstate_accuracy":gs_ok/max(gs_n,1),
        "full_triplet_correct":trip_ok,"full_triplet_eligible":trip_n,"full_triplet_accuracy":trip_ok/max(trip_n,1),
        "ols_slope":float(slope),"ols_intercept_mev_atom":float(inter),"ols_r2":float(ols_r2),
        "predictive_r2":float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
        "spearman":float(spearmanr(y,p).statistic),"pearson":float(pearsonr(y,p).statistic)}
    return summary,sub.reset_index(drop=True),pair_frame


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              formulas_val: list[str]) -> tuple[float, float, float]:
    """Selecciona alpha Ridge solo con MAE y ranking de inner-validation.

    Args:
        x_train/y_train: Pares inner-train.
        x_val/y_val: Pares inner-validation.
        formulas_val: Fórmula de cada par validación.
    Returns:
        Alpha, MAE validación y exactitud pairwise.
    """
    choices=[]
    for alpha in RIDGE_ALPHAS:
        model=Ridge(alpha=alpha,fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000).fit(x_train,y_train)
        pred=model.predict(x_val);stats=rank_stats(y_val,pred,formulas_val)
        choices.append((pair_mae(y_val,pred,formulas_val),-stats["pairwise_accuracy"],-alpha,alpha,stats["pairwise_accuracy"]))
    best=min(choices)
    return float(best[3]),float(best[0]),float(best[4])


def fit_krr(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            formulas_val: list[str], train_pairs: tuple[np.ndarray,np.ndarray],
            val_pairs: tuple[np.ndarray,np.ndarray], gamma_multipliers: list[float] | None = None,
            lambdas: list[float] | None = None) -> tuple[float,float,float,float]:
    """Selecciona gamma/lambda de kernel pairwise en inner-validation.

    Args:
        x_train/y_train: Estructuras/targets de train.
        x_val/y_val: Estructuras/targets de validation.
        formulas_val: Fórmula por par validation.
        train_pairs/val_pairs: Índices de pares locales.
        gamma_multipliers/lambdas: Grillas predefinidas.
    Returns:
        Gamma, lambda, MAE validation y accuracy validation.
    """
    mults=gamma_multipliers or KRR_GAMMA_MULTIPLIERS; regs=lambdas or KRR_LAMBDAS
    base=1/max(float(np.median(pdist(x_train,"sqeuclidean"))),1e-12);ti,tj=train_pairs;vi,vj=val_pairs
    choices=[]
    for mult in mults:
        gamma=base*mult;kt=pair_rbf(x_train,x_train,gamma);kv=pair_rbf(x_val,x_train,gamma)
        K=induced_pair_kernel(kt,ti,tj,ti,tj);Kv=induced_pair_kernel(kv,vi,vj,ti,tj)
        for lam in regs:
            dual=np.linalg.solve(K+lam*np.eye(len(K)),y_train);pred=Kv@dual;stats=rank_stats(y_val,pred,formulas_val)
            choices.append((pair_mae(y_val,pred,formulas_val),-stats["pairwise_accuracy"],-lam,gamma,lam,stats["pairwise_accuracy"]))
    best=min(choices)
    return float(best[3]),float(best[4]),float(best[0]),float(best[5])


def fit_outer_scores(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_outer: np.ndarray,
                     params: dict[str,float], train_pairs: tuple[np.ndarray,np.ndarray],
                     outer_pairs: tuple[np.ndarray,np.ndarray]) -> tuple[np.ndarray,np.ndarray]:
    """Ajusta probe seleccionado en train y predice scores/ΔE outer.

    Args:
        kind: Ridge o KRR.
        x_train/x_outer: Embeddings ya escalados.
        y_train: Targets de train.
        params: Hiperparámetros elegidos en validation.
        train_pairs/outer_pairs: Índices locales.
    Returns:
        Scores por estructura outer y predicciones por par outer.
    """
    ti,tj=train_pairs;oi,oj=outer_pairs
    if kind=="Ridge":
        model=Ridge(alpha=params["alpha"],fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000).fit(x_train[tj]-x_train[ti],y_train)
        scores=x_outer@model.coef_.reshape(-1);pair_pred=scores[oj]-scores[oi]
    else:
        gamma=params["gamma"];lam=params["lambda"];K=pair_rbf(x_train,x_train,gamma)
        Kp=induced_pair_kernel(K,ti,tj,ti,tj);dual=np.linalg.solve(Kp+lam*np.eye(len(Kp)),y_train)
        Ko=pair_rbf(x_outer,x_train,gamma);scores=(Ko[:,tj]-Ko[:,ti])@dual
        pair_pred=scores[oj]-scores[oi]
    return scores,np.asarray(pair_pred,float)


def fold_comparison(data: pd.DataFrame, fold_rows_map: dict[str,np.ndarray], feat: np.ndarray,
                    means: dict[tuple[int,str],float], fold: int, kind: str) -> tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    """Evalúa una matriz de features capa a capa y selecciona BEST en validación.

    Args:
        data: Cohorte ABS₃.
        fold_rows_map: Índices por partición.
        feat: Embedding de una capa.
        means: Ridge M2 composicional.
        fold: Fold externo.
        kind: Ridge o KRR.
    Returns:
        Filas internas, OOF por capa, y resumen de selección/predicciones.
    """
    tr,va,ot=(fold_rows_map[k] for k in ["train","val","outer"])
    scaler=StandardScaler().fit(feat[tr]);z=scaler.transform(feat)
    ti,tj,yt,_=local_pairs(data,tr);vi,vj,yv,fv=local_pairs(data,va);oi,oj,yo,fo=local_pairs(data,ot)
    candidates=[];outer=[]
    layer_names=list(feat_layers_current)
    for order,layer in enumerate(layer_names):
        # The caller supplies one layer at a time; this name is attached externally.
        x=z
        if kind=="Ridge":
            alpha,val_mae,val_acc=fit_ridge(x[tr[tj]-tr[tj]] if False else x[tr][tj]-x[tr][ti],yt,
                x[va][vj]-x[va][vi],yv,fv)
            params={"alpha":alpha};regularizer=alpha
            val_pred=Ridge(alpha=alpha,fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000).fit(x[tr][tj]-x[tr][ti],yt).predict(x[va][vj]-x[va][vi])
        else:
            gamma,lam,val_mae,val_acc=fit_krr(x[tr],yt,x[va],yv,fv,(ti,tj),(vi,vj))
            params={"gamma":gamma,"lambda":lam};regularizer=lam
            kt=pair_rbf(x[tr],x[tr],gamma);kv=pair_rbf(x[va],x[tr],gamma)
            dual=np.linalg.solve(induced_pair_kernel(kt,ti,tj,ti,tj)+lam*np.eye(len(yt)),yt)
            val_pred=induced_pair_kernel(kv,vi,vj,ti,tj)@dual
        candidates.append({"layer":layer,"layer_order":order,"kind":kind,"fold":fold,"val_mae":val_mae,
            "val_pairwise_accuracy":val_acc,"regularizer":regularizer,**params})
        scores,pairpred=fit_outer_scores(kind,x[tr],yt,x[ot],params,(ti,tj),(oi,oj))
        outer.append({"layer":layer,"scores":scores,"pair_pred":pairpred,"y":yo,"pairs":(oi,oj),"pair_formulas":fo,
            "indices":ot,"params":params,"val_pred":val_pred})
    # Candidate order is the explicit insertion order; tie favors larger regularization then shallower layer.
    selected=min(candidates,key=lambda r:(r["val_mae"],-r["val_pairwise_accuracy"],-r["regularizer"],r["layer_order"]))
    final_candidate=next(r for r in candidates if r["layer"]=="FINAL_POST_LN_F")
    best_outer=next(x for x in outer if x["layer"]==selected["layer"])
    final_outer=next(x for x in outer if x["layer"]=="FINAL_POST_LN_F")
    return candidates,outer,{"selected":selected,"best_outer":best_outer,"final_outer":final_outer,"final_candidate":final_candidate}


# External layer name used only to annotate a one-layer fold call; never affects selection.
feat_layers_current: list[str] = []


def make_pair_rows(kind: str, model: str, result: dict[str,Any], data: pd.DataFrame, fold: int,
                   role: str, layer: str) -> tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any],pd.DataFrame,pd.DataFrame]:
    """Serializa OOF de una capa y métricas de fold.

    Args:
        kind/model: Probe y extractor.
        result: Elemento outer de la capa.
        data: Cohorte.
        fold: Fold externo.
        role: ALL_LAYER_DIAGNOSTIC, FINAL o BEST_LAYER.
        layer: ID de capa.
    Returns:
        Filas por estructura/par y métricas agregadas/composición.
    """
    stats,structures,pairs=metrics_for_predictions(data,result["indices"],result["scores"],CURRENT_MU,fold)
    structures["model"]=model;structures["probe"]=kind;structures["layer"]=layer;structures["evaluation_role"]=role;structures["fold"]=fold
    pairs["model"]=model;pairs["probe"]=kind;pairs["layer"]=layer;pairs["evaluation_role"]=role
    stats.update({"model":model,"probe":kind,"layer":layer,"evaluation_role":role,"fold":fold})
    return structures.to_dict("records"),pairs.to_dict("records"),stats,structures,pairs


CURRENT_MU: dict[tuple[int,str],float] = {}


def structural_tables(data: pd.DataFrame, layers: dict[str,dict[str,np.ndarray]]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Calcula separabilidad intrafórmula y descomposición de varianza por capa.

    Args:
        data: Cohorte.
        layers: Modelo→capa→embedding.
    Returns:
        Tabla de distancias/correlaciones y varianza within/between.
    """
    distance_rows=[];variance_rows=[]
    for model,states in layers.items():
        for name,h in states.items():
            residual=h-h.reshape(70,3,768).mean(axis=1).repeat(3,axis=0)
            total=np.sum((h-h.mean(axis=0))**2);within=np.sum(residual**2);between=max(total-within,0)
            variance_rows.append({"model":model,"layer":name,"within_ss_fraction":within/total,
                "between_ss_fraction":between/total,"within_between_ratio":within/max(between,1e-12),
                "exploratory_all_development":True})
            pair_distance=[];pair_gap=[]
            for formula,group in data.groupby("composition_id",sort=False):
                idx=group.index.to_list()
                for a,b in [(0,1),(0,2),(1,2)]:
                    va=h[idx[a]];vb=h[idx[b]];norm=np.linalg.norm(va)+np.linalg.norm(vb)
                    cos=float(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-12));dist=float(np.linalg.norm(va-vb))
                    normalized=dist/max(norm,1e-12);gap=abs(float(data.loc[idx[b],"energy_dft_ev_atom"]-data.loc[idx[a],"energy_dft_ev_atom"]))*1000
                    pair_distance.append(normalized);pair_gap.append(gap)
                    distance_rows.append({"model":model,"layer":name,"formula":str(formula),"structure_i":str(data.loc[idx[a],"structure_id"]),
                        "structure_j":str(data.loc[idx[b],"structure_id"]),"cosine_similarity":cos,"l2_distance":dist,
                        "normalized_l2":normalized,"norm_i":float(np.linalg.norm(va)),"norm_j":float(np.linalg.norm(vb)),
                        "absolute_delta_dft_mev_atom":gap,"near_identical":bool(normalized<1e-6),"exploratory_all_development":True})
            sp=spearmanr(pair_distance,pair_gap).statistic;pe=pearsonr(pair_distance,pair_gap).statistic
            variance_rows[-1].update({"spearman_distance_abs_gap":float(sp),"pearson_distance_abs_gap":float(pe),
                "distance_gap_note":"descriptivo exploratorio con toda la cohorte de desarrollo; no generalización OOF"})
    return pd.DataFrame(distance_rows),pd.DataFrame(variance_rows)


def decode_geometry(data: pd.DataFrame, folds: pd.DataFrame, layers: dict[str,dict[str,np.ndarray]]) -> pd.DataFrame:
    """Prueba Ridge por capa para recuperar siete descriptores geométricos.

    Args:
        data: Cohorte.
        folds: Splits históricos.
        layers: Embeddings por extractor/capa.
    Returns:
        Métricas OOF por fold/capa/descriptor.
    """
    output=[];target=data[GEOM].to_numpy(float)
    for fold in range(5):
        roles=fold_roles(folds,fold,data);tr,va,ot=(roles[k] for k in ["train","val","outer"])
        yscaler=StandardScaler().fit(target[tr]);y=yscaler.transform(target)
        for model,states in layers.items():
            for layer,h0 in states.items():
                xscaler=StandardScaler().fit(h0[tr]);x=xscaler.transform(h0);scores=[]
                for alpha in RIDGE_ALPHAS:
                    estimator=Ridge(alpha=alpha,fit_intercept=True,solver="lsqr",tol=1e-10,max_iter=10000).fit(x[tr],y[tr])
                    scores.append((float(np.mean((estimator.predict(x[va])-y[va])**2)),alpha))
                alpha=min(scores)[1];estimator=Ridge(alpha=alpha,fit_intercept=True,solver="lsqr",tol=1e-10,max_iter=10000).fit(x[tr],y[tr])
                pred=yscaler.inverse_transform(estimator.predict(x[ot]))
                for j,descriptor in enumerate(GEOM):
                    err=pred[:,j]-target[ot,j];base=np.sum((target[ot,j]-target[tr,j].mean())**2)
                    output.append({"fold":fold,"model":model,"layer":layer,"descriptor":descriptor,"alpha":alpha,
                        "mae":float(np.mean(abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),
                        "mae_normalized_train_sd":float(np.mean(abs(err))/max(float(target[tr,j].std()),1e-12)),
                        "predictive_r2":float(1-np.sum(err**2)/base),"n_outer_structures":len(ot),"scaler_fit":"inner_train_only"})
    return pd.DataFrame(output)


def pairwise_krr_candidates(xtr: np.ndarray, ytr: np.ndarray, xval: np.ndarray, yval: np.ndarray,
                            fval: list[str], ti: np.ndarray, tj: np.ndarray, vi: np.ndarray, vj: np.ndarray,
                            multipliers: list[float] | None = None, lambdas: list[float] | None = None) -> list[dict[str,Any]]:
    """Devuelve todos los candidatos KRR y sus scores de validación.

    Args:
        xtr/ytr, xval/yval: Features/targets de estructura y pares.
        fval: Fórmula por par validation.
        ti/tj/vi/vj: Índices locales.
        multipliers/lambdas: Grillas.
    Returns:
        Candidatos con métricas internas y parámetros.
    """
    base=1/max(float(np.median(pdist(xtr,"sqeuclidean"))),1e-12);rows=[]
    for mult in multipliers or KRR_GAMMA_MULTIPLIERS:
        gamma=base*mult;K=induced_pair_kernel(pair_rbf(xtr,xtr,gamma),ti,tj,ti,tj)
        Kv=induced_pair_kernel(pair_rbf(xval,xtr,gamma),vi,vj,ti,tj)
        for lam in lambdas or KRR_LAMBDAS:
            dual=np.linalg.solve(K+lam*np.eye(len(K)),ytr);pred=Kv@dual;acc=rank_stats(yval,pred,fval)["pairwise_accuracy"]
            rows.append({"gamma":gamma,"gamma_multiplier":mult,"lambda":lam,"val_mae":pair_mae(yval,pred,fval),"val_accuracy":acc,"val_pred":pred})
    return rows


def fit_layer_all(data: pd.DataFrame, roles: dict[str,np.ndarray], feat: np.ndarray, fold: int,
                  model_name: str, layer_name: str, mu: dict[tuple[int,str],float]) -> tuple[dict[str,Any],dict[str,Any],list[dict[str,Any]]]:
    """Evalúa una capa con Ridge y KRR, seleccionando hiperparámetros internos.

    Args:
        data/roles/feat: Cohorte, fold partitions y embedding de capa.
        fold/model_name/layer_name: Identificadores.
        mu: Ridge M2 fijo.
    Returns:
        Predicciones de ambos probes, métricas inner y tabla de candidatos.
    """
    tr,va,ot=(roles[k] for k in ["train","val","outer"]);scaler=StandardScaler().fit(feat[tr]);z=scaler.transform(feat)
    ti,tj,yt,_=local_pairs(data,tr);vi,vj,yv,fv=local_pairs(data,va);oi,oj,yo,fo=local_pairs(data,ot)
    val_rows=[];outs={}
    # Ridge candidate table, alpha chosen by macro validation ΔE MAE.
    ridge_candidates=[]
    for alpha in RIDGE_ALPHAS:
        est=Ridge(alpha=alpha,fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000).fit(z[tr][tj]-z[tr][ti],yt)
        pred=est.predict(z[va][vj]-z[va][vi]);acc=rank_stats(yv,pred,fv)["pairwise_accuracy"]
        ridge_candidates.append({"alpha":alpha,"val_mae":pair_mae(yv,pred,fv),"val_accuracy":acc,"val_pred":pred})
    br=min(ridge_candidates,key=lambda c:(c["val_mae"],-c["val_accuracy"],-c["alpha"]))
    params={"alpha":float(br["alpha"])}
    scores,pred=fit_outer_scores("Ridge",z[tr],yt,z[ot],params,(ti,tj),(oi,oj))
    outs["Ridge"]={"scores":scores,"pred":pred,"indices":ot,"y":yo,"formulas":fo,"params":params}
    val_rows.append({"fold":fold,"model":model_name,"layer":layer_name,"probe":"Ridge","val_mae":br["val_mae"],"val_pairwise_accuracy":br["val_accuracy"],**params})
    for c in ridge_candidates:val_rows.append({"fold":fold,"model":model_name,"layer":layer_name,"probe":"Ridge_CANDIDATE","val_mae":c["val_mae"],"val_pairwise_accuracy":c["val_accuracy"],"alpha":c["alpha"]})
    # KRR candidates; tie-break regularization first, then gamma multiplier.
    kc=pairwise_krr_candidates(z[tr],yt,z[va],yv,fv,ti,tj,vi,vj)
    bk=min(kc,key=lambda c:(c["val_mae"],-c["val_accuracy"],-c["lambda"],c["gamma_multiplier"]))
    params={"gamma":float(bk["gamma"]),"lambda":float(bk["lambda"])}
    scores,pred=fit_outer_scores("KRR",z[tr],yt,z[ot],params,(ti,tj),(oi,oj))
    outs["KRR"]={"scores":scores,"pred":pred,"indices":ot,"y":yo,"formulas":fo,"params":params}
    val_rows.append({"fold":fold,"model":model_name,"layer":layer_name,"probe":"KRR","val_mae":bk["val_mae"],"val_pairwise_accuracy":bk["val_accuracy"],**params})
    for c in kc:val_rows.append({"fold":fold,"model":model_name,"layer":layer_name,"probe":"KRR_CANDIDATE","val_mae":c["val_mae"],"val_pairwise_accuracy":c["val_accuracy"],"gamma":c["gamma"],"gamma_multiplier":c["gamma_multiplier"],"lambda":c["lambda"]})
    return outs, {"train_pairs":(ti,tj),"train_y":yt,"val_pairs":(vi,vj),"val_y":yv,"val_formulas":fv,"outer_pairs":(oi,oj)},val_rows


def evaluate_all_layers(data: pd.DataFrame, folds: pd.DataFrame, layer_cache: dict[str,dict[str,np.ndarray]],
                        mu: dict[tuple[int,str],float]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Ejecuta selección interna y OOF para cada capa y probe.

    Args:
        data/folds: Cohorte y asignaciones.
        layer_cache: Matriz por extractor y capa.
        mu: μ M2 fijo por fold/fórmula.
    Returns:
        Predicciones de estructuras, pares, métricas por fold/composición,
        selección de capa y métricas de validación.
    """
    global CURRENT_MU
    CURRENT_MU=mu
    structure_rows=[];pair_rows=[];fold_metrics=[];comp_metrics=[];selections=[];validation_rows=[]
    for fold in range(5):
        roles=fold_roles(folds,fold,data)
        for model,states in layer_cache.items():
            layer_results: dict[str,dict[str,Any]]={"Ridge":{},"KRR":{}}
            for layer,feat in states.items():
                outs,meta,valrows=fit_layer_all(data,roles,feat,fold,model,layer,mu);validation_rows.extend(valrows)
                layer_results["Ridge"][layer]=outs["Ridge"];layer_results["KRR"][layer]=outs["KRR"]
            for probe in ["Ridge","KRR"]:
                # Per-layer best parameters are selected only in inner validation; selection criterion frozen.
                val_sub=[r for r in validation_rows if r.get("fold")==fold and r.get("model")==model and r.get("probe")==probe]
                chosen={}
                for layer in states:
                    rows=[r for r in val_sub if r.get("layer")==layer]
                    primary=next(r for r in rows if r.get("probe")==probe)
                    reg=float(primary.get("alpha",primary.get("lambda",0)))
                    chosen[layer]=(float(primary["val_mae"]),-float(primary["val_pairwise_accuracy"]),-reg,list(states).index(layer),primary)
                best_layer=min(chosen,key=lambda l:chosen[l][:4])
                for layer in states:
                    result=layer_results[probe][layer]
                    stats,structures,pairs=metrics_for_predictions(data,result["indices"],result["scores"],mu,fold)
                    role="ALL_LAYER_DIAGNOSTIC"
                    for frame,rows in [(structures,structure_rows),(pairs,pair_rows)]:
                        frame["model"]=model;frame["probe"]=probe;frame["layer"]=layer;frame["evaluation_role"]=role;frame["fold"]=fold
                        rows.extend(frame.to_dict("records"))
                    stats.update({"model":model,"probe":probe,"layer":layer,"evaluation_role":role,"fold":fold})
                    fold_metrics.append(stats)
                    comp_metrics.extend(composition_metrics(data,structures,pairs,model,probe,layer,role,fold))
                # Explicit FINAL and BEST aliases point to the same already selected outer prediction.
                for role,layer in [("FINAL","FINAL_POST_LN_F"),("BEST_LAYER",best_layer)]:
                    result=layer_results[probe][layer]
                    stats,structures,pairs=metrics_for_predictions(data,result["indices"],result["scores"],mu,fold)
                    for frame,rows in [(structures,structure_rows),(pairs,pair_rows)]:
                        frame["model"]=model;frame["probe"]=probe;frame["layer"]=role;frame["source_layer"]=layer;frame["evaluation_role"]=role;frame["fold"]=fold
                        rows.extend(frame.to_dict("records"))
                    stats.update({"model":model,"probe":probe,"layer":role,"source_layer":layer,"evaluation_role":role,"fold":fold})
                    fold_metrics.append(stats);comp_metrics.extend(composition_metrics(data,structures,pairs,model,probe,role,role,fold))
                    if role=="BEST_LAYER":
                        selections.append({"fold":fold,"model":model,"probe":probe,"selected_layer":layer,
                            "inner_validation_deltaE_mae_mev_atom":chosen[layer][0],"inner_validation_pairwise_accuracy":-chosen[layer][1],
                            "selected_alpha":chosen[layer][4].get("alpha"),"selected_gamma":chosen[layer][4].get("gamma"),
                            "selected_lambda":chosen[layer][4].get("lambda"),"tie_break":"higher validation pair accuracy; greater regularization; shallower layer"})
        LOG.info("Capa OOF: fold=%d",fold)
    return (pd.DataFrame(structure_rows),pd.DataFrame(pair_rows),pd.DataFrame(fold_metrics),pd.DataFrame(comp_metrics),
            pd.DataFrame(selections),pd.DataFrame(validation_rows))


def composition_metrics(data: pd.DataFrame, structures: pd.DataFrame, pairs: pd.DataFrame,
                       model: str, probe: str, layer: str, role: str, fold: int) -> list[dict[str,Any]]:
    """Calcula métricas por fórmula para bootstrap apareado.

    Args:
        data: Cohorte.
        structures/pairs: Predicciones outer.
        model/probe/layer/role/fold: Identificadores.
    Returns:
        Filas por fórmula con errores y conteos.
    """
    out=[]
    for formula,g in structures.groupby("composition_id",sort=False):
        pp=pairs[pairs.formula.astype(str).eq(str(formula))]
        energies=g.energy_error_mev_atom.to_numpy(float)
        out.append({"model":model,"probe":probe,"layer":layer,"evaluation_role":role,"fold":fold,"formula":str(formula),
            "energy_mae_mev_atom":float(np.mean(np.abs(energies))),"deltaE_mae_estimand_A_mev_atom":float(pp.absolute_error_mev_atom.mean()),
            "pairwise_correct":int(pp.pair_correct.sum()),"pairwise_eligible":int(pp.eligible_pair.sum()),
            "groundstate_correct":int(ground_triplet(g)[0]),"groundstate_eligible":int(ground_triplet(g)[1]),
            "full_triplet_correct":int(full_triplet(g)[0]),"full_triplet_eligible":int(full_triplet(g)[1])})
    return out


def ground_triplet(g: pd.DataFrame) -> tuple[int,int]:
    """Determina acierto ground-state con gaps DFT/predicho >1 meV/átomo.

    Args:
        g: Tres estructuras de una fórmula.
    Returns:
        Correcto y elegible como enteros.
    """
    y=g.energy_dft_ev_atom.to_numpy(float)*1000;p=g.energy_pred_ev_atom.to_numpy(float)*1000
    order=np.argsort(y,kind="stable");eligible=int(y[order[1]]-y[order[0]]>1)
    correct=int(eligible and np.argmin(y)==np.argmin(p) and np.sort(p)[1]-np.sort(p)[0]>1)
    return correct,eligible


def full_triplet(g: pd.DataFrame) -> tuple[int,int]:
    """Determina exactitud de orden completo con gap DFT elegible.

    Args:
        g: Tres estructuras de una fórmula.
    Returns:
        Correcto y elegible.
    """
    y=g.energy_dft_ev_atom.to_numpy(float)*1000;p=g.energy_pred_ev_atom.to_numpy(float)*1000
    elegible=int(all(abs(y[b]-y[a])>1 for a,b in [(0,1),(0,2),(1,2)]))
    return int(elegible and np.array_equal(np.argsort(y),np.argsort(p))),elegible


def aggregate_metrics(fold_metrics: pd.DataFrame, role: str = "ALL_LAYER_DIAGNOSTIC") -> pd.DataFrame:
    """Agrega OOF por capa con macro MAE y exactitudes pooled.

    Args:
        fold_metrics: Métricas fold a fold.
        role: Tipo de comparación que se agrega.
    Returns:
        Métricas OOF por modelo/probe/capa.
    """
    rows=[];sub=fold_metrics[fold_metrics.evaluation_role.eq(role)]
    for keys,g in sub.groupby(["model","probe","layer"],sort=False):
        y=g;den=lambda key:int(y[key].sum());num=lambda key:int(y[key].sum())
        rows.append({"model":keys[0],"probe":keys[1],"layer":keys[2],"folds":g.fold.nunique(),
            "structures":int(g.structures.sum()),"compositions":int(g.compositions.sum()),"pairs":int(g.pairs.sum()),
            "energy_mae_mev_atom":float(np.average(g.energy_mae_mev_atom,weights=g.compositions)),
            "energy_rmse_global_mev_atom":float(np.sqrt(np.sum(g.energy_rmse_global_mev_atom**2*g.structures)/g.structures.sum())),
            "deltaE_mae_estimand_A_mev_atom":float(np.average(g.deltaE_mae_estimand_A_mev_atom,weights=g.compositions)),
            "deltaE_rmse_global_mev_atom":float(np.sqrt(np.sum(g.deltaE_rmse_global_mev_atom**2*g.pairs)/g.pairs.sum())),
            "pairwise_correct":den("pairwise_correct"),"pairwise_eligible":den("pairwise_eligible"),
            "pairwise_accuracy":den("pairwise_correct")/max(den("pairwise_eligible"),1),
            "unresolved_eligible":den("unresolved_eligible"),"groundstate_correct":den("groundstate_correct"),
            "groundstate_eligible":den("groundstate_eligible"),"groundstate_accuracy":den("groundstate_correct")/max(den("groundstate_eligible"),1),
            "full_triplet_correct":den("full_triplet_correct"),"full_triplet_eligible":den("full_triplet_eligible"),
            "full_triplet_accuracy":den("full_triplet_correct")/max(den("full_triplet_eligible"),1),
            "ols_slope":float(np.average(g.ols_slope,weights=g.pairs)),"predictive_r2":float(np.average(g.predictive_r2,weights=g.pairs))})
    return pd.DataFrame(rows)


def refine_global_metrics(summary: pd.DataFrame, structures: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Recalcula RMSE y OLS globales directamente sobre las 210 estructuras/210 pares.

    Args:
        summary: Agregado inicialmente desde métricas por fold.
        structures: Predicciones OOF estructurales.
        pairs: Predicciones OOF pareadas.
    Returns:
        Agregado con RMSE/OLS calculados sobre la población OOF combinada.
    """
    out=summary.copy()
    for idx,row in out.iterrows():
        s=structures[(structures.model.eq(row.model))&(structures.probe.eq(row.probe))&
            (structures.layer.eq(row.layer))&(structures.evaluation_role.eq("ALL_LAYER_DIAGNOSTIC"))]
        p=pairs[(pairs.model.eq(row.model))&(pairs.probe.eq(row.probe))&
            (pairs.layer.eq(row.layer))&(pairs.evaluation_role.eq("ALL_LAYER_DIAGNOSTIC"))]
        if len(s):out.loc[idx,"energy_rmse_global_mev_atom"]=float(np.sqrt(np.mean(s.energy_error_mev_atom.to_numpy(float)**2)))
        if len(p)>1:
            x=p.delta_dft_mev_atom.to_numpy(float);y=p.delta_pred_mev_atom.to_numpy(float);slope,inter=np.polyfit(x,y,1)
            out.loc[idx,"deltaE_rmse_global_mev_atom"]=float(np.sqrt(np.mean((y-x)**2)))
            out.loc[idx,"ols_slope"]=float(slope);out.loc[idx,"ols_intercept_mev_atom"]=float(inter)
            out.loc[idx,"ols_r2"]=float(1-np.sum((y-(slope*x+inter))**2)/np.sum((y-y.mean())**2))
            out.loc[idx,"predictive_r2"]=float(1-np.sum((y-x)**2)/np.sum((x-x.mean())**2))
    return out


def bootstrap_comparisons(comp: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap apareado por composición de FINAL contra BEST y GEO híbrido.

    Args:
        comp: Tabla por fórmula para roles FINAL y BEST_LAYER.
    Returns:
        Contrastes con intervalos percentiles IC95%.
    """
    rng=np.random.default_rng(BOOT_SEED);rows=[];metrics=["energy_mae_mev_atom","deltaE_mae_estimand_A_mev_atom"]
    for model in ["MAT","MATTER"]:
        for probe in ["Ridge","KRR"]:
            q=comp[(comp.model==model)&(comp.probe==probe)&comp.layer.isin(["FINAL","BEST_LAYER"])]
            wide=q.pivot(index="formula",columns="layer",values="deltaE_mae_estimand_A_mev_atom").dropna()
            if len(wide)!=70:continue
            dif=wide.BEST_LAYER.to_numpy()-wide.FINAL.to_numpy();draws=np.mean(dif[rng.integers(0,len(dif),(BOOT_N,len(dif)))],axis=1)
            rows.append({"contrast":f"{model} BEST_LAYER − FINAL {probe}","metric":"deltaE_mae_estimand_A_mev_atom","estimate":float(dif.mean()),
                "ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
            for metric in ["pairwise_accuracy","groundstate_accuracy","full_triplet_accuracy"]:
                stem={"pairwise_accuracy":"pairwise","groundstate_accuracy":"groundstate","full_triplet_accuracy":"full_triplet"}[metric]
                nums=[];dens=[]
                for layer in ["BEST_LAYER","FINAL"]:
                    f=comp[(comp.model==model)&(comp.probe==probe)&(comp.layer==layer)].sort_values("formula")
                    nums.append(f[stem+"_correct"].to_numpy(float));dens.append(f[stem+"_eligible"].to_numpy(float))
                indexes=rng.integers(0,70,(BOOT_N,70));deltas=[]
                for a in range(BOOT_N):
                    b=float(nums[0][indexes[a]].sum()/max(dens[0][indexes[a]].sum(),1));c=float(nums[1][indexes[a]].sum()/max(dens[1][indexes[a]].sum(),1));deltas.append(b-c)
                estimate=float(nums[0].sum()/max(dens[0].sum(),1)-nums[1].sum()/max(dens[1].sum(),1))
                rows.append({"contrast":f"{model} BEST_LAYER − FINAL {probe}","metric":metric,"estimate":estimate,
                    "ci95_low":float(np.quantile(deltas,.025)),"ci95_high":float(np.quantile(deltas,.975)),"replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
    return pd.DataFrame(rows)


def bootstrap_conditional(pair_predictions: pd.DataFrame, structure_predictions: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap valor incremental de kernels híbridos frente a GEO KRR.

    Args:
        pair_predictions: ΔE OOF por par y condición híbrida.
        structure_predictions: Energías OOF reconstruidas por estructura.
    Returns:
        Contrastes apareados por composición para errores y ranking.
    """
    formula_rows=[]
    for (model,condition,formula),pg in pair_predictions.groupby(["model","condition","formula"],sort=False):
        sg=structure_predictions[(structure_predictions.model.eq(model))&(structure_predictions.condition.eq(condition))&structure_predictions.formula.eq(formula)]
        y=pg.delta_dft_mev_atom.to_numpy(float);p=pg.delta_pred_mev_atom.to_numpy(float)
        eligible=np.abs(y)>5;resolved=np.abs(p)>1;correct=eligible&resolved&(np.sign(y)==np.sign(p))
        energies=sg.sort_values("family").energy_dft_ev_atom.to_numpy(float)*1000
        pred_energy=sg.sort_values("family").energy_pred_ev_atom.to_numpy(float)*1000
        order=np.argsort(energies,kind="stable");gs_eligible=int(energies[order[1]]-energies[order[0]]>1)
        gs_correct=int(gs_eligible and np.argmin(energies)==np.argmin(pred_energy) and np.sort(pred_energy)[1]-np.sort(pred_energy)[0]>1)
        trip_eligible=int(all(abs(energies[b]-energies[a])>1 for a,b in [(0,1),(0,2),(1,2)]))
        trip_correct=int(trip_eligible and np.array_equal(np.argsort(energies),np.argsort(pred_energy)))
        formula_rows.append({"model":model,"condition":condition,"formula":formula,
            "energy_mae_mev_atom":float(np.mean(np.abs(sg.energy_error_mev_atom))),
            "deltaE_mae_estimand_A_mev_atom":float(np.mean(np.abs(y-p))),
            "pairwise_correct":int(correct.sum()),"pairwise_eligible":int(eligible.sum()),
            "groundstate_correct":gs_correct,"groundstate_eligible":gs_eligible,
            "full_triplet_correct":trip_correct,"full_triplet_eligible":trip_eligible})
    comp=pd.DataFrame(formula_rows);rng=np.random.default_rng(BOOT_SEED);rows=[]
    metrics=["energy_mae_mev_atom","deltaE_mae_estimand_A_mev_atom","pairwise_accuracy","groundstate_accuracy","full_triplet_accuracy"]
    for model in ["MAT","MATTER"]:
        for condition in ["GEO_FINAL","GEO_BEST"]:
            left=comp[(comp.model.eq(model))&(comp.condition.eq(condition))].set_index("formula").sort_index()
            right=comp[(comp.model.eq("GEO"))&(comp.condition.eq("GEO_KRR"))].set_index("formula").sort_index()
            common=left.index.intersection(right.index)
            if len(common)!=70:continue
            ix=rng.integers(0,70,(BOOT_N,70))
            for metric in metrics:
                if metric.endswith("accuracy"):
                    stem={"pairwise_accuracy":"pairwise","groundstate_accuracy":"groundstate","full_triplet_accuracy":"full_triplet"}[metric]
                    ln=left.loc[common,stem+"_correct"].to_numpy(float);ld=left.loc[common,stem+"_eligible"].to_numpy(float)
                    rn=right.loc[common,stem+"_correct"].to_numpy(float);rd=right.loc[common,stem+"_eligible"].to_numpy(float)
                    draws=ln[ix].sum(axis=1)/np.maximum(ld[ix].sum(axis=1),1)-rn[ix].sum(axis=1)/np.maximum(rd[ix].sum(axis=1),1)
                    estimate=float(ln.sum()/max(ld.sum(),1)-rn.sum()/max(rd.sum(),1))
                else:
                    delta=left.loc[common,metric].to_numpy(float)-right.loc[common,metric].to_numpy(float)
                    draws=delta[ix].mean(axis=1);estimate=float(delta.mean())
                rows.append({"contrast":f"{model} {condition} − GEO GEO_KRR","metric":metric,"estimate":estimate,
                    "ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),
                    "replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
    return pd.DataFrame(rows)


def make_figures(separability: pd.DataFrame, variance: pd.DataFrame, geometry: pd.DataFrame,
                 inner: pd.DataFrame, selection: pd.DataFrame, aggregate: pd.DataFrame,
                 pairs: pd.DataFrame, conditional: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    """Genera nueve figuras por capa con títulos, unidades y notas de población."""
    IMG.mkdir(parents=True,exist_ok=True);colors={"MAT":"#2878B5","MATTER":"#D28BC2","GEO":"#7A8B5A"}
    palette={"MatGPT":colors["MAT"],"MatterGPT":colors["MATTER"]}
    layer_order={}
    for model,g in variance.groupby("model",sort=False):layer_order[model]={name:i for i,name in enumerate(g.layer)}
    # 1: separability
    fig,ax=plt.subplots(figsize=(10,5.5))
    for model in ["MAT","MATTER"]:
        q=variance[variance.model.eq(model)];x=np.arange(len(q));ax.plot(x,q.within_between_ratio,marker="o",color=colors[model],label=f"{model} — L2/(entre)")
    ax.set_xticks(range(len(next(iter(layer_order.values())))),list(next(iter(layer_order.values()))),rotation=40,ha="right")
    ax.set_ylabel("Razón de suma de cuadrados dentro / entre composición");ax.set_title("ABS₃: separabilidad intracomposicional por capa")
    ax.grid(alpha=.25);ax.legend();fig.text(.01,.01,"Cada valor resume 210 estructuras (70 fórmulas × 3 polimorfos); análisis descriptivo de todas las representaciones de desarrollo, no métrica OOF de energía.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(IMG/"layerwise_structural_separability.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 2: within/between
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,model in zip(axs,["MAT","MATTER"]):
        q=variance[variance.model.eq(model)];x=np.arange(len(q));ax.plot(x,q.within_ss_fraction,marker="o",label="Dentro de composición",color=colors[model]);ax.plot(x,q.between_ss_fraction,marker="s",ls="--",label="Entre composiciones",color="#777777")
        ax.set_xticks(x,q.layer,rotation=45,ha="right");ax.set_ylim(0,1);ax.set_title(model);ax.set_ylabel("Fracción de suma de cuadrados total");ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("ABS₃: variación de embeddings dentro y entre composiciones")
    fig.text(.01,.01,"Mismas 210 estructuras en cada capa; variación intracomposición se calcula tras restar la media de cada fórmula.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(IMG/"layerwise_within_between_variance.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 3: geometry decoding
    g=geometry.groupby(["model","layer"],as_index=False).agg(mae=("mae_normalized_train_sd","mean"),r2=("predictive_r2","mean"))
    fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for model in ["MAT","MATTER"]:
        q=g[g.model.eq(model)];x=[layer_order[model].get(s,0) for s in q.layer]
        axs[0].plot(x,q.mae,marker="o",color=colors[model],label=model);axs[1].plot(x,q.r2,marker="o",color=colors[model],label=model)
    names=list(next(iter(layer_order.values())));axs[0].set_ylabel("MAE promedio / SD train (7 descriptores)");axs[1].set_ylabel("R² predictivo medio");axs[1].set_xlabel("Estado contextualizado (la capa final incluye ln_f)")
    axs[1].set_xticks(range(len(names)),names,rotation=40,ha="right");axs[0].legend();axs[1].legend();axs[0].grid(alpha=.2);axs[1].grid(alpha=.2);fig.suptitle("ABS₃: decodificación geométrica OOF por capa")
    fig.text(.01,.01,"Ridge; 5 folds; 70 fórmulas/210 estructuras OOF. Scalers de inputs y targets ajustados solo en las 45 fórmulas inner-train. R² > 0 indica mejora sobre la media de train.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(IMG/"layerwise_geometry_decoding.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 4: inner validation
    fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,model in zip(axs,["MAT","MATTER"]):
        for probe,color,style in [("Ridge",colors[model],"-"),("KRR", "#444444","--")]:
            q=inner[(inner.model.eq(model))&(inner.probe.eq(probe))].groupby("layer").val_mae.mean()
            names=list(layer_order[model]);ax.plot(range(len(names)),[q.get(n,np.nan) for n in names],marker="o",ls=style,color=color,label=probe)
        ax.set_title(model);ax.set_xticks(range(len(layer_order[model])),list(layer_order[model]),rotation=45,ha="right");ax.set_ylabel("MAE ΔE inner-validation (meV/átomo)");ax.grid(alpha=.2);ax.legend()
    fig.suptitle("ABS₃: selección de probe/capa usando solo validación interna")
    fig.text(.01,.01,"Cada curva promedia cinco validaciones de 11 fórmulas (33 estructuras); no muestra outer. Los hiperparámetros y la capa se seleccionan exclusivamente aquí; selección final se evalúa una vez en outer.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(IMG/"inner_validation_deltaE_by_layer.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 5 selections
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,(model,probe) in zip(axs.flat,[("MAT","Ridge"),("MAT","KRR"),("MATTER","Ridge"),("MATTER","KRR")]):
        q=selection[(selection.model.eq(model))&(selection.probe.eq(probe))];names=list(layer_order[model]);mapn={n:i for i,n in enumerate(names)}
        ax.scatter(q.fold,[mapn[s] for s in q.selected_layer],color=colors[model],s=45);ax.set_yticks(range(len(names)),names,fontsize=7);ax.set_xticks(range(5),[f"F{i}" for i in range(5)]);ax.set_title(f"{model} — {probe}");ax.grid(alpha=.2)
    fig.suptitle("ABS₃: capa seleccionada por inner-validation en cada fold")
    fig.text(.01,.01,"Cada marcador = capa seleccionada con 45 fórmulas inner-train y 11 validation. Outer (14 fórmulas/fold) no participó. La selección puede variar entre folds.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(IMG/"selected_layer_by_fold.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 6 final vs best: 4 panels
    compare=aggregate[aggregate.layer.isin(["FINAL","BEST_LAYER"])]
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(metric,title) in zip(axs.flat,[("energy_mae_mev_atom","MAE energía absoluta"),("deltaE_mae_estimand_A_mev_atom","MAE ΔE estimando A"),("pairwise_accuracy","Exactitud pairwise elegible"),("groundstate_accuracy","Exactitud de estado fundamental")]):
        xlabels=[];values=[];cols=[]
        for model in ["MAT","MATTER"]:
            for probe in ["Ridge","KRR"]:
                for layer in ["FINAL","BEST_LAYER"]:
                    row=compare[(compare.model.eq(model))&(compare.probe.eq(probe))&(compare.layer.eq(layer))]
                    if row.empty:continue
                    xlabels.append(f"{model}\n{probe}\n{layer}");values.append(float(row.iloc[0][metric]));cols.append(colors[model])
        bars=ax.bar(range(len(values)),values,color=cols);ax.set_xticks(range(len(values)),xlabels,rotation=35,ha="right",fontsize=7);ax.set_title(title);ax.grid(axis="y",alpha=.2)
        if "accuracy" in metric:ax.set_ylim(0,1);ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))
        else:ax.set_ylabel("meV/átomo")
    fig.suptitle("ABS₃: FINAL histórico frente a BEST LAYER elegido internamente")
    fig.text(.01,.01,"OOF de desarrollo; 70 fórmulas, 210 estructuras y 5 folds. BEST se seleccionó por inner-validation, no por estos resultados. Accuracy usa denominadores elegibles agrupados.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(IMG/"final_vs_best_layer.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 7 energy gap parity
    chosen=pairs[pairs.layer.isin(["FINAL","BEST_LAYER"])].copy();fig,axs=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
    for ax,(model,probe) in zip(axs.flat,[("MAT","Ridge"),("MAT","KRR"),("MATTER","Ridge"),("MATTER","KRR")]):
        for layer,ls,color in [("FINAL","--","#555555"),("BEST_LAYER","-",colors[model])]:
            q=chosen[(chosen.model.eq(model))&(chosen.probe.eq(probe))&(chosen.layer.eq(layer))];x=q.delta_dft_mev_atom.to_numpy();y=q.delta_pred_mev_atom.to_numpy()
            ax.scatter(x,y,s=9,alpha=.15,color=color,label=f"{layer} puntos")
            slope,inter=np.polyfit(x,y,1);grid=np.array([x.min(),x.max()]);ax.plot(grid,slope*grid+inter,ls=ls,color=color,label=f"{layer} OLS m={slope:.2f}")
        ax.set_title(f"{model} — {probe}");ax.grid(alpha=.2);ax.legend(fontsize=6)
    lim=max(chosen.delta_dft_mev_atom.abs().max(),chosen.delta_pred_mev_atom.abs().max());
    for ax in axs.flat:ax.plot([-lim,lim],[-lim,lim],"k:",lw=1);ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect("equal")
    for ax in axs[-1,:]:ax.set_xlabel("ΔE DFT firmada Eⱼ−Eᵢ (meV/átomo)")
    for ax in axs[:,0]:ax.set_ylabel("ΔE predicha firmada Eⱼ−Eᵢ (meV/átomo)")
    fig.suptitle("ABS₃: calibración ΔE de FINAL frente a BEST LAYER")
    fig.text(.5,.015,"Cada punto = par intracomposición; 210 pares OOF por condición. Líneas OLS calculadas con todos los pares mostrados en cada panel/capa; selección de capa solo interna.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(IMG/"final_vs_best_gap_compression.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 8 conditional incremental value
    c=conditional[conditional.model.isin(["GEO","MAT","MATTER"])];fig,ax=plt.subplots(figsize=(9,5.5))
    labels=[];values=[];colors_list=[]
    for model in ["GEO","MAT","MATTER"]:
        q=c[c.model.eq(model)]
        for condition in ["GEO_KRR","GEO_FINAL","GEO_BEST"]:
            row=q[q.condition.eq(condition)]
            if not row.empty:labels.append(f"{model}\n{condition}");values.append(float(row.iloc[0].deltaE_mae_estimand_A_mev_atom));colors_list.append(colors["GEO"] if condition=="GEO_KRR" else colors[model])
    bars=ax.bar(range(len(values)),values,color=colors_list);ax.bar_label(bars,fmt="%.1f");ax.set_xticks(range(len(values)),labels,rotation=25,ha="right");ax.set_ylabel("MAE ΔE OOF (meV/átomo)");ax.set_title("ABS₃: valor incremental de embeddings sobre kernel geométrico")
    ax.text(.01,-.25,"η, gamma y lambda se eligieron en inner-validation; barras OOF por 70 fórmulas. Celdas relajadas por DFT. IC apareados se reportan en bootstrap_paired.csv.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(IMG/"conditional_value_by_layer.png",dpi=300,bbox_inches="tight");plt.close(fig)
    # 9 bootstrap
    q=bootstrap[bootstrap.metric.eq("deltaE_mae_estimand_A_mev_atom")];fig,ax=plt.subplots(figsize=(9,5.5));y=np.arange(len(q))
    ax.errorbar(q.estimate,y,xerr=[q.estimate-q.ci95_low,q.ci95_high-q.estimate],fmt="o",color="#2878B5",capsize=3);ax.axvline(0,color="black",ls="--");ax.set_yticks(y,q.contrast);ax.set_xlabel("BEST LAYER − FINAL: ΔE MAE (meV/átomo)");ax.set_title("ABS₃: bootstrap apareado por composición")
    ax.text(.01,-.18,"10 000 réplicas, semilla fija 20261103; se promedian los tres pares dentro de fórmula y se remuestrean 70 fórmulas completas. IC95% percentil, desarrollo OOF; no test externo.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(IMG/"bootstrap_best_vs_final.png",dpi=300,bbox_inches="tight");plt.close(fig)


def main() -> None:
    """Ejecuta probing, análisis, salidas y recursos reproducibles."""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--permutations",type=int,default=100)
    parser.add_argument("--skip-permutations",action="store_true");args=parser.parse_args()
    if args.permutations<0:parser.error("--permutations no puede ser negativo")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    EXP.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/"scripts/ABS3/abs3_representation_sufficiency_audit"))
    from run_audit import load_inputs
    start=time.perf_counter();cpu=time.process_time()
    with threadpool_limits(limits=1):
        data,folds,terminal_features,old_embedding_manifest,means=load_inputs()
        cache={m:load_layer_cache(m,data.structure_id.to_numpy(str)) for m in ["MAT","MATTER"]}
        sep,var=structural_tables(data,cache);atomic_csv(EXP/"layer_separability.csv",sep);atomic_csv(EXP/"within_between_variance.csv",var)
        geometry=decode_geometry(data,folds,cache);atomic_csv(EXP/"layer_geometry_probe.csv",geometry)
        structures,pairs,fold_metrics,comp_metrics,selections,valrows=evaluate_all_layers(data,folds,cache,means)
        atomic_csv(EXP/"oof_predictions_structures.csv",structures);atomic_csv(EXP/"oof_predictions_pairs.csv",pairs)
        atomic_csv(EXP/"metrics_per_fold.csv",fold_metrics);atomic_csv(EXP/"metrics_per_composition.csv",comp_metrics)
        atomic_csv(EXP/"layer_selection_per_fold.csv",selections);atomic_csv(EXP/"layer_energy_inner_validation.csv",valrows)
        agg=aggregate_metrics(fold_metrics);atomic_csv(EXP/"layer_ridge_results.csv",agg[agg.probe.eq("Ridge")]);atomic_csv(EXP/"layer_krr_results.csv",agg[agg.probe.eq("KRR")])
        # Store optimized hyperparameters and a selected-layer OOF summary.
        atomic_csv(EXP/"layer_energy_results.csv",agg)
        # Conditional kernels: search layer, eta and lambda only on validation, fit 45 train groups.
        conditional_rows=[];conditional_preds=[];conditional_structures=[];cond_fold_rows=[]
        graw=data[GEOM].to_numpy(float)
        for fold in range(5):
            roles=fold_roles(folds,fold,data);tr,va,ot=(roles[k] for k in ["train","val","outer"])
            ti,tj,yt,_=local_pairs(data,tr);vi,vj,yv,fv=local_pairs(data,va);oi,oj,yo,fo=local_pairs(data,ot)
            gsc=StandardScaler().fit(graw[tr]);gz=gsc.transform(graw);gbase=1/max(float(np.median(pdist(gz[tr],"sqeuclidean"))),1e-12)
            kgtr=pair_rbf(gz[tr],gz[tr],gbase);kgv=pair_rbf(gz[va],gz[tr],gbase);kgo=pair_rbf(gz[ot],gz[tr],gbase)
            Kg=induced_pair_kernel(kgtr,ti,tj,ti,tj);Kgv=induced_pair_kernel(kgv,vi,vj,ti,tj);Kgo=induced_pair_kernel(kgo,oi,oj,ti,tj)
            geo_choices=[]
            for lam in KRR_LAMBDAS:
                dual=np.linalg.solve(Kg+lam*np.eye(len(Kg)),yt);pval=Kgv@dual
                accuracy=rank_stats(yv,pval,fv)["pairwise_accuracy"]
                geo_choices.append((pair_mae(yv,pval,fv),-accuracy,-lam,lam,dual))
            geo_best=min(geo_choices);geo_lam,geo_dual=geo_best[3],geo_best[4];geo_pred=Kgo@geo_dual
            cond_fold_rows.append({"fold":fold,"model":"GEO","condition":"GEO_KRR","layer":"GEOMETRY","eta":0.0,"lambda":geo_lam,"val_mae":geo_best[0],"outer_pred":geo_pred})
            for model,states in cache.items():
                candidates=[]
                for layer,h0 in states.items():
                    hsc=StandardScaler().fit(h0[tr]);hz=hsc.transform(h0);hbase=1/max(float(np.median(pdist(hz[tr],"sqeuclidean"))),1e-12)
                    khtr=pair_rbf(hz[tr],hz[tr],hbase);khv=pair_rbf(hz[va],hz[tr],hbase);kho=pair_rbf(hz[ot],hz[tr],hbase)
                    Khe=induced_pair_kernel(khtr,ti,tj,ti,tj);Khv=induced_pair_kernel(khv,vi,vj,ti,tj);Kho=induced_pair_kernel(kho,oi,oj,ti,tj)
                    for eta in ETA_GRID:
                        for lam in KRR_LAMBDAS:
                            dual=np.linalg.solve(Kg+eta*Khe+lam*np.eye(len(Kg)),yt);vp=(Kgv+eta*Khv)@dual
                            accuracy=rank_stats(yv,vp,fv)["pairwise_accuracy"]
                            candidates.append((pair_mae(yv,vp,fv),-accuracy,-lam,list(states).index(layer),layer,eta,lam,dual,Kho))
                best=min(candidates,key=lambda c:c[:4])
                _,_,_,_,layer,eta,lam,dual,Kho=best;op=(Kgo+eta*Kho)@dual
                cond_fold_rows.append({"fold":fold,"model":model,"condition":"GEO_BEST","layer":layer,"eta":eta,"lambda":lam,"val_mae":best[0],"outer_pred":op})
                final_layer="FINAL_POST_LN_F";final_c=min((c for c in candidates if c[4]==final_layer),key=lambda c:c[:4])
                opf=(Kgo+final_c[5]*final_c[8])@final_c[7]
                cond_fold_rows.append({"fold":fold,"model":model,"condition":"GEO_FINAL","layer":final_layer,"eta":final_c[5],"lambda":final_c[6],"val_mae":final_c[0],"outer_pred":opf})
            for item in cond_fold_rows[-5:]:
                p=item["outer_pred"]
                for k,(a,b) in enumerate(zip(oi,oj)):
                    conditional_preds.append({"fold":fold,"model":item["model"],"condition":item["condition"],"layer":item["layer"],"formula":fo[k],
                        "delta_dft_mev_atom":yo[k],"delta_pred_mev_atom":p[k],"absolute_error_mev_atom":abs(yo[k]-p[k])})
                for formula in dict.fromkeys(fo):
                    pi=np.flatnonzero(np.asarray(fo)==formula)
                    if len(pi)!=3:raise ValueError(f"Kernel híbrido: número de pares incorrecto para {formula}")
                    d01,d02,_=p[pi];scores=np.asarray([-(d01+d02)/3, (2*d01-d02)/3, (2*d02-d01)/3])
                    g=data[data.composition_id.astype(str).eq(str(formula))].sort_values("family",kind="stable")
                    for j,row in enumerate(g.itertuples()):
                        pred_e=means[(fold,str(formula))]+scores[j]/1000
                        conditional_structures.append({"fold":fold,"model":item["model"],"condition":item["condition"],"layer":item["layer"],
                            "formula":str(formula),"structure_id":str(row.structure_id),"family":str(row.family),
                            "energy_dft_ev_atom":float(row.energy_dft_ev_atom),"energy_pred_ev_atom":float(pred_e),
                            "energy_error_mev_atom":float((pred_e-row.energy_dft_ev_atom)*1000)})
            LOG.info("Kernel aditivo validado: fold=%d",fold)
        cp=pd.DataFrame(conditional_preds);atomic_csv(EXP/"conditional_value_oof_pairs.csv",cp)
        cs=pd.DataFrame(conditional_structures);atomic_csv(EXP/"conditional_value_oof_structures.csv",cs)
        for (model,condition),g in cp.groupby(["model","condition"]):
            mae_by=g.groupby("formula").absolute_error_mev_atom.mean()
            chosen=[r for r in cond_fold_rows if r["model"]==model and r["condition"]==condition]
            unique_layers=list(dict.fromkeys(str(r["layer"]) for r in chosen))
            conditional_rows.append({"model":model,"condition":condition,"layer":unique_layers[0] if len(unique_layers)==1 else ";".join(unique_layers),
                "eta":float(np.mean([r.get("eta",0.0) for r in chosen])),"lambda":float(np.mean([r["lambda"] for r in chosen])),
                "deltaE_mae_estimand_A_mev_atom":float(mae_by.mean()),"compositions":int(mae_by.size)})
        conditional=pd.DataFrame(conditional_rows);atomic_csv(EXP/"conditional_embedding_value.csv",conditional)
        # 100 permutations within composition for FINAL and BEST selected by the internal-energy probe.
        if args.skip_permutations: perms=pd.DataFrame()
        else:
            rng=np.random.default_rng(PAIR_SEED);permutation_rows=[]
            for model,states in cache.items():
                for fold in range(5):
                    roles=fold_roles(folds,fold,data);tr,va,ot=(roles[k] for k in ["train","val","outer"])
                    sel=selections[(selections.model.eq(model))&(selections.fold.eq(fold))]
                    for probe,kind in [("Ridge","Ridge"),("KRR","KRR")]:
                        selected_layer=str(sel[sel.probe.eq(probe)].selected_layer.iloc[0])
                        for layer_role,layer in [("FINAL","FINAL_POST_LN_F"),("BEST_LAYER",selected_layer)]:
                            h0=states[layer];perm_feat=h0.copy();permutation_maps=[]
                            # Each draw independently permutes three representations within every formula.
                            for rep in range(args.permutations):
                                perm_feat[:]=h0
                                for _,g in data.groupby("composition_id",sort=False):
                                    ids=list(g.index);perm=rng.permutation(ids);perm_feat[ids]=h0[perm]
                                scaler=StandardScaler().fit(perm_feat[tr]);z=scaler.transform(perm_feat)
                                ti,tj,yt,_=local_pairs(data,tr);vi,vj,yv,fv=local_pairs(data,va);oi,oj,yo,fo=local_pairs(data,ot)
                                hyper=sel[sel.probe.eq(probe)].iloc[0]
                                if kind=="Ridge":params={"alpha":float(hyper.selected_alpha)}
                                else:params={"gamma":float(hyper.selected_gamma),"lambda":float(hyper.selected_lambda)}
                                scores,pred=fit_outer_scores(kind,z[tr],yt,z[ot],params,(ti,tj),(oi,oj))
                                stats,_,_=metrics_for_predictions(data,ot,scores,means,fold)
                                permutation_rows.append({"model":model,"probe":probe,"layer_role":layer_role,"layer":layer,"fold":fold,"permutation":rep,
                                    "deltaE_mae_estimand_A_mev_atom":stats["deltaE_mae_estimand_A_mev_atom"],"pairwise_accuracy":stats["pairwise_accuracy"],
                                    "pairwise_correct":stats["pairwise_correct"],"pairwise_eligible":stats["pairwise_eligible"],"control":"permuted_within_formula_train_validation_outer"})
                            LOG.info("Permutaciones: %s fold=%d %s %s n=%d",model,fold,probe,layer_role,args.permutations)
            perms=pd.DataFrame(permutation_rows);atomic_csv(EXP/"permutation_controls.csv",perms)
        # Bootstrap FINAL vs BEST layer, using all 70 formula-level OOF rows.
        boot=pd.concat([bootstrap_comparisons(comp_metrics),bootstrap_conditional(cp,cs)],ignore_index=True);atomic_csv(EXP/"bootstrap_paired.csv",boot)
        layer_metrics=refine_global_metrics(aggregate_metrics(fold_metrics),structures,pairs);atomic_csv(EXP/"layer_energy_results.csv",layer_metrics)
        atomic_csv(EXP/"layer_ridge_results.csv",layer_metrics[layer_metrics.probe.eq("Ridge")])
        atomic_csv(EXP/"layer_krr_results.csv",layer_metrics[layer_metrics.probe.eq("KRR")])
        make_figures(sep,var,geometry,valrows,selections,layer_metrics,pairs,conditional,boot)
        # gap compression by layer and DFT gap bins, with pair denominator.
        pair_eval=pairs[pairs.layer.isin(["FINAL","BEST_LAYER"])].copy()
        bins=[0,5,10,25,50,100,np.inf];labels=["0-5","5-10","10-25","25-50","50-100",">100"]
        pair_eval["gap_bin_mev_atom"]=pd.cut(pair_eval.delta_dft_mev_atom.abs(),bins=bins,labels=labels,right=False,include_lowest=True)
        gaps=pair_eval.groupby(["model","probe","layer","gap_bin_mev_atom"],observed=False).agg(n_pairs=("absolute_error_mev_atom","size"),
            mae_mev_atom=("absolute_error_mev_atom","mean"),rmse_mev_atom=("absolute_error_mev_atom",lambda x:float(np.sqrt(np.mean(x*x))))).reset_index()
        atomic_csv(EXP/"gap_compression_analysis.csv",gaps)
        summary={"status":"COMPLETE_WITH_LIMITATIONS","cohort":{"compositions":70,"structures":210,"polymorphs_per_composition":3,"folds":5,"sizes":{"inner_train":45,"inner_validation":11,"outer":14}},
            "layer_selection_protocol":"por fold/model/probe: mínimo MAE ΔE inner-validation; desempates mayor pairwise, mayor regularización, menor profundidad; refit final inner-train (45 formulas)",
            "test_accessed":False,"backbones_modified":False,"ranking_90pct_reached":bool(layer_metrics.pairwise_accuracy.max()>=0.9),
            "selected_layers":json.loads(selections.to_json(orient="records")),"terminal_reproduction":"FINAL exacto vs caché histórica; ver token_alignment_audit.json",
            "selected_procedure_metrics":json.loads(layer_metrics[layer_metrics.layer.isin(["FINAL","BEST_LAYER"])].to_json(orient="records")),
            "limitations":["La selección de BEST usa validación y su OOF es desarrollo CV, no test externo.","Distancias por todas las 70 composiciones son descriptivas y no evidencia de generalización.","El Ridge M2 composicional es fijo; la energía absoluta permanece anclada a él.","El kernel híbrido selecciona eta/capa en validation; IC y outer siguen siendo exploratorios."]}
        atomic_json(EXP/"summary.json",summary)
        elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu
        atomic_json(EXP/"resources.json",{"wall_seconds":elapsed,"process_cpu_seconds":cpu_elapsed,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "device":"cpu","cpu_threads":1,"backbones_modified":False,"test_accessed":False,"layer_count":{"MAT":len(cache["MAT"]),"MATTER":len(cache["MATTER"])},
            "permutations_per_model_probe_fold_layer":args.permutations if not args.skip_permutations else 0,"bootstrap_replicates":BOOT_N,"bootstrap_seed":BOOT_SEED,
            "random_seed_permutation":PAIR_SEED,"mps":False,"cuda":False})
        report=build_report(layer_metrics,selections,boot,conditional,alignment=json.loads((EXP/"token_alignment_audit.json").read_text()),perms=perms)
        (EXP/"reporte_cientifico.md").write_text(report,encoding="utf-8");(REPORT/"reporte_cientifico.md").write_text(report,encoding="utf-8")
        readme="# Auditoría layerwise ABS₃\n\nAuditoría OOF de estados L0–L12/FINAL de MatGPT y MatterGPT originales. Ejecutar `python scripts/ABS3/abs3_transformer_layerwise_audit/extract_layers.py` y después `python scripts/ABS3/abs3_transformer_layerwise_audit/run_layerwise.py`. La selección de capa usa solo inner-validation; el test independiente queda cerrado.\n"
        (EXP/"README.md").write_text(readme,encoding="utf-8");(REPORT/"README.md").write_text(readme,encoding="utf-8")
        LOG.info("Layerwise finalizado en %.1fs; OOF filas estructura=%d pares=%d",elapsed,len(structures),len(pairs))


def build_report(metrics: pd.DataFrame, selections: pd.DataFrame, bootstrap: pd.DataFrame,
                 conditional: pd.DataFrame, alignment: dict[str,Any], perms: pd.DataFrame) -> str:
    """Construye informe científico español con resultados y controles.

    Args:
        metrics: Métricas agregadas por capa.
        selections: Capa seleccionada por fold.
        bootstrap: Contrastes pareados.
        conditional: Kernels geométricos/aditivos.
        alignment: Comparación FINAL histórica.
        perms: Resultados de permutación.
    Returns:
        Markdown del informe.
    """
    lines=["# Auditoría de representaciones por capa: MatGPT vs MatterGPT", "",
        "## Diseño y controles", "",
        "Cohorte de 70 composiciones/210 estructuras; cinco pliegues históricos agrupados por fórmula (45 train, 11 validation, 14 outer). Los extractores originales se mantuvieron congelados. MatGPT usó SLICES y MatterGPT el conditioning `[1.0, 2.0]`. Los targets por par son ΔE = Eⱼ−Eᵢ en meV/átomo. Scalers se ajustaron solo con inner-train; Ridge M2 quedó fijo para reconstrucción absoluta. No se consultó el test independiente.", "",
        "La capa se eligió por fold y probe usando exclusivamente el mínimo de MAE ΔE de inner-validation; desempates por pairwise accuracy, mayor regularización y menor profundidad. Después se ajustó sobre las 45 fórmulas inner-train y se evaluó una sola vez en outer. FINAL y BEST_LAYER se comparan como desarrollo OOF; no son evaluación externa.", "",
        "## Compatibilidad de extracción", "",
        f"FINAL MatGPT reproduce la caché histórica con error absoluto máximo {alignment['MAT']['max_abs_diff']:.3g} y coseno mínimo {alignment['MAT']['min_cosine']:.12f}. MatterGPT: máximo {alignment['MATTER']['max_abs_diff']:.3g}, coseno mínimo {alignment['MATTER']['min_cosine']:.12f}. Ambos cumplen tolerancias; hashes de checkpoints y tokenización constan en manifiestos.", "",
        "## Capas elegidas por validación interna", "",
        "| Modelo | Probe | Capas seleccionadas en F0–F4 |", "|---|---|---|"]
    for (model,probe),g in selections.groupby(["model","probe"],sort=False):lines.append(f"| {model} | {probe} | {', '.join(g.sort_values('fold').selected_layer.astype(str))} |")
    lines += ["", "## Comparación FINAL vs BEST_LAYER OOF", "", "| Modelo | Probe | Capa | ΔE MAE A | Energía MAE | Pairwise | Ground state | m OLS |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for model in ["MAT","MATTER"]:
        for probe in ["Ridge","KRR"]:
            for layer in ["FINAL","BEST_LAYER"]:
                r=metrics[(metrics.model.eq(model))&(metrics.probe.eq(probe))&(metrics.layer.eq(layer))]
                if r.empty:continue
                q=r.iloc[0];lines.append(f"| {model} | {probe} | {layer} | {q.deltaE_mae_estimand_A_mev_atom:.2f} meV/átomo | {q.energy_mae_mev_atom:.2f} meV/átomo | {q.pairwise_correct}/{q.pairwise_eligible} ({q.pairwise_accuracy:.1%}) | {q.groundstate_correct}/{q.groundstate_eligible} ({q.groundstate_accuracy:.1%}) | {q.ols_slope:.3f} |")
    lines += ["", "ΔE MAE sigue estimando A: primero media de los tres errores absolutos por fórmula y después macro-promedio de 70 fórmulas. Pairwise elegible usa |ΔE_DFT|>5 meV/átomo; predicciones |ΔE|≤1 son no resueltas e incorrectas. Ground state requiere gap DFT >1 meV/átomo y predicción sin empate >1. Full-triplet requiere todos los gaps DFT >1 meV/átomo.", "",
        "## Bootstrap pareado", "", "| Contraste | Métrica | Diferencia | IC95% |", "|---|---|---:|---:|"]
    for r in bootstrap.itertuples():lines.append(f"| {r.contrast} | {r.metric} | {r.estimate:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] |")
    lines += ["", "Bootstrap por fórmula: 10 000 réplicas, semilla 20261103. Intervalos descriptivos de CV de desarrollo; no tratar pares dentro de fórmula como independientes.", "",
        "## Kernel geométrico y complemento por capa", "", "| Modelo | Condición | Capa | η | ΔE MAE OOF |", "|---|---|---|---:|---:|"]
    for r in conditional.itertuples():lines.append(f"| {r.model} | {r.condition} | {r.layer} | {getattr(r,'eta',np.nan):.3g} | {r.deltaE_mae_estimand_A_mev_atom:.2f} meV/átomo |")
    lines += ["", "Los kernels aditivos se seleccionaron dentro de cada fold. Una mejora puntual no implica valor incremental estadísticamente concluyente; contrastes apareados están en bootstrap_paired.csv.", "",
        "## Interpretación y límites", "",
        "Las curvas layer-wise describen el comportamiento de probes con normalización interna. La capa con menor validación puede variar entre folds; solo el OOF de BEST_LAYER estima el procedimiento de selección. Una pendiente más cercana a 1 no sustituye MAE/RMSE ni ranking. La geometría procede de celdas finales relajadas por DFT. No se puede atribuir una diferencia MatGPT–MatterGPT exclusivamente al backbone.", "",
        f"El máximo pairwise OOF entre capas evaluadas fue {metrics.pairwise_accuracy.max():.1%}; meta aspiracional 90% {'alcanzada descriptivamente' if metrics.pairwise_accuracy.max()>=.9 else 'no alcanzada'}. Permutaciones completadas: {0 if perms.empty else perms.permutation.nunique()} repeticiones por combinación de modelo/probe/fold/capa.", "",
        "## Artefactos y reproducibilidad", "",
        "Cachés layerwise: embeddings_matgpt_layers.npz y embeddings_mattergpt_layers.npz. Manifiestos: architecture_manifest.json, checkpoint_manifest.json, layer_extraction_manifest.json, token_alignment_audit.json y embedding_cache_manifest.json. Predicciones y métricas se guardan por estructura, par, fórmula, fold, capa y probe. Recursos y duración están en resources.json. Test histórico cerrado."]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    main()
