"""Etapas complementarias de structural probing, few-shot y controles negativos ABS3."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_audit import (ALPHAS, BOOT_N, BOOT_SEED, EXP, GEOM, GAMMA_MULTIPLIERS,
                       KRR_LAMBDAS, ROOT, T0, atomic_csv, atomic_json, build_pairs_for_indices,
                       fit_predict_probe, fold_rows, load_inputs, mae, metric_block, pair_kernel, rbf, sha256)

LOG = logging.getLogger("abs3.representation_audit.extended")
ELEMENTS=["Ba","Ca","Cd","Cu","Ge","Hf","Mg","Nb","Pb","S","Si","Sn","Sr","Ti","V","Zn","Zr"]


def composition_features(d: pd.DataFrame) -> np.ndarray:
    """Genera 17 fracciones estequiométricas por estructura desde la fórmula reducida."""
    result=[]
    for formula in d.reduced_formula.astype(str):
        counts={el:0.0 for el in ELEMENTS}
        for el,num in re.findall(r"([A-Z][a-z]?)(\d*)",formula):
            if el in counts: counts[el]+=float(num or 1)
        total=sum(counts.values())
        if not total: raise ValueError(f"No se pudo parsear fórmula: {formula}")
        result.append([counts[e]/total for e in ELEMENTS])
    return np.asarray(result,float)


def pair_metrics(d: pd.DataFrame, ix: np.ndarray, pi: np.ndarray, pj: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str,float]:
    """Métricas pairwise con macro-MAE por composición y reglas históricas."""
    by=[]; eligible=correct=unresolved=0
    for formula, inds in d.loc[ix].groupby("composition_id",sort=False).groups.items():
        global_ids=list(inds); local={g:k for k,g in enumerate(ix)}
        mask=np.asarray([d.loc[ix[a],"composition_id"]==formula for a in pi])
        by.append(float(np.mean(np.abs(y[mask]-p[mask]))))
        for yy,pp in zip(y[mask],p[mask]):
            if abs(yy)>5:
                eligible+=1; unresolved+=int(abs(pp)<=1); correct+=int(abs(pp)>1 and np.sign(yy)==np.sign(pp))
    slope,inter=np.polyfit(y,p,1) if len(y)>1 else (np.nan,np.nan)
    return {"deltaE_mae_estimand_A_mev_atom":float(np.mean(by)),"deltaE_rmse_global_mev_atom":float(np.sqrt(np.mean((y-p)**2))),
            "pairwise_correct":int(correct),"pairwise_eligible":int(eligible),"pairwise_accuracy":correct/eligible if eligible else np.nan,
            "unresolved_eligible":int(unresolved),"ols_slope":float(slope),"ols_intercept":float(inter),
            "deltaE_predictive_r2":float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}


def run_geometry_decoding(d: pd.DataFrame, folds: pd.DataFrame, feat: dict[str,np.ndarray]) -> pd.DataFrame:
    """OOF geometry decoding with Ridge/KRR, composition-only, and constant controls."""
    out=[]; comp=composition_features(d); target=d[GEOM].to_numpy(float)
    for fold in range(5):
        ix=fold_rows(folds,fold,d); tr,va,ot=(ix[k] for k in ["train","val","outer"])
        ys=StandardScaler().fit(target[tr]); y=ys.transform(target)
        for ext,h0 in feat.items():
            xs=StandardScaler().fit(h0[tr]); h=xs.transform(h0)
            # Uniform target-space validation score across descriptor units.
            best=None
            for alpha in ALPHAS:
                m=Ridge(alpha=alpha,fit_intercept=True,solver="lsqr",tol=1e-10,max_iter=10000).fit(h[tr],y[tr])
                pv=m.predict(h[va]); score=float(np.mean((pv-y[va])**2))
                if best is None or score<best[0]: best=(score,alpha)
            _,alpha=best;m=Ridge(alpha=alpha,fit_intercept=True,solver="lsqr",tol=1e-10,max_iter=10000).fit(h[tr],y[tr])
            pred=ys.inverse_transform(m.predict(h[ot]))
            for j,col in enumerate(GEOM):
                err=pred[:,j]-target[ot,j]; var=np.sum((target[ot,j]-target[tr,j].mean())**2)
                out.append({"fold":fold,"extractor":ext,"probe":"Ridge","descriptor":col,"selected_alpha":alpha,
                    "mae":float(np.mean(abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),"predictive_r2":float(1-np.sum(err**2)/var),
                    "correlation":float(np.corrcoef(pred[:,j],target[ot,j])[0,1]),"train_sd_normalized_mae":float(np.mean(abs(err))/max(target[tr,j].std(),1e-12))})
            # RBF KRR on structures; same train-only scaler and validation model selection.
            base=1/max(float(np.median(pdist(h[tr],"sqeuclidean"))),1e-12);best=None
            for gm in GAMMA_MULTIPLIERS:
                gamma=base*gm;K=rbf(h[tr],h[tr],gamma);Kv=rbf(h[va],h[tr],gamma)
                for lam in KRR_LAMBDAS:
                    alpha_dual=np.linalg.solve(K+lam*np.eye(len(K)),y[tr]);pv=Kv@alpha_dual
                    score=float(np.mean((pv-y[va])**2));key=(score,-lam,-gamma)
                    if best is None or key<best[0]:best=(key,gamma,lam)
            _,gamma,lam=best;K=rbf(h[tr],h[tr],gamma);Ko=rbf(h[ot],h[tr],gamma)
            pred=ys.inverse_transform(Ko@np.linalg.solve(K+lam*np.eye(len(K)),y[tr]))
            for j,col in enumerate(GEOM):
                err=pred[:,j]-target[ot,j];var=np.sum((target[ot,j]-target[tr,j].mean())**2)
                out.append({"fold":fold,"extractor":ext,"probe":"KRR_RBF","descriptor":col,"selected_gamma":gamma,"selected_lambda":lam,
                    "mae":float(np.mean(abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),"predictive_r2":float(1-np.sum(err**2)/var),
                    "correlation":float(np.corrcoef(pred[:,j],target[ot,j])[0,1]),"train_sd_normalized_mae":float(np.mean(abs(err))/max(target[tr,j].std(),1e-12))})
        # Constant and stoichiometry-only OOF controls.
        for name,xx in [("constant",None),("composition_Ridge",comp)]:
            if name=="constant": pred=np.tile(target[tr].mean(axis=0),(len(ot),1))
            else:
                sx=StandardScaler().fit(xx[tr]); x=sx.transform(xx)
                best=min(((float(np.mean((Ridge(alpha=a).fit(x[tr],y[tr]).predict(x[va])-y[va])**2)),a) for a in ALPHAS),key=lambda z:z[0])
                model=Ridge(alpha=best[1]).fit(x[tr],y[tr]);pred=ys.inverse_transform(model.predict(x[ot]))
            if name=="constant":pass
            for j,col in enumerate(GEOM):
                err=pred[:,j]-target[ot,j];var=np.sum((target[ot,j]-target[tr,j].mean())**2)
                out.append({"fold":fold,"extractor":name,"probe":name,"descriptor":col,
                    "mae":float(np.mean(abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),"predictive_r2":float(1-np.sum(err**2)/var),
                    "correlation":float(np.corrcoef(pred[:,j],target[ot,j])[0,1]),"train_sd_normalized_mae":float(np.mean(abs(err))/max(target[tr,j].std(),1e-12))})
    return pd.DataFrame(out)


def run_additive_kernel(d: pd.DataFrame, folds: pd.DataFrame, feats: dict[str,np.ndarray]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Evalúa K_GEO + eta*K_embedding con eta seleccionado en inner-validation."""
    rows=[];predrows=[];geo=d[GEOM].to_numpy(float)
    for fold in range(5):
        ix=fold_rows(folds,fold,d);tr,va,ot=(ix[k] for k in ["train","val","outer"])
        bases={}
        for name,raw in [("GEO",geo),("MAT",feats["MAT"]),("MATTER",feats["MATTER"])]:
            sc=StandardScaler().fit(raw[tr]);z=sc.transform(raw)
            bases[name]=(z,1/max(float(np.median(pdist(z[tr],"sqeuclidean"))),1e-12))
        # local structure indices preserve sorted composition/family order.
        _,yt,ids_tr,ti,tj=build_pairs_for_indices(d,bases["GEO"][0],tr)
        _,yv,ids_va,vi,vj=build_pairs_for_indices(d,bases["GEO"][0],va)
        _,yo,ids_ot,oi,oj=build_pairs_for_indices(d,bases["GEO"][0],ot)
        for ext in ["MAT","MATTER"]:
            zg,ggeo=bases["GEO"];ze,ge=bases[ext]
            best=None
            for mg in GAMMA_MULTIPLIERS:
                Kg=rbf(zg[tr],zg[tr],ggeo*mg);Kgv=rbf(zg[va],zg[tr],ggeo*mg);Kgo=rbf(zg[ot],zg[tr],ggeo*mg)
                for me in GAMMA_MULTIPLIERS:
                    Ke=rbf(ze[tr],ze[tr],ge*me);Kev=rbf(ze[va],ze[tr],ge*me);Keo=rbf(ze[ot],ze[tr],ge*me)
                    for eta in [0.0,0.1,1.0,10.0]:
                        kt=Kg+eta*Ke;kv=Kgv+eta*Kev;ko=Kgo+eta*Keo
                        K=pair_kernel(kt,ti,tj,ti,tj);Kv=pair_kernel(kv,vi,vj,ti,tj);Ko=pair_kernel(ko,oi,oj,ti,tj)
                        for lam in KRR_LAMBDAS:
                            dual=np.linalg.solve(K+lam*np.eye(len(K)),yt);pval=Kv@dual;score=mae(yv,pval);key=(score,-lam,-eta,-mg,-me)
                            if best is None or key<best[0]:best=(key,eta,mg,me,lam,Ko,dual)
            _,eta,mg,me,lam,Ko,dual=best;po=Ko@dual
            rows.append({"fold":fold,"extractor":ext,"eta":eta,"gamma_geo_multiplier":mg,"gamma_embedding_multiplier":me,"lambda":lam,
                         "validation_deltaE_mae":best[0][0],"outer_deltaE_mae":mae(yo,po)})
            for p,(a,b) in enumerate(zip(oi,oj)):
                predrows.append({"fold":fold,"extractor":ext,"eta":eta,"formula":d.loc[ids_ot[a],"composition_id"],
                    "structure_i":d.loc[ids_ot[a],"structure_id"],"structure_j":d.loc[ids_ot[b],"structure_id"],
                    "delta_dft_mev_atom":float(yo[p]),"delta_pred_mev_atom":float(po[p])})
    return pd.DataFrame(rows),pd.DataFrame(predrows)


def run_learning_curves(d: pd.DataFrame, folds: pd.DataFrame, feats: dict[str,np.ndarray]) -> pd.DataFrame:
    """Curvas few-shot reproducibles; 10 subconjuntos para n<45, conjunto completo n=45."""
    rows=[];rng=np.random.default_rng(77123);geo=d[GEOM].to_numpy(float)
    for fold in range(5):
        ix=fold_rows(folds,fold,d); train_comps=np.array(sorted(d.iloc[ix["train"]].composition_id.unique()))
        for n in [10,20,30,45]:
            reps=1 if n==45 else 10
            for rep in range(reps):
                chosen=set(train_comps if n==45 else rng.choice(train_comps,size=n,replace=False).tolist())
                tr=np.flatnonzero(d.composition_id.isin(chosen).to_numpy());va=ix["val"];ot=ix["outer"]
                for ext,raw in [("GEO",geo),("MAT",feats["MAT"]),("MATTER",feats["MATTER"])]:
                    sc=StandardScaler().fit(raw[tr]);z=sc.transform(raw)
                    xt,yt,ids_t,ti,tj=build_pairs_for_indices(d,z,tr);xv,yv,ids_v,vi,vj=build_pairs_for_indices(d,z,va);xo,yo,ids_o,oi,oj=build_pairs_for_indices(d,z,ot)
                    for probe in ["RIDGE","KRR"]:
                        pv,po,params=fit_predict_probe(probe,xt,yt,xv,yv,xo,yo,
                            d.loc[ids_t,"composition_id"].to_numpy(),d.loc[ids_v,"composition_id"].to_numpy(),
                            z[ids_t],z[ids_v],z[ids_o],(ti,tj),(vi,vj),(oi,oj))
                        if probe=="RIDGE":ptr=xt@params["beta"]
                        else:
                            kt=rbf(params["x_reference"],params["x_reference"],params["gamma"])
                            ptr=pair_kernel(kt,ti,tj,ti,tj)@params["dual_alpha"]
                        for split,yy,pp in [("train",yt,ptr),("validation",yv,pv),("outer",yo,po)]:
                            m=pair_metrics(d,ids_t if split=="train" else ids_v if split=="validation" else ids_o,
                                ti if split=="train" else vi if split=="validation" else oi,
                                tj if split=="train" else vj if split=="validation" else oj,yy,pp)
                            rows.append({"fold":fold,"n_train_compositions":n,"subset_replicate":rep,"extractor":ext,"probe":probe,"split":split,**m})
    return pd.DataFrame(rows)


def run_permutations(d: pd.DataFrame, folds: pd.DataFrame, feats: dict[str,np.ndarray], n_perm: int=100) -> pd.DataFrame:
    """Permuta embeddings entre polimorfos dentro de composición y repite selección OOF."""
    rng=np.random.default_rng(5512026);rows=[]
    for perm in range(n_perm):
        permuted={}
        for ext,h in feats.items():
            hp=h.copy()
            for _,g in d.groupby("composition_id",sort=False):
                ix=list(g.index);hp[ix]=h[rng.permutation(ix)]
            permuted[ext]=hp
        for fold in range(5):
            ix=fold_rows(folds,fold,d);tr,va,ot=(ix[k] for k in ["train","val","outer"])
            for ext in ["MAT","MATTER"]:
                raw=permuted[ext];sc=StandardScaler().fit(raw[tr]);z=sc.transform(raw)
                xt,yt,ids_t,ti,tj=build_pairs_for_indices(d,z,tr);xv,yv,ids_v,vi,vj=build_pairs_for_indices(d,z,va);xo,yo,ids_o,oi,oj=build_pairs_for_indices(d,z,ot)
                for probe in ["RIDGE","KRR"]:
                    pv,po,params=fit_predict_probe(probe,xt,yt,xv,yv,xo,yo,d.loc[ids_t,"composition_id"].to_numpy(),d.loc[ids_v,"composition_id"].to_numpy(),
                        z[ids_t],z[ids_v],z[ids_o],(ti,tj),(vi,vj),(oi,oj))
                    met=pair_metrics(d,ids_o,oi,oj,yo,po)
                    rows.append({"permutation":perm,"fold":fold,"extractor":ext,"probe":probe,**met})
        if (perm+1)%10==0: LOG.info("permutaciones completadas %d/%d",perm+1,n_perm)
    return pd.DataFrame(rows)


def bootstrap_effects(mainpairs: pd.DataFrame, additive: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap apareado por fórmula para comparaciones de métodos y kernels aditivos."""
    rng=np.random.default_rng(BOOT_SEED);rows=[]
    # Reduce three pairs to a composition-level MAE before resampling.
    groups=[]
    for ext in mainpairs.condition.unique():
        for probe in mainpairs.probe.unique():
            q=mainpairs[(mainpairs.condition==ext)&(mainpairs.probe==probe)]
            v=q.groupby("formula").absolute_error_mev_atom.mean().to_dict()
            if len(v)==70:groups.append((ext,probe,v))
    for ext,probe,base in groups:
        pass
    by={(e,p):v for e,p,v in groups}
    contrasts=[]
    for ext in ["MAT","MATTER"]:
        if (ext,"Ridge") in by and ("GEO","Ridge") in by: contrasts.append((f"{ext} Ridge − GEO Ridge",by[(ext,"Ridge")],by[("GEO","Ridge")]))
        if (ext,"KRR_RBF") in by and ("GEO","KRR_RBF") in by: contrasts.append((f"{ext} KRR − GEO KRR",by[(ext,"KRR_RBF")],by[("GEO","KRR_RBF")]))
        if (ext,"Ridge") in by and (ext,"KRR_RBF") in by: contrasts.append((f"{ext} KRR − Ridge",by[(ext,"KRR_RBF")],by[(ext,"Ridge")]))
    for label,a,b in contrasts:
        keys=sorted(set(a)&set(b));diff=np.array([a[k]-b[k] for k in keys]);draws=np.mean(diff[rng.integers(0,len(diff),(BOOT_N,len(diff)))],axis=1)
        rows.append({"contrast":label,"metric":"deltaE_mae_mev_atom","estimate":float(diff.mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
    if len(additive):
        # Mat/Matter hybrid eta-selected KRR versus geometry-only RBF from main OOF.
        for ext in ["MAT","MATTER"]:
            q=additive[additive.extractor.eq(ext)]
            baseline=mainpairs[(mainpairs.condition=="GEO")&(mainpairs.probe=="KRR_RBF")]
            av=q.groupby("formula").apply(lambda z:np.mean(np.abs(z.delta_dft_mev_atom-z.delta_pred_mev_atom)),include_groups=False).to_dict()
            bv=baseline.groupby("formula").absolute_error_mev_atom.mean().to_dict()
            keys=sorted(set(av)&set(bv));diff=np.array([av[k]-bv[k] for k in keys]);draws=np.mean(diff[rng.integers(0,len(diff),(BOOT_N,len(diff)))],axis=1)
            rows.append({"contrast":f"{ext}+G additive KRR − GEO KRR","metric":"deltaE_mae_mev_atom","estimate":float(diff.mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"replicates":BOOT_N,"seed":BOOT_SEED,"unit":"composition"})
    return pd.DataFrame(rows)


def main() -> None:
    """Ejecuta etapas geométricas, kernels aditivos, curvas few-shot y permutaciones."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    d,folds,feats,manifest,mu=load_inputs()
    with threadpool_limits(limits=1):
        geo=run_geometry_decoding(d,folds,feats);atomic_csv(EXP/"geometry_probe_results.csv",geo)
        additive_cfg,additive_pred=run_additive_kernel(d,folds,feats)
        atomic_csv(EXP/"conditional_value.csv",additive_cfg);atomic_csv(EXP/"conditional_value_predictions.csv",additive_pred)
        curves=run_learning_curves(d,folds,feats);atomic_csv(EXP/"learning_curves.csv",curves)
        perms=run_permutations(d,folds,feats,100);atomic_csv(EXP/"permutation_controls.csv",perms)
        pairs=pd.read_csv(EXP/"predicciones_oof_por_par.csv")
        boot=bootstrap_effects(pairs,additive_pred);atomic_csv(EXP/"bootstrap_paired.csv",boot)
        atomic_json(EXP/"extended_resources.json",{"wall_seconds":time.monotonic()-T0,"device":"cpu","cpu_threads":1,
            "permutations":100,"learning_curve_subsets":int(curves.subset_replicate.nunique()),"bootstrap_replicates":BOOT_N,"bootstrap_seed":BOOT_SEED,
            "historical_test_accessed":False,"backbones_modified":False})
        LOG.info("Etapas extendidas completas: geometría=%d, curvas=%d, permutaciones=%d",len(geo),len(curves),len(perms))


if __name__=="__main__": main()
