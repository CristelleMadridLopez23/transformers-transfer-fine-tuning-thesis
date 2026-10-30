"""Analiza controles H3-C por fórmula, energía absoluta y separaciones internas."""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau,spearmanr

from reproduce import FAMILIES,project_root,sha256

sys.path.insert(0,str(project_root()))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.metrics import metric_tables

LOG=logging.getLogger("h3c_final_analysis")
KINDS=("pretrained","random","composition")
BINS=((0,5),(5,10),(10,25),(25,50),(50,100),(100,float("inf")))


def pooled_ols(y:np.ndarray,p:np.ndarray) -> dict[str,float]:
    """Obtiene ajuste OLS con intercepto y razón de dispersión física."""
    slope,intercept=np.polyfit(y,p,1)
    fitted=slope*y+intercept
    return {"ols_slope":float(slope),"ols_intercept":float(intercept),
            "ols_r2":float(1-np.sum((p-fitted)**2)/np.sum((p-p.mean())**2)) if np.var(p)>0 else float("nan"),
            "predictive_r2":float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
            "std_ratio":float(np.std(p)/np.std(y))}


def analyze(root:Path) -> dict[str,Any]:
    """Une tres controles, verifica 210 IDs pareados y guarda tablas estadísticas."""
    report=root/"reports/MatGPT/MatGPT_H3C_Final"
    old=report/"reproduction/replayed_predictions.csv"
    if json.loads((report/"reproduction/reproduction_manifest.json").read_text())["status"]!="PASS":
        raise ValueError("Reproducción histórica no aprobada")
    frames=[]
    pretrained=pd.read_csv(old)
    pretrained["model"]="pretrained";frames.append(pretrained)
    for kind in ("random","composition"):
        for seed in (42,123,2026):
            name=f"random_fit_seed_{seed}" if kind=="random" else f"composition_seed_{seed}"
            path=report/f"controls/{name}/predictions.csv"
            manifest=json.loads(path.with_name("manifest.json").read_text())
            if manifest["status"]!="PASS" or manifest["predictions_sha256"]!=sha256(path):
                raise ValueError(f"Control no íntegro: {path}")
            frames.append(pd.read_csv(path))
    predictions=pd.concat(frames,ignore_index=True)
    if len(predictions)!=1890 or predictions.groupby(["model","seed"]).size().ne(210).any():
        raise ValueError("Cobertura de controles incompleta")
    source=pretrained.sort_values(["seed","structure_id"])
    for (_,seed),part in predictions.groupby(["model","seed"]):
        matched=part.sort_values("structure_id")
        reference=source[source.seed==seed].sort_values("structure_id")
        if matched.structure_id.tolist()!=reference.structure_id.tolist() or not np.allclose(matched.energy_per_atom_dft_ev,reference.energy_per_atom_dft_ev,atol=1e-8):
            raise ValueError("IDs/etiquetas distintos entre controles")
    out=report/"analysis";out.mkdir(exist_ok=True)
    predictions.to_csv(out/"predictions_by_structure.csv",index=False)
    legacy=predictions.rename(columns={"energy_per_atom_dft_ev":"E_DFT_eV_atom",
                                       "energy_per_atom_pred_ev":"E_pred_eV_atom"})
    comp_hist,pairs_hist,fold_hist=metric_tables(legacy)
    comp_hist.to_csv(out/"composition_metrics_historical_1mev.csv",index=False)
    pairs_hist.to_csv(out/"pairs_historical_1mev.csv",index=False)
    fold_hist.to_csv(out/"fold_seed_metrics_historical_1mev.csv",index=False)
    comps=[];pairs=[];pooled=[];bins=[]
    for (model,seed),frame in predictions.groupby(["model","seed"]):
        y=frame.energy_per_atom_dft_ev.to_numpy(float)
        p=frame.energy_per_atom_pred_ev.to_numpy(float)
        yt=frame.energy_total_dft_ev.to_numpy(float)
        pt=frame.energy_total_pred_ev.to_numpy(float)
        if np.max(np.abs(pt/frame.n_atoms-p))>1e-8:
            raise ValueError("Predicción total/E por átomo inconsistente")
        all_yres=[];all_pres=[];all_anch=[];all_pair_y=[];all_pair_p=[];gs_correct=[];gs_eligible=[];raw_gs=[]
        for formula,part in frame.groupby("composition_id",sort=True):
            group=part.set_index("polymorph_family").loc[list(FAMILIES)]
            true=group.energy_per_atom_dft_ev.to_numpy(float)
            pred=group.energy_per_atom_pred_ev.to_numpy(float)
            dft_mean=true.mean();pred_mean=pred.mean()
            ry=true-dft_mean;rp=pred-pred_mean
            all_yres.extend(ry);all_pres.extend(rp)
            anchor=int(np.argmin(true))
            all_anch.extend(1000*((pred-pred[anchor])-(true-true[anchor])))
            dft_tie=(np.partition(true,1)[1]-true.min())<=.001
            pred_tie=(np.partition(pred,1)[1]-pred.min())<=.001
            correct=int(not dft_tie and not pred_tie and np.argmin(true)==np.argmin(pred))
            gs_correct.append(correct);gs_eligible.append(int(not dft_tie))
            raw=int(np.argmin(true)==np.argmin(pred)) if len(np.unique(pred))==3 else np.nan
            raw_gs.append(raw)
            error=(pred-true)*1000
            record={"model":model,"seed":seed,"composition_id":formula,"fold":int(part.fold.iloc[0]),
                    "n_structures":3,"mean_dft_ev_atom":dft_mean,"mean_pred_ev_atom":pred_mean,
                    "mean_error_mev_atom":float(error.mean()),"mean_abs_error_mev_atom":float(abs(error.mean())),
                    "residual_mae_mev_atom":float(np.mean(np.abs(rp-ry))*1000),
                    "residual_rmse_mev_atom":float(np.sqrt(np.mean((rp-ry)**2))*1000),
                    "absolute_mae_mev_atom":float(np.mean(np.abs(error))),
                    "gs_correct":correct,"gs_eligible":int(not dft_tie),"predicted_gs_abstention":int(pred_tie),
                    "gs_raw_argmin_correct":raw,
                    "spearman_triplet":float(spearmanr(true,pred).statistic) if np.std(pred)>1e-12 else float("nan"),
                    "kendall_triplet":float(kendalltau(true,pred).statistic) if np.std(pred)>1e-12 else float("nan")}
            comps.append(record)
            for i,j in itertools.combinations(range(3),2):
                delta_y=1000*(true[j]-true[i]);delta_p=1000*(pred[j]-pred[i])
                all_pair_y.append(delta_y);all_pair_p.append(delta_p)
                eligible=abs(delta_y)>5
                pair_correct=int(eligible and abs(delta_p)>1 and np.sign(delta_p)==np.sign(delta_y))
                pairs.append({"model":model,"seed":seed,"composition_id":formula,"fold":int(part.fold.iloc[0]),
                              "family_a":FAMILIES[i],"family_b":FAMILIES[j],
                              "delta_dft_mev_atom":delta_y,"delta_pred_mev_atom":delta_p,
                              "eligible_gt5":int(eligible),"correct_gt5":pair_correct,
                              "absolute_error_mev_atom":abs(delta_p-delta_y),
                              "zero_error_mev_atom":abs(delta_y)})
        pair_y=np.array(all_pair_y);pair_p=np.array(all_pair_p)
        eligible=np.abs(pair_y)>5
        pooled.append({"model":model,"seed":seed,"physical_structures":210,"compositions":70,
                       "derived_pairs":210,"eligible_pairs_gt5":int(eligible.sum()),
                       "total_mae_ev":float(np.mean(np.abs(pt-yt))),
                       "total_rmse_ev":float(np.sqrt(np.mean((pt-yt)**2))),
                       "atom_mae_mev":float(np.mean(np.abs(p-y))*1000),
                       "atom_rmse_mev":float(np.sqrt(np.mean((p-y)**2))*1000),
                       "mean_composition_mae_mev":float(np.mean([item["mean_abs_error_mev_atom"] for item in comps if item["model"]==model and item["seed"]==seed])),
                       "residual_mae_mev":float(np.mean(np.abs(np.array(all_pres)-np.array(all_yres)))*1000),
                       "signed_pair_mae_mev":float(np.mean(np.abs(pair_p-pair_y))),
                       "zero_pair_mae_mev":float(np.mean(np.abs(pair_y))),
                       "anchored_delta_mae_mev":float(np.mean(np.abs(all_anch))),
                       "gs_correct":int(sum(gs_correct)),"gs_eligible":int(sum(gs_eligible)),
                       "gs_accuracy":float(sum(gs_correct)/sum(gs_eligible)),
                       "gs_raw_argmin_accuracy":float(np.mean(raw_gs)) if np.isfinite(raw_gs).all() else float("nan"),
                       "pair_correct_gt5":int(np.sum((np.sign(pair_p[eligible])==np.sign(pair_y[eligible]))&(np.abs(pair_p[eligible])>1))),
                       "pair_accuracy_gt5":float(np.mean((np.sign(pair_p[eligible])==np.sign(pair_y[eligible]))&(np.abs(pair_p[eligible])>1))),
                       **{f"atom_{key}":value for key,value in pooled_ols(y,p).items()},
                       **{f"residual_{key}":value for key,value in pooled_ols(np.array(all_yres),np.array(all_pres)).items()},
                       **{f"pair_{key}":value for key,value in pooled_ols(pair_y,pair_p).items()}})
        for low,high in BINS:
            mask=(np.abs(pair_y)>=low if low==0 else np.abs(pair_y)>low)&(np.abs(pair_y)<=high)
            use=mask&(np.abs(pair_y)>5)
            bins.append({"model":model,"seed":seed,"lower_mev":low,"upper_mev":high,
                         "n_pairs":int(mask.sum()),"eligible_pairs":int(use.sum()),
                         "mae_model_mev":float(np.mean(np.abs(pair_p[mask]-pair_y[mask]))) if mask.any() else np.nan,
                         "mae_zero_mev":float(np.mean(np.abs(pair_y[mask]))) if mask.any() else np.nan,
                         "pair_accuracy":float(np.mean((np.sign(pair_p[use])==np.sign(pair_y[use]))&(np.abs(pair_p[use])>1))) if use.any() else np.nan,
                         "sparse_lt10":int(mask.sum()<10)})
    composition=pd.DataFrame(comps);pair=pd.DataFrame(pairs);summary=pd.DataFrame(pooled);binframe=pd.DataFrame(bins)
    pair_composition=pair.groupby(["model","seed","composition_id"],as_index=False).agg(
        pair_mae_mev_atom=("absolute_error_mev_atom","mean"),
        pair_eligible_gt5=("eligible_gt5","sum"),pair_correct_gt5=("correct_gt5","sum"))
    composition=composition.merge(pair_composition,on=["model","seed","composition_id"],validate="one_to_one")
    composition.to_csv(out/"composition_metrics.csv",index=False)
    pair.to_csv(out/"pairs_gt5.csv",index=False)
    summary.to_csv(out/"summary_by_seed.csv",index=False)
    binframe.to_csv(out/"delta_bins.csv",index=False)
    # Paired bootstrap: average training seeds within formula, then resample 70 formulas.
    boot=[];rng=np.random.default_rng(20261028)
    for other in ("random","composition"):
        for metric in ("absolute_mae_mev_atom","mean_abs_error_mev_atom","residual_mae_mev_atom","pair_mae_mev_atom","gs_correct"):
            wide=composition[composition.model.isin(["pretrained",other])].pivot_table(
                index="composition_id",columns="model",values=metric,aggfunc="mean")
            if len(wide)!=70 or wide.isna().any().any():raise ValueError("Bootstrap no emparejado")
            diff=(wide.pretrained-wide[other]).to_numpy(float)
            samples=diff[rng.integers(0,70,size=(2000,70))].mean(axis=1)
            boot.append({"comparison":f"pretrained-minus-{other}","metric":metric,
                         "mean_difference":float(diff.mean()),"ci_low":float(np.percentile(samples,2.5)),
                         "ci_high":float(np.percentile(samples,97.5)),"resampling_unit":"composition",
                         "physical_compositions":70,"seed_repetitions":3})
        paired=composition[composition.model.isin(["pretrained",other])].groupby(
            ["model","composition_id"],as_index=False)[["pair_correct_gt5","pair_eligible_gt5"]].mean()
        p=paired[paired.model=="pretrained"].sort_values("composition_id")
        q=paired[paired.model==other].sort_values("composition_id")
        if p.composition_id.tolist()!=q.composition_id.tolist():raise ValueError("Ranking no emparejado")
        numerator=p.pair_correct_gt5.to_numpy(float)-q.pair_correct_gt5.to_numpy(float)
        denominator=p.pair_eligible_gt5.to_numpy(float)
        if not np.allclose(denominator,q.pair_eligible_gt5):raise ValueError("Pares elegibles distintos")
        sample_idx=rng.integers(0,70,size=(2000,70))
        draws=numerator[sample_idx].sum(axis=1)/denominator[sample_idx].sum(axis=1)
        boot.append({"comparison":f"pretrained-minus-{other}","metric":"pair_accuracy_gt5",
                     "mean_difference":float(numerator.sum()/denominator.sum()),
                     "ci_low":float(np.percentile(draws,2.5)),"ci_high":float(np.percentile(draws,97.5)),
                     "resampling_unit":"composition","physical_compositions":70,"seed_repetitions":3})
    pd.DataFrame(boot).to_csv(out/"paired_bootstrap.csv",index=False)
    result={"status":"PASS","physical_structures":210,"compositions":70,"seeds":[42,123,2026],
            "models":list(KINDS),"historical_test_accessed":False,
            "tables":{path.name:sha256(path) for path in out.glob("*.csv")}}
    (out/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    LOG.info(json.dumps({"event":"analysis","status":"PASS","rows":len(predictions)}))
    return result


def main() -> None:
    """Ejecuta análisis con raíz del repositorio configurable."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    analyze(args.repo.resolve())


if __name__=="__main__":main()
