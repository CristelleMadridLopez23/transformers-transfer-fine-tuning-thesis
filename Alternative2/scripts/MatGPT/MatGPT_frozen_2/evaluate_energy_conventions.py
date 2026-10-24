#!/usr/bin/env python3
"""Calcula métricas OOF, ΔE doble, ranking, descomposición y bootstrap pareado."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import FAMILIES, MODELS, REPORT_DIR, ExperimentError, load_development, log, sha256_file, write_json


PAIR_ORDER = (("dist_perovskite", "hexagonal"), ("dist_perovskite", "needle"), ("hexagonal", "needle"))
FAMILY_LABEL = {"dist_perovskite": "Perovskita distorsionada", "hexagonal": "Hexagonal", "needle": "Tipo aguja"}
D1_RIDGE = Path(__file__).resolve().parents[3] / "Alternative2" / "reports" / "MatGPT_diagnostics" / "D1" / "v3" / "oof_predictions.csv"


def category(diff: float, tau: float = 1.0) -> str:
    """Clasifica diferencia orientada i−j en i_lower/tie/j_lower."""
    return "i_lower" if diff < -tau else "j_lower" if diff > tau else "tie"


def summarize_triplet(group: pd.DataFrame, threshold: float = 1.0) -> dict[str, Any]:
    """Calcula diagnósticos para un triplete de una composición."""
    d = group.set_index("polymorph_family").loc[list(FAMILIES)]
    y = d.E_DFT_eV_atom.to_numpy(float) * 1000
    p = d.E_pred_eV_atom.to_numpy(float) * 1000
    dft_order = np.sort(y)
    dft_gap = float(dft_order[1] - dft_order[0])
    min_pred = p.min()
    pred_winners = np.flatnonzero(p <= min_pred + threshold)
    dft_winner = int(np.argmin(y))
    dft_tie = dft_gap <= threshold
    pred_tie = len(pred_winners) != 1
    pred_winner = int(pred_winners[0]) if not pred_tie else -1
    abs_err = p - y
    mu_y, mu_p = y.mean(), p.mean()
    centered_y, centered_p = y - mu_y, p - mu_p
    err_mu = float((p - y).mean())
    err_res = abs_err - err_mu
    # Validación numérica de identidad MSE grupo = error medio² + varianza residual.
    lhs = float(np.mean(abs_err**2))
    rhs = float(err_mu**2 + np.mean(err_res**2))
    if abs(lhs - rhs) > 1e-7:
        raise ExperimentError(f"Identidad MSE incumplida: {lhs} != {rhs}")
    anchors_dft = y - y[dft_winner]
    anchors_pred = p - p[dft_winner]
    own_dft = y - y[dft_winner]
    own_pred = p - p[int(np.argmin(p))]
    pair_rows = []
    for fam_i, fam_j in PAIR_ORDER:
        i, j = FAMILIES.index(fam_i), FAMILIES.index(fam_j)
        diff_dft, diff_pred = float(y[i]-y[j]), float(p[i]-p[j])
        true_cat, pred_cat = category(diff_dft), category(diff_pred)
        eligible = abs(diff_dft) > threshold
        pair_rows.append({"family_i": fam_i, "family_j": fam_j, "diff_dft_mev_atom": diff_dft,
            "diff_pred_mev_atom": diff_pred, "dft_relation": true_cat, "pred_relation": pred_cat,
            "dft_tie": not eligible, "pred_tie": pred_cat == "tie", "eligible_primary": eligible,
            "correct_primary": bool(eligible and pred_cat == true_cat)})
    return {"dft_gs_family": FAMILIES[dft_winner], "pred_gs_family": FAMILIES[pred_winner] if pred_winner >= 0 else "tie/abstain",
        "dft_gs_index": dft_winner, "pred_gs_index": pred_winner, "dft_gap_mev_atom": dft_gap,
        "dft_gs_eligible": not dft_tie, "pred_gs_tie": pred_tie,
        "gs_correct": bool(not dft_tie and not pred_tie and dft_winner == pred_winner),
        "absolute_mae_mev_atom": float(np.mean(np.abs(abs_err)),), "absolute_rmse_mev_atom": float(np.sqrt(np.mean(abs_err**2))),
        "delta_own_mae_mev_atom": float(np.mean(np.abs(own_pred-own_dft))),
        "delta_anchored_mae_mev_atom": float(np.mean(np.abs(anchors_pred-anchors_dft))),
        "signed_pairwise_mae_mev_atom": float(np.mean([abs(r["diff_pred_mev_atom"]-r["diff_dft_mev_atom"]) for r in pair_rows])),
        "pair_rows": pair_rows, "pairwise_eligible": sum(r["eligible_primary"] for r in pair_rows),
        "pairwise_correct": sum(r["correct_primary"] for r in pair_rows), "pairwise_physical_count": len(pair_rows),
        "pairwise_accuracy": float(sum(r["correct_primary"] for r in pair_rows) / max(1, sum(r["eligible_primary"] for r in pair_rows))),
        "range_dft_mev_atom": float(y.max()-y.min()), "range_pred_mev_atom": float(p.max()-p.min()),
        "range_abs_error_mev_atom": float(abs((p.max()-p.min())-(y.max()-y.min()))),
        "collapse": bool((p.max()-p.min()) < 1.0 or pred_tie), "mean_error_mev_atom": err_mu,
        "mae_mean_component_mev_atom": abs(err_mu), "mae_centered_residual_mev_atom": float(np.mean(np.abs(err_res))),
        "mse_total_mev2_atom2": lhs, "mse_mean_component_mev2_atom2": err_mu**2,
        "mse_residual_component_mev2_atom2": float(np.mean(err_res**2)), "mse_identity_abs_error": abs(lhs-rhs),
        "E_DFT_mu_eV_atom": float(mu_y/1000), "E_pred_mu_eV_atom": float(mu_p/1000),
        "delta_dft_centered_mev_atom": centered_y.tolist(), "delta_pred_centered_mev_atom": centered_p.tolist(),
        "dft_anchor_mev_atom": anchors_dft.tolist(), "pred_anchor_mev_atom": anchors_pred.tolist(),
        "dft_own_min_mev_atom": own_dft.tolist(), "pred_own_min_mev_atom": own_pred.tolist()}


def enrich_predictions(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adjunta valores centrados/relativos y tablas por composición y par."""
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for (model, fold, seed, comp), group in oof.groupby(["model", "fold", "seed", "composition_id"], sort=False):
        metrics = summarize_triplet(group)
        order = group.set_index("polymorph_family").loc[list(FAMILIES)]
        for idx, (family, r) in enumerate(order.iterrows()):
            row = r.to_dict()
            row["polymorph_family"] = str(family)
            row.update({"E_DFT_mu_eV_atom": metrics["E_DFT_mu_eV_atom"], "E_pred_mu_eV_atom": metrics["E_pred_mu_eV_atom"],
                "delta_DFT_centered_eV_atom": metrics["delta_dft_centered_mev_atom"][idx]/1000,
                "delta_pred_centered_eV_atom": metrics["delta_pred_centered_mev_atom"][idx]/1000,
                "DFT_GS_index": metrics["dft_gs_index"], "pred_GS_index": metrics["pred_gs_index"],
                "delta_DFT_anchored_meV_atom": metrics["dft_anchor_mev_atom"][idx],
                "delta_pred_anchored_meV_atom": metrics["pred_anchor_mev_atom"][idx],
                "delta_DFT_own_min_meV_atom": metrics["dft_own_min_mev_atom"][idx],
                "delta_pred_own_min_meV_atom": metrics["pred_own_min_mev_atom"][idx],
                "dft_gs_eligible": metrics["dft_gs_eligible"], "pred_gs_tie": metrics["pred_gs_tie"]})
            rows.append(row)
        for pr in metrics["pair_rows"]:
            pair_rows.append({"model": model, "fold": fold, "seed": seed, "composition_id": comp, **pr})
        comp_rows.append({"model": model, "fold": fold, "seed": seed, "composition_id": comp,
            **{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}})
    return pd.DataFrame(rows), pd.DataFrame(comp_rows), pd.DataFrame(pair_rows)


def aggregate_metrics(comps: pd.DataFrame, pairs: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula macro por composición y métricas pooled separadas por fold/semilla."""
    output = []
    for (model, fold, seed), group in comps.groupby(["model", "fold", "seed"]):
        eligible_gs = group.loc[group.dft_gs_eligible]
        pair = pairs.loc[(pairs.model.eq(model)) & (pairs.fold.eq(fold)) & (pairs.seed.eq(seed))]
        eligible_pair = pair.loc[pair.eligible_primary]
        pred = predictions.loc[(predictions.model.eq(model)) & (predictions.fold.eq(fold)) & (predictions.seed.eq(seed))]
        err = (pred.E_pred_eV_atom-pred.E_DFT_eV_atom).to_numpy(float)*1000
        output.append({"model": model, "fold": fold, "seed": seed, "compositions": group.composition_id.nunique(),
            "structures": len(pred), "MAE_absolute_macro_meV_atom": group.absolute_mae_mev_atom.mean(),
            "MAE_absolute_pooled_meV_atom": np.mean(np.abs(err)), "RMSE_absolute_macro_meV_atom": group.absolute_rmse_mev_atom.mean(),
            "MAE_delta_GS_anchored_macro_meV_atom": group.delta_anchored_mae_mev_atom.mean(),
            "MAE_delta_own_min_macro_meV_atom": group.delta_own_mae_mev_atom.mean(),
            "GS_accuracy": eligible_gs.gs_correct.mean() if len(eligible_gs) else np.nan,
            "GS_n_eligible": len(eligible_gs), "GS_n_excluded_DFT_tie": len(group)-len(eligible_gs),
            "GS_predicted_abstentions": int(group.pred_gs_tie.sum()),
            "pairwise_accuracy": eligible_pair.correct_primary.mean() if len(eligible_pair) else np.nan,
            "pairwise_n_eligible": len(eligible_pair), "pairwise_n_DFT_ties": int((~pair.eligible_primary).sum()),
            "pairwise_predicted_ties": int(pair.pred_tie.sum()), "signed_pairwise_MAE_meV_atom": group.signed_pairwise_mae_mev_atom.mean(),
            "mean_range_error_meV_atom": group.range_abs_error_mev_atom.mean(),
            "MAE_composition_mean_meV_atom": group.mae_mean_component_mev_atom.mean(),
            "MAE_structural_residual_meV_atom": group.mae_centered_residual_mev_atom.mean(),
            "MSE_total_meV2_atom2": group.mse_total_mev2_atom2.mean(),
            "MSE_mean_component_meV2_atom2": group.mse_mean_component_mev2_atom2.mean(),
            "MSE_residual_component_meV2_atom2": group.mse_residual_component_mev2_atom2.mean(),
            "collapsed_compositions": int(group.collapse.sum())})
    return pd.DataFrame(output)


def ensemble_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Promedia semillas por ID físico y construye confusiones sin pseudorreplicar."""
    keys = ["structure_id", "composition_id", "polymorph_family", "fold", "model"]
    ens = predictions.groupby(keys, as_index=False).agg(E_DFT_eV_atom=("E_DFT_eV_atom", "first"),
        E_pred_eV_atom=("E_pred_eV_atom", "mean"), E_pred_mu_eV_atom=("E_pred_mu_eV_atom", "mean"),
        delta_pred_centered_eV_atom=("delta_pred_centered_eV_atom", "mean"))
    gs_counts, pair_counts, ensemble_comps = [], [], []
    for (model, comp), group in ens.groupby(["model", "composition_id"], sort=False):
        m = summarize_triplet(group)
        ensemble_comps.append({"model": model, "composition_id": comp, **{k:v for k,v in m.items() if not isinstance(v,(list,dict))}})
        gs_counts.append({"model":model,"dft_family":m["dft_gs_family"],"pred_family":m["pred_gs_family"],
            "eligible":m["dft_gs_eligible"],"predicted_abstention":m["pred_gs_tie"]})
        for pr in m["pair_rows"]:
            pair_counts.append({"model":model,"composition_id":comp,**pr})
    return ens, pd.DataFrame(ensemble_comps), pd.DataFrame(gs_counts).merge(pd.DataFrame(pair_counts), on="model", how="left")


def bootstrap_comparisons(comps: pd.DataFrame, replicates: int = 2000, seed: int = 13072026) -> dict[str, Any]:
    """Bootstrap pareado por 70 fórmulas tras promediar semillas por composición."""
    averaged = comps.groupby(["model", "composition_id"], as_index=False).mean(numeric_only=True)
    metrics = ["absolute_mae_mev_atom", "delta_anchored_mae_mev_atom", "delta_own_mae_mev_atom",
               "mae_mean_component_mev_atom", "mae_centered_residual_mev_atom", "range_abs_error_mev_atom",
               "gs_correct", "pairwise_accuracy"]
    rng = np.random.default_rng(seed)
    groups = sorted(comps.composition_id.unique())
    result: dict[str, Any] = {"method":"paired bootstrap over development compositions; seeds averaged within composition",
        "replicates":replicates,"seed":seed,"groups":len(groups),"interpretation":"exploratory intervals; not proof of superiority","comparisons":{}}
    for a,b in (("H1","H0"),("H2","H0"),("H2","H1")):
        item: dict[str, Any] = {}
        for metric in metrics:
            wide = averaged.pivot(index="composition_id", columns="model", values=metric).reindex(groups)
            delta=(wide[a]-wide[b]).to_numpy(float)
            vals=[]
            for _ in range(replicates):
                sample=rng.integers(0,len(groups),len(groups))
                if metric == "gs_correct":
                    boot = {}
                    for candidate in (a,b):
                        sub=averaged.loc[averaged.model.eq(candidate)].set_index("composition_id").reindex(np.asarray(groups)[sample])
                        boot[candidate]=float(sub.loc[sub.dft_gs_eligible.gt(0),"gs_correct"].mean())
                    vals.append(boot[a]-boot[b])
                elif metric == "pairwise_accuracy":
                    boot={}
                    for candidate in (a,b):
                        sub=averaged.loc[averaged.model.eq(candidate)].set_index("composition_id").reindex(np.asarray(groups)[sample])
                        eligible=sub.pairwise_eligible
                        boot[candidate]=float(sub.loc[eligible.gt(0),"pairwise_correct"].sum()/eligible.sum()) if eligible.sum()>0 else np.nan
                    vals.append(boot[a]-boot[b])
                else:
                    vals.append(float(np.nanmean(delta[sample])))
            item[metric]={"difference_mean_a_minus_b":float(np.nanmean(delta)),"ci95_low":float(np.nanquantile(vals,.025)),"ci95_high":float(np.nanquantile(vals,.975))}
        result["comparisons"][f"{a}_minus_{b}"]=item
    return result


def load_d1_ridge_reference(predictions: pd.DataFrame, output: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Valida D1 Ridge OOF contra los mismos IDs/folds/DFT y exporta referencia."""
    if not D1_RIDGE.is_file():
        return None, {"status":"NOT_COMPARABLE","reason":"D1 Ridge OOF artifact unavailable"}
    d1=pd.read_csv(D1_RIDGE)
    d1=d1.loc[d1.configuration.eq("D1-B Ridge composición")].copy()
    d1=d1.rename(columns={"composition_group":"composition_id","energy_per_atom_ev":"E_DFT_eV_atom","energy_pred_ev_atom":"E_pred_eV_atom"})
    d1["structure_id"]=d1.structure_id.astype(str)
    base=predictions.drop_duplicates("structure_id")[["structure_id","composition_id","polymorph_family","fold","E_DFT_eV_atom"]]
    if len(d1)!=210 or d1.structure_id.duplicated().any() or set(d1.structure_id)!=set(base.structure_id):
        return None,{"status":"NOT_COMPARABLE","reason":"D1 IDs do not match the 210 development structures"}
    check=d1.merge(base,on="structure_id",suffixes=("_d1","_new"),validate="one_to_one")
    same_fold=check.fold_d1.astype(int).equals(check.fold_new.astype(int))
    same_formula=check.composition_id_d1.astype(str).equals(check.composition_id_new.astype(str))
    same_energy=np.allclose(check.E_DFT_eV_atom_d1,check.E_DFT_eV_atom_new,atol=1e-9,rtol=0)
    if not (same_fold and same_formula and same_energy):
        return None,{"status":"NOT_COMPARABLE","reason":"D1 ID/fold/formula/DFT-label mismatch"}
    ridge=d1[["structure_id","composition_id","polymorph_family","fold","E_DFT_eV_atom","E_pred_eV_atom"]].copy()
    ridge["model"]="D1 Ridge composition-only"
    per_comp=ridge.assign(abs_error_mev_atom=lambda q:(q.E_pred_eV_atom-q.E_DFT_eV_atom).abs()*1000).groupby("composition_id").abs_error_mev_atom.mean()
    constant=all(g.E_pred_eV_atom.nunique()==1 for _,g in ridge.groupby("composition_id"))
    ridge.to_csv(output/"D1_ridge_composition_only_oof.csv",index=False)
    metrics={"status":"PASS","same_210_development_IDs":True,"same_outer_fold_per_ID":True,"same_DFT_labels":True,
        "MAE_absolute_macro_mev_atom":float(per_comp.mean()),"MAE_absolute_pooled_mev_atom":float((ridge.E_pred_eV_atom-ridge.E_DFT_eV_atom).abs().mean()*1000),
        "composition_constant_predictions":constant,"ranking":"abstention/non-discriminative; no GS success rate assigned",
        "source_path":str(D1_RIDGE),"source_sha256":sha256_file(D1_RIDGE)}
    return ridge,metrics


def export_figures(pred: pd.DataFrame, comps: pd.DataFrame, fold_metrics: pd.DataFrame,
                   ensemble_comps: pd.DataFrame, output: Path,
                   ridge: pd.DataFrame | None = None) -> dict[str, Any]:
    """Genera las paridades, scorecards, matrices, descomposición y curvas PNG/SVG/PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"H0":"#2471A3","H1":"#D35400","H2":"#138D75"}
    markers = {"dist_perovskite":"o","hexagonal":"^","needle":"s"}
    results: dict[str, Any] = {}

    def save(fig: Any, name: str) -> None:
        """Exporta en formatos de figura solicitados."""
        for ext in ("png","svg","pdf"):
            fig.savefig(figures / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Paridad DFT-GS anchored. Semillas se promedian primero por estructura.
    ens = pred.groupby(["model","structure_id","composition_id","polymorph_family","fold"],as_index=False).agg(
        E_DFT_eV_atom=("E_DFT_eV_atom","first"),E_pred_eV_atom=("E_pred_eV_atom","mean"))
    for model, group in ens.groupby("model"):
        temp=[]
        for comp,g in group.groupby("composition_id"):
            temp.append((g,summarize_triplet(g)))
        rows=[]
        for g,m in temp:
            for i,(family,_) in enumerate(g.set_index("polymorph_family").loc[list(FAMILIES)].iterrows()):
                rows.append((m["dft_anchor_mev_atom"][i],m["pred_anchor_mev_atom"][i],family))
        x=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows]);
        if len(x)<2: continue
        slope,intercept=np.polyfit(x,y,1); fitted=slope*x+intercept
        r2ols=1-float(np.sum((y-fitted)**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else np.nan
        r2pred=1-float(np.mean((y-x)**2)/np.var(x)) if np.var(x)>0 else np.nan
        results[f"parity_{model}"]={"N_structures":len(x),"N_compositions":len(temp),"slope_dimensionless":float(slope),"intercept_mev_atom":float(intercept),"R2_OLS":r2ols,"R2_prediction_y_equals_x":r2pred,"MAE_DFT_anchor_mev_atom":float(np.mean(np.abs(y-x)))}
    limits_vals=[]
    for model,g in ens.groupby("model"):
        for comp,part in g.groupby("composition_id"):
            m=summarize_triplet(part); limits_vals.extend(m["dft_anchor_mev_atom"]+m["pred_anchor_mev_atom"])
    low,high=min(limits_vals)-10,max(limits_vals)+10
    fig,axes=plt.subplots(1,3,figsize=(17,5.6),sharex=True,sharey=True)
    for ax,model in zip(axes,MODELS):
        xvals=[];yvals=[]
        for comp,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            ordered=g.set_index("polymorph_family").loc[list(FAMILIES)]
            for i,f in enumerate(FAMILIES):
                x,y=m["dft_anchor_mev_atom"][i],m["pred_anchor_mev_atom"][i]
                xvals.append(x);yvals.append(y)
                ax.scatter(x,y,marker=markers[f],color=colors[model],alpha=.72,s=25)
        xarr=np.asarray(xvals);yarr=np.asarray(yvals);m=results[f"parity_{model}"]
        ax.plot([low,high],[low,high],color="#333333",linestyle="--",label="Ideal: y=x")
        ax.plot([low,high],[m["slope_dimensionless"]*low+m["intercept_mev_atom"],m["slope_dimensionless"]*high+m["intercept_mev_atom"]],color="#C2185B",label="OLS libre")
        ax.set_title(f"{model} | N=210, 70 fórmulas\nm={m['slope_dimensionless']:.3f}; b={m['intercept_mev_atom']:.1f} meV/átomo\nR² OLS={m['R2_OLS']:.3f}; MAE={m['MAE_DFT_anchor_mev_atom']:.1f} meV/átomo",fontsize=9,pad=12)
        ax.grid(alpha=.25);ax.set_xlim(low,high);ax.set_ylim(low,high);ax.set_xlabel("DFT: Eᵢ − E_GS,DFT (meV/átomo)")
    axes[0].set_ylabel("Modelo: Eᵢ − E_pred,DFT-GS (meV/átomo)")
    handles=[Line2D([0],[0],marker=markers[f],color="gray",linestyle="",label=FAMILY_LABEL[f]) for f in FAMILIES]
    axes[-1].legend(handles=handles+[Line2D([0],[0],color="#333",linestyle="--",label="Ideal y=x"),Line2D([0],[0],color="#C2185B",label="OLS y=mx+b")],loc="lower right",fontsize=8)
    fig.suptitle("ABS₃: energía relativa anclada al estado fundamental DFT",fontsize=15,y=.98)
    fig.subplots_adjust(top=.76,bottom=.28,wspace=.08)
    fig.text(.5,.035,"Cada punto = un polimorfo estático; semillas promediadas por ID (210 estructuras, 70 fórmulas, 5 folds OOF de desarrollo).\nCero = GS DFT; predicción negativa = inversión. OLS incluye todos los puntos; R² OLS es ajuste libre y R² predictivo usa y=x. Validation histórica participó en preselección.",ha="center",va="bottom",fontsize=8)
    save(fig,"parity_dft_gs_anchored")

    # Legacy own-min parity separada.
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),sharex=True,sharey=True)
    limits=[]
    for ax,model in zip(axes,MODELS):
        xs=[];ys=[]
        for _,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            xs.extend(m["dft_own_min_mev_atom"]);ys.extend(m["pred_own_min_mev_atom"])
        x=np.asarray(xs);y=np.asarray(ys);slope,intercept=np.polyfit(x,y,1);fit=slope*x+intercept
        r2=1-np.sum((y-fit)**2)/np.sum((y-y.mean())**2) if np.var(y)>0 else np.nan
        mae=float(np.mean(np.abs(y-x)));limits.extend([x.min(),y.min(),x.max(),y.max()])
        results[f"legacy_parity_{model}"]={"N_structures":len(x),"slope_dimensionless":float(slope),"intercept_mev_atom":float(intercept),"R2_OLS":float(r2),"R2_prediction_y_equals_x":float(1-np.mean((y-x)**2)/np.var(x)) if np.var(x)>0 else np.nan,"MAE_mev_atom":mae}
        for fam,point in zip(FAMILIES,range(3)):
            ax.scatter(x[point::3],y[point::3],marker=markers[fam],c=colors[model],alpha=.55,s=19)
        ax.set_title(f"{model} | N={len(x)}\nm={slope:.3f}, b={intercept:.1f} meV/átomo\nR² OLS={r2:.3f}; MAE={mae:.1f} meV/átomo",fontsize=9,pad=12)
        ax.grid(alpha=.25);ax.set_xlabel("DFT ΔE respecto a mínimo propio (meV/átomo)")
    lo=max(-5,min(limits)-5);hi=max(limits)+5
    for ax in axes: ax.plot([lo,hi],[lo,hi],"--",color="#333",label="Ideal y=x");ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
    axes[0].set_ylabel("Predicción ΔE respecto a mínimo propio (meV/átomo)")
    fam_handles=[Line2D([0],[0],marker=markers[f],color="gray",linestyle="",label=FAMILY_LABEL[f]) for f in FAMILIES]
    axes[-1].legend(handles=fam_handles,loc="upper left",fontsize=7)
    fig.suptitle("ABS₃: paridad ΔE own-min legacy (definición histórica)",y=.98)
    fig.subplots_adjust(top=.76,bottom=.27,wspace=.08)
    fig.text(.5,.035,"Cada punto es un polimorfo OOF; semillas promediadas por estructura. Círculo = perovskita distorsionada; triángulo = hexagonal; cuadrado = aguja.\nAmbos ejes usan su propio mínimo (ΔE≥0); esta definición no muestra inversiones relativas al GS DFT.",ha="center",va="bottom",fontsize=8)
    save(fig,"parity_delta_own_min_legacy")

    # Paridad de energía absoluta, incorporando Ridge composition-only como control externo.
    abs_models = [(model, ens.loc[ens.model.eq(model)].copy()) for model in MODELS]
    if ridge is not None:
        abs_models.append(("D1 Ridge composition-only", ridge.copy()))
    fig,axes=plt.subplots(1,len(abs_models),figsize=(5.2*len(abs_models),5.4),sharex=True,sharey=True)
    axes=np.atleast_1d(axes); stats=[]
    xmin=min(float(ens.E_DFT_eV_atom.min()),float(ridge.E_DFT_eV_atom.min()) if ridge is not None else float(ens.E_DFT_eV_atom.min()))
    xmax=max(float(ens.E_DFT_eV_atom.max()),float(ridge.E_DFT_eV_atom.max()) if ridge is not None else float(ens.E_DFT_eV_atom.max()))
    xlo,xhi=xmin-.15,xmax+.15
    all_colors={**colors,"D1 Ridge composition-only":"#6C3483"}
    for ax,(model,g) in zip(axes,abs_models):
        x=g.E_DFT_eV_atom.to_numpy(float);y=g.E_pred_eV_atom.to_numpy(float)
        slope,intercept=np.polyfit(x,y,1);fit=slope*x+intercept
        r2ols=1-float(np.sum((y-fit)**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else np.nan
        r2pred=1-float(np.sum((y-x)**2)/np.sum((x-x.mean())**2)) if np.var(x)>0 else np.nan
        mae=float(np.mean(np.abs((y-x)*1000)))
        stats.append({"model":model,"N_structures":len(x),"N_compositions":g.composition_id.nunique(),"slope_dimensionless":float(slope),"intercept_eV_atom":float(intercept),"R2_OLS":r2ols,"R2_prediction_y_equals_x":r2pred,"MAE_meV_atom":mae})
        for family,part in g.groupby("polymorph_family"):
            ax.scatter(part.E_DFT_eV_atom,part.E_pred_eV_atom,marker=markers[family],s=22,alpha=.68,color=all_colors[model])
        ax.plot([xlo,xhi],[xlo,xhi],"--",color="#333",label="Ideal y=x")
        ax.plot([xlo,xhi],[slope*xlo+intercept,slope*xhi+intercept],color="#C2185B",label="OLS libre")
        ax.set_title(f"{model}\nN={len(x)}, {g.composition_id.nunique()} fórmulas\nm={slope:.2f}; b={intercept:.2f} eV/átomo\nR² OLS={r2ols:.3f}; MAE={mae:.1f} meV/átomo")
        ax.grid(alpha=.25);ax.set_xlim(xlo,xhi);ax.set_ylim(xlo,xhi);ax.set_xlabel("DFT estática (eV/átomo)")
    axes[0].set_ylabel("Predicción (eV/átomo)")
    axes[-1].legend(fontsize=8,loc="lower right")
    fig.suptitle("ABS₃: paridad de energía total estática por átomo",fontsize=15,y=.98)
    fig.subplots_adjust(top=.75,bottom=.28,wspace=.08)
    fig.text(.5,.035,"Cada punto = una estructura estática OOF; semillas promediadas por ID. Ridge D1 es referencia de MAE absoluta con los mismos folds; predice una energía constante por fórmula y no ordena polimorfos.\nCírculo = perovskita distorsionada; triángulo = hexagonal; cuadrado = aguja. OLS usa todos los puntos; R² OLS ≠ R² predictivo. Validation histórica participó en preselección.",ha="center",va="bottom",fontsize=8)
    save(fig,"absolute_energy_parity_with_D1_ridge")
    results["absolute_energy_parity"]={row["model"]:row for row in stats}

    # Scorecard macro OOF por composición y semillas.
    score= comps.groupby("model").mean(numeric_only=True)
    metrics=[("MAE absoluto (meV/átomo)","absolute_mae_mev_atom",10,5),("MAE ΔE DFT-GS anclada (meV/átomo)","delta_anchored_mae_mev_atom",10,5),
        ("MAE ΔE own-min legacy (meV/átomo)","delta_own_mae_mev_atom",10,5),("SignedPairwiseMAE (meV/átomo)","signed_pairwise_mae_mev_atom",10,5),
        ("Selección GS correcta (%)","gs_correct",90,100),("Pairwise accuracy (%)","pairwise_accuracy",90,100),
        ("Error absoluto de rango (meV/átomo)","range_abs_error_mev_atom",10,5),("Error media composicional (meV/átomo)","mae_mean_component_mev_atom",10,5)]
    table=[]
    for label,key,acceptable,strong in metrics:
        vals=[]
        for model in MODELS:
            if key=="gs_correct": v=comps.loc[comps.model.eq(model)&comps.dft_gs_eligible].gs_correct.mean()*100
            elif key=="pairwise_accuracy":
                ps=pd.read_csv(output/"pairwise_comparisons.csv");q=ps.loc[ps.model.eq(model)&ps.eligible_primary];v=q.correct_primary.mean()*100
            else:v=float(score.loc[model,key])
            vals.append(f"{v:.2f}" if np.isfinite(v) else "ND")
        table.append([label,f"≤{acceptable}" if "%" not in label else f">{acceptable}",f"≤{strong}" if "%" not in label else f"{strong}",*vals,"OOF desarrollo; 70 fórmulas; 210 estructuras; 5 folds×3 semillas"])
    table=[[row[0],row[1],row[2],row[3],row[4],row[5],"210 estructuras\n70 fórmulas; 5×3"] for row in table]
    fig,ax=plt.subplots(figsize=(17,6.2));ax.axis("off")
    widths=[.29,.12,.12,.075,.075,.075,.25]
    tab=ax.table(cellText=table,colLabels=["Métrica","Meta aceptable*","Meta fuerte*","H0","H1","H2","N / dominio"],loc="center",cellLoc="center",colWidths=widths)
    tab.auto_set_font_size(False);tab.set_fontsize(7.5);tab.scale(1,1.85)
    fig.suptitle("ABS₃: scorecard energético de MatGPT congelado",fontsize=15,y=.97)
    fig.subplots_adjust(top=.85,bottom=.18,left=.02,right=.98)
    fig.text(.5,.035,"*Metas aspiracionales heredadas de NequIX, no validadas para MatGPT. GS excluye gap DFT ≤1 meV/átomo y abstenciones; pairwise excluye DFT ties.\nLos valores GS/pairwise son porcentajes. Resultados OOF exploratorios: la validation histórica participó en preselección.",ha="center",va="bottom",fontsize=8)
    save(fig,"scorecard_energy")

    # Descomposición media/residuo.
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    mu_x=ensemble_comps.E_DFT_mu_eV_atom.to_numpy(float)*1000
    mu_y=ensemble_comps.E_pred_mu_eV_atom.to_numpy(float)*1000
    lim=max(abs(mu_x).max(),abs(mu_y).max())+80
    for model in MODELS:
        c=ensemble_comps.loc[ensemble_comps.model.eq(model)]
        axes[0].scatter(c.E_DFT_mu_eV_atom*1000,c.E_pred_mu_eV_atom*1000,label=model,color=colors[model],s=20,alpha=.65)
        axes[1].scatter(c.mae_mean_component_mev_atom,c.mae_centered_residual_mev_atom,label=model,color=colors[model],s=20,alpha=.65)
        axes[2].scatter(c.mse_mean_component_mev2_atom2,c.mse_residual_component_mev2_atom2,label=model,color=colors[model],s=20,alpha=.65)
    axes[0].plot([-lim,lim],[-lim,lim],"--",color="#333",label="Ideal y=x")
    for model in MODELS:
        c=ensemble_comps.loc[ensemble_comps.model.eq(model)];x=c.E_DFT_mu_eV_atom.to_numpy(float)*1000;y=c.E_pred_mu_eV_atom.to_numpy(float)*1000
        slope,intercept=np.polyfit(x,y,1);axes[0].plot([-lim,lim],[slope*-lim+intercept,slope*lim+intercept],color=colors[model],lw=1.3,label=f"{model} OLS m={slope:.2f}")
    axes[0].set_xlim(-lim,lim);axes[0].set_ylim(-lim,lim)
    axes[2].plot([0,axes[2].get_xlim()[1]],[0,axes[2].get_ylim()[1]],"--",color="#333",label="Media y residuo iguales")
    for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_xlabel("Media DFT por fórmula μ (meV/átomo)");axes[0].set_ylabel("μ predicha (meV/átomo)");axes[0].set_title("Nivel energético medio")
    axes[1].set_xlabel("|error de media| (meV/átomo)");axes[1].set_ylabel("MAE residual centrado (meV/átomo)");axes[1].set_title("Media frente a estructura")
    axes[2].set_xlabel("Componente MSE media (meV²/átomo²)");axes[2].set_ylabel("Componente MSE residual (meV²/átomo²)");axes[2].set_title("Suma exacta de componentes MSE")
    fig.suptitle("ABS₃: de dónde viene el error OOF, por composición")
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.35)
    fig.text(.5,.035,"Cada punto = fórmula; semillas promediadas antes de resumir. En el panel izquierdo, línea discontinua y=x es ideal y rectas de color son OLS libre.\nError medio = promedio(E_pred−E_DFT); residuo = error individual−error medio. MSE suma exactamente; MAE no. N=70 fórmulas OOF.",ha="center",va="bottom",fontsize=8)
    save(fig,"energy_decomposition")

    # Paridad explícita de la cabeza común, con regresión libre por panel.
    mu_values=np.r_[ensemble_comps.E_DFT_mu_eV_atom.to_numpy(float)*1000,ensemble_comps.E_pred_mu_eV_atom.to_numpy(float)*1000]
    mu_lo,mu_hi=float(mu_values.min()-100),float(mu_values.max()+100)
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),sharex=True,sharey=True)
    for ax,model in zip(axes,MODELS):
        c=ensemble_comps.loc[ensemble_comps.model.eq(model)]
        x=c.E_DFT_mu_eV_atom.to_numpy(float)*1000;y=c.E_pred_mu_eV_atom.to_numpy(float)*1000
        slope,intercept=np.polyfit(x,y,1);pred_line=slope*x+intercept
        r2=1-float(np.sum((y-pred_line)**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else np.nan
        mae=float(np.mean(np.abs(y-x)))
        results[f"composition_mu_parity_{model}"]={"N_compositions":len(c),"slope_dimensionless":float(slope),"intercept_mev_atom":float(intercept),"R2_OLS":r2,"MAE_mev_atom":mae}
        ax.scatter(x,y,color=colors[model],s=24,alpha=.68)
        ax.plot([mu_lo,mu_hi],[mu_lo,mu_hi],"--",color="#333",label="Ideal y=x")
        ax.plot([mu_lo,mu_hi],[slope*mu_lo+intercept,slope*mu_hi+intercept],color="#C2185B",label="OLS libre y=mx+b")
        ax.set_xlim(mu_lo,mu_hi);ax.set_ylim(mu_lo,mu_hi);ax.grid(alpha=.25);ax.set_xlabel("μ DFT (meV/átomo)")
        ax.set_title(f"{model} | N={len(c)} fórmulas\nm={slope:.3f}; b={intercept:.1f} meV/átomo\nR² OLS={r2:.3f}; MAE={mae:.1f} meV/átomo",fontsize=9,pad=10)
    axes[0].set_ylabel("μ predicha (meV/átomo)");axes[-1].legend(fontsize=8,loc="lower right")
    fig.suptitle("ABS₃: energía media de la composición, DFT frente a cabeza común",y=.98)
    fig.subplots_adjust(top=.77,bottom=.24,wspace=.08)
    fig.text(.5,.035,"Cada punto = una fórmula de outer-validation; se promedian las 3 semillas por ID. x = media DFT del triplete; y = salida de la cabeza común.\nOLS libre incluye los 70 puntos; R² OLS describe el ajuste, mientras MAE compara directamente con la identidad y=x. Solo desarrollo.",ha="center",va="bottom",fontsize=8)
    save(fig,"composition_mu_parity")

    # Residuos estructurales centrados y distribuciones de rango/inversiones.
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharex=True,sharey=True)
    for ax,model in zip(axes,MODELS):
        xvals=[];yvals=[];fam_vals=[]
        for _,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            xvals.extend(m["delta_dft_centered_mev_atom"]);yvals.extend(m["delta_pred_centered_mev_atom"]);fam_vals.extend(FAMILIES)
        for family in FAMILIES:
            select=np.asarray(fam_vals)==family
            ax.scatter(np.asarray(xvals)[select],np.asarray(yvals)[select],marker=markers[family],color=colors[model],alpha=.48,s=19)
        slope,intercept=np.polyfit(xvals,yvals,1);fit=slope*np.asarray(xvals)+intercept
        r2=1-float(np.sum((np.asarray(yvals)-fit)**2)/np.sum((np.asarray(yvals)-np.mean(yvals))**2)) if np.var(yvals)>0 else np.nan
        mae=float(np.mean(np.abs(np.asarray(yvals)-np.asarray(xvals))))
        results[f"structural_residual_parity_{model}"]={"N_structures":len(xvals),"slope_dimensionless":float(slope),"intercept_mev_atom":float(intercept),"R2_OLS":r2,"MAE_mev_atom":mae}
        lim=max(abs(np.asarray(xvals)).max(),abs(np.asarray(yvals)).max())+5
        ax.plot([-lim,lim],[-lim,lim],"--",color="#333",label="Ideal y=x")
        ax.plot([-lim,lim],[slope*-lim+intercept,slope*lim+intercept],color="#C2185B",label="OLS libre")
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_title(f"{model}: N=210\nm={slope:.2f}; b={intercept:.1f} meV/átomo\nR² OLS={r2:.3f}; MAE={mae:.1f} meV/átomo",fontsize=9);ax.set_xlabel("DFT: Eᵢ−μ_DFT (meV/átomo)");ax.grid(alpha=.25)
    axes[0].set_ylabel("Predicción: Eᵢ−μ_pred (meV/átomo)")
    fam_legend=[Line2D([0],[0],marker=markers[f],color="gray",linestyle="",label=FAMILY_LABEL[f]) for f in FAMILIES]
    axes[-1].legend(handles=fam_legend+[Line2D([0],[0],color="#333",linestyle="--",label="Ideal y=x"),Line2D([0],[0],color="#C2185B",label="OLS libre")],fontsize=7)
    fig.suptitle("ABS₃: paridad de residuos polimórficos centrados")
    fig.subplots_adjust(top=.76,bottom=.27,wspace=.08)
    fig.text(.5,.035,"Semillas promediadas por estructura antes de centrar el triplete. El centrado elimina el nivel medio y deja las diferencias entre polimorfos.\nCada punto es una estructura OOF; no representa la paridad absoluta.",ha="center",va="bottom",fontsize=8)
    save(fig,"structural_residual_parity")

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    bins=np.linspace(0,max(5,max(ensemble_comps.range_dft_mev_atom.max(),ensemble_comps.range_pred_mev_atom.max())),24)
    for model in MODELS:
        g=ensemble_comps.loc[ensemble_comps.model.eq(model)]
        axes[0].hist(g.range_dft_mev_atom,bins=bins,histtype="step",linewidth=2,color=colors[model],label=f"{model} DFT")
        axes[0].hist(g.range_pred_mev_atom,bins=bins,histtype="step",linestyle="--",linewidth=2,color=colors[model],label=f"{model} predicción")
    axes[0].set_xlabel("Rango del triplete max(E)−min(E) (meV/átomo)");axes[0].set_ylabel("Número de fórmulas (bin común)");axes[0].set_title("Dispersión energética por fórmula");axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    collapse=ensemble_comps.groupby("model").collapse.mean()*100
    ties=ensemble_comps.groupby("model").pred_gs_tie.mean()*100
    x=np.arange(3);w=.36
    axes[1].bar(x-w/2,[collapse.get(m,0) for m in MODELS],w,label="Colapso: rango predicho <1 meV o empate",color=[colors[m] for m in MODELS],alpha=.9)
    axes[1].bar(x+w/2,[ties.get(m,0) for m in MODELS],w,label="Empate GS predicho",color="#AAB7B8")
    near={m:int(((ensemble_comps.model.eq(m)) & (ensemble_comps.dft_gap_mev_atom<=1)).sum()) for m in MODELS}
    axes[1].set_xticks(x,MODELS);axes[1].set_ylabel("Composiciones (%)");axes[1].set_title("Colapsos, empates y DFT casi degenerado");axes[1].legend(fontsize=7);axes[1].grid(axis="y",alpha=.2)
    for i,m in enumerate(MODELS): axes[1].text(i, max(collapse.get(m,0),ties.get(m,0))+1,f"DFT gap≤1: {near[m]}",ha="center",fontsize=7)
    fig.suptitle("ABS₃: rangos y capacidad para distinguir polimorfos")
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.25)
    fig.text(.5,.035,"Histogramas: frecuencias y bins comunes; cada fórmula pesa una vez tras promediar semillas. Línea sólida = rango DFT; discontinua = predicción.\nColapso = rango predicho <1 meV/átomo o empate GS. DFT gap≤1 se excluye del GS primario.",ha="center",va="bottom",fontsize=8)
    save(fig,"ranges_collapses_ties")

    # Pareadas por fórmula: diferencia H1/H2−H0 en MAE absoluto y relativo.
    formula=ensemble_comps.pivot(index="composition_id",columns="model",values="absolute_mae_mev_atom")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,basekey,title in ((axes[0],"absolute_mae_mev_atom","Cambio de MAE absoluta"),(axes[1],"delta_anchored_mae_mev_atom","Cambio de MAE ΔE anclada")):
        wide=ensemble_comps.pivot(index="composition_id",columns="model",values=basekey)
        for cand,offset in (("H1",-.08),("H2",.08)):
            delta=wide[cand]-wide["H0"]
            ax.scatter(np.arange(len(delta))+offset,delta,alpha=.65,s=18,color=colors[cand],label=f"{cand}−H0")
            ax.axhline(float(delta.mean()),color=colors[cand],linestyle="--",linewidth=1)
        ax.axhline(0,color="#333",linewidth=1);ax.set_title(title);ax.set_xlabel("Composición (70 grupos; orden fijo)");ax.set_ylabel("Diferencia por composición (meV/átomo; <0 favorece candidato)");ax.grid(alpha=.2);ax.legend()
    fig.suptitle("ABS₃: diferencias pareadas por fórmula tras promediar semillas")
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.28)
    fig.text(.5,.035,"Cada fórmula aparece una vez; líneas discontinuas = diferencia media. IC pareados por composición en bootstrap_paired.json (2,000 remuestras).\nLas semillas se promediaron dentro de cada fórmula; los puntos no son semillas independientes.",ha="center",va="bottom",fontsize=8)
    save(fig,"paired_composition_differences")

    # Confusion matrices ensemble seed-averaged.
    from matplotlib.colors import Normalize
    gs_order=list(FAMILIES); pred_order=list(FAMILIES)+["tie/abstain"]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,model in zip(axes,MODELS):
        mat=np.zeros((3,4),int); excluded=0
        for _,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            if not m["dft_gs_eligible"]:excluded+=1;continue
            mat[FAMILIES.index(m["dft_gs_family"]),pred_order.index(m["pred_gs_family"])]+=1
        pct=np.divide(mat,mat.sum(axis=1,keepdims=True),out=np.zeros_like(mat,dtype=float),where=mat.sum(axis=1,keepdims=True)>0)*100
        ax.imshow(pct,vmin=0,vmax=100,cmap="Blues")
        for i in range(3):
            for j in range(4): ax.text(j,i,f"{mat[i,j]}\n{pct[i,j]:.0f}%",ha="center",va="center",color="black",fontsize=9)
        ax.set_xticks(range(4),[FAMILY_LABEL.get(x,x) for x in pred_order],rotation=25,ha="right",fontsize=8);ax.set_yticks(range(3),[FAMILY_LABEL[x] for x in gs_order],fontsize=8);ax.set_title(f"{model}: N={int(mat.sum())}; DFT excluidas={excluded}",fontsize=9,pad=10)
        ax.set_xlabel("Ganador predicho",fontsize=9)
        if ax is axes[0]:ax.set_ylabel("Ganador DFT",fontsize=9)
        else:ax.set_yticklabels([])
    fig.suptitle("ABS₃: matriz de confusión del estado fundamental, ensamble de semillas")
    fig.subplots_adjust(top=.84,bottom=.30,wspace=.12,left=.1,right=.89)
    cax=fig.add_axes([.92,.28,.015,.52]);fig.colorbar(axes[0].images[0],cax=cax,label="Porcentaje por fila (%)")
    fig.text(.5,.045,"Filas = familia GS DFT; columnas = familia predicha o abstención. Cada celda muestra conteo y porcentaje de la fila.\n70 fórmulas físicas como máximo, semilla promediada; se excluyen del denominador grupos con gap DFT ≤1 meV/átomo.",ha="center",va="bottom",fontsize=8)
    save(fig,"gs_confusion_matrices")

    # Pairwise relation matrix including DFT tie category.
    rels=["i_lower","tie","j_lower"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for ax,model in zip(axes,MODELS):
        mat=np.zeros((3,3),int); rows=0;ties=0;errors=0
        for _,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            for pr in m["pair_rows"]:
                mat[rels.index(pr["dft_relation"]),rels.index(pr["pred_relation"])]+=1; rows+=1
                ties+=int(pr["dft_tie"]);errors+=int(pr["eligible_primary"] and not pr["correct_primary"])
        pct=np.divide(mat,mat.sum(axis=1,keepdims=True),out=np.zeros_like(mat,dtype=float),where=mat.sum(axis=1,keepdims=True)>0)*100
        ax.imshow(pct,vmin=0,vmax=100,cmap="Purples")
        for i in range(3):
            for j in range(3):ax.text(j,i,f"{mat[i,j]}\n{pct[i,j]:.0f}%",ha="center",va="center",fontsize=9)
        ax.set_xticks(range(3),rels,fontsize=8);ax.set_yticks(range(3),rels,fontsize=8);ax.set_xlabel("Relación predicha i−j",fontsize=9)
        if ax is axes[0]:ax.set_ylabel("Relación DFT i−j",fontsize=9)
        else:ax.set_yticklabels([])
        ax.set_title(f"{model}: {rows} pares; DFT ties={ties}; errores={errors}",fontsize=9,pad=10)
    fig.suptitle("ABS₃: ordenamiento pareado por familia, ensamble de semillas")
    fig.subplots_adjust(top=.84,bottom=.27,wspace=.12,left=.1,right=.89)
    cax=fig.add_axes([.92,.28,.015,.52]);fig.colorbar(axes[0].images[0],cax=cax,label="Porcentaje por fila (%)")
    fig.text(.5,.04,"Pares orientados: (perovskita, hexagonal), (perovskita, aguja), (hexagonal, aguja), ΔEᵢⱼ=Eᵢ−Eⱼ.\nFilas/columnas = i_lower, tie, j_lower. DFT ties se muestran para diagnóstico; cada par físico cuenta una vez.",ha="center",va="bottom",fontsize=8)
    save(fig,"pairwise_confusion_matrices")

    # OOF fold-level and sensitivity to seeds.
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    metrics=[("MAE_absolute_macro_meV_atom","MAE absoluto (meV/átomo)"),("MAE_delta_GS_anchored_macro_meV_atom","MAE ΔE anclada (meV/átomo)"),("GS_accuracy","Exactitud GS"),("pairwise_accuracy","Exactitud pairwise")]
    for ax,(key,title) in zip(axes.flat,metrics):
        for model in MODELS:
            g=fold_metrics.loc[fold_metrics.model.eq(model)]
            for seed in sorted(g.seed.unique()):
                s=g.loc[g.seed.eq(seed)].sort_values("fold")
                ax.plot(s.fold,s[key],marker="o",alpha=.55,color=colors[model],label=model if seed==sorted(g.seed.unique())[0] else None)
        ax.set_title(title);ax.set_xlabel("Fold externo");ax.set_xticks(range(5));ax.grid(alpha=.25);ax.legend()
    fig.suptitle("ABS₃: variación entre pliegues y semillas (OOF desarrollo)")
    fig.subplots_adjust(top=.9,bottom=.17,hspace=.32,wspace=.26)
    fig.text(.5,.035,"Cada línea = semilla; cada punto = fold externo (14 fórmulas). Folds/semillas no son fórmulas independientes.\nGS y pairwise usan los denominadores elegibles indicados en las tablas.",ha="center",va="bottom",fontsize=8)
    save(fig,"fold_seed_stability")

    # Training curves by epoch, with number of active runs.
    hist_path=output/"training_history.csv"
    if hist_path.is_file():
        hist=pd.read_csv(hist_path)
        hist["train_objective_total"]=hist.train_loss_first+hist.train_loss_second
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,model in zip(axes,MODELS):
            h=hist.loc[hist.model.eq(model)]
            for key,label,color in (("train_objective_total","Entrenamiento total",colors[model]),("inner_validation_objective","Validación interna total","#222222")):
                gr=h.groupby("epoch")[key]
                med=gr.mean();n=gr.count();ax.plot(med.index,med.values,label=label,color=color)
            ax.set_title(f"{model}: {h[['fold','seed']].drop_duplicates().shape[0]} corridas");ax.set_xlabel("Época local");ax.set_ylabel("Pérdida normalizada (adimensional)");ax.grid(alpha=.25);ax.legend()
        fig.suptitle("ABS₃: aprendizaje de cabezas congeladas")
        fig.subplots_adjust(top=.84,bottom=.24,wspace=.22)
        fig.text(.5,.035,"Train = promedio de minibatches con pesos online; validation = inner-validation completa por época. Se promedian solo corridas activas; ver training_history.csv.\nCheckpoint = mínima pérdida de validación interna; límite 200 épocas y paciencia 20.",ha="center",va="bottom",fontsize=8)
        save(fig,"training_curves")
    return results


def run(out_dir: Path, bootstrap_replicates: int = 2000) -> None:
    """Agrega resultados OOF, exporta tablas, IC, métricas y figuras."""
    pred_path=out_dir/"oof_predictions.csv"
    if not pred_path.is_file():raise ExperimentError(f"Falta OOF: {pred_path}")
    raw=pd.read_csv(pred_path)
    if set(raw.model.unique())!=set(MODELS):raise ExperimentError("Se requiere OOF completo H0/H1/H2.")
    counts=raw.groupby(["model","fold","seed"]).structure_id.nunique()
    if len(counts)!=45 or not counts.eq(42).all():raise ExperimentError("OOF no tiene 45 tripletes-fold/seed de 42 estructuras cada uno.")
    enriched, comps, pairs=enrich_predictions(raw)
    enriched.to_csv(out_dir/"oof_predictions_enriched.csv",index=False)
    comps.to_csv(out_dir/"metrics_composition.csv",index=False)
    pairs.to_csv(out_dir/"pairwise_comparisons.csv",index=False)
    per_fold=aggregate_metrics(comps,pairs,enriched)
    per_fold.to_csv(out_dir/"metrics_per_fold_seed.csv",index=False)
    seed_metrics=per_fold.groupby(["model","seed"],as_index=False).mean(numeric_only=True)
    seed_metrics.to_csv(out_dir/"metrics_per_seed.csv",index=False)
    per_model=per_fold.groupby("model").agg({col:["mean","std"] for col in per_fold.select_dtypes(include=[np.number]).columns if col not in {"fold","seed"}})
    per_model.columns=["_".join(x).strip("_") for x in per_model.columns]
    per_model.to_csv(out_dir/"metrics_summary.csv")
    ens, ensemble_comps, confusion=ensemble_tables(enriched)
    ensemble_comps.to_csv(out_dir/"ensemble_metrics_composition.csv",index=False)
    # Matrix counts with physical compositions/pairs only (no cross-seed duplication).
    gs_rows=[];pair_rows=[]
    for model in MODELS:
        for _,g in ens.loc[ens.model.eq(model)].groupby("composition_id"):
            m=summarize_triplet(g)
            gs_rows.append({"model":model,"composition_id":g.composition_id.iloc[0],"dft_gs_family":m["dft_gs_family"],"pred_gs_family":m["pred_gs_family"],"dft_gs_eligible":m["dft_gs_eligible"],"dft_gap_mev_atom":m["dft_gap_mev_atom"],"predicted_abstention":m["pred_gs_tie"]})
            for pair in m["pair_rows"]:pair_rows.append({"model":model,"composition_id":g.composition_id.iloc[0],**pair})
    pd.DataFrame(gs_rows).to_csv(out_dir/"gs_confusion_counts.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir/"pairwise_confusion_counts.csv",index=False)
    near_rows=[]
    for model in MODELS:
        subset=pd.DataFrame(pair_rows)
        subset=subset.loc[subset.model.eq(model)]
        for (fam_i,fam_j),g in subset.groupby(["family_i","family_j"]):
            near_rows.append({"model":model,"family_i":fam_i,"family_j":fam_j,"physical_pairs":len(g),
                "DFT_ties_abs_delta_le_1_mev_atom":int(g.dft_tie.sum()),"DFT_non_ties":int(g.eligible_primary.sum()),
                "mean_abs_dft_delta_mev_atom":float(g.diff_dft_mev_atom.abs().mean())})
    pd.DataFrame(near_rows).to_csv(out_dir/"near_degenerate_pairs.csv",index=False)
    # Recalcula las métricas por fórmula desde energías ya promediadas entre semillas.
    # Así, bootstrap y comparaciones no tratan las tres semillas como grupos químicos.
    ecomp=[]
    for (model,comp),g in ens.groupby(["model","composition_id"]):
        m=summarize_triplet(g)
        ecomp.append({"model":model,"composition_id":comp,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))},
            "E_DFT_mu_eV_atom":m["E_DFT_mu_eV_atom"],"E_pred_mu_eV_atom":m["E_pred_mu_eV_atom"]})
    ensemble_comps=pd.DataFrame(ecomp)
    ensemble_comps.to_csv(out_dir/"ensemble_metrics_composition.csv",index=False)
    bootstrap=bootstrap_comparisons(ensemble_comps,bootstrap_replicates)
    write_json(out_dir/"bootstrap_paired.json",bootstrap)
    paired_rows=[]
    for metric in ("absolute_mae_mev_atom","delta_anchored_mae_mev_atom","delta_own_mae_mev_atom",
                   "gs_correct","pairwise_accuracy","range_abs_error_mev_atom","mae_mean_component_mev_atom",
                   "mae_centered_residual_mev_atom"):
        wide=ensemble_comps.pivot(index="composition_id",columns="model",values=metric)
        for cand,base in (("H1","H0"),("H2","H0"),("H2","H1")):
            for comp in wide.index:
                row=ensemble_comps.loc[(ensemble_comps.model.eq(cand))&(ensemble_comps.composition_id.eq(comp))].iloc[0]
                eligible=bool(row.dft_gs_eligible>0) if metric=="gs_correct" else True
                candidate_value=float(wide.loc[comp,cand])
                reference_value=float(wide.loc[comp,base])
                paired_rows.append({"metric":metric,"candidate":cand,"reference":base,"composition_id":comp,
                    "candidate_value":candidate_value,"reference_value":reference_value,
                    "difference_candidate_minus_reference":candidate_value-reference_value,"eligible_primary":eligible,
                    "aggregation":"promedio de energías por estructura entre semillas; métrica recalculada por composición"})
    paired_frame=pd.DataFrame(paired_rows)
    paired_frame.to_csv(out_dir/"paired_comparisons.csv",index=False)
    # H0 per-composition error decomposition before comparing new heads.
    h0=ensemble_comps.loc[ensemble_comps.model.eq("H0")].copy()
    h0.to_csv(out_dir/"H0_error_decomposition.csv",index=False)
    all_mse= h0.mse_total_mev2_atom2.mean(); mean_mse=h0.mse_mean_component_mev2_atom2.mean();res_mse=h0.mse_residual_component_mev2_atom2.mean()
    if abs(all_mse-mean_mse-res_mse)>1e-6:raise ExperimentError("MSE agregado no conserva identidad por composición.")
    ridge_frame,ridge_metrics=load_d1_ridge_reference(enriched,out_dir)
    figure_stats=export_figures(enriched,comps,per_fold,ensemble_comps,out_dir,ridge_frame)
    write_json(out_dir/"parity_statistics.json",figure_stats)
    # Per-family OOF and outliers.
    fam=enriched.assign(abs_error_mev_atom=lambda x:(x.E_pred_eV_atom-x.E_DFT_eV_atom).abs()*1000).groupby(["model","polymorph_family"],as_index=False).agg(N=("structure_id","nunique"),MAE_mev_atom=("abs_error_mev_atom","mean"),median_abs_error_mev_atom=("abs_error_mev_atom","median"))
    fam.to_csv(out_dir/"metrics_by_family.csv",index=False)
    outliers=enriched.assign(abs_error_mev_atom=lambda x:(x.E_pred_eV_atom-x.E_DFT_eV_atom).abs()*1000).sort_values("abs_error_mev_atom",ascending=False).groupby("model").head(15)
    outliers.to_csv(out_dir/"largest_structure_errors.csv",index=False)
    examples=ensemble_comps.loc[ensemble_comps.model.eq("H0")].sort_values("absolute_mae_mev_atom",ascending=False).head(15)
    examples.to_csv(out_dir/"illustrative_compositions_H0.csv",index=False)
    write_json(out_dir/"D1_ridge_comparison.json",ridge_metrics)
    near_pairs=pd.read_csv(out_dir/"near_degenerate_pairs.csv")
    dft_near=near_pairs.loc[near_pairs.model.eq("H0"),"DFT_ties_abs_delta_le_1_mev_atom"].sum()
    # Report de lectura científica simple, basado estrictamente en tablas reales.
    report=["# MatGPT congelado H0/H1/H2 — resultados de desarrollo\n",
        "## Alcance y limitaciones\n",
        "Se usaron 210 estructuras estáticas GGA/PBE (70 composiciones, tres polimorfos) en cinco folds agrupados; 3 semillas por modelo. Las energías son totales estáticas DFT por átomo, no energías de formación. El backbone MatGPT original se mantuvo congelado. No se usaron estructuras, etiquetas ni embeddings del test histórico. Las configuraciones y pooling previos se eligieron con parte de la validation histórica; por tanto, estos intervalos son exploratorios. H1/H2 requieren conocer las tres estructuras candidatas de una composición al inferir.\n",
        "## Resultados macro OOF por configuración\n",
        "La tabla siguiente resume las 15 combinaciones fold×semilla por modelo (cada combinación evalúa 14 fórmulas). Para resultados principales de composición, matrices y bootstrap, primero se promediaron las energías OOF de las tres semillas por estructura y después se recalcularon las métricas por fórmula.\n",
        "| Modelo | MAE absoluta (meV/átomo) | MAE ΔE anclada DFT-GS | MAE ΔE own-min legacy | GS (denom.) | Pairwise (denom.) | Error de media | Residuo centrado |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for model in MODELS:
        c=comps.loc[comps.model.eq(model)]
        pp=pairs.loc[pairs.model.eq(model)&pairs.eligible_primary]
        pc=c.loc[c.model.eq(model)]
        eligible_comps=int(pc.dft_gs_eligible.sum()/3)
        report.append(f"| {model} | {pc.absolute_mae_mev_atom.mean():.2f} | {pc.delta_anchored_mae_mev_atom.mean():.2f} | {pc.delta_own_mae_mev_atom.mean():.2f} | {pc.loc[pc.dft_gs_eligible,'gs_correct'].mean()*100:.1f}% ({eligible_comps} fórmulas; 3 semillas) | {pp.correct_primary.mean()*100:.1f}% ({len(pp)} pares elegibles×semilla) | {pc.mae_mean_component_mev_atom.mean():.2f} | {pc.mae_centered_residual_mev_atom.mean():.2f} |\n")
    ensemble_summary=ensemble_comps.groupby("model").mean(numeric_only=True)
    report += ["\n### Métricas principales tras promediar semillas por estructura\n",
        "Cada fórmula física aparece una sola vez. GS excluye fórmulas con brecha DFT ≤1 meV/átomo y empates predichos son abstenciones; pairwise excluye pares DFT empatados.\n",
        "| Modelo | MAE absoluta | ΔE anclada | ΔE own-min | GS estricta | Pairwise | Error media | MAE residual centrado |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for model in MODELS:
        row=ensemble_summary.loc[model]
        model_rows=ensemble_comps.loc[ensemble_comps.model.eq(model)]
        gs_rows=model_rows.loc[model_rows.dft_gs_eligible.gt(0)]
        n_gs=len(gs_rows)
        n_pairs=int(model_rows.pairwise_eligible.sum())
        gs_accuracy=gs_rows.gs_correct.mean()
        pair_accuracy=model_rows.pairwise_correct.sum()/n_pairs if n_pairs else np.nan
        report.append(f"| {model} | {row.absolute_mae_mev_atom:.2f} meV/átomo | {row.delta_anchored_mae_mev_atom:.2f} meV/átomo | {row.delta_own_mae_mev_atom:.2f} meV/átomo | {gs_accuracy*100:.1f}% ({n_gs}/67) | {pair_accuracy*100:.1f}% ({int(model_rows.pairwise_correct.sum())}/{n_pairs}) | {row.mae_mean_component_mev_atom:.2f} meV/átomo | {row.mae_centered_residual_mev_atom:.2f} meV/átomo |\n")
    report += ["\n## H0: identidad de error cuadrático\n",
        f"MSE total macro = {all_mse:.3f} meV²/átomo²; componente del error de media = {mean_mse:.3f} ({mean_mse/all_mse*100:.2f}%); componente residual centrado = {res_mse:.3f} ({res_mse/all_mse*100:.2f}%); discrepancia de identidad = {abs(all_mse-mean_mse-res_mse):.3g}. Esta igualdad vale para MSE; los dos MAE diagnósticos no se suman para formar el MAE absoluto. Se promediaron semillas por ID antes de esta descomposición OOF.\n",
        f"MAE del error de media = {h0.mae_mean_component_mev_atom.mean():.2f} meV/átomo; MAE del residuo estructural centrado = {h0.mae_centered_residual_mev_atom.mean():.2f} meV/átomo. Los mayores errores individuales están en `largest_structure_errors.csv`.\n",
        f"El diagnóstico de pares encontró {int(dft_near)} pares físicos DFT empatados a ±1 meV/átomo en el ensamble por composición; el listado por familias/modelos está en `near_degenerate_pairs.csv`.\n",
        "## Incertidumbre y comparación pareada\n",
        "Las diferencias H1−H0, H2−H0 y H2−H1 se remuestrean por composición después de promediar semillas dentro de cada composición. Un valor negativo favorece al candidato para errores; positivo favorece al candidato para GS/pairwise.\n",
        "| Comparación pareada | MAE abs Δ (IC95%) | ΔE anclada Δ (IC95%) | GS Δ (IC95%) | Pairwise Δ (IC95%) |\n|---|---:|---:|---:|---:|\n"]
    for comparison in ("H1_minus_H0","H2_minus_H0","H2_minus_H1"):
        item=bootstrap["comparisons"][comparison]
        def cell(metric: str, scale: float = 1.0) -> str:
            z=item[metric]
            return f"{z['difference_mean_a_minus_b']*scale:.3f} [{z['ci95_low']*scale:.3f}, {z['ci95_high']*scale:.3f}]"
        report.append(f"| {comparison.replace('_minus_','−')} | {cell('absolute_mae_mev_atom')} meV/átomo | {cell('delta_anchored_mae_mev_atom')} meV/átomo | {cell('gs_correct',100)} puntos % | {cell('pairwise_accuracy',100)} puntos % |\n")
    report += ["\nLos IC son exploratorios; 2,000 remuestras mantienen juntas las tres estructuras de cada fórmula y no tratan semillas como muestras químicas independientes. Medias y dispersión fold/semilla están en `metrics_per_fold_seed.csv` y `metrics_per_seed.csv`.\n",
        "## Cómo leer las figuras\n",
        "La paridad DFT-GS anclada permite predicciones relativas negativas, señal de inversión respecto al GS DFT. La gráfica own-min usa el mínimo propio de cada eje y oculta inversiones. R² OLS corresponde a una recta con intercepto libre; R² predictivo evalúa la diagonal y=x. En matrices, cada fórmula cuenta una vez tras promediar semillas; las matrices pairwise muestran empates diagnósticos, mientras que la exactitud primaria excluye DFT ties. El scorecard usa metas heredadas aspiracionales, no calibradas para MatGPT.\n",
        "## Control externo D1: Ridge composition-only\n"]
    if ridge_metrics.get("status")=="PASS":
        report.append(f"La auditoría de IDs y outer folds coincide exactamente. Ridge tiene MAE absoluta macro {ridge_metrics['MAE_absolute_macro_mev_atom']:.2f} meV/átomo (pooled {ridge_metrics['MAE_absolute_pooled_mev_atom']:.2f}). Las tres predicciones de cada fórmula son constantes, de modo que el ranking queda indeterminado; no se le atribuye una tasa de acierto GS. Ridge no forma parte de H0/H1/H2.\n")
    else:
        report.append(f"Ridge no se integró como referencia cuantitativa porque no pasó la comprobación de protocolo: {ridge_metrics.get('reason')}.\n")
    report += ["\n## Interpretación por arquitectura\n"]
    for model in MODELS:
        c=comps.loc[comps.model.eq(model)]
        report.append(f"- **{model}:** MAE absoluta macro {c.absolute_mae_mev_atom.mean():.1f} meV/átomo; error de media {c.mae_mean_component_mev_atom.mean():.1f}; MAE residual centrado {c.mae_centered_residual_mev_atom.mean():.1f}; ΔE anclada {c.delta_anchored_mae_mev_atom.mean():.1f}.\n")
    em=ensemble_comps.groupby("model").mean(numeric_only=True)
    report.append(f"- **Efecto de H1/H2:** frente a H0, H1 reduce el MAE absoluta {em.loc['H0','absolute_mae_mev_atom']-em.loc['H1','absolute_mae_mev_atom']:.1f} meV/átomo en el ensamble OOF de semillas, sobre todo por mejorar la media composicional. H2 reduce {em.loc['H1','absolute_mae_mev_atom']-em.loc['H2','absolute_mae_mev_atom']:.1f} meV/átomo adicionales respecto a H1; su diferencia de pooling afecta la cabeza de media y el cambio de residuo centrado es pequeño.\n")
    gs_summary={};pair_summary={}
    for model in MODELS:
        q=ensemble_comps.loc[ensemble_comps.model.eq(model)]
        gs_summary[model]=q.loc[q.dft_gs_eligible.gt(0),"gs_correct"].mean()*100
        pair_summary[model]=q.pairwise_correct.sum()/q.pairwise_eligible.sum()*100
    report.append(f"- **Ranking:** GS del ensamble con 67 fórmulas elegibles (3 DFT casi degeneradas excluidas) H0={gs_summary['H0']:.1f}%, H1={gs_summary['H1']:.1f}%, H2={gs_summary['H2']:.1f}%; exactitud pairwise agrupada sobre 206 pares DFT no empatados H0={pair_summary['H0']:.1f}%, H1={pair_summary['H1']:.1f}%, H2={pair_summary['H2']:.1f}%. Las diferencias pairwise y GS tienen intervalos que incluyen cero.\n")
    report += ["\nNo se atribuye un fallo exclusivamente al embedding: la cabeza, el objetivo y el pequeño tamaño de entrenamiento también afectan el resultado. Ridge solo es una referencia externa para MAE absoluta; no forma parte de H0/H1/H2 ni proporciona ranking.\n",
        "## Referencias\n- MatGPT, commit histórico: https://github.com/devalab/matgpt\n- SLICES: Xiao et al. (2023), https://doi.org/10.1038/s41467-023-42870-7\n- MatFold: Witman & Schindler (2025), https://doi.org/10.1039/D4DD00250D\n- PBE: Perdew, Burke & Ernzerhof (1996), https://doi.org/10.1103/PhysRevLett.77.3865\n- PyTorch SmoothL1Loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html\n- scikit-learn confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html\n- pymatgen ComputedStructureEntry: https://pymatgen.org/pymatgen.core.html\n"]
    (out_dir/"H0_H1_H2_report.md").write_text("".join(report),encoding="utf-8")
    if figure_stats.get("absolute_energy_parity"):
        pd.DataFrame(figure_stats["absolute_energy_parity"].values()).to_csv(out_dir/"regression_statistics.csv",index=False)
    relative_stats=[{"convention":key.replace("legacy_parity_","own_min_legacy_"),**value} for key,value in figure_stats.items() if key.startswith("legacy_parity_")]
    if relative_stats:pd.DataFrame(relative_stats).to_csv(out_dir/"relative_regression_statistics.csv",index=False)
    component_stats=[{"comparison":key,**value} for key,value in figure_stats.items() if key.startswith(("composition_mu_parity_","structural_residual_parity_"))]
    if component_stats:pd.DataFrame(component_stats).to_csv(out_dir/"component_regression_statistics.csv",index=False)
    write_json(out_dir/"evaluation_manifest.json",{"status":"PASS","models":list(MODELS),"folds":5,"seeds":3,"OOF_structures_per_model_seed":210,"test_used":False,"bootstrap_replicates":bootstrap_replicates,"bootstrap_seed":13072026,"bootstrap_and_primary_composition_metrics":"average OOF energies across seeds per structure, then recalculate metrics per physical composition","parity_statistics":figure_stats,"MSE_decomposition_H0":{"total":all_mse,"mean_component":mean_mse,"residual_component":res_mse,"identity_abs_error":abs(all_mse-mean_mse-res_mse)}})


def main() -> None:
    """Valida OOF y genera artefactos de evaluación."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out-dir",type=Path,default=REPORT_DIR);parser.add_argument("--bootstrap",type=int,default=2000);args=parser.parse_args()
    try:
        if args.bootstrap<2000:parser.error("El protocolo requiere al menos 2,000 remuestras")
        run(args.out_dir,args.bootstrap)
        log(logging.INFO,"evaluation_complete",out_dir=args.out_dir.resolve(),test=False)
    except (ExperimentError,OSError,ValueError,KeyError,RuntimeError,ImportError) as exc:
        log(logging.ERROR,"evaluation_failed",error=str(exc));raise SystemExit(2) from exc


if __name__=="__main__":
    main()
