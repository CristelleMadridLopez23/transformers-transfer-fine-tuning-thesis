#!/usr/bin/env python3
"""Genera las figuras PNG científicas de H3 desde sus tablas OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import FAMILIES, H3Error, REPORT_DIR, write_json

OUT = REPORT_DIR / "figures"
COLORS = {"H0": "#5b6573", "H2": "#2878b5", "R4": "#8064a2", "D1": "#9a8f3a",
          "H3-A": "#1b9e77", "H3-B": "#e07a1f", "H3-C": "#c83c61"}
MARKERS = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}
MODELS = ("H0", "H2", "R4", "D1", "H3-A", "H3-B", "H3-C")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga las predicciones OOF, tablas por composición/fold y alpha.

    Returns:
        Predicciones ensembeled, métricas de composición, métricas fold/semilla,
        parejas físicas y selección Ridge interna.
    """
    root = REPORT_DIR / "comparative_analysis"
    required = [root / "oof_predictions_all_models.csv", root / "metrics_by_composition_ensemble.csv",
                root / "metrics_per_fold_seed.csv", root / "pairwise_observations_ensemble.csv",
                root / "ridge_alpha_inner_validation.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise H3Error(f"Faltan salidas para figuras: {missing}")
    raw = pd.read_csv(required[0])
    predictions = raw.groupby(["model", "structure_id", "composition_id", "polymorph_family", "fold"], as_index=False).agg(
        E_DFT_eV_atom=("E_DFT_eV_atom", "first"), E_pred_eV_atom=("E_pred_eV_atom", "mean"))
    return (predictions, pd.read_csv(required[1]), pd.read_csv(required[2]),
            pd.read_csv(required[3]), pd.read_csv(required[4]))


def _ols_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS libre y R² del ajuste sobre exactamente los puntos dibujados."""
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - fitted) ** 2) / denom) if denom else float("nan")
    mae = float(np.mean(np.abs(y - x)))
    return float(slope), float(intercept), r2, mae


def _save(fig: plt.Figure, name: str, made: list[str], note: str | None = None) -> None:
    """Guarda PNG y agrega el archivo a su manifiesto."""
    if note:
        fig.text(.02, .008, note, fontsize=8, color="#333333", ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    made.append(name)


def _scatter_parity(pred: pd.DataFrame, model: str, x: np.ndarray, y: np.ndarray,
                    xlabel: str, ylabel: str, title: str, units: str,
                    note: str, made: list[str], name: str) -> None:
    """Crea paneles comparables DFT en x y predicción en y con OLS libre."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    stats = []
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    padding = .04 * max(high - low, 1e-9)
    lim = (low - padding, high + padding)
    for ax, model in zip(axes_flat, MODELS):
        frame = pred.loc[pred.model.eq(model)].copy()
        xx, yy = x[frame.index], y[frame.index]
        slope, intercept, r2, mae = _ols_stats(xx, yy)
        for family, marker in MARKERS.items():
            mask = frame.polymorph_family.eq(family).to_numpy()
            ax.scatter(xx[mask], yy[mask], marker=marker, s=16, alpha=.68, color=COLORS[model],
                       edgecolor="white", linewidth=.25, label=family)
        grid = np.linspace(*lim, 100)
        ax.plot(grid, grid, color="#252525", linestyle="--", linewidth=1.2, label="Ideal y=x")
        ax.plot(grid, slope * grid + intercept, color="#d33682", linestyle="-.", linewidth=1.2, label="OLS libre")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=.2)
        r2_text = f"{r2:.3f}" if np.isfinite(r2) else "no definido"
        ax.set_title(f"{model} | N=210, 70 comp.\nm={slope:.3f}; b={intercept:.3g} {units}\nR² OLS={r2_text}; MAE={mae:.2f} {units}", fontsize=9)
        stats.append({"model": model, "N_structures": len(frame), "N_compositions": frame.composition_id.nunique(),
                      "slope_dimensionless": slope, "intercept": intercept, "R2_OLS": r2, "MAE_vs_reference": mae})
    for ax in axes_flat[len(MODELS):]: ax.axis("off")
    axes[1, 0].set_xlabel(xlabel); axes[1, 0].set_ylabel(ylabel)
    axes[0, 0].set_ylabel(ylabel); axes[0, 1].set_xlabel(xlabel); axes[0, 2].set_xlabel(xlabel); axes[0, 3].set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, fontsize=8, bbox_to_anchor=(.5, 1.01))
    fig.suptitle(title, y=1.06, fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, .95))
    _save(fig, name, made, note)
    pd.DataFrame(stats).to_csv(REPORT_DIR / "comparative_analysis" / name.replace(".png", "_statistics.csv"), index=False)


def _composition_mean_parity(means: pd.DataFrame, made: list[str]) -> None:
    """Muestra exactamente una marca/OLS por composición para la cabeza μ."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True)
    flat=axes.ravel(); stats=[]
    x=means.E_DFT_mu_eV_atom.to_numpy(float)*1000; y=means.E_pred_mu_eV_atom.to_numpy(float)*1000
    low,high=float(min(x.min(),y.min())),float(max(x.max(),y.max()));pad=.04*(high-low);lims=(low-pad,high+pad)
    for ax,model in zip(flat,MODELS):
        group=means.loc[means.model.eq(model)].copy();xx=group.E_DFT_mu_eV_atom.to_numpy(float)*1000;yy=group.E_pred_mu_eV_atom.to_numpy(float)*1000
        slope,intercept,r2,mae=_ols_stats(xx,yy);grid=np.linspace(*lims,100)
        ax.scatter(xx,yy,s=23,color=COLORS[model],marker="o",alpha=.72,label="Una media por composición")
        ax.plot(grid,grid,"k--",lw=1.2,label="Ideal y=x");ax.plot(grid,slope*grid+intercept,color="#d33682",ls="-.",lw=1.2,label="OLS libre")
        ax.set(xlim=lims,ylim=lims,title=f"{model} | N=70\nm={slope:.3f}; b={intercept:.1f} meV/átomo\nR² OLS={r2:.3f}; MAE={mae:.1f} meV/átomo")
        ax.grid(alpha=.2);stats.append({"model":model,"N_compositions":len(group),"slope_dimensionless":slope,"intercept_mev_atom":intercept,"R2_OLS":r2,"MAE_mev_atom":mae})
    for ax in flat[len(MODELS):]:ax.axis("off")
    axes[0,0].set_ylabel("Media predicha μ̂ (meV/átomo)");axes[1,0].set_ylabel("Media predicha μ̂ (meV/átomo)")
    for ax in axes[1,:3]:ax.set_xlabel("Media DFT μ (meV/átomo)")
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,1.02),fontsize=9)
    fig.suptitle("Paridad de energía media por composición",y=1.07,fontsize=15);fig.tight_layout(rect=(0,0,1,.95))
    _save(fig,"02_composition_mean_parity.png",made,"Una marca = una composición; N=70 por modelo después de promediar semillas y polimorfos. H3 Ridge usa L03_MEAN de las tres estructuras; H0/H2/R4/D1 tienen media descriptiva de sus energías predichas.")
    pd.DataFrame(stats).to_csv(REPORT_DIR/"comparative_analysis/02_composition_mean_parity_statistics.csv",index=False)


def generate() -> list[str]:
    """Crea 24 figuras diagnósticas en PNG a partir de predicciones observadas.

    Returns:
        Nombres de los archivos PNG generados.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    pred, comp, run_metrics, pairs, alpha = _load()
    made: list[str] = []
    pred = pred.reset_index(drop=True)
    # Figure 1: absolute energy parity, all observed physical structures.
    _scatter_parity(pred, "all", pred.E_DFT_eV_atom.to_numpy() * 1000,
                    pred.E_pred_eV_atom.to_numpy() * 1000, "DFT, E (meV/átomo)",
                    "Predicción, E (meV/átomo)", "H3: paridad de energía absoluta OOF",
                    "meV/átomo", "Cada punto = estructura del outer-fold; promedio de 3 semillas antes del panel. N=210 estructuras/70 composiciones por modelo; OLS libre describe asociación, y=x mide predicción.", made, "01_absolute_energy_parity.png")

    # Composition means and centered structural residuals for all models.
    triplets = pred.groupby(["model", "composition_id"], sort=False).agg(
        E_DFT_mu=("E_DFT_eV_atom", "mean"), E_pred_mu=("E_pred_eV_atom", "mean"),
        E_DFT_min=("E_DFT_eV_atom", "min")).reset_index()
    triplets = triplets.rename(columns={"E_DFT_mu":"E_DFT_mu_eV_atom","E_pred_mu":"E_pred_mu_eV_atom"})
    _composition_mean_parity(triplets, made)
    pred["delta_DFT_centered_mev_atom"] = pred.E_DFT_eV_atom * 1000 - pred.groupby(["model", "composition_id"]).E_DFT_eV_atom.transform("mean") * 1000
    pred["delta_pred_centered_mev_atom"] = pred.E_pred_eV_atom * 1000 - pred.groupby(["model", "composition_id"]).E_pred_eV_atom.transform("mean") * 1000
    _scatter_parity(pred, "all", pred.delta_DFT_centered_mev_atom.to_numpy(), pred.delta_pred_centered_mev_atom.to_numpy(),
                    "Residuo DFT centrado (meV/átomo)", "Residuo predicho centrado (meV/átomo)",
                    "Paridad de desviaciones estructurales centradas", "meV/átomo",
                    "N=210 polimorfos/70 composiciones por modelo; cada triplete se centra con su propia media. El intercepto ideal es 0; la línea negra es igualdad y la OLS usa todos los puntos.", made, "03_structural_residual_parity.png")

    # Anchored and own-min conventions are calculated independently.
    pred["dft_anchor"] = pred.E_DFT_eV_atom * 1000 - pred.groupby(["model", "composition_id"]).E_DFT_eV_atom.transform("min") * 1000
    pred["pred_at_dft_gs"] = np.nan
    # Explicit map keeps each model/composition's DFT-ground-state reference unambiguous.
    for (model, composition), group in pred.groupby(["model", "composition_id"], sort=False):
        idx = group.E_DFT_eV_atom.idxmin()
        pred.loc[group.index, "pred_at_dft_gs"] = pred.loc[idx, "E_pred_eV_atom"] * 1000
        pred.loc[group.index, "pred_own_min"] = group.E_pred_eV_atom.min() * 1000
    pred["pred_anchor"] = pred.E_pred_eV_atom * 1000 - pred.pred_at_dft_gs
    pred["dft_own_min"] = pred.dft_anchor
    pred["pred_own_min_rel"] = pred.E_pred_eV_atom * 1000 - pred.pred_own_min
    _scatter_parity(pred, "all", pred.dft_anchor.to_numpy(), pred.pred_anchor.to_numpy(),
                    "DFT ΔE anclada a DFT-GS (meV/átomo)", "Predicción anclada al mismo DFT-GS (meV/átomo)",
                    "ΔE con referencia DFT-GS (protocolo principal)", "meV/átomo",
                    "Negativos predichos indican inversión respecto al GS DFT. N=210/70 por modelo; se conservan referencias (0,0), no se recorta la cola negativa.", made, "04_delta_dft_gs_anchored_parity.png")
    _scatter_parity(pred, "all", pred.dft_own_min.to_numpy(), pred.pred_own_min_rel.to_numpy(),
                    "ΔE DFT respecto a su mínimo (meV/átomo)", "ΔE predicha respecto a su propio mínimo (meV/átomo)",
                    "ΔE own-min (convención histórica separada)", "meV/átomo",
                    "Convención propia de cada vector; no equivale al anclaje al GS DFT. N=210 estructuras/70 composiciones/modelo.", made, "05_delta_own_min_legacy_parity.png")

    # Figure 6: exact MSE decomposition terms by composition.
    fig, ax = plt.subplots(figsize=(12, 6.2)); width = .36
    order = list(MODELS); x = np.arange(len(order))
    means = [float(comp.loc[comp.model.eq(m), "MSE_mu_mev2_atom2"].mean()) for m in order]
    residuals = [float(comp.loc[comp.model.eq(m), "MSE_residual_mev2_atom2"].mean()) for m in order]
    ax.bar(x - width/2, means, width, label="Error cuadrático de media μ", color=[COLORS[m] for m in order], alpha=.9)
    ax.bar(x + width/2, residuals, width, label="Error cuadrático residual centrado", color="#b8c2cc", edgecolor=[COLORS[m] for m in order], linewidth=1.2)
    ax.set_xticks(x, order); ax.set_ylabel("MSE por composición (meV²/átomo²)"); ax.set_title("Descomposición aditiva del MSE, no del MAE")
    ax.grid(axis="y", alpha=.2); ax.legend()
    _save(fig, "06_mse_mean_residual_decomposition.png", made, "Cada barra es el promedio macro de 70 composiciones; MSE_total = error_mu² + MSE_residuo por triplete (error máximo de identidad en algebraic_checks.json). No sumar los MAE.")

    # Figure 7: predicted vs DFT ranges by composition.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True); axes = axes.ravel()
    max_range = float(max(comp.DFT_range_mev_atom.max(), comp.predicted_range_mev_atom.max()))
    for ax, model in zip(axes, MODELS):
        g = comp.loc[comp.model.eq(model)]; ax.scatter(g.DFT_range_mev_atom, g.predicted_range_mev_atom, s=21, alpha=.72, color=COLORS[model])
        ax.plot([0,max_range],[0,max_range],"k--",label="Ideal y=x"); ax.set_title(f"{model} | N=70 comp.")
        ax.grid(alpha=.2); ax.set_xlim(0,max_range*1.03); ax.set_ylim(0,max_range*1.03)
    for ax in axes[len(MODELS):]: ax.axis("off")
    axes[0].set_ylabel("Rango predicho (meV/átomo)"); axes[4].set_ylabel("Rango predicho (meV/átomo)")
    for ax in axes[4:7]: ax.set_xlabel("Rango DFT (meV/átomo)")
    fig.suptitle("Rango energético predicho frente al rango DFT por composición")
    _save(fig,"07_polymorph_ranges.png",made,"Un punto = una composición; rangos max(E)−min(E). N=70 por modelo; diagonal = igualdad de rangos. Las predicciones D1 colapsan a rango cero por diseño.")

    # Figure 8/9: paired distributions by composition.
    for number, metric, title, filename, ylabel in [
        (8,"MAE_absolute_mev_atom","MAE absoluta por composición","08_absolute_mae_by_model.png","MAE absoluta (meV/átomo)"),
        (9,"MAE_delta_GS_anchored_mev_atom","MAE de ΔE anclada a DFT-GS","09_relative_mae_by_model.png","MAE ΔE anclada (meV/átomo)")]:
        fig, ax = plt.subplots(figsize=(12,6))
        values=[comp.loc[comp.model.eq(m),metric].dropna().to_numpy() for m in MODELS]
        boxes=ax.boxplot(values,tick_labels=MODELS,patch_artist=True,showfliers=True)
        for box,m in zip(boxes["boxes"],MODELS): box.set_facecolor(COLORS[m]); box.set_alpha(.55)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y",alpha=.2)
        _save(fig,filename,made,"Caja=Q1–Q3, línea=mediana, bigotes=1.5×IQR, puntos extremos conservados. Una observación por composición después de promediar las 3 semillas.")

    # Figure 10 fold/seed stability.
    fig, axes = plt.subplots(1,2,figsize=(15,5.5))
    for model in MODELS:
        g=run_metrics.loc[run_metrics.model.eq(model)].copy()
        if g.empty: continue
        offset=(MODELS.index(model)-3)*.035
        x=g.fold.to_numpy()+offset
        axes[0].scatter(x,g.MAE_absolute_mev_atom,s=27,color=COLORS[model],label=model,alpha=.8)
        axes[1].scatter(x,g.GS_accuracy,s=27,color=COLORS[model],label=model,alpha=.8)
    axes[0].set(ylabel="MAE absoluta macro (meV/átomo)",xlabel="Outer fold",title="MAE por fold × semilla")
    axes[1].set(ylabel="GS accuracy elegible",xlabel="Outer fold",title="GS por fold × semilla");
    for ax in axes: ax.grid(alpha=.2); ax.set_xticks(range(5))
    axes[1].legend(ncol=2,fontsize=8)
    _save(fig,"10_fold_seed_stability.png",made,"Puntos H0/H2/H3: folds × semillas; R4/D1: una predicción determinista por fold (seed=-1). No se cuentan semillas como composiciones independientes.")

    # Figure 11 GS confusion, DFT ties omitted, predicted ties retained as abstention.
    classes=list(FAMILIES)+["tie/abstain"]
    obs=pd.read_csv(REPORT_DIR/"comparative_analysis/gs_confusion_observations.csv")
    fig,axes=plt.subplots(2,4,figsize=(16,8)); axes=axes.ravel()
    for ax,model in zip(axes,MODELS):
        sub=obs.loc[obs.model.eq(model)]; mat=np.zeros((3,4),int)
        for i,true in enumerate(FAMILIES):
            for j,predicted in enumerate(classes): mat[i,j]=int(((sub.true_class==true)&(sub.predicted_class==predicted)).sum())
        pct=mat/np.maximum(mat.sum(axis=1,keepdims=True),1)*100
        ax.imshow(pct,cmap="Blues",vmin=0,vmax=100,aspect="auto")
        for i in range(3):
            for j in range(4): ax.text(j,i,f"{mat[i,j]}\n{pct[i,j]:.0f}%",ha="center",va="center",fontsize=8)
        ax.set_xticks(range(4),["P","H","N","Empate"],rotation=20); ax.set_yticks(range(3),["P","H","N"])
        ax.set_title(f"{model} | N={len(sub)} composiciones")
    for ax in axes[len(MODELS):]: ax.axis("off")
    fig.suptitle("Confusión del polimorfo GS; filas DFT, columnas predicción",y=1.02)
    image = axes[0].images[0]
    colorbar = fig.colorbar(image, ax=axes.tolist(), fraction=.025, pad=.025)
    colorbar.set_label("Porcentaje dentro de cada fila (%)")
    _save(fig,"11_ground_state_confusion.png",made,"P=perovskita distorsionada, H=hexagonal, N=needle. Se excluyen composiciones con gap DFT≤1 meV/átomo; empates predichos quedan como abstención. Cada composición se cuenta una vez tras el ensemble de semillas.")

    # Figure 12 pairwise confusion, physical oriented pairs (P,H),(P,N),(H,N).
    pair_classes=["i_lower","tie","j_lower"]
    pair_obs=pd.read_csv(REPORT_DIR/"comparative_analysis/pairwise_confusion_observations.csv")
    fig,axes=plt.subplots(2,4,figsize=(16,8)); axes=axes.ravel()
    for ax,model in zip(axes,MODELS):
        sub=pair_obs.loc[pair_obs.model.eq(model)]; mat=np.zeros((3,3),int)
        for i,t in enumerate(pair_classes):
            for j,p in enumerate(pair_classes): mat[i,j]=int(((sub.DFT_class==t)&(sub.pred_class==p)).sum())
        pct=mat/np.maximum(mat.sum(axis=1,keepdims=True),1)*100
        ax.imshow(pct,cmap="Purples",vmin=0,vmax=100,aspect="auto")
        for i in range(3):
            for j in range(3): ax.text(j,i,f"{mat[i,j]}\n{pct[i,j]:.0f}%",ha="center",va="center",fontsize=8)
        ax.set_xticks(range(3),["i<j","empate","j<i"]); ax.set_yticks(range(3),["i<j","empate","j<i"])
        ax.set_title(f"{model} | {len(sub)} pares")
    for ax in axes[len(MODELS):]: ax.axis("off")
    fig.suptitle("Orden por pares; filas relación DFT, columnas relación predicha",y=1.02)
    image = axes[0].images[0]
    colorbar = fig.colorbar(image, ax=axes.tolist(), fraction=.025, pad=.025)
    colorbar.set_label("Porcentaje dentro de cada fila (%)")
    _save(fig,"12_pairwise_ordering_confusion.png",made,"Orientación fija (P,H), (P,N), (H,N); diferencia E_i−E_j, tolerancia empate 1 meV/átomo. Incluye ties DFT para diagnóstico; N máximo=210 pares físicos/modelo tras promediar semillas.")

    # Figure 13/14 family and macro ranking metrics.
    fig,ax=plt.subplots(figsize=(12,6)); eligible=comp.loc[comp.GS_eligible]
    family_rows=[]
    for model in MODELS:
        g=eligible.loc[eligible.model.eq(model)]
        for family in FAMILIES:
            truth=g.loc[g.GS_DFT_class.eq(family)]
            val=float(truth.GS_predicted_class.eq(family).mean()) if len(truth) else np.nan
            family_rows.append({"model":model,"DFT_GS_family":family,"N":len(truth),"accuracy":val})
    fam=pd.DataFrame(family_rows)
    for k,family in enumerate(FAMILIES):
        g=fam.loc[fam.DFT_GS_family.eq(family)]
        ax.bar(np.arange(len(MODELS))+(k-1)*.24,g.accuracy,width=.23,label=family,color=["#456990","#49a078","#b36a5e"][k])
    ax.set_xticks(range(len(MODELS)),MODELS); ax.set_ylabel("GS accuracy por familia DFT"); ax.set_ylim(0,1); ax.legend(); ax.grid(axis="y",alpha=.2)
    ax.set_title("Selección GS por familia del mínimo DFT")
    _save(fig,"13_ground_state_by_family.png",made,"Una composición por punto de denominador; DFT gaps≤1 meV/átomo excluidos. N por familia varía y se exporta en family_gs_accuracy.csv.")
    fam.to_csv(REPORT_DIR/"comparative_analysis/family_gs_accuracy.csv",index=False)
    summary=pd.read_csv(REPORT_DIR/"comparative_analysis/comparative_metrics.csv").set_index("model").reindex(MODELS)
    fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(MODELS)); w=.36
    ax.bar(x-w/2,summary.GS_balanced_accuracy,w,label="Balanced accuracy GS",color="#2878b5")
    ax.bar(x+w/2,summary.GS_macro_f1,w,label="Macro-F1 GS",color="#e07a1f")
    ax.set_xticks(x,MODELS); ax.set_ylim(0,1); ax.set_ylabel("Métrica (0–1)"); ax.set_title("Ranking GS: métricas macro"); ax.grid(axis="y",alpha=.2); ax.legend()
    _save(fig,"14_ground_state_balanced_f1.png",made,"Una observación física por composición; 67/70 elegibles tras excluir 3 gaps DFT≤1 meV/átomo. Abstenciones predichas no reciben acierto.")

    # Figure 15 pairwise signed error distributions.
    fig,ax=plt.subplots(figsize=(12,6)); vals=[pairs.loc[pairs.model.eq(m),"diff_pred_mev_atom"].to_numpy()-pairs.loc[pairs.model.eq(m),"diff_DFT_mev_atom"].to_numpy() for m in MODELS]
    bins=np.linspace(-500,500,61)
    for m,v in zip(MODELS,vals): ax.hist(v,bins=bins,density=True,histtype="step",linewidth=1.6,color=COLORS[m],label=m)
    ax.axvline(0,color="black",linestyle="--",label="Error cero"); ax.set(xlabel="Error firmado en diferencia de energía (meV/átomo)",ylabel="Densidad (bins comunes de 16.7 meV/átomo)",title="Distribución de errores de orden por pares")
    ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    _save(fig,"15_pairwise_signed_error_distribution.png",made,"Hasta 210 pares orientados por modelo; una predicción física por par tras ensemble. Histograma normalizado por densidad y bins comunes; valores fuera del rango se reportan en las colas, no se eliminan del CSV.")

    # Figure 16 ties/collapse/abstentions.
    fig,axes=plt.subplots(1,3,figsize=(15,5)); summary2=summary
    axes[0].bar(MODELS,summary2.GS_predicted_abstentions,color=[COLORS[m] for m in MODELS]);axes[0].set_title("Empates GS predichos");axes[0].set_ylabel("Composiciones (N=70)")
    axes[1].bar(MODELS,summary2.Energy_collapse_compositions,color=[COLORS[m] for m in MODELS]);axes[1].set_title("Colapso: rango predicho <1 meV/átomo")
    tie_counts=pair_obs.assign(is_tie=pair_obs.pred_class.eq("tie")).groupby("model").is_tie.sum().reindex(MODELS)
    axes[2].bar(MODELS,tie_counts,color=[COLORS[m] for m in MODELS]);axes[2].set_title("Empates predichos entre pares");axes[2].set_ylabel("Pares (N≤210)")
    for ax in axes: ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.2)
    _save(fig,"16_ties_and_collapse.png",made,"Conteos físicos del ensemble de semillas; tres pares canónicos por composición. D1 es constante por composición y por ello abstiene en GS y colapsa el rango.")

    # Figure 17 mean-error vs centered residual error.
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True);axes=axes.ravel()
    for ax,m in zip(axes,MODELS):
        g=comp.loc[comp.model.eq(m)]; ax.scatter(g.MAE_mu_mev_atom,g.MAE_delta_centered_mev_atom,s=24,alpha=.75,color=COLORS[m])
        ax.set_title(f"{m} | N=70");ax.grid(alpha=.2)
    for ax in axes[len(MODELS):]:ax.axis("off")
    axes[0].set_ylabel("MAE residual estructural centrado (meV/átomo)");axes[4].set_ylabel("MAE residual estructural centrado (meV/átomo)")
    for ax in axes[4:7]:ax.set_xlabel("Error absoluto de μ composicional (meV/átomo)")
    fig.suptitle("Error de nivel medio frente a error estructural por composición")
    _save(fig,"17_mean_vs_residual_error.png",made,"Cada punto = una composición; el error de media y el residuo se miden en meV/átomo. Son diagnósticos separados; sus MAE no se suman para obtener el MAE absoluto.")

    # Figure 18 training curves across active H3-C runs.
    history_files=sorted((REPORT_DIR/"H3_C_ridge_retrained_delta/runs").glob("fold_*_seed_*/training_history.csv"))
    histories=[pd.read_csv(p) for p in history_files]
    hist=pd.concat(histories,ignore_index=True);hist.to_csv(REPORT_DIR/"comparative_analysis/H3C_training_curves_data.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,6))
    for _,g in hist.groupby(["fold","seed"]): ax.plot(g.epoch,g.inner_validation_loss_delta,color="#e07a1f",alpha=.17,lw=.9)
    grouped=hist.groupby("epoch").inner_validation_loss_delta.agg(["mean","std","count"])
    ax.plot(grouped.index,grouped["mean"],color="#c83c61",lw=2.4,label="Media de ejecuciones activas")
    ax.fill_between(grouped.index,grouped["mean"]-grouped["std"],grouped["mean"]+grouped["std"],color="#c83c61",alpha=.16,label="±1 DE entre ejecuciones activas")
    ax.set(xlabel="Época local de MLP8",ylabel="SmoothL1 de residuo normalizado",title="H3-C: selección de época por inner-validation")
    ax.grid(alpha=.2);ax.legend()
    _save(fig,"18_h3c_training_curves.png",made,"15 ejecuciones (5 folds×3 semillas); líneas finas = una corrida, banda/media = corridas que aún tienen esa época. La selección de checkpoint usa solo inner-validation; no se promedian épocas posteriores al early stopping.")

    # Figure 19 Ridge alpha sensitivity.
    fig,ax=plt.subplots(figsize=(9,5.5))
    for fold,g in alpha.groupby("fold"):
        ax.plot(g.alpha,g.inner_validation_mae_ev_atom*1000,marker="o",label=f"Fold {fold}")
    ax.set_xscale("log");ax.set(xlabel="Ridge α (escala logarítmica)",ylabel="MAE inner-validation (meV/átomo)",title="Selección anidada de α Ridge")
    ax.grid(alpha=.22);ax.legend(ncol=5)
    _save(fig,"19_ridge_alpha_sensitivity.png",made,"N=45 composiciones inner-train y 11 inner-validation por fold; un target medio por fórmula. La curva no incluye outer-validation y α elegido se refit en las 56 composiciones outer-train.")

    # Figure 20 H3 vs H2 paired composition ΔMAE.
    fig,ax=plt.subplots(figsize=(11,5.8)); comparisons=[("H3-A","H2"),("H3-B","H2"),("H3-C","H2")]
    data_box=[];labels=[]
    for candidate,ref in comparisons:
        a=comp.loc[comp.model.eq(candidate)].set_index("composition_id");b=comp.loc[comp.model.eq(ref)].set_index("composition_id")
        data_box.append((a.MAE_absolute_mev_atom-b.MAE_absolute_mev_atom).to_numpy());labels.append(f"{candidate} − {ref}")
    box=ax.boxplot(data_box,tick_labels=labels,patch_artist=True,showfliers=True)
    for patch,c in zip(box["boxes"],[COLORS["H3-A"],COLORS["H3-B"],COLORS["H3-C"]]):patch.set_facecolor(c);patch.set_alpha(.55)
    ax.axhline(0,color="black",linestyle="--",label="Sin cambio");ax.set_ylabel("Δ MAE absoluta por composición (meV/átomo)");ax.set_title("Cambio pareado por fórmula frente a H2");ax.grid(axis="y",alpha=.2);ax.legend()
    _save(fig,"20_paired_mae_change_vs_h2.png",made,"N=70 pares de composiciones; valores negativos favorecen H3. Un punto/caja por fórmula, bootstrap agrupado por fórmula en paired_bootstrap_ci.json; outliers conservados.")

    # Figure 21 H2 vs H3-A per-structure absolute errors.
    ens=pred.pivot(index="structure_id",columns="model",values="E_pred_eV_atom").reset_index()
    dft=pred.loc[pred.model.eq("H2"),["structure_id","E_DFT_eV_atom"]].drop_duplicates()
    err=ens.merge(dft,on="structure_id");e2=np.abs(err.H2-err.E_DFT_eV_atom)*1000;ea=np.abs(err["H3-A"]-err.E_DFT_eV_atom)*1000
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(e2,ea,s=22,alpha=.68,color=COLORS["H3-A"]);m=max(float(e2.max()),float(ea.max()));ax.plot([0,m],[0,m],"k--",label="Igual error")
    ax.set(xlabel="|error H2| (meV/átomo)",ylabel="|error H3-A| (meV/átomo)",title="Recombinación R4 media + H2 residuo");ax.grid(alpha=.2);ax.legend()
    _save(fig,"21_absolute_error_before_after_recombination.png",made,"Cada punto = una estructura de desarrollo, promedio de semillas H2; N=210. Debajo de y=x la recombinación reduce error absoluto. El ranking se conserva exactamente por construcción.")

    # Figure 22 relation mean MAE to absolute MAE.
    fig,ax=plt.subplots(figsize=(9,6))
    for m in MODELS:
        g=comp.loc[comp.model.eq(m)];ax.scatter(g.MAE_mu_mev_atom,g.MAE_absolute_mev_atom,s=25,alpha=.68,color=COLORS[m],label=m)
    ax.set(xlabel="MAE de media composicional (meV/átomo)",ylabel="MAE absoluto final (meV/átomo)",title="¿Cuánto error absoluto acompaña al nivel medio?");ax.grid(alpha=.2);ax.legend(ncol=2)
    _save(fig,"22_mean_error_vs_absolute_error.png",made,"Cada punto = composición (70/modelo), predicciones promediadas entre semillas. Para D1 la energía se repite en los tres polimorfos; el ranking es abstención.")

    # Figure 23 centered residual distributions.
    fig,ax=plt.subplots(figsize=(12,6));values=[]
    for m in MODELS: values.append(pred.loc[pred.model.eq(m),"delta_pred_centered_mev_atom"].to_numpy())
    v=ax.violinplot(values,showmeans=False,showmedians=True,showextrema=True)
    for body,m in zip(v["bodies"],MODELS):body.set_facecolor(COLORS[m]);body.set_alpha(.5)
    ax.set_xticks(range(1,len(MODELS)+1),MODELS);ax.set_ylabel("Residuo estructural predicho centrado (meV/átomo)");ax.set_title("Distribución de residuos predichos");ax.grid(axis="y",alpha=.2)
    _save(fig,"23_predicted_structural_residual_distributions.png",made,"Violin = distribución de residuos centrados de 210 estructuras por modelo; línea = mediana. H3-A/B heredan exactamente los residuos H2; D1 tiene residuos cero.")

    # Figure 24 compact energy scorecard values.
    cols=["MAE_absolute_mev_atom_macro","MAE_mu_mev_atom_macro","MAE_delta_centered_mev_atom_macro",
          "MAE_delta_GS_anchored_mev_atom_macro","MAE_delta_own_min_legacy_mev_atom_macro",
          "GS_accuracy","Pairwise_accuracy_macro_composition"]
    labels=["MAE abs.\n(meV/átomo)","MAE μ\n(meV/átomo)","MAE residuo\n(meV/átomo)","MAE ΔE anclada\n(meV/átomo)","MAE ΔE own-min\n(meV/átomo)","GS\n(0–1)","Pairwise\n(0–1)"]
    fig,ax=plt.subplots(figsize=(15,5.5));table=summary[cols].to_numpy()
    # Use transposed values so metrics are readable as rows and models as columns.
    im=ax.imshow(table.T,aspect="auto",cmap="YlGnBu")
    ax.set_xticks(range(len(MODELS)),MODELS);ax.set_yticks(range(len(labels)),labels)
    for i in range(table.T.shape[0]):
        for j in range(table.T.shape[1]):
            val=table.T[i,j]; text=f"{val:.1f}" if i<5 else f"{val:.2f}"
            ax.text(j,i,text,ha="center",va="center",fontsize=8,color="black")
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label("Valor observado; unidades en cada fila")
    ax.set_title("Scorecard energético/ranking OOF de desarrollo")
    _save(fig,"24_energy_ranking_scorecard.png",made,"Ensemble de 3 semillas antes de calcular métricas; 210 estructuras/70 composiciones. GS usa 67 composiciones elegibles; pairwise usa 206 pares con DFT gap>1 meV/átomo. No incluye test histórico.")

    write_json(REPORT_DIR / "comparative_analysis/figures_manifest.json", {
        "status":"PASS","count":len(made),"figures":made,"format":"PNG only",
        "source":"development OOF predictions and derived tables; no synthetic values",
        "test_used":False,"ensemble":"average energies across three seeds per physical structure before composition-level metrics"})
    return made


def main() -> None:
    """Genera PNG y escribe manifiesto de integridad visual."""
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        print(json.dumps({"status":"PASS","figures":generate()},indent=2,ensure_ascii=False))
    except (H3Error,OSError,ValueError,KeyError) as exc:
        raise SystemExit(f"H3 figure generation FAILED: {exc}") from exc


if __name__=="__main__":
    main()
