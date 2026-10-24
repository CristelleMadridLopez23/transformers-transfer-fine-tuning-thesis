#!/usr/bin/env python3
"""Genera 18 figuras PNG legibles del holdout final H3-TEST."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import FAMILIES, FAMILY_LABELS, REPORT_DIR, H3TestError, write_json
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.metrics import summarize_model

COLORS = {"H2": "#2878b5", "H3-A": "#cc3f68", "DFT": "#4d5561"}
MARKERS = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}
MODEL_COLS = {"H2": "E_pred_H2_eV_atom", "H3-A": "E_pred_H3A_eV_atom"}
OUT = REPORT_DIR / "figures"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga etiquetas ya evaluadas, OOF development y tablas bloqueadas.

    Returns:
        Tabla por estructura, métricas test, OOF desarrollo y comparación dev/test.
    """
    pred = REPORT_DIR / "predictions/test_predictions_with_dft.csv"
    metrics = REPORT_DIR / "metrics/test_metrics_summary.csv"
    dev = REPORT_DIR.parent / "hybrid_ridge_mlp8/comparative_analysis/comparative_metrics.csv"
    comp = REPORT_DIR / "comparisons/development_vs_test.csv"
    required = [pred, metrics, dev, comp]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise H3TestError(f"Faltan artefactos para figuras: {missing}")
    frame = pd.read_csv(pred)
    if len(frame) != 24 or frame.groupby("composition_id").size().ne(3).any():
        raise H3TestError("Las figuras requieren 24 estructuras/8 tripletes completos.")
    return frame, pd.read_csv(metrics), pd.read_csv(dev), pd.read_csv(comp)


def _save(fig: plt.Figure, name: str, note: str, files: list[str]) -> None:
    """Guarda PNG con nota y registra el nombre de archivo."""
    fig.text(.015, .012, note, ha="left", va="bottom", fontsize=8, color="#343a40", wrap=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    files.append(name)


def _relative(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcula ΔE anclada y own-min por composición sin recorte de negativos."""
    result = frame.copy()
    result["dft_anchor_mev"] = np.nan
    for formula, group in result.groupby("composition_id", sort=False):
        dft_gs = group.E_DFT_eV_atom.idxmin()
        result.loc[group.index, "dft_anchor_mev"] = (group.E_DFT_eV_atom - result.loc[dft_gs, "E_DFT_eV_atom"]) * 1000
        for model, col in MODEL_COLS.items():
            result.loc[group.index, f"{model}_anchor_mev"] = (group[col] - result.loc[dft_gs, col]) * 1000
            result.loc[group.index, f"{model}_own_min_mev"] = (group[col] - group[col].min()) * 1000
    return result


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Estima OLS libre, R² OLS, R² predictivo y MAE contra identidad."""
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope*x + intercept
    denom = np.sum((y-y.mean())**2)
    r2_ols = 1-np.sum((y-fitted)**2)/denom if denom else np.nan
    denom_pred = np.sum((x-x.mean())**2)
    r2_pred = 1-np.sum((y-x)**2)/denom_pred if denom_pred else np.nan
    return float(slope), float(intercept), float(r2_ols), float(r2_pred), float(np.mean(np.abs(y-x)))


def _parity(frame: pd.DataFrame, xcol: str, ycols: dict[str, str], x_label: str,
            title: str, unit: str, note: str, name: str,
            regression_rows: list[dict[str, Any]], files: list[str]) -> None:
    """Dibuja paridad DFT-x/predicción-y comparable entre H2 y H3-A."""
    arrays = [frame[xcol].to_numpy(float)] + [frame[col].to_numpy(float) for col in ycols.values()]
    low = min(float(np.nanmin(a)) for a in arrays); high = max(float(np.nanmax(a)) for a in arrays)
    pad = max((high-low)*.06, 1e-5); limits = (low-pad, high+pad)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    grid = np.linspace(*limits, 100)
    for ax, (model, ycol) in zip(axes, ycols.items()):
        x = frame[xcol].to_numpy(float); y = frame[ycol].to_numpy(float)
        slope, intercept, r2_ols, r2_pred, mae = _ols(x, y)
        if "polymorph_family" in frame.columns:
            for family, marker in MARKERS.items():
                mask = frame.polymorph_family.eq(family).to_numpy()
                ax.scatter(x[mask], y[mask], marker=marker, s=47, alpha=.78, color=COLORS[model],
                           edgecolor="white", linewidth=.5, label=FAMILY_LABELS[family])
        else:
            ax.scatter(x, y, marker="o", s=52, alpha=.8, color=COLORS[model],
                       edgecolor="white", linewidth=.5, label="Una marca por composición")
        ax.plot(grid, grid, color="#202020", ls="--", lw=1.3, label="Ideal: y=x")
        ax.plot(grid, slope*grid+intercept, color="#d33682", ls="-.", lw=1.5, label="OLS libre")
        ax.set(xlim=limits, ylim=limits, title=(f"{model} | N=24 estructuras, 8 fórmulas\n"
            f"m={slope:.3f}; b={intercept:.2f} {unit}\nR² OLS={r2_ols:.3f}; R² pred={r2_pred:.3f}; MAE={mae:.2f} {unit}"))
        ax.title.set_fontsize(10)
        ax.grid(alpha=.22)
        regression_rows.append({"figure": name, "model": model, "N_structures": len(frame),
            "N_compositions": frame.composition_id.nunique(), "slope_dimensionless": slope,
            "intercept": intercept, "intercept_unit": unit, "R2_OLS": r2_ols,
            "R2_predictive_y_equals_x": r2_pred, "MAE_vs_DFT": mae, "MAE_unit": unit})
    axes[0].set_ylabel(f"Predicción ({unit})"); axes[0].set_xlabel(x_label)
    axes[1].set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=5, fontsize=8)
    fig.suptitle(title, y=1.12, fontsize=13); fig.tight_layout(rect=(0, .055, 1, .93))
    _save(fig, name, note, files)


def _confusion_figure(frame: pd.DataFrame, files: list[str]) -> None:
    """Genera matrices de GS y pares con conteos y porcentaje por fila."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    gs_classes = list(FAMILIES) + ["tie/abstain"]
    gs_labels = ["P", "H", "A", "Empate"]
    gs_mats = []
    for model in ("H2", "H3-A"):
        cmat = np.zeros((3, 4), int)
        for _, group in frame.groupby("composition_id"):
            group = group.set_index("polymorph_family").loc[list(FAMILIES)]
            truth = str(group.E_DFT_GS_family.iloc[0]) if "E_DFT_GS_family" in group else str(group.loc[group.E_DFT_eV_atom.idxmin()].name)
            sorted_dft = np.sort(group.E_DFT_eV_atom.to_numpy(float))
            if (sorted_dft[1]-sorted_dft[0])*1000 <= 1.0:
                continue
            values = group[MODEL_COLS[model]].to_numpy(float)
            nearest = np.flatnonzero((values-values.min())*1000 <= 1.0)
            predicted = "tie/abstain" if len(nearest) != 1 else FAMILIES[int(nearest[0])]
            cmat[FAMILIES.index(truth), gs_classes.index(predicted)] += 1
        gs_mats.append(cmat)
    for ax, model, cmat in zip(axes, ("H2", "H3-A"), gs_mats):
        row_sum = cmat.sum(axis=1, keepdims=True)
        pct = np.divide(cmat, row_sum, out=np.zeros_like(cmat, dtype=float), where=row_sum>0)*100
        image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{cmat[i,j]}\n{pct[i,j]:.0f}%" if row_sum[i,0] else "0\n—",
                        ha="center", va="center", fontsize=9)
        ax.set(xticks=range(4), xticklabels=gs_labels, yticks=range(3), yticklabels=["P", "H", "A"],
               xlabel="Polimorfo predicho", ylabel="Ground state DFT", title=f"{model} | N elegible={int(row_sum.sum())}")
    cb = fig.colorbar(image, ax=axes, fraction=.04, pad=.04); cb.set_label("Porcentaje dentro de fila (%)")
    fig.suptitle("Estado fundamental por composición", y=1.02)
    _save(fig, "06_ground_state_confusion.png", "Cada fila es el polimorfo de menor energía DFT; columnas = predicción. P=perovskita distorsionada, H=hexagonal, A=tipo aguja. Se excluyen composiciones con brecha DFT ≤1 meV/átomo; predicción empatada es abstención.", files)

    pair_classes = ["i_lower", "tie", "j_lower"]
    pair_labels = ["i<j", "Empate", "j<i"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, model in zip(axes, ("H2", "H3-A")):
        cmat = np.zeros((3,3), int)
        for _, group in frame.groupby("composition_id"):
            group = group.set_index("polymorph_family").loc[list(FAMILIES)]
            y = group.E_DFT_eV_atom.to_numpy(float); p = group[MODEL_COLS[model]].to_numpy(float)
            for i,j in combinations(range(3),2):
                truth = "i_lower" if (y[i]-y[j])*1000 < -1 else "j_lower" if (y[i]-y[j])*1000 > 1 else "tie"
                guess = "i_lower" if (p[i]-p[j])*1000 < -1 else "j_lower" if (p[i]-p[j])*1000 > 1 else "tie"
                cmat[pair_classes.index(truth), pair_classes.index(guess)] += 1
        row_sum = cmat.sum(axis=1, keepdims=True)
        pct = np.divide(cmat, row_sum, out=np.zeros_like(cmat,dtype=float), where=row_sum>0)*100
        image = ax.imshow(pct, cmap="Purples", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3): ax.text(j,i,f"{cmat[i,j]}\n{pct[i,j]:.0f}%" if row_sum[i,0] else "0\n—",ha="center",va="center",fontsize=9)
        ax.set(xticks=range(3),xticklabels=pair_labels,yticks=range(3),yticklabels=pair_labels,
               xlabel="Relación predicha",ylabel="Relación DFT",title=f"{model} | 24 pares físicos")
    cb=fig.colorbar(image,ax=axes,fraction=.04,pad=.04);cb.set_label("Porcentaje dentro de fila (%)")
    fig.suptitle("Orden de las tres parejas canónicas por composición",y=1.02)
    _save(fig,"07_pairwise_confusion.png", "Filas = relación DFT; columnas = predicción. Orden fijo (perovskita, hexagonal), (perovskita, aguja), (hexagonal, aguja); diferencia Eᵢ−Eⱼ. Tolerancia ±1 meV/átomo. N máximo=24 pares, ties DFT incluidos para diagnóstico.",files)


def generate() -> dict[str, Any]:
    """Crea 18 figuras PNG y exporta las estadísticas de regresión.

    Returns:
        Manifiesto de figuras generadas exclusivamente desde tablas reales.
    """
    if OUT.exists() and any(OUT.glob("*.png")):
        raise H3TestError("Ya hay figuras PNG; se preservan los resultados y no se sobrescriben.")
    OUT.mkdir(parents=True, exist_ok=True)
    frame, test_metrics, dev_metrics, comparison = _load()
    frame = _relative(frame)
    frame["E_DFT_absolute_mev_atom"] = frame.E_DFT_eV_atom * 1000.0
    frame["E_pred_H2_absolute_mev_atom"] = frame.E_pred_H2_eV_atom * 1000.0
    frame["E_pred_H3A_absolute_mev_atom"] = frame.E_pred_H3A_eV_atom * 1000.0
    made: list[str] = []; regression: list[dict[str, Any]] = []
    # 1. Absolute parity.
    _parity(frame,"E_DFT_absolute_mev_atom",{"H2":"E_pred_H2_absolute_mev_atom","H3-A":"E_pred_H3A_absolute_mev_atom"},"Energía DFT (meV/átomo)","ABS₃ test: paridad de energía absoluta","meV/átomo",
        "Cada punto = una estructura estática del holdout; H2 y H3-A promedian los mismos 5 folds × 3 semillas. Formas identifican familia; y=x es la referencia predictiva.","01_absolute_energy_parity.png",regression,made)
    # 2. Composition mean parity.
    means=frame.groupby("composition_id",as_index=False).agg(E_DFT_mu=("E_DFT_eV_atom","mean"),
        E_pred_H2=("E_pred_H2_eV_atom","mean"),E_pred_H3A=("E_pred_H3A_eV_atom","mean"))
    for column in ("E_DFT_mu", "E_pred_H2", "E_pred_H3A"):
        means[column] *= 1000.0
    means_file=REPORT_DIR/"metrics/test_composition_energy_means.csv";means.to_csv(means_file,index=False)
    mean_frame=means.rename(columns={"E_DFT_mu":"x","E_pred_H2":"H2","E_pred_H3A":"H3-A"})
    _parity(mean_frame,"x",{"H2":"H2","H3-A":"H3-A"},"Media DFT por fórmula μ (meV/átomo)",
        "ABS₃ test: paridad de la media energética composicional","meV/átomo",
        "Una marca = una de 8 fórmulas; las energías se promedian primero entre sus 3 polimorfos. Los paneles usan el mismo ensemble final.","02_composition_mean_parity.png",regression,made)
    # 3. Centered residual parity.
    frame["DFT_delta_centered_mev"]=(frame.E_DFT_eV_atom-frame.groupby("composition_id").E_DFT_eV_atom.transform("mean"))*1000
    frame["H2_delta_centered_mev"]=(frame.E_pred_H2_eV_atom-frame.groupby("composition_id").E_pred_H2_eV_atom.transform("mean"))*1000
    frame["H3A_delta_centered_mev"]=(frame.E_pred_H3A_eV_atom-frame.groupby("composition_id").E_pred_H3A_eV_atom.transform("mean"))*1000
    residual_frame=frame.rename(columns={"DFT_delta_centered_mev":"x","H2_delta_centered_mev":"H2","H3A_delta_centered_mev":"H3-A"})
    _parity(residual_frame,"x",{"H2":"H2","H3-A":"H3-A"},"Residuo DFT centrado δ (meV/átomo)",
        "ABS₃ test: paridad de residuos estructurales centrados","meV/átomo",
        "N=24 polimorfos/8 fórmulas. Cada triplete se centra con su propia media predicha y DFT; origen y=0 representa desviación media cero.","03_structural_residual_parity.png",regression,made)
    # 4–5. Two distinct ΔE definitions.
    _parity(frame,"dft_anchor_mev",{"H2":"H2_anchor_mev","H3-A":"H3-A_anchor_mev"},
        "ΔE DFT sobre el mínimo DFT de cada fórmula (meV/átomo)","ABS₃ test: ΔE anclada al mismo ground state DFT","meV/átomo",
        "ΔEᵢ=Eᵢ−E_min,DFT dentro de la fórmula en ambos ejes. Valores negativos de predicción muestran inversión; N=24, 8 fórmulas.","04_dft_gs_anchored_delta_parity.png",regression,made)
    frame["dft_own_min_mev"]=(frame.E_DFT_eV_atom-frame.groupby("composition_id").E_DFT_eV_atom.transform("min"))*1000
    _parity(frame,"dft_own_min_mev",{"H2":"H2_own_min_mev","H3-A":"H3-A_own_min_mev"},
        "ΔE DFT respecto a su mínimo DFT (meV/átomo)","ABS₃ test: ΔE own-min, convención histórica separada","meV/átomo",
        "Ambos ejes usan su propio mínimo por fórmula; esta métrica no equivale a anclar predicción al GS DFT. N=24/8.","05_own_min_delta_parity.png",regression,made)
    _confusion_figure(frame,made)
    # 8. Energy triplets.
    fig,axes=plt.subplots(4,2,figsize=(13,15),sharex=True);axes=axes.ravel()
    labels=[FAMILY_LABELS[f] for f in FAMILIES]
    for ax,(formula,group) in zip(axes,frame.groupby("composition_id",sort=True)):
        group=group.set_index("polymorph_family").loc[list(FAMILIES)];x=np.arange(3)
        ax.plot(x,group.E_DFT_eV_atom.to_numpy(),marker="o",color=COLORS["DFT"],label="DFT")
        ax.plot(x,group.E_pred_H2_eV_atom.to_numpy(),marker="o",color=COLORS["H2"],label="H2")
        ax.plot(x,group.E_pred_H3A_eV_atom.to_numpy(),marker="o",color=COLORS["H3-A"],label="H3-A")
        ax.set_title(formula);ax.set_xticks(x, ["P","H","A"]);ax.grid(axis="y",alpha=.2);ax.set_ylabel("Energía (eV/átomo)")
    handles,legend=[*axes[0].get_legend_handles_labels()];fig.legend(handles,legend,loc="upper center",ncol=3,bbox_to_anchor=(.5,1.01))
    fig.suptitle("Energías absolutas por triplete de polimorfos",y=1.03)
    _save(fig,"08_energy_triplets_by_composition.png","Cada panel = una de 8 fórmulas; puntos conectados por modelo sobre tres familias. Las energías son estáticas por átomo. P=perovskita, H=hexagonal, A=aguja; escala vertical independiente por panel.",made)
    # 9–10. Development vs test.
    for number,label,dev_col,test_col,unit in [
        (9,"MAE absoluto macro","MAE_absolute_mev_atom_macro","MAE_absolute_mev_atom_macro_composition","meV/átomo"),
        (10,"MAE ΔE anclada DFT-GS","MAE_delta_GS_anchored_mev_atom_macro","MAE_delta_DFT_GS_anchored_mev_atom_macro","meV/átomo")]:
        fig,ax=plt.subplots(figsize=(9,5.4));width=.34;positions=np.arange(2)
        for i,model in enumerate(("H2","H3-A")):
            d=float(dev_metrics.set_index("model").loc[model,dev_col]);t=float(test_metrics.set_index("model").loc[model,test_col])
            ax.bar(i-width/2,d,width,color=COLORS[model],alpha=.4,label="Desarrollo OOF" if i==0 else None)
            ax.bar(i+width/2,t,width,color=COLORS[model],alpha=1,label="Test ensemble" if i==0 else None)
            ax.text(i-width/2,d,f"{d:.1f}",ha="center",va="bottom",fontsize=8);ax.text(i+width/2,t,f"{t:.1f}",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(positions,["H2","H3-A"]);ax.set_ylabel(f"{label} ({unit})");ax.set_title(f"Desarrollo OOF y holdout: {label}");ax.grid(axis="y",alpha=.2);ax.legend()
        _save(fig,f"{number:02d}_development_test_{'absolute_mae' if number==9 else 'anchored_delta_mae'}.png",
              "Desarrollo = OOF de 70 composiciones; test = ensemble de 5 modelos Ridge × 15 cabezas H2 en 8 composiciones. El color identifica H2/H3-A y el tono distingue partición. Estimadores distintos; diferencia descriptiva, no prueba domain shift.",made)
    # 11–12. Per-composition errors.
    metric_comp=pd.read_csv(REPORT_DIR/"metrics/test_metrics_by_composition.csv")
    for number,metric,title,filename in [(11,"MAE_absolute_mev_atom","MAE absoluto por fórmula","11_absolute_error_by_composition.png"),
        (12,"MAE_delta_GS_anchored_mev_atom","MAE ΔE anclada por fórmula","12_relative_error_by_composition.png")]:
        pivot=metric_comp.pivot(index="composition_id",columns="model",values=metric).reindex(sorted(metric_comp.composition_id.unique()))
        fig,ax=plt.subplots(figsize=(12,5.8));x=np.arange(len(pivot));ax.scatter(x-.08,pivot["H2"],color=COLORS["H2"],label="H2",s=40)
        ax.scatter(x+.08,pivot["H3-A"],color=COLORS["H3-A"],label="H3-A",s=40)
        for i,formula in enumerate(pivot.index):ax.plot([i-.08,i+.08],[pivot.loc[formula,"H2"],pivot.loc[formula,"H3-A"]],color="#adb5bd",lw=.8)
        ax.set_xticks(x,pivot.index,rotation=30,ha="right");ax.set_ylabel("MAE (meV/átomo)");ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.legend()
        _save(fig,filename,"Una observación por composición (8); puntos pareados H2/H3-A, línea une la misma fórmula. No se omiten colas ni composiciones.",made)
    # 13. DFT vs predicted ranges.
    fig,ax=plt.subplots(figsize=(8,6));comp_rows=metric_comp.loc[metric_comp.model.eq("H2")].merge(
        metric_comp.loc[metric_comp.model.eq("H3-A"),["composition_id","predicted_range_mev_atom"]],on="composition_id",suffixes=("_H2","_H3A"))
    maxv=float(max(comp_rows.DFT_range_mev_atom.max(),comp_rows.predicted_range_mev_atom_H2.max(),comp_rows.predicted_range_mev_atom_H3A.max()))
    for model,col,marker in (("H2","predicted_range_mev_atom_H2","o"),("H3-A","predicted_range_mev_atom_H3A","s")):
        ax.scatter(comp_rows.DFT_range_mev_atom,comp_rows[col],color=COLORS[model],marker=marker,s=55,label=model)
        for row in comp_rows.itertuples(index=False):ax.annotate(row.composition_id,(row.DFT_range_mev_atom,getattr(row,col)),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.plot([0,maxv],[0,maxv],"k--",label="Igualdad de rangos");ax.set(xlabel="Rango DFT max−min (meV/átomo)",ylabel="Rango predicho max−min (meV/átomo)",title="Rangos polimórficos por fórmula");ax.grid(alpha=.2);ax.legend()
    _save(fig,"13_dft_vs_predicted_energy_ranges.png","Cada punto = una fórmula de test; la diagonal indica rango predicho igual al DFT. Mismo conjunto de 8 composiciones; rangos dentro de cada triplete.",made)
    # 14. MSE decomposition.
    fig,ax=plt.subplots(figsize=(8.5,5.5));summary=test_metrics.set_index("model").loc[["H2","H3-A"]]
    x=np.arange(2);w=.34
    ax.bar(x-w/2,summary.MSE_mu_mev2_atom2,w,color=[COLORS[m] for m in summary.index],label="MSE del nivel μ")
    ax.bar(x+w/2,summary.MSE_residual_mev2_atom2,w,color="#adb5bd",edgecolor=[COLORS[m] for m in summary.index],label="MSE residual centrado")
    ax.set_xticks(x,summary.index);ax.set_ylabel("Promedio por fórmula (meV²/átomo²)");ax.set_title("Descomposición de MSE por composición");ax.grid(axis="y",alpha=.2);ax.legend()
    _save(fig,"14_mse_mean_residual_decomposition.png","MSE_total = MSE_mu + MSE_residuo para tripletes completos. N=8 fórmulas; identidad cuadrática exacta salvo tolerancia numérica. Los MAE no son aditivos.",made)
    # 15. Paired absolute-error change per structure.
    frame["delta_abs_error_h3_minus_h2"]=(np.abs(frame.E_pred_H3A_eV_atom-frame.E_DFT_eV_atom)-np.abs(frame.E_pred_H2_eV_atom-frame.E_DFT_eV_atom))*1000
    fig,ax=plt.subplots(figsize=(11,5.5));formula_order=sorted(frame.composition_id.unique());positions={f:i for i,f in enumerate(formula_order)}
    for family,marker in MARKERS.items():
        sub=frame.loc[frame.polymorph_family.eq(family)];ax.scatter([positions[f] for f in sub.composition_id],sub.delta_abs_error_h3_minus_h2,
            marker=marker,s=48,label=FAMILY_LABELS[family],alpha=.8)
    ax.axhline(0,color="black",ls="--",lw=1,label="Sin cambio");ax.set_xticks(range(8),formula_order,rotation=30,ha="right")
    ax.set_ylabel("|error H3-A| − |error H2| (meV/átomo)");ax.set_title("Cambio pareado de error absoluto por estructura");ax.grid(axis="y",alpha=.2);ax.legend(ncol=2)
    _save(fig,"15_paired_absolute_error_difference.png","N=24 estructuras/8 fórmulas; cada marcador compara la misma etiqueta DFT. Negativo favorece H3-A; no es una muestra independiente por fórmula.",made)
    # 16. Error distributions, raw points retained.
    fig,ax=plt.subplots(figsize=(8.5,5.5));data=[]
    for model,col in MODEL_COLS.items():data.append(np.abs(frame[col]-frame.E_DFT_eV_atom).to_numpy()*1000)
    boxes=ax.boxplot(data,tick_labels=list(MODEL_COLS),patch_artist=True,showfliers=True,showmeans=True)
    for box,model in zip(boxes["boxes"],MODEL_COLS):box.set_facecolor(COLORS[model]);box.set_alpha(.48)
    rng=np.random.default_rng(13072026)
    for i,(values,model) in enumerate(zip(data,MODEL_COLS)):
        ax.scatter(i+1+rng.uniform(-.08,.08,len(values)),values,s=16,alpha=.55,color=COLORS[model])
    ax.set_ylabel("Error absoluto (meV/átomo)");ax.set_title("Distribución de errores absolutos en test");ax.grid(axis="y",alpha=.2)
    _save(fig,"16_absolute_error_distribution.png","Cada punto = estructura (N=24); caja Q1–Q3, línea mediana, bigotes 1.5×IQR, diamante media, extremos visibles. Interpretar junto a errores por fórmula; el triplete induce dependencia.",made)
    # 17. Fold/seed variability from same predictions, descriptive only.
    run_rows=[]
    perrun=pd.read_csv(REPORT_DIR/"predictions/test_predictions_prelabels.csv")
    for (fold,seed),group in perrun.groupby(["fold","seed"]):
        merged=frame[["structure_id","composition_id","polymorph_family","E_DFT_eV_atom"]].merge(
            group,on=["structure_id","composition_id","polymorph_family"],validate="one_to_one")
        for model,col in (("H2","E_pred_H2_eV_atom"),("H3-A","E_pred_H3A_eV_atom")):
            run_summary,_,_=summarize_model(merged,col)
            run_rows.append({"fold":fold,"seed":seed,"model":model,"MAE_absolute_mev_atom":run_summary["MAE_absolute_mev_atom_macro_composition"],
                             "GS_accuracy":run_summary["GS_accuracy"],"pairwise_accuracy":run_summary["pairwise_accuracy"]})
    runs=pd.DataFrame(run_rows);runs.to_csv(REPORT_DIR/"metrics/test_metrics_by_fold_seed.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for model in ("H2","H3-A"):
        subset=runs.loc[runs.model.eq(model)]
        for seed in (42,123,2026):
            s=subset.loc[subset.seed.eq(seed)].sort_values("fold")
            axes[0].plot(s.fold,s.MAE_absolute_mev_atom,marker="o",color=COLORS[model],alpha=.35,
                         label=f"{model}, semilla {seed}" if seed==42 else None)
        axes[1].scatter(subset.fold+(0 if model=="H2" else .12),subset.GS_accuracy,color=COLORS[model],label=model,s=40)
    axes[0].set(xlabel="Modelo entrenado en fold",ylabel="MAE absoluto macro (meV/átomo)",title="MAE del ensemble parcial por fold/semilla")
    axes[1].set(xlabel="Fold",ylabel="GS accuracy (fracción elegible)",title="Ground state por fold × semilla")
    for ax in axes:ax.grid(alpha=.2);ax.set_xticks(range(5))
    axes[0].legend(fontsize=7,ncol=2);axes[1].legend()
    _save(fig,"17_fold_seed_variability.png","15 variantes evaluadas sobre las mismas 8 fórmulas; son miembros del ensemble, no 15 muestras independientes. Resultados descriptivos; ensemble principal promedia todos los miembros.",made)
    # 18. Scorecard.
    test_idx=test_metrics.set_index("model")
    rows=[("MAE absoluto macro (meV/átomo)","MAE_absolute_mev_atom_macro_composition","{:.2f}"),
          ("RMSE absoluto pooled (meV/átomo)","RMSE_absolute_mev_atom","{:.2f}"),
          ("MAE media composicional (meV/átomo)","MAE_mu_mev_atom_macro","{:.2f}"),
          ("MAE residuo centrado (meV/átomo)","MAE_delta_centered_mev_atom_macro","{:.2f}"),
          ("MAE ΔE anclada a DFT-GS (meV/átomo)","MAE_delta_DFT_GS_anchored_mev_atom_macro","{:.2f}"),
          ("MAE ΔE own-min (meV/átomo)","MAE_delta_own_min_mev_atom_macro","{:.2f}"),
          ("Acierto de estado fundamental","GS_accuracy","{:.1%}"),
          ("Acierto de ordenamiento por pares","pairwise_accuracy","{:.1%}")]
    cell=[]
    for label,key,fmt in rows:cell.append([label,fmt.format(float(test_idx.loc["H2",key])),fmt.format(float(test_idx.loc["H3-A",key]))])
    fig,ax=plt.subplots(figsize=(12,5.8));ax.axis("off")
    table=ax.table(cellText=cell,colLabels=["Métrica","H2","H3-A"],cellLoc="center",colLoc="center",loc="center")
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.55)
    for col,model in ((1,"H2"),(2,"H3-A")):
        table[(0,col)].set_facecolor(COLORS[model]);table[(0,col)].set_text_props(color="white",weight="bold")
    table[(0,0)].set_facecolor("#dee2e6");table[(0,0)].set_text_props(weight="bold")
    for row in range(1,len(cell)+1):table[(row,0)].set_text_props(ha="left")
    ax.set_title("H3-TEST: scorecard del holdout de ocho composiciones",pad=16)
    _save(fig,"18_test_energy_ranking_scorecard.png","N=24 estructuras/8 composiciones. Valores sobre ensemble 5 folds × 3 semillas; GS denominador elegible y pairwise excluye pares DFT empatados a 1 meV/átomo. H3-A/H2 ΔE y ranking deben coincidir por construcción.",made)

    stats_path=REPORT_DIR/"metrics/figure_regression_statistics.csv"
    pd.DataFrame(regression).to_csv(stats_path,index=False)
    non_png=[str(p) for p in OUT.rglob("*") if p.is_file() and p.suffix.lower()!=".png"]
    if non_png or len(made)!=18:
        raise H3TestError(f"Figura(s) inesperadas; se esperan 18 PNG exclusivamente: {non_png}")
    manifest={"status":"PASS","files":made,"figure_count":len(made),"format":"PNG only",
              "test_labels_used_for_scoring_only":True,"regression_statistics_csv":str(stats_path.relative_to(REPORT_DIR)),
              "figure_generator_sha256":__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
              "figure_generator_revised_after_scoring":True,
              "revision_scope":"Render-only: corrected eV-to-meV display, parity-statistics title spacing, Spanish scorecard and clear development/test legend; predictions/metrics unchanged."}
    write_json(OUT/"figures_manifest.json",manifest)
    return manifest


def main() -> None:
    """Genera y verifica el conjunto de figuras PNG."""
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        result=generate();print(json.dumps(result,ensure_ascii=False))
    except (H3TestError,OSError,ValueError,KeyError,RuntimeError) as exc:
        raise SystemExit(f"H3-TEST figures FAILED: {exc}") from exc


if __name__=="__main__":main()
