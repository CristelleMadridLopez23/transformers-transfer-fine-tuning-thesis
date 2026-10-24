#!/usr/bin/env python3
"""Genera la batería científica comparable R1-R4 en formato PNG solamente."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (FAMILIES, FAMILY_LABELS, H2_REPORT, REPORT_DIR, ExperimentError, log)

COLORS = {"H0": "#475569", "H1": "#2563EB", "H2": "#059669", "R1-A-ABS": "#D97706",
    "R1-D-COMBINED": "#C2410C", "R2-SMALL4": "#9333EA", "R2-REF8": "#7C3AED",
    "R3-DROP30": "#DB2777", "R3-WD1E2": "#BE123C", "R3-EARLY8": "#F472B6",
    "R4-SELECTED-ABSOLUTE": "#0891B2", "R4-SELECTED-2COMP": "#0E7490",
    "R4-FIXED-FINAL_END": "#64748B", "R4-FIXED-FINAL_MEAN": "#14B8A6"}
MARKERS = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}
LABELS = {"H0": "H0", "H1": "H1", "H2": "H2", "R1-A-ABS": "R1-A Ridge END energía",
    "R1-D-COMBINED": "R1-D Ridge END μ+δ", "R2-SMALL4": "R2 ancho 4", "R2-REF8": "R2 referencia ancho 8",
    "R3-DROP30": "R3 dropout 0.30", "R3-WD1E2": "R3 weight decay 0.01", "R3-EARLY8": "R3 paciencia 8",
    "R4-SELECTED-ABSOLUTE": "R4 Ridge capa/pooling elegido (E)",
    "R4-SELECTED-2COMP": "R4 Ridge μ+δ elegido", "R4-FIXED-FINAL_END": "R4 control final END",
    "R4-FIXED-FINAL_MEAN": "R4 control final MEAN"}


def save_png(fig: plt.Figure, path: Path) -> None:
    """Guarda figura con extensión PNG explícita y cierra el recurso."""
    if path.suffix.lower() != ".png":
        raise ExperimentError(f"Solo se permite PNG en figuras: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, format="png", bbox_inches="tight")
    plt.close(fig)
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ExperimentError(f"Firma PNG inválida: {path}")


def parity_panel(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str, title: str, units: str,
                 reference: str = "DFT", note_metric: str = "MAE",
                 shared_limits: tuple[float,float] | None = None) -> dict[str, float]:
    """Pinta parity y=x, OLS y=mx+b, símbolos de familia y estadísticos."""
    xx, yy = frame[x].to_numpy(float), frame[y].to_numpy(float)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[finite], yy[finite]
    for family in FAMILIES:
        select = frame.loc[finite, "polymorph_family"].eq(family).to_numpy()
        ax.scatter(xx[select], yy[select], marker=MARKERS[family], color=frame.loc[finite, "_color"].iloc[0],
                   alpha=.58, s=16, edgecolors="none")
    slope, intercept = np.polyfit(xx, yy, 1) if len(xx) >= 2 else (np.nan, np.nan)
    fit = slope * xx + intercept
    r2_ols = r2_score(yy, fit) if len(xx) >= 2 else np.nan
    r2_pred = r2_score(xx, yy) if len(xx) >= 2 else np.nan
    mae = float(np.mean(np.abs(yy - xx)))
    if shared_limits is None:
        lo, hi = min(xx.min(), yy.min()), max(xx.max(), yy.max())
        pad = .04 * max(hi - lo, 1e-8)
        lo, hi = lo - pad, hi + pad
    else:
        lo,hi=shared_limits
    ax.plot([lo, hi], [lo, hi], "--", color="#111827", linewidth=1, label="Acuerdo ideal y=x")
    ax.plot([lo, hi], [slope*lo+intercept, slope*hi+intercept], color="#DB2777", linewidth=1.3, label="OLS libre y=mx+b")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=title,
           xlabel=f"{reference} ({units})", ylabel=f"Predicción ({units})")
    ax.text(.03, .97, f"N={len(xx)}\nm={slope:.3f}; b={intercept:.2f} {units}\nR² OLS={r2_ols:.3f}; R² y=x={r2_pred:.3f}\n{note_metric}={mae:.2f} {units}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            bbox={"boxstyle":"round,pad=.25", "facecolor":"white", "edgecolor":"#CBD5E1", "alpha":.88})
    ax.grid(alpha=.22)
    return {"N": len(xx), "slope": float(slope), "intercept": float(intercept), "R2_OLS": float(r2_ols),
            "R2_prediction_y_equals_x": float(r2_pred), "MAE": mae}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga únicamente artefactos OOF ya evaluados en development."""
    root = REPORT_DIR / "comparisons"
    p = pd.read_csv(root / "ensemble_predictions.csv")
    m = pd.read_csv(root / "metrics_by_composition.csv")
    folds = pd.read_csv(root / "metrics_per_fold_seed.csv")
    return p, m, folds


def _valid_energy_models(pred: pd.DataFrame) -> list[str]:
    """Selecciona candidatos que representan energía total, no componente aislado."""
    candidates = ["H0", "H1", "H2", "R1-A-ABS", "R1-D-COMBINED", "R2-SMALL4", "R2-REF8",
        "R3-DROP30", "R3-WD1E2", "R3-EARLY8", "R4-SELECTED-ABSOLUTE", "R4-SELECTED-2COMP",
        "R4-FIXED-FINAL_END", "R4-FIXED-FINAL_MEAN"]
    return [name for name in candidates if name in set(pred.model)]


def run() -> list[str]:
    """Exporta 13 familias comunes y seis diagnósticos específicos en PNG."""
    pred, comp, fold = _load()
    out = REPORT_DIR / "comparisons" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    models = _valid_energy_models(pred)
    figures: list[str] = []

    # 1. Absolute energy parity; overlays D1 only as a non-ranking energy reference.
    parity_models = [m for m in ["H0", "H1", "H2", "R1-A-ABS", "R1-D-COMBINED", "R2-REF8", "R3-WD1E2", "R4-SELECTED-2COMP"] if m in models]
    d1_path = PROJECT / "Alternative2/reports/MatGPT_diagnostics/D1/v3/oof_predictions.csv"
    d1 = pd.read_csv(d1_path) if d1_path.is_file() else None
    if d1 is not None:
        d1 = d1.loc[d1.configuration.eq("D1-B Ridge composición")].copy()
        d1 = d1.rename(columns={"composition_group":"composition_id", "energy_per_atom_ev":"E_DFT_eV_atom",
            "energy_pred_ev_atom":"E_pred_eV_atom"})
        d1["structure_id"] = d1.structure_id.astype(str)
    abs_frames=[pred.loc[pred.model.eq(m)] for m in parity_models]+([d1] if d1 is not None else [])
    abs_values=np.concatenate([pd.concat([g.E_DFT_eV_atom,g.E_pred_eV_atom]).to_numpy(float) for g in abs_frames])
    abs_pad=.04*(abs_values.max()-abs_values.min())
    abs_limits=(float(abs_values.min()-abs_pad),float(abs_values.max()+abs_pad))
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)
    abs_stats = []
    for ax, model in zip(axes.flat, parity_models):
        g = pred.loc[pred.model.eq(model)].copy()
        g["_color"] = COLORS.get(model, "#333333")
        result = parity_panel(ax, g, "E_DFT_eV_atom", "E_pred_eV_atom", LABELS.get(model, model), "eV/átomo", note_metric="MAE", shared_limits=abs_limits)
        abs_stats.append({"model": model, **result})
    if d1 is not None:
        ax = axes.flat[len(parity_models)]
        d1 = d1.copy(); d1["_color"] = "#64748B"
        result = parity_panel(ax, d1, "E_DFT_eV_atom", "E_pred_eV_atom", "D1 Ridge composición-only", "eV/átomo", note_metric="MAE", shared_limits=abs_limits)
        ax.text(.03,.03,"Una energía por composición;\nranking indeterminado",transform=ax.transAxes,fontsize=7,
                bbox={"boxstyle":"round,pad=.2","facecolor":"white","edgecolor":"#CBD5E1","alpha":.88})
        abs_stats.append({"model":"D1 Ridge composición-only", **result})
    used_panels=len(parity_models)+(1 if d1 is not None else 0)
    for ax in axes.flat[used_panels:]:
        ax.axis("off")
    fig.suptitle("ABS₃: paridad de energía total estática DFT y predicción | OOF development; test excluido", fontsize=16)
    fig.text(.5, -.01, "Cada punto = una de 210 estructuras OOF (70 composiciones); forma = familia polimórfica. Línea discontinua: y=x; rosa: OLS libre ajustado con todos los puntos del panel. m es adimensional; b y MAE están en eV/átomo. D1 repite una predicción por composición y su ranking es indeterminado.", ha="center", fontsize=9)
    save_png(fig, out / "absolute_energy_parity_with_D1_ridge.png"); figures.append("absolute_energy_parity_with_D1_ridge.png")
    pd.DataFrame(abs_stats).to_csv(REPORT_DIR/"comparisons"/"absolute_parity_regression_statistics.csv",index=False)

    # 2. Composition mean parity.
    mu = pred.groupby(["model", "composition_id"], as_index=False).agg(E_DFT_mu=("E_DFT_eV_atom", "mean"),
        E_pred_mu=("E_pred_eV_atom", "mean"), polymorph_family=("polymorph_family", "first"))
    if "E_pred_mu_eV_atom" in pred:
        explicit = pred.groupby(["model", "composition_id"]).E_pred_mu_eV_atom.mean()
        mu = mu.set_index(["model", "composition_id"])
        mu["E_pred_mu"] = [explicit.get(k, np.nan) if np.isfinite(explicit.get(k, np.nan)) else val
            for k, val in zip(mu.index, mu.E_pred_mu)]
        mu = mu.reset_index()
    mu_models = [m for m in ["H0", "H1", "H2", "R1-B-MU", "R1-D-COMBINED", "R4-SELECTED-2COMP", "R2-REF8"] if m in set(mu.model)]
    mu_values=np.concatenate([pd.concat([mu.loc[mu.model.eq(m),"E_DFT_mu"],mu.loc[mu.model.eq(m),"E_pred_mu"]]).to_numpy(float) for m in mu_models])
    mu_pad=.04*(mu_values.max()-mu_values.min());mu_limits=(float(mu_values.min()-mu_pad),float(mu_values.max()+mu_pad))
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
    for ax, model in zip(axes.flat, mu_models):
        g = mu.loc[mu.model.eq(model)].copy(); g["_color"] = COLORS.get(model, "#555555")
        result=parity_panel(ax, g, "E_DFT_mu", "E_pred_mu", LABELS.get(model, model), "eV/átomo", reference="Media DFT del triplete",shared_limits=mu_limits)
        abs_stats.append({"model":model,"quantity":"composition_mean",**result})
    for ax in axes.flat[len(mu_models):]: ax.axis("off")
    fig.suptitle("ABS₃: acceso al nivel energético medio por composición | N=70 fórmulas", fontsize=15)
    fig.text(.5, -.015, "Cada punto = una fórmula (media de tres polimorfos); se promedian primero las semillas por estructura. H0 usa la media descriptiva de sus tres predicciones. Para R1-B, la predicción μ repetida no predice diferencias polimórficas.", ha="center", fontsize=9)
    save_png(fig, out / "composition_mu_parity.png"); figures.append("composition_mu_parity.png")
    pd.DataFrame(abs_stats).to_csv(REPORT_DIR/"comparisons"/"parity_regression_statistics.csv",index=False)

    # 3. MSE decomposition.
    summary = comp.groupby("model").agg(mse_mean=("MSE_mean_mev2_atom2", "mean"), mse_resid=("MSE_residual_mev2_atom2", "mean"), mse_total=("MSE_total_mev2_atom2", "mean"))
    use = [m for m in models if m in summary.index]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(use)); width = .25
    ax.bar(x-width, summary.loc[use, "mse_mean"], width, label="MSE del error de media composicional", color="#2563EB")
    ax.bar(x, summary.loc[use, "mse_resid"], width, label="MSE del residuo centrado", color="#F59E0B")
    ax.bar(x+width, summary.loc[use, "mse_total"], width, label="MSE total", color="#64748B", alpha=.7)
    ax.set_xticks(x, use, rotation=20, ha="right"); ax.set_ylabel("MSE (meV²/átomo²)"); ax.set_title("ABS₃: ¿el error viene del nivel medio o de las diferencias entre polimorfos?")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.text(.5, -.02, "Cada resumen promedia 70 composiciones tras promediar semillas por estructura. Por identidad, MSE total = MSE de media + MSE de residuo centrado; los MAE no son aditivos.", ha="center", fontsize=9)
    save_png(fig, out / "energy_decomposition.png"); figures.append("energy_decomposition.png")

    # 4. Fold/seed stability.
    show = [m for m in models if m in set(fold.model)]
    metric_fields = [("MAE_absolute_macro_mev_atom", "MAE absoluta (meV/átomo)"),
        ("MAE_delta_GS_anchored_macro_mev_atom", "MAE ΔE anclada a GS DFT (meV/átomo)"),
        ("GS_accuracy", "Selección GS (% elegibles)"), ("pairwise_accuracy", "Pairwise (% pares no empatados)")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (field, label) in zip(axes.flat, metric_fields):
        vals = [fold.loc[fold.model.eq(model), field].dropna().to_numpy() * (100 if field.endswith("accuracy") else 1) for model in show]
        ax.boxplot(vals, tick_labels=show, showmeans=True, showfliers=True)
        ax.set_title(label); ax.tick_params(axis="x", labelrotation=35); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Estabilidad entre outer folds y semillas | medias de 15 fold×semilla, no 15 composiciones independientes", fontsize=14)
    fig.text(.5, -.015, "Cada dato = una ejecución fold×semilla con 14 composiciones outer. GS excluye brechas DFT ≤1 meV/átomo y abstenciones no cuentan como acierto; pairwise excluye pares DFT empatados. Las semillas reutilizan las mismas composiciones/folds.", ha="center", fontsize=9)
    save_png(fig, out / "fold_seed_stability.png"); figures.append("fold_seed_stability.png")

    # 5. GS confusion: counts + percentages.
    gs_mat = pd.read_csv(REPORT_DIR / "comparisons" / "confusion_matrices.csv")
    gs_models = [m for m in ["H0", "H2", "R2-REF8", "R4-SELECTED-2COMP"] if m in set(gs_mat.model)]
    classes = list(FAMILIES) + ["tie/abstain"]
    fig, axes = plt.subplots(1, max(1, len(gs_models)), figsize=(5*len(gs_models), 5), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes.flat, gs_models):
        g = gs_mat.loc[(gs_mat.model.eq(model)) & gs_mat.primary_GS_row]
        matrix = np.zeros((3, 4))
        for i, true in enumerate(FAMILIES):
            for j, pred_cls in enumerate(classes):
                row = g.loc[g.true_class.eq(true) & g.predicted_class.eq(pred_cls)]
                matrix[i, j] = float(row.row_percent.iloc[0]) if len(row) and np.isfinite(row.row_percent.iloc[0]) else 0
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        counts = g.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=FAMILIES, columns=classes).fillna(0)
        for i in range(3):
            for j in range(4): ax.text(j, i, f"{int(counts.iloc[i,j])}\n{matrix[i,j]:.0f}%", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), ["Perovskita", "Hexagonal", "Aguja", "Abstención"], rotation=35, ha="right")
        ax.set_yticks(range(3), ["Perovskita", "Hexagonal", "Aguja"]); ax.set_ylabel("Ganador DFT"); ax.set_title(f"{model} | filas elegibles n=67")
    fig.suptitle("ABS₃: matriz de selección del estado fundamental | cada fórmula cuenta una vez", fontsize=14)
    fig.text(.5, -.015, "Filas = ganador DFT; columnas = ganador predicho. Celdas muestran conteo y porcentaje por fila. Tres fórmulas near-degenerate DFT (gap ≤1 meV/átomo) se excluyen del denominador primario; empates predichos son abstenciones.", ha="center", fontsize=9)
    save_png(fig, out / "gs_confusion_matrices.png"); figures.append("gs_confusion_matrices.png")

    # 6. Paired composition differences vs H2.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    comp_pivot = comp.pivot(index="composition_id", columns="model", values="MAE_absolute_mev_atom")
    delta_pivot = comp.pivot(index="composition_id", columns="model", values="MAE_delta_GS_anchored_mev_atom")
    paired_models = [m for m in ["R1-A-ABS", "R1-D-COMBINED", "R2-SMALL4", "R3-DROP30", "R3-WD1E2", "R3-EARLY8", "R4-SELECTED-2COMP"] if m in comp_pivot]
    for ax, wide, title, unit in [(axes[0], comp_pivot, "Diferencia de MAE absoluta por fórmula", "meV/átomo"),
                                 (axes[1], delta_pivot, "Diferencia de MAE ΔE anclada por fórmula", "meV/átomo")]:
        for offset, model in enumerate(paired_models):
            difference = wide[model] - wide["H2"]
            ax.scatter(np.arange(len(difference)) + offset*.07, difference, s=13, alpha=.6, color=COLORS.get(model), label=f"{LABELS.get(model,model)} − H2")
            ax.axhline(difference.mean(), color=COLORS.get(model), linestyle="--", linewidth=.8)
        ax.axhline(0, color="#111827", linewidth=1); ax.set_title(title); ax.set_xlabel("Composición (70 grupos)"); ax.set_ylabel(f"Candidato − H2 ({unit}; negativo favorece candidato)"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("ABS₃: diferencias pareadas por composición y semillas agregadas", fontsize=14)
    fig.text(.5, -.02, "Cada punto = una composición. Los IC pareados agrupados se informan en paired_bootstrap.json; diferencia de medias de grupos, no de seeds independientes.", ha="center", fontsize=9)
    save_png(fig, out / "paired_composition_differences.png"); figures.append("paired_composition_differences.png")

    # 7. Pairwise confusion.
    pair_mat = pd.read_csv(REPORT_DIR / "comparisons" / "pairwise_confusion_matrices.csv")
    pair_models = [m for m in ["H0", "H2", "R2-REF8", "R4-SELECTED-2COMP"] if m in set(pair_mat.model)]
    labels = ["i_lower", "tie", "j_lower"]
    fig, axes = plt.subplots(1, len(pair_models), figsize=(4.7*len(pair_models), 4.7), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes.flat, pair_models):
        g = pair_mat.loc[pair_mat.model.eq(model)]
        matrix = g.pivot(index="true_class", columns="predicted_class", values="row_percent").reindex(index=labels, columns=labels).fillna(0)
        counts = g.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=labels, columns=labels).fillna(0)
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Purples")
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{int(counts.iloc[i,j])}\n{matrix.iloc[i,j]:.0f}%", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels); ax.set_xlabel("Relación predicha"); ax.set_ylabel("Relación DFT"); ax.set_title(f"{model} | 210 pares físicos")
    fig.suptitle("ABS₃: matriz de ordenamiento por pares, dirección física fija", fontsize=14)
    fig.text(.5, -.015, "Orientación: (perovskita, hexagonal), (perovskita, aguja), (hexagonal, aguja). Conteos y porcentaje por fila; DFT ties aparecen como clase diagnóstica, no entran en pairwise accuracy primaria.", ha="center", fontsize=9)
    save_png(fig, out / "pairwise_confusion_matrices.png"); figures.append("pairwise_confusion_matrices.png")

    # 8/9. Delta parity own-min and DFT-GS anchored.
    delta_parity_stats=[]
    for output_name, xcol, ycol, axis_label, title in [
        ("parity_delta_own_min_legacy.png", "dft_own", "pred_own", "ΔE desde el mínimo propio de cada triplete (meV/átomo)", "ABS₃: ΔE legacy own-min"),
        ("parity_dft_gs_anchored.png", "dft_anchor", "pred_anchor", "ΔE desde el GS DFT del triplete (meV/átomo)", "ABS₃: ΔE anclada al GS DFT")]:
        figures_data = pred.copy()
        figures_data["dft_anchor"] = np.nan; figures_data["pred_anchor"] = np.nan
        figures_data["dft_own"] = np.nan; figures_data["pred_own"] = np.nan
        for _, ids in figures_data.groupby(["model", "composition_id"]).groups.items():
            g = figures_data.loc[ids]
            dft_mev = g.E_DFT_eV_atom.to_numpy()*1000; pred_mev = g.E_pred_eV_atom.to_numpy()*1000
            dft_min = int(np.argmin(dft_mev)); pred_min = int(np.argmin(pred_mev))
            figures_data.loc[ids, "dft_anchor"] = dft_mev-dft_mev[dft_min]
            figures_data.loc[ids, "pred_anchor"] = pred_mev-pred_mev[dft_min]
            figures_data.loc[ids, "dft_own"] = dft_mev-dft_mev.min()
            figures_data.loc[ids, "pred_own"] = pred_mev-pred_mev[pred_min]
        fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)
        chosen = [m for m in ["H0", "H1", "H2", "R1-D-COMBINED", "R2-SMALL4", "R2-REF8", "R3-WD1E2", "R4-SELECTED-2COMP", "R4-FIXED-FINAL_MEAN"] if m in set(figures_data.model)]
        rel_values=np.concatenate([pd.concat([figures_data.loc[figures_data.model.eq(m),xcol],figures_data.loc[figures_data.model.eq(m),ycol]]).dropna().to_numpy(float) for m in chosen])
        rel_pad=.04*(rel_values.max()-rel_values.min());rel_limits=(float(rel_values.min()-rel_pad),float(rel_values.max()+rel_pad))
        for ax, model in zip(axes.flat, chosen):
            g = figures_data.loc[figures_data.model.eq(model)].copy(); g["_color"] = COLORS.get(model)
            result=parity_panel(ax, g, xcol, ycol, LABELS.get(model, model), "meV/átomo", reference=axis_label,shared_limits=rel_limits)
            delta_parity_stats.append({"quantity":output_name,"model":model,**result})
        fig.suptitle(title + " | OOF development; test excluido", fontsize=16)
        fig.text(.5, -.01, "Cada punto = estructura física OOF; círculos perovskita distorsionada, triángulos hexagonal, cuadrados aguja. Línea ideal y=x; OLS libre descriptivo. En ΔE DFT-GS anclada, valores predichos negativos indican inversión del orden y se conservan.", ha="center", fontsize=9)
        save_png(fig, out / output_name); figures.append(output_name)
    pd.DataFrame(delta_parity_stats).to_csv(REPORT_DIR/"comparisons"/"relative_parity_regression_statistics.csv",index=False)

    # 10. Ranges and collapse.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    use = [m for m in models if m in comp.model.unique()]
    for model in use:
        g = comp.loc[comp.model.eq(model)]
        axes[0].scatter(g.DFT_range_mev_atom, g.predicted_range_mev_atom, s=14, alpha=.6, color=COLORS.get(model), label=LABELS.get(model, model))
    lim = max(comp.DFT_range_mev_atom.max(), comp.predicted_range_mev_atom.max())
    axes[0].plot([0,lim],[0,lim],"--",color="#111827",label="Rango ideal"); axes[0].set(xlabel="Rango DFT max−min (meV/átomo)",ylabel="Rango predicho max−min (meV/átomo)",title="Rango del triplete"); axes[0].legend(fontsize=6,ncol=2); axes[0].grid(alpha=.2)
    collapse = comp.groupby("model").predicted_near_collapse.mean()*100
    ties = comp.groupby("model").GS_predicted_abstention.mean()*100
    x=np.arange(len(use)); axes[1].bar(x-.18,[collapse.get(m,0) for m in use],.36,label="Rango predicho <1 meV/átomo",color=[COLORS.get(m) for m in use]); axes[1].bar(x+.18,[ties.get(m,0) for m in use],.36,label="Empate GS predicho",color="#9CA3AF")
    axes[1].set_xticks(x,use,rotation=35,ha="right"); axes[1].set_ylabel("Composiciones (%)"); axes[1].set_title("Colapso y abstenciones"); axes[1].legend(fontsize=8); axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("ABS₃: dispersión y capacidad de distinguir polimorfos", fontsize=14)
    fig.text(.5,-.02,"Cada punto/barra = fórmula; semillas se promedian antes. El rango bajo indica poca separación predictiva; una abstención ocurre si hay dos mínimos predichos dentro de 1 meV/átomo.",ha="center",fontsize=9)
    save_png(fig,out/"ranges_collapses_ties.png");figures.append("ranges_collapses_ties.png")

    # 11. Energy scorecard (all values are measured, aspirations clearly labeled).
    score = comp.groupby("model").agg(abs_mae=("MAE_absolute_mev_atom","mean"),anchor=("MAE_delta_GS_anchored_mev_atom","mean"),own=("MAE_delta_own_min_legacy_mev_atom","mean"),mu=("MAE_mu_mev_atom","mean"),delta=("MAE_delta_centered_mev_atom","mean"),gs=("GS_correct","mean"),pair=("pairwise_accuracy","mean"),range=("absolute_range_error_mev_atom","mean"))
    score = score.reindex(use)
    fig, ax = plt.subplots(figsize=(14, max(5, len(use)*.46)), constrained_layout=True)
    norm = score.copy(); norm["abs_mae"] /= 10; norm["anchor"] /= 10; norm["own"] /= 10; norm["mu"] /= 10; norm["delta"] /= 10; norm["gs"] *= 100; norm["pair"] *= 100; norm["range"] /= 10
    im=ax.imshow(norm.to_numpy(float),aspect="auto",cmap="RdYlGn_r",vmin=0,vmax=10)
    ax.set_xticks(range(len(norm.columns)),["MAE abs\n÷10", "ΔE anclada\n÷10", "ΔE own-min\n÷10", "MAE μ\n÷10", "MAE δ\n÷10", "GS %", "Pairwise %", "Error rango\n÷10"])
    ax.set_yticks(range(len(norm.index)),[LABELS.get(m,m) for m in norm.index])
    for i,model in enumerate(score.index):
        vals=score.loc[model].to_numpy(float)
        for j,val in enumerate(vals): ax.text(j,i,f"{val*100:.1f}%" if j in (5,6) else f"{val:.1f}",ha="center",va="center",fontsize=7)
    ax.set_title("Scorecard ABS₃: error cuantitativo y discriminación polimórfica")
    fig.colorbar(im,ax=ax,label="Escala de color: unidad normalizada para legibilidad; celdas muestran valores reales")
    fig.text(.5,-.01,"MAE en meV/átomo; GS excluye 3 brechas DFT ≤1; pairwise usa pares no empatados. Metas aspiracionales históricas (MAE energía/relativa ≤10, fuerte ≤5; GS >90%/100%) no están validadas para MatGPT y no aplican como garantía.",ha="center",fontsize=8)
    save_png(fig,out/"scorecard_energy.png");figures.append("scorecard_energy.png")

    # 12. Centered structural residual parity.
    fig, axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    chosen=[m for m in ["H0","H2","R1-D-COMBINED","R2-REF8","R3-WD1E2","R4-SELECTED-2COMP"] if m in set(pred.model)]
    residual_data=pred.copy()
    for model,g in residual_data.groupby("model"):
        idx=g.index
        residual_data.loc[idx,"dft_centered"]=(g.E_DFT_eV_atom-g.groupby("composition_id").E_DFT_eV_atom.transform("mean"))*1000
        residual_data.loc[idx,"pred_centered"]=(g.E_pred_eV_atom-g.groupby("composition_id").E_pred_eV_atom.transform("mean"))*1000
    res_values=np.concatenate([pd.concat([residual_data.loc[residual_data.model.eq(m),"dft_centered"],residual_data.loc[residual_data.model.eq(m),"pred_centered"]]).to_numpy(float) for m in chosen])
    res_pad=.04*(res_values.max()-res_values.min());res_limits=(float(res_values.min()-res_pad),float(res_values.max()+res_pad))
    results=[]
    for ax,model in zip(axes.flat,chosen):
        g=residual_data.loc[residual_data.model.eq(model)].copy()
        g["_color"]=COLORS.get(model)
        results.append({"model":model,**parity_panel(ax,g,"dft_centered","pred_centered",LABELS.get(model,model),"meV/átomo",reference="Residuo centrado DFT",shared_limits=res_limits)})
    for ax in axes.flat[len(chosen):]:ax.axis("off")
    fig.suptitle("ABS₃: paridad de desviaciones polimórficas centradas | μ se elimina",fontsize=14)
    fig.text(.5,-.01,"Cada punto = una estructura OOF. x = E_DFT−μ_DFT; y = E_pred−μ_pred, centrados dentro de cada triplete. La figura evalúa las diferencias entre polimorfos, no la energía total.",ha="center",fontsize=9)
    save_png(fig,out/"structural_residual_parity.png");figures.append("structural_residual_parity.png")

    # 13. Training curves and ridge alpha selection.
    fig, axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    hist_paths=list((REPORT_DIR/"R2_head_capacity").glob("*/training_history.csv"))+list((REPORT_DIR/"R3_regularization").glob("*/training_history.csv"))
    for path in hist_paths:
        h=pd.read_csv(path)
        if h.empty:continue
        summary_h=h.groupby("epoch").agg(train=("train_loss_mu","mean"),val=("inner_validation_objective","mean"),active=("seed","count"))
        axes[0].plot(summary_h.index,summary_h.train,label=f"{path.parent.name} train")
        axes[0].plot(summary_h.index,summary_h.val,linestyle="--",label=f"{path.parent.name} inner-val")
    axes[0].set(xlabel="Época local",ylabel="Pérdida (objetivo normalizado H2)",title="MLP: train por minibatch vs validación completa");axes[0].legend(fontsize=6);axes[0].grid(alpha=.2)
    alpha_path=REPORT_DIR/"R1_linear_probes"/"validation_selection.csv"
    if alpha_path.is_file():
        a=pd.read_csv(alpha_path);a=a.loc[a.model.eq("R1-A-ABS")]
        for fold_id,g in a.groupby("fold"):axes[1].plot(g.alpha,g.inner_validation_MAE_mev_atom,marker="o",label=f"fold {fold_id}")
        axes[1].set_xscale("log")
    axes[1].set(xlabel="Ridge α (log)",ylabel="MAE inner-validation (meV/átomo)",title="Ridge: selección de regularización");axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle("ABS₃: selección y generalización interna | test histórico no usado",fontsize=14)
    fig.text(.5,-.02,"Train resume minibatches con pesos de época; inner-validation evalúa el triplete completo al final de época. Solo se agregan corridas activas en cada época. Ridge no tiene epochs; el panel muestra selección interna de α.",ha="center",fontsize=9)
    save_png(fig,out/"training_curves.png");figures.append("training_curves.png")

    # Extra A. Ridge alpha sensitivity.
    alpha_frame=pd.read_csv(REPORT_DIR/"R1_linear_probes"/"validation_selection.csv")
    fig, axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for model, g in alpha_frame.groupby("model"):
        if not str(model).startswith("R1-"):continue
        axes[0].plot(g.alpha,g.inner_validation_MAE_mev_atom,marker="o",alpha=.65,label=model)
    axes[0].set_xscale("log");axes[0].set(xlabel="Ridge α (log)",ylabel="MAE inner-validation (meV/átomo)",title="Sensibilidad de alpha por target/fold");axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    selected=alpha_frame.loc[alpha_frame.selected.astype(bool)].groupby(["model","alpha"]).size().unstack(fill_value=0)
    selected.plot(kind="bar",stacked=True,ax=axes[1]);axes[1].set(title="Alpha seleccionado en folds",xlabel="Target",ylabel="Número de folds");axes[1].legend(title="α",fontsize=7);axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("R1: regularización lineal elegida solo con inner-validation",fontsize=13)
    save_png(fig,out/"R1_ridge_alpha_sensitivity.png");figures.append("R1_ridge_alpha_sensitivity.png")

    # Extra B. head width.
    width_models=[m for m in ["R2-SMALL4","R2-REF8","H2"] if m in comp.model.unique()]
    width_data=comp.loc[comp.model.isin(width_models)]
    fig,ax=plt.subplots(figsize=(9,5))
    metrics=[("MAE_absolute_mev_atom","E absoluta"),("MAE_delta_GS_anchored_mev_atom","ΔE anclada a DFT-GS"),
        ("MAE_mu_mev_atom","Media μ"),("MAE_delta_centered_mev_atom","Residuo δ centrado")]
    for metric,label in metrics:
        vals=[width_data.loc[width_data.model.eq(model),metric].mean() for model in width_models]
        ax.plot(width_models,vals,marker="o",linewidth=1.5,label=label)
    ax.set_ylabel("MAE (meV/átomo)");ax.set_title("R2: capacidad por objetivo");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.text(.5,-.02,"MLP con dos cabezas, mismo pooling/pérdida/folds/semillas. Ancho 4: 6,162 parámetros; ancho 8: 12,322. H2 histórico sirve como referencia, R2-REF8 fue reentrenado para emparejamiento.",ha="center",fontsize=9)
    save_png(fig,out/"R2_head_width_comparison.png");figures.append("R2_head_width_comparison.png")

    # Extra C. train-validation gap by epochs.
    fig,ax=plt.subplots(figsize=(10,5))
    for path in hist_paths:
        h=pd.read_csv(path)
        if h.empty:continue
        g=h.groupby("epoch").agg(train=("train_loss_mu","mean"),val=("inner_validation_objective","mean"))
        ax.plot(g.index,g.val-g.train,label=path.parent.name)
    ax.axhline(0,color="#111827",linewidth=1);ax.set(xlabel="Época",ylabel="Pérdida inner-val − train",title="R3: brecha de generalización durante entrenamiento");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.text(.5,-.02,"Loss train = promedio de minibatches con pesos online; inner-val completa por composición. Brecha positiva indica objetivo interno mayor que train, no sustituye métricas OOF.",ha="center",fontsize=9)
    save_png(fig,out/"R3_regularization_gap.png");figures.append("R3_regularization_gap.png")

    # Extra D. layer/pooling heatmap from inner-validation selection audit.
    r4_sel=pd.read_csv(REPORT_DIR/"R4_layers_pooling"/"validation_selection.csv")
    r4_sel=r4_sel.loc[r4_sel.model.str.startswith("R4-ABSOLUTE-")]
    r4_sel["variant"]=r4_sel.feature_variant
    heat=r4_sel.groupby(["variant","fold"]).inner_validation_MAE_mev_atom.min().unstack()
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(heat.to_numpy(),aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(heat.columns)),[f"fold {f}" for f in heat.columns]);ax.set_yticks(range(len(heat.index)),heat.index);ax.set_title("R4: MAE de energía absoluta en inner-validation por capa/pooling");ax.set_xlabel("Outer fold (validación interna de selección)");ax.set_ylabel("Representación de MatGPT")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):ax.text(j,i,f"{heat.iloc[i,j]:.1f}",ha="center",va="center",color="white",fontsize=8)
    fig.colorbar(im,ax=ax,label="MAE inner-validation (meV/átomo)")
    fig.text(.5,-.02,"Celdas muestran el mejor alpha de la grilla para cada fold/representación. Las filas son salidas crudas de bloques antes de ln_f salvo controles FINAL_*, que sí son post-ln_f. Estos son criterios internos, no estimaciones OOF.",ha="center",fontsize=9)
    save_png(fig,out/"R4_layer_pooling_heatmap.png");figures.append("R4_layer_pooling_heatmap.png")

    # Extra E. embedding stability / variability by layer and pooling.
    feature_cache=np.load(REPORT_DIR/"features_cache"/"layer_features.npz",allow_pickle=False)
    names=[f"L{layer:02d}_{pool}" for layer in (3,6,9,12) for pool in ("END","MEAN")]+["FINAL_END","FINAL_MEAN"]
    variability=[]
    for name in names:
        from scipy.spatial.distance import pdist
        x=np.asarray(feature_cache[name],dtype=np.float64)
        norm=np.linalg.norm(x,axis=1)
        cosine_distance=pdist(x,metric="cosine")
        variability.append({"representation":name,"mean_embedding_norm":norm.mean(),"std_embedding_norm":norm.std(),
            "mean_coordinate_std":x.std(axis=0).mean(),"mean_pairwise_cosine_distance":float(np.mean(cosine_distance))})
    feature_cache.close();pd.DataFrame(variability).to_csv(REPORT_DIR/"R4_layers_pooling"/"feature_stability.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,4.7));stab=pd.DataFrame(variability)
    ax.bar(stab.representation,stab.mean_coordinate_std,color="#0E7490");ax.set_ylabel("Desviación estándar media por coordenada");ax.set_title("R4: variabilidad entre 210 embeddings congelados (solo development)");ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.2)
    fig.text(.5,-.02,"Barra = promedio de desviación estándar de cada coordenada entre estructuras. Describe dispersión de embeddings, no demuestra por sí sola información energética.",ha="center",fontsize=9)
    save_png(fig,out/"R4_features_stability.png");figures.append("R4_features_stability.png")

    # Extra F. ranking balanced metrics.
    ranking=comp.groupby("model").agg(gs=("GS_correct","mean"),pair=("pairwise_accuracy","mean"),collapse=("predicted_near_collapse","mean"))
    ranking=ranking.reindex([m for m in models if m in ranking.index])
    fig,ax=plt.subplots(figsize=(13,5));x=np.arange(len(ranking));width=.25
    ax.bar(x-width,ranking.gs*100,width,label="GS correcto (sobre elegibles)",color="#2563EB")
    ax.bar(x,ranking.pair*100,width,label="Pairwise por composición",color="#059669")
    ax.bar(x+width,ranking.collapse*100,width,label="Composiciones colapsadas",color="#F59E0B")
    ax.set_xticks(x,ranking.index,rotation=35,ha="right");ax.set_ylabel("Porcentaje (%)");ax.set_title("Ranking y colapso: exactitud sin ocultar abstenciones");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.text(.5,-.02,"GS excluye DFT gap ≤1 meV/átomo y empates predichos no son aciertos; Pairwise exactitud por fórmula sobre pares DFT no empatados. Para modelos μ-only, ranking es indeterminado/abstención.",ha="center",fontsize=9)
    save_png(fig,out/"ranking_balanced_metrics.png");figures.append("ranking_balanced_metrics.png")

    forbidden={".pdf",".svg",".eps",".jpg",".jpeg",".html",".pptx"}
    found=[str(p) for p in REPORT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    if found:
        raise ExperimentError(f"Se encontraron formatos de figura prohibidos dentro del experimento: {found[:5]}")
    write_json_path=REPORT_DIR/"comparisons"/"figures_manifest.json"
    import json
    write_json_path.write_text(json.dumps({"status":"PASS","figure_count":len(figures),"figures_png":figures,
        "forbidden_figure_formats_found":found,"test_used":False},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return figures


def main() -> None:
    """Genera gráficos y comprueba que ningún formato distinto de PNG se exportó."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        figures=run();log("png_figures_complete",count=len(figures),test_used=False,formats=["png"])
    except (ExperimentError,OSError,ValueError,KeyError,RuntimeError) as exc:
        log("figure_generation_failed",error=str(exc));raise SystemExit(2) from exc


if __name__=="__main__":main()
