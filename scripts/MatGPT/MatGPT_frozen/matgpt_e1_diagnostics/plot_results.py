#!/usr/bin/env python3
"""Crea figuras PNG en español para E1.1 o la comparación de pooling E1.2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, DiagnosticError, log, logger_for

LOGGER = logger_for("matgpt_e1_diagnostics")
COLORS = {"mlp8": "#3B73A8", "mlp32": "#D05A47", "ridge": "#6A9F58", "last_token": "#3569A8", "masked_mean": "#C14E80"}


def style() -> None:
    """Aplica tipografía legible y cuadrícula tenue a figuras científicas.

    Returns:
        None; actualiza los parámetros globales de Matplotlib.
    """
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.titlesize": 15, "legend.fontsize": 8, "axes.grid": True,
                         "grid.alpha": 0.22, "savefig.dpi": 180})


def note(fig: plt.Figure, text: str) -> None:
    """Coloca una nota metodológica debajo de los ejes.

    Args:
        fig: Figura activa de Matplotlib.
        text: Nota de muestra, partición, definición de variable y límites.

    Returns:
        None.
    """
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8, color="#39465e", wrap=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Guarda PNG sin sustituir una figura existente.

    Args:
        fig: Figura renderizada que se va a exportar.
        path: Destino `.png` que debe estar libre.

    Returns:
        None.
    """
    if path.exists():
        raise DiagnosticError(f"La figura ya existe y no se sobrescribe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def bar_metric(metrics: pd.DataFrame, column: str, ylabel: str, title: str, target: Path, note_text: str) -> None:
    """Grafica media ± desviación entre semillas por configuración.

    Args:
        metrics: Métricas separadas por split, configuración y semilla.
        column: Columna numérica que se representa.
        ylabel: Etiqueta del eje y con unidad o definición adimensional.
        title: Título científico del panel.
        target: Ruta nueva de exportación PNG.
        note_text: Nota metodológica visible en la figura.

    Returns:
        None; guarda el PNG sin sobrescribir versiones previas.
    """
    val = metrics[metrics.split.eq("val")].copy()
    val["label"] = val["head_type"].astype(str) + " | λ=" + val.pairwise_lambda.fillna(0).map(lambda value: f"{value:g}")
    grouped = val.groupby(["configuration", "label"], dropna=False)[column].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(max(9, len(grouped) * 0.72), 5.3))
    colors = [COLORS.get(value, "#777777") for value in grouped.label.str.split(" | ", regex=False).str[0]]
    errors = grouped["std"].fillna(0).to_numpy()
    ax.bar(np.arange(len(grouped)), grouped["mean"], yerr=errors, capsize=3, color=colors, edgecolor="white")
    ax.set_xticks(np.arange(len(grouped)), grouped.configuration, rotation=70, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.98, "Barras: media entre semillas; bigotes: ±1 desviación estándar.\nUna barra por configuración, seleccionada con validation.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd3df"})
    heads = list(dict.fromkeys(grouped.label.str.split(" | ", regex=False).str[0].tolist()))
    ax.legend(handles=[Patch(facecolor=COLORS.get(head, "#777777"), label=head) for head in heads],
              title="Color: tipo de cabeza", loc="upper left")
    fig.tight_layout(rect=(0.02, 0.12, 0.99, 0.98))
    note(fig, note_text)
    save_figure(fig, target)
    plt.close(fig)


def e1_1_figures(directory: Path, out: Path) -> list[Path]:
    """Genera diagnósticos E1.1 desde métricas y predicciones train/validation.

    Args:
        directory: Directorio de resultados de `train_heads.py`.
        out: Directorio nuevo para figuras PNG.

    Returns:
        Rutas de las figuras creadas.
    """
    metrics_path, pred_path, history_path = directory / "metrics.csv", directory / "predictions_train_val.csv", directory / "training_history.csv"
    if not all(path.is_file() for path in (metrics_path, pred_path, history_path)):
        raise DiagnosticError("E1.1 requiere metrics.csv, predictions_train_val.csv y training_history.csv.")
    metrics = pd.read_csv(metrics_path)
    pred = pd.read_csv(pred_path, low_memory=False)
    history = pd.read_csv(history_path)
    val = pred[pred.split.eq("val")]
    n_formulas = int(val.composition.nunique())
    n_structures = int(val.structure_id.nunique())
    n_mlp_seeds = int(val.loc[val.head_type.astype(str).str.startswith("mlp"), "seed"].nunique())
    note_text = f"Validación por composición: N={n_structures} estructuras, {n_formulas} composiciones, {n_mlp_seeds} semillas MLP. Una observación es una estructura/polimorfo; barras resumen semillas con ±1 DE. Métricas de test no consultadas."
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    entries = [
        ("energy_mae_ev_atom", "MAE absoluta (eV/átomo)", "E1.1: error absoluto de energía DFT estática"),
        ("relative_energy_mae_mev_atom", "MAE de ΔE relativa (meV/átomo)", "E1.1: error de energía relativa dentro de composición"),
        ("pairwise_ordering_accuracy", "Exactitud de ordenamiento (fracción)", "E1.1: ordenamiento de pares de polimorfos"),
        ("ground_state_accuracy", "Exactitud de selección del estado fundamental (fracción)", "E1.1: selección del mínimo DFT por composición"),
    ]
    for index, (column, ylabel, title) in enumerate(entries, 1):
        path = out / f"{index:02d}_{column}.png"
        bar_metric(metrics, column, ylabel, title, path, note_text)
        generated.append(path)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ranges = pred[pred.split.eq("val")].copy()
    formula_ranges = ranges.groupby(["configuration", "seed", "composition"], as_index=False).agg(
        dft_range=("dft_relative_energy_mev_atom", "max"), predicted_range=("predicted_relative_energy_mev_atom", "max"),
        head_type=("head_type", "first"), pairwise_lambda=("pairwise_lambda", "first"))
    for head, group in formula_ranges.groupby("head_type", sort=True):
        ax.scatter(group.dft_range, group.predicted_range, s=36, alpha=0.52, color=COLORS.get(head, "#555555"), label=head)
    upper = max(float(formula_ranges.dft_range.max()), float(formula_ranges.predicted_range.quantile(0.98)))
    ax.plot([0, upper], [0, upper], "--", color="#333333", label="Calibración ideal: rango predicho = rango DFT")
    ax.set(xlabel="Rango ΔE DFT por composición (meV/átomo)", ylabel="Rango ΔE predicho por composición (meV/átomo)",
           title="E1.1: amplitud energética de los polimorfos")
    ax.legend(title="Cabeza; una marca = composición y semilla", loc="upper left")
    fig.tight_layout(rect=(0.03, 0.13, 0.98, 0.96))
    note(fig, f"Validación: {len(formula_ranges)} observaciones composición-configuración-semilla de {n_formulas} composiciones. Ambos rangos se calculan como máximo menos mínimo dentro de cada composición; la diagonal indica igualdad de amplitudes.")
    path = out / "05_rangos_energeticos.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {0.0: "#555555", 0.1: "#3366AA", 0.3: "#D18A28", 1.0: "#9B4D96"}
    history = history.copy()
    history["head_type"] = history.configuration.str.extract(r"(mlp\d+)", expand=False)
    history["pairwise_lambda"] = history.configuration.str.extract(r"lambda([0-9.]+)", expand=False).astype(float)
    for (head, weight), group in history.groupby(["head_type", "pairwise_lambda"], dropna=True):
        summary = group.groupby("epoch").agg(train_mean=("train_objective_normalized", "mean"), train_std=("train_objective_normalized", "std"),
            val_mean=("validation_objective_normalized", "mean"), val_std=("validation_objective_normalized", "std")).reset_index()
        color = colors.get(float(weight), "#777777")
        ax.plot(summary.epoch, summary.train_mean, color=color, linestyle="--", alpha=0.85, label=f"{head}, λ={weight:g}, train")
        ax.fill_between(summary.epoch, summary.train_mean-summary.train_std.fillna(0), summary.train_mean+summary.train_std.fillna(0), color=color, alpha=0.08)
        ax.plot(summary.epoch, summary.val_mean, color=color, linestyle="-", alpha=0.85, label=f"{head}, λ={weight:g}, validación")
        ax.fill_between(summary.epoch, summary.val_mean-summary.val_std.fillna(0), summary.val_mean+summary.val_std.fillna(0), color=color, alpha=0.16)
    ax.set(xlabel="Época local de la cabeza", ylabel="Pérdida Huber normalizada (adimensional)", title="E1.1: optimización de cabezas congeladas")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.97))
    note(fig, "Líneas discontinuas: minilotes de entrenamiento agrupados por composición. Líneas continuas: conjunto completo de validación. Bandas: ±1 desviación entre semillas; no representan incertidumbre entre composiciones. El punto de control se selecciona por la pérdida de validación normalizada; λ pondera diferencias intracomposición.")
    path = out / "06_curvas_entrenamiento.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    valm = metrics[metrics.split.eq("val")].copy()
    for ax, metric, label in zip(axes, ("energy_mae_ev_atom", "relative_energy_mae_mev_atom"), ("MAE absoluta (eV/átomo)", "MAE relativa (meV/átomo)")):
        for head, group in valm.groupby("head_type"):
            grouped = group.groupby("pairwise_lambda")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(grouped.pairwise_lambda.fillna(0), grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=3,
                        color=COLORS.get(head, "#555555"), label=head)
        ax.set_xlabel("λ de pérdida pairwise (adimensional)")
        ax.set_ylabel(label)
        ax.legend(title="Cabeza")
    fig.suptitle("E1.1: efecto de supervisar diferencias energéticas")
    fig.tight_layout(rect=(0.03, 0.14, 0.98, 0.91))
    note(fig, f"Validación por composición; N={n_structures} estructuras y {n_formulas} composiciones. Media ±1 DE entre semillas MLP; baselines deterministas tienen una sola ejecución. λ=0 es el control; mejorar MAE relativa no garantiza preservar el ranking ni el rango energético.")
    path = out / "07_efecto_perdida_pairwise.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)

    fam = pred[pred.split.eq("val")].assign(abs_error=lambda frame: (frame.energy_pred_ev_atom-frame.energy_dft_ev_atom).abs())
    table = fam.groupby(["configuration", "polymorph_family"], as_index=False).abs_error.mean().pivot(index="configuration", columns="polymorph_family", values="abs_error")
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(table) * 0.42)))
    mesh = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)), [str(x) for x in table.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(table.index)), table.index)
    ax.set(xlabel="Familia estructural", ylabel="Configuración", title="E1.1: MAE por familia estructural")
    cb = fig.colorbar(mesh, ax=ax); cb.set_label("MAE (eV/átomo)")
    fig.tight_layout(rect=(0.03, 0.13, 0.98, 0.97))
    note(fig, f"Validación: N={n_structures} estructuras en {n_formulas} composiciones. Cada celda promedia |E_pred − E_DFT| por estructura y semilla para esa familia; no es un promedio por composición.")
    path = out / "08_mae_por_familia.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)
    return generated


def pooling_figures(last_dir: Path, mean_dir: Path, out: Path) -> list[Path]:
    """Compara métricas y paridad relativa para dos ejecuciones de pooling.

    Args:
        last_dir: Resultados de la representación del token `<end>`.
        mean_dir: Resultados masked mean con la misma cabeza y pérdida.
        out: Directorio versionado de figuras y estadísticas OLS.

    Returns:
        Rutas de las figuras PNG creadas.
    """
    frames = []
    for pooling, directory in (("last_token", last_dir), ("masked_mean", mean_dir)):
        file = directory / "metrics.csv"
        predfile = directory / "predictions_train_val.csv"
        if not file.is_file() or not predfile.is_file():
            raise DiagnosticError(f"Faltan resultados para {pooling}: {directory}")
        metrics = pd.read_csv(file); metrics["pooling_panel"] = pooling
        frames.append(metrics)
    metric = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    val_counts = metric[metric.split.eq("val")].groupby("pooling_panel").agg(n=("n_structures", "max"), compositions=("n_compositions", "max")) if {"n_structures", "n_compositions"}.issubset(metric.columns) else None
    for number, (column, ylabel, title) in enumerate((("energy_mae_ev_atom", "MAE (eV/átomo)", "Error absoluto por pooling"),
          ("relative_energy_mae_mev_atom", "MAE relativa (meV/átomo)", "Error de ΔE por pooling"),
          ("pairwise_ordering_accuracy", "Exactitud (fracción)", "Ordenamiento por pooling"),
          ("ground_state_accuracy", "Exactitud (fracción)", "Estado fundamental por pooling")), 1):
        val = metric[metric.split.eq("val")]
        agg = val.groupby("pooling_panel")[column].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), capsize=4,
               color=[COLORS.get(x, "#777777") for x in agg.index])
        ax.set(ylabel=ylabel, title=f"E1.2: {title}")
        ax.legend(handles=[Patch(facecolor=COLORS[pooling], label="Token final <end>" if pooling == "last_token" else "Promedio enmascarado")
                           for pooling in agg.index], title="Estrategia de pooling")
        fig.tight_layout(rect=(0.02, 0.15, 0.98, 0.95))
        sample_note = "Validation; media ±1 DE entre semillas. "
        if val_counts is not None and pooling in val_counts.index:
            sample_note += f"N={int(val_counts.loc[pooling, 'n'])} estructuras y {int(val_counts.loc[pooling, 'compositions'])} composiciones. "
        sample_note += "Misma arquitectura, pérdida, checkpoint, etiquetas, IDs y split; solo cambia el pooling. Test no usado para seleccionar."
        note(fig, sample_note)
        path = out / f"{number:02d}_pooling_{column}.png"
        save_figure(fig, path); plt.close(fig); generated.append(path)

    pred_frames = []
    for pooling, directory in (("last_token", last_dir), ("masked_mean", mean_dir)):
        frame = pd.read_csv(directory / "predictions_train_val.csv", low_memory=False)
        frame = frame[frame.split.eq("val")].copy()
        frame["pooling_panel"] = pooling
        pred_frames.append(frame)
    pred = pd.concat(pred_frames, ignore_index=True)
    pred = pred.groupby(["pooling_panel", "structure_id", "composition", "polymorph_family"], as_index=False).agg(
        dft_relative_energy_mev_atom=("dft_relative_energy_mev_atom", "first"),
        predicted_relative_energy_mev_atom=("predicted_relative_energy_mev_atom", "mean"), seed_count=("seed", "nunique"))
    upper = float(max(pred.dft_relative_energy_mev_atom.max(), pred.predicted_relative_energy_mev_atom.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    parity_stats: list[dict[str, object]] = []
    for ax, (pooling, frame) in zip(axes, pred.groupby("pooling_panel", sort=False)):
        x, y = frame.dft_relative_energy_mev_atom.to_numpy(float), frame.predicted_relative_energy_mev_atom.to_numpy(float)
        family_colors = {"dist_perovskite":"#4C78A8", "hexagonal":"#F2A541", "needle":"#8E5EA2"}
        ax.scatter(x, y, alpha=0.75, s=34, c=frame.polymorph_family.map(family_colors))
        ax.plot([0, upper], [0, upper], "--", color="#333333")
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            fit = slope*x + intercept
            r2den = np.sum((y-y.mean())**2)
            r2 = 1 - np.sum((y-fit)**2)/r2den if r2den > 0 else float("nan")
            ax.plot([0, upper], [intercept, slope*upper+intercept], color="#C43C6F")
            parity_stats.append({"pooling": pooling, "split": "val", "n_structures": len(x),
                "n_compositions": int(frame.composition.nunique()), "seed_aggregation": "media por structure_id",
                "ols_slope_dimensionless": float(slope), "ols_intercept_mev_atom": float(intercept),
                "ols_r2": float(r2), "mae_vs_dft_mev_atom": float(np.mean(np.abs(y-x)))})
            stats = f"N = {len(x)}\nPendiente m = {slope:.4f} (adimensional)\nIntercepto b = {intercept:.3f} meV/átomo\nR² OLS = {r2:.4f}\nMAE vs DFT = {np.mean(np.abs(y-x)):.3f} meV/átomo"
            ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"boxstyle":"round", "facecolor":"white", "alpha":0.9, "edgecolor":"#cbd3df"})
        ax.set_title("Token final <end>" if pooling == "last_token" else "Promedio enmascarado de tokens SLICES")
        handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=family)
                   for family, color in family_colors.items()]
        handles += [Line2D([0], [0], color="#333333", linestyle="--", label="Acuerdo ideal: y = x"),
                    Line2D([0], [0], color="#C43C6F", linestyle="-", label="OLS libre: y = m x + b")]
        ax.legend(handles=handles, title="Familia, referencia y ajuste", loc="lower right", fontsize=7)
        ax.set_xlim(0, upper); ax.set_ylim(0, upper)
    axes[0].set_ylabel("ΔE predicha (meV/átomo)")
    for ax in axes: ax.set_xlabel("ΔE DFT (meV/átomo)")
    fig.suptitle("E1.2: paridad de energías relativas por estrategia de pooling")
    fig.tight_layout(rect=(0.03, 0.17, 0.98, 0.9))
    ncomp = pred.composition.nunique()
    note(fig, f"Validación: N={len(pred)//2} estructuras estáticas por panel y {ncomp} composiciones. Un punto = una estructura; y es la media de predicciones entre {int(pred.seed_count.max())} semillas. ΔEᵢ = Eᵢ − mínimo dentro de cada composición; x = DFT, y = predicción. OLS usa todos los puntos del panel; el test no interviene en la selección.")
    path = out / "05_paridad_energia_relativa_pooling.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)
    stats_path = out / "pooling_parity_ols.csv"
    if stats_path.exists():
        raise DiagnosticError(f"Las estadísticas OLS ya existen y no se sobrescriben: {stats_path}")
    pd.DataFrame(parity_stats).to_csv(stats_path, index=False)
    return generated


def geometry_figures(pair_path: Path, out: Path) -> list[Path]:
    """Grafica similitudes, distancias y sensibilidad geométrica.

    Args:
        pair_path: CSV de pares intracomposición e intercomposición muestreados.
        out: Directorio versionado de figuras PNG.

    Returns:
        Rutas de las figuras creadas.
    """
    if not pair_path.is_file():
        raise DiagnosticError(f"No existe tabla geométrica: {pair_path}")
    pairs = pd.read_csv(pair_path, low_memory=False)
    required = {"pooling", "pair_scope", "cosine_similarity", "euclidean_distance", "sequence_length_difference"}
    if not required.issubset(pairs.columns):
        raise DiagnosticError(f"Faltan columnas para figuras geométricas: {sorted(required-set(pairs.columns))}")
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    within = pairs[pairs.pair_scope.eq("intra_composition")]
    between = pairs[pairs.pair_scope.eq("inter_composition_sample")]
    for number, (value, ylabel, title) in enumerate((("cosine_similarity", "Similitud coseno (adimensional)", "Similitud coseno intra frente a intercomposición"),
          ("euclidean_distance", "Distancia euclidiana (unidades del embedding)", "Distancias de embeddings intra frente a intercomposición")), 1):
        fig, ax = plt.subplots(figsize=(8.3, 5.2))
        datasets, labels, colors = [], [], []
        for pooling in ("last_token", "masked_mean"):
            for scope, frame in (("Intra composición", within), ("Inter composición, muestra", between)):
                values = frame.loc[frame.pooling.eq(pooling), value].dropna().to_numpy(float)
                if len(values):
                    datasets.append(values); labels.append(f"{pooling}\n{scope}")
                    colors.append(COLORS[pooling])
        artists = ax.violinplot(datasets, showmedians=True, showextrema=True)
        for body, color in zip(artists["bodies"], colors):
            body.set_facecolor(color); body.set_alpha(0.58)
        ax.set_xticks(np.arange(1, len(labels)+1), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout(rect=(0.03, 0.15, 0.98, 0.95))
        n_intra = int((within.pooling == "last_token").sum())
        n_inter = int((between.pooling == "last_token").sum())
        note(fig, f"Por pooling: {n_intra} pares exhaustivos de polimorfos de igual composición; {n_inter} pares de distinta composición en muestra reproducible. La línea central es la mediana; la forma resume densidad, no incertidumbre.")
        path = out / f"{number:02d}_embedding_{value}.png"
        save_figure(fig, path); plt.close(fig); generated.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, pooling in zip(axes, ("last_token", "masked_mean")):
        local = within[within.pooling.eq(pooling)]
        for family, group in local.groupby("family_i", sort=True):
            ax.scatter(group.absolute_dft_energy_difference_mev_atom, group.euclidean_distance, s=30, alpha=0.72, label=str(family))
        ax.set_title("Token final <end>" if pooling == "last_token" else "Promedio enmascarado")
        ax.legend(title="Familia del primer polimorfo", fontsize=7)
    axes[0].set_ylabel("Distancia euclidiana entre embeddings")
    for ax in axes: ax.set_xlabel("|ΔE DFT entre polimorfos| (meV/átomo)")
    fig.suptitle("Distancia de embeddings y diferencia energética intracomposición")
    fig.tight_layout(rect=(0.03, 0.15, 0.98, 0.9))
    n_intra_panel = int((within.pooling == "last_token").sum())
    note(fig, f"Cada marca representa un par no ordenado dentro de una composición; N={n_intra_panel} pares en cada panel. La asociación es descriptiva y no implica causalidad ni suficiencia física del embedding.")
    path = out / "03_embedding_distancia_vs_delta_dft.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)

    length = pairs.dropna(subset=["sequence_length_difference"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, pooling in zip(axes, ("last_token", "masked_mean")):
        local = length[(length.pooling.eq(pooling)) & (length.pair_scope.eq("intra_composition"))]
        ax.scatter(local.sequence_length_difference, local.euclidean_distance, s=30, alpha=0.68, color=COLORS[pooling])
        ax.set_title("Token final <end>" if pooling == "last_token" else "Promedio enmascarado")
    axes[0].set_ylabel("Distancia euclidiana entre embeddings")
    for ax in axes: ax.set_xlabel("Diferencia absoluta de longitud SLICES (tokens)")
    fig.suptitle("Sensibilidad de las distancias a la longitud SLICES")
    fig.tight_layout(rect=(0.03, 0.15, 0.98, 0.9))
    n_length = int(((length.pair_scope == "intra_composition") & (length.pooling == "last_token")).sum())
    note(fig, f"Pares intracomposición con longitud disponible: N={n_length} en cada panel. Cada secuencia se mide por número de tokens SLICES; se excluyen <start> y <end>.")
    path = out / "04_embedding_longitud_secuencia.png"
    save_figure(fig, path); plt.close(fig); generated.append(path)
    return generated


def main() -> None:
    """Lee resultados existentes y exporta figuras PNG, sin evaluar test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Directorio de resultados E1.1 para ocho figuras.")
    parser.add_argument("--last-dir", type=Path, default=None, help="Resultados de pooling last-token E1.2.")
    parser.add_argument("--mean-dir", type=Path, default=None, help="Resultados de pooling masked-mean E1.2.")
    parser.add_argument("--embedding-distances", type=Path, default=None, help="Tabla CSV de embedding_diagnostics.py.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        style()
        if args.experiment_dir is not None:
            output = args.out_dir or args.experiment_dir / "figures"
            generated = e1_1_figures(args.experiment_dir, output)
        elif args.last_dir is not None and args.mean_dir is not None:
            output = args.out_dir or ALT / "reports" / "matgpt_e1_2" / "figures"
            generated = pooling_figures(args.last_dir, args.mean_dir, output)
        elif args.embedding_distances is not None:
            output = args.out_dir or ALT / "reports" / "matgpt_e1_2" / "figures"
            generated = geometry_figures(args.embedding_distances, output)
        else:
            raise DiagnosticError("Especifica --experiment-dir o ambos --last-dir y --mean-dir.")
        log(LOGGER, logging.INFO, "figures_written", count=len(generated), paths=[str(path) for path in generated])
    except (DiagnosticError, OSError, ValueError, KeyError) as exc:
        log(LOGGER, logging.ERROR, "figure_generation_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
