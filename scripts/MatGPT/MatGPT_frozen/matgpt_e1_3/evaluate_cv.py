#!/usr/bin/env python3
"""Calcula métricas OOF pareadas, bootstrap por composición y figuras PNG E1.3."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CANDIDATES, E13Error, OUT, SEEDS, TIE_TOLERANCE_EV,
                    get_logger, group_metrics, log, sha256_file, write_json)

LOGGER = get_logger("matgpt_e1_3_evaluate")
COLORS = {"matgpt_e13_mlp8_mean_pair1": "#007C91", "matgpt_e13_mlp8_last_pair1": "#C43C78", "matgpt_e13_mlp32_mean_abs": "#D98E04"}
FAMILY_MARKERS = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}


def aggregate_predictions(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve filas de métricas por grupo y resumen de cada unidad solicitada."""
    summaries: list[dict[str, Any]] = []
    group_rows: list[pd.DataFrame] = []
    for values, part in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        summary, by_group = group_metrics(part)
        summaries.append({**dict(zip(keys, values)), **summary})
        group_rows.append(by_group.assign(**dict(zip(keys, values))))
    return pd.DataFrame(summaries), pd.concat(group_rows, ignore_index=True)


def metric_summary(ordered: pd.DataFrame, group_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crea métricas fold×seed, OOF por seed y tabla detallada composición×seed.

    Args:
        ordered: Predicciones OOF ya cotejadas con el fold externo.
        group_table: Mapa único de composición a fold.

    Returns:
        Métricas fold×seed, métricas OOF por seed y tabla composición×seed.
    """
    fold_seed, comp_fold_seed = aggregate_predictions(ordered, ["configuration", "seed", "fold"])
    oof_seed, comp_seed = aggregate_predictions(ordered, ["configuration", "seed"])
    composition = comp_seed.merge(group_table[["composition_group", "fold"]].drop_duplicates(), on="composition_group", how="left", validate="many_to_one")
    # Marcadores de outlier Tukey dentro de cada configuración/semilla para errores absolutos.
    composition["absolute_mae_outlier_tukey"] = False
    for _, idx in composition.groupby(["configuration", "seed"]).groups.items():
        values = composition.loc[idx, "absolute_mae_ev_atom"].dropna()
        if len(values):
            q1, q3 = values.quantile([0.25, 0.75])
            composition.loc[idx, "absolute_mae_outlier_tukey"] = composition.loc[idx, "absolute_mae_ev_atom"] > q3 + 1.5 * (q3-q1)
    return fold_seed, oof_seed, composition


def family_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAE/RMSE por familia, fold, semilla y candidato."""
    frame = oof.assign(abs_error=lambda x: (x.energy_pred_ev_atom-x.energy_dft_ev_atom).abs(),
                       squared_error=lambda x: (x.energy_pred_ev_atom-x.energy_dft_ev_atom)**2)
    return frame.groupby(["configuration", "seed", "fold", "polymorph_family"], as_index=False).agg(
        n_structures=("structure_id", "size"), mae_ev_atom=("abs_error", "mean"),
        rmse_ev_atom=("squared_error", lambda x: float(np.sqrt(x.mean()))))


def paired_by_composition(composition: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Promedia semillas dentro de composición y calcula bootstrap pareado de grupos."""
    metrics = ["absolute_mae_ev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "range_error_mev_atom"]
    averaged = composition.groupby(["configuration", "composition_group"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in metrics},
        pairwise_correct=("pairwise_correct", "mean"), pairwise_eligible=("pairwise_eligible", "mean"),
        ground_state_evaluable=("ground_state_evaluable", "mean"), predicted_range_mev_atom=("predicted_range_mev_atom", "mean"),
        dft_range_mev_atom=("dft_range_mev_atom", "mean"), collapsed_lt_1mev=("collapsed_lt_1mev", "mean"))
    averaged["absolute_range_error_mev_atom"] = averaged.range_error_mev_atom.abs()
    pair_rows: list[dict[str, Any]] = []
    ci_payload: dict[str, Any] = {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "resampling_unit": "70 development compositions; all three polymorphs kept together; seeds averaged within composition",
        "interpretation": "Exploratory paired intervals; not a guarantee of superiority; E1.1/E1.2 candidates were selected on these historical validation groups.",
        "comparisons": {}}
    groups = sorted(averaged.composition_group.unique())
    if len(groups) != 70:
        raise E13Error(f"Bootstrap requires 70 OOF development compositions, got {len(groups)}.")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {candidate: averaged[averaged.configuration.eq(candidate)].set_index("composition_group").reindex(groups) for candidate in CANDIDATES}
    for left, right in itertools.combinations(CANDIDATES, 2):
        ldf, rdf = samples[left], samples[right]
        if len(ldf) != 70 or len(rdf) != 70:
            raise E13Error(f"No hay 70 composiciones pareadas para {left} vs {right}.")
        out: dict[str, Any] = {}
        for metric in metrics + ["absolute_range_error_mev_atom"]:
            lval, rval = ldf[metric].to_numpy(float), rdf[metric].to_numpy(float)
            diff = lval - rval
            for i, group in enumerate(groups):
                pair_rows.append({"comparison": f"{left} - {right}", "composition_group": group,
                    "metric": metric, "left_value": lval[i], "right_value": rval[i], "paired_difference_left_minus_right": diff[i]})
            replicate_differences: list[float] = []
            for _ in range(BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, len(groups), size=len(groups))
                if metric == "ground_state_accuracy":
                    lmean = np.nanmean(lval[idx]) if np.isfinite(lval[idx]).any() else np.nan
                    rmean = np.nanmean(rval[idx]) if np.isfinite(rval[idx]).any() else np.nan
                elif metric == "pairwise_accuracy":
                    lc, le = ldf.pairwise_correct.to_numpy(float)[idx], ldf.pairwise_eligible.to_numpy(float)[idx]
                    rc, re = rdf.pairwise_correct.to_numpy(float)[idx], rdf.pairwise_eligible.to_numpy(float)[idx]
                    lmean = lc.sum()/le.sum() if le.sum() else np.nan
                    rmean = rc.sum()/re.sum() if re.sum() else np.nan
                else:
                    lmean, rmean = np.nanmean(lval[idx]), np.nanmean(rval[idx])
                if np.isfinite(lmean) and np.isfinite(rmean):
                    replicate_differences.append(float(lmean-rmean))
            observed = float(np.nanmean(diff)) if np.isfinite(diff).any() else None
            ci = np.quantile(replicate_differences, [0.025, 0.975]).tolist() if replicate_differences else [None, None]
            out[metric] = {"mean_paired_difference_left_minus_right": observed, "ci95_percentile": ci,
                           "bootstrap_valid_replicates": len(replicate_differences), "n_compositions": len(groups)}
        ci_payload["comparisons"][f"{left}__vs__{right}"] = out
    return pd.DataFrame(pair_rows), ci_payload


def _figure_fold_metrics(folds: pd.DataFrame, dest: Path) -> None:
    """Dibuja los valores fold×seed sin confundir semillas con grupos."""
    items = [("absolute_mae_ev_atom_pooled", "MAE absoluta pooled (eV/átomo)"),
             ("delta_e_mae_mev_atom_macro", "MAE de ΔE macro (meV/átomo)"),
             ("ground_state_accuracy", "Exactitud de estado fundamental"),
             ("pairwise_ordering_accuracy", "Exactitud de orden por pares")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (metric, title) in zip(axes.flat, items):
        for x, candidate in enumerate(CANDIDATES):
            part = folds[folds.configuration.eq(candidate)]
            for seed_index, seed in enumerate(SEEDS):
                vals = part[part.seed.eq(seed)][metric].dropna().to_numpy(float)
                jitter = (seed_index-1)*0.055
                ax.scatter(np.full(len(vals), x+jitter), vals, color=COLORS[candidate], marker=["o", "s", "^"][seed_index], alpha=0.78,
                           label=f"semilla {seed}" if x == 0 else None)
            mean = part[metric].mean()
            ax.plot([x-0.18, x+0.18], [mean, mean], color="black", lw=2)
        ax.set_title(title)
        ax.set_xticks(range(len(CANDIDATES)), [c.replace("matgpt_e13_", "") for c in CANDIDATES], rotation=20, ha="right")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("E1.3: métricas por pliegue externo y semilla", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "Puntos = un fold×seed; líneas negras = media aritmética de 15 valores por candidato. 5 folds, 3 semillas; 14 composiciones por fold. Test histórico excluido.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(dest / "01_metricas_por_fold.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure_paired_compositions(composition: pd.DataFrame, dest: Path) -> None:
    """Compara ΔE MAE por las mismas composiciones tras promediar semillas."""
    avg = composition.groupby(["configuration", "composition_group"], as_index=False).delta_e_mae_mev_atom.mean()
    pivot = avg.pivot(index="configuration", columns="composition_group", values="delta_e_mae_mev_atom").reindex(list(CANDIDATES))
    fig, ax = plt.subplots(figsize=(15, 4.8))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)), [x.replace("matgpt_e13_", "") for x in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=70, ha="right", fontsize=7)
    ax.set_title("MAE de ΔE por composición — comparación pareada E1.3")
    ax.set_xlabel("Composición química completa (70 grupos de desarrollo)")
    ax.set_ylabel("Configuración; valores promediados en 3 semillas")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("MAE de ΔE (meV/átomo; menor es mejor)")
    fig.text(0.5, 0.01, "Cada columna representa la misma composición para los tres candidatos; cada composición conserva sus tres polimorfos. ΔE = Eᵢ − mínimo del grupo calculado por separado en DFT y ML. Test histórico excluido.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(dest / "02_heatmap_deltae_pareado.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure_ranges(composition: pd.DataFrame, dest: Path) -> None:
    """Compara rango DFT y predicho por composición y candidato."""
    avg = composition.groupby(["configuration", "composition_group"], as_index=False).agg(
        dft_range_mev_atom=("dft_range_mev_atom", "mean"), predicted_range_mev_atom=("predicted_range_mev_atom", "mean"))
    lim = max(float(avg.dft_range_mev_atom.max()), float(avg.predicted_range_mev_atom.max())) * 1.03
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, candidate in zip(axes, CANDIDATES):
        part = avg[avg.configuration.eq(candidate)]
        ax.scatter(part.dft_range_mev_atom, part.predicted_range_mev_atom, s=28, alpha=0.75, color=COLORS[candidate], edgecolors="white", linewidths=0.35)
        ax.plot([0, lim], [0, lim], "--", color="#333333", label="acuerdo ideal: predicho = DFT")
        ax.set_title(candidate.replace("matgpt_e13_", ""))
        ax.set_xlabel("Rango DFT por composición (meV/átomo)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Rango predicho por composición (meV/átomo)")
    for ax in axes: ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.suptitle("E1.3: rangos polimórficos DFT y MatGPT OOF", weight="bold")
    fig.text(0.5, 0.01, "Punto = composición; predicciones promediadas entre 3 semillas después de evaluaciones outer-validation. Rango = máximo − mínimo dentro de cada composición. No se mezclan fórmulas; test excluido.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(dest / "03_rangos_dft_predichos.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure_delta_parity(oof: pd.DataFrame, dest: Path) -> None:
    """Grafica ΔE DFT frente a ΔE predicho con OLS libre por candidato."""
    avg = oof.groupby(["configuration", "structure_id"], as_index=False).agg(
        composition_group=("composition_group", "first"), polymorph_family=("polymorph_family", "first"),
        delta_e_dft_mev_atom=("delta_e_dft_mev_atom", "first"), delta_e_pred_mev_atom=("delta_e_pred_mev_atom", "mean"))
    limit = max(float(avg.delta_e_dft_mev_atom.max()), float(avg.delta_e_pred_mev_atom.max())) * 1.03
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    stats: dict[str, Any] = {}
    for ax, candidate in zip(axes, CANDIDATES):
        part = avg[avg.configuration.eq(candidate)]
        x, y = part.delta_e_dft_mev_atom.to_numpy(float), part.delta_e_pred_mev_atom.to_numpy(float)
        if len(x) < 2 or np.ptp(x) == 0:
            raise E13Error(f"No se puede estimar OLS con los puntos ΔE del candidato {candidate}.")
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope*x + intercept
        ss_res, ss_tot = float(np.sum((y-fitted)**2)), float(np.sum((y-y.mean())**2))
        r2 = 1-ss_res/ss_tot if ss_tot > 0 else None
        mae = float(np.mean(np.abs(y-x)))
        stats[candidate] = {"N_structures": len(part), "slope_dimensionless": float(slope), "intercept_mev_atom": float(intercept), "r2_ols": r2, "mae_vs_dft_mev_atom": mae}
        for family, marker in FAMILY_MARKERS.items():
            sample = part[part.polymorph_family.eq(family)]
            ax.scatter(sample.delta_e_dft_mev_atom, sample.delta_e_pred_mev_atom, marker=marker, s=27, alpha=0.75,
                       color=COLORS[candidate], edgecolors="white", linewidths=0.35, label=family)
        ax.plot([0, limit], [0, limit], "--", color="#222222", label="ideal: y = x")
        xx = np.linspace(0, limit, 100)
        ax.plot(xx, slope*xx+intercept, "-", color="#F2C14E", lw=2, label="OLS libre: y = m x + b")
        ax.text(0.03, 0.97, f"N = {len(part)} estructuras\nm = {slope:.3f}\nb = {intercept:.2f} meV/átomo\nR² = {r2:.3f}\nMAE = {mae:.2f} meV/átomo", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#ccd2d8"})
        ax.set_title(candidate.replace("matgpt_e13_", ""))
        ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.grid(alpha=0.2)
        ax.set_xlabel("DFT: ΔEᵢ = Eᵢ − Eₘᵢₙ,DFT (meV/átomo)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("MatGPT: ΔEᵢ = Eᵢ − Eₘᵢₙ,ML (meV/átomo)")
    fig.suptitle("E1.3: paridad OOF de energías relativas polimórficas", weight="bold")
    fig.text(0.5, 0.005, "Cada marcador = una estructura; forma = familia polimórfica. Predicciones promediadas entre semillas para 210 estructuras de desarrollo; cada grupo procede de outer-validation. Los mínimos DFT y ML se calculan por composición. Test histórico excluido.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(dest / "04_paridad_deltae_oof.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_json(dest.parent / "deltae_ols_statistics.json", stats)


def _figure_seeds(fold_seed: pd.DataFrame, dest: Path) -> None:
    """Muestra variación entre semillas de MAE y selección GS en OOF."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    metrics = [("absolute_mae_ev_atom_macro", "MAE absoluta macro (eV/átomo)"), ("ground_state_accuracy", "Exactitud estado fundamental")]
    for ax, (metric, title) in zip(axes, metrics):
        for x, candidate in enumerate(CANDIDATES):
            part = fold_seed[fold_seed.configuration.eq(candidate)]
            for j, seed in enumerate(SEEDS):
                values = part[part.seed.eq(seed)][metric].dropna().to_numpy(float)
                ax.scatter(np.full(len(values), x+(j-1)*0.06), values, marker=["o", "s", "^"][j], color=COLORS[candidate], alpha=0.68,
                           label=f"semilla {seed}" if x == 0 else None)
        ax.set_title(title); ax.set_xticks(range(len(CANDIDATES)), [x.replace("matgpt_e13_", "") for x in CANDIDATES], rotation=20, ha="right"); ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E1.3: sensibilidad a semilla por fold externo", weight="bold")
    fig.text(0.5, 0.01, "Puntos = 5 folds por semilla; 15 valores por candidato antes de valores indefinidos. Las semillas no se cuentan como composiciones independientes. Test excluido.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(dest / "05_sensibilidad_semilla.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure_collapse(composition: pd.DataFrame, dest: Path) -> None:
    """Expone colapso, empates y grupos próximos a la tolerancia DFT."""
    avg = composition.groupby(["configuration", "composition_group"], as_index=False).agg(
        pred_range=("predicted_range_mev_atom", "mean"), dft_range=("dft_range_mev_atom", "mean"),
        predicted_tie_fraction=("predicted_tied", "mean"), dft_tie_fraction=("dft_tied", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, candidate in enumerate(CANDIDATES):
        part = avg[avg.configuration.eq(candidate)]
        axes[0].scatter(np.full(len(part), i), part.pred_range, color=COLORS[candidate], alpha=0.7, s=25)
        axes[1].scatter(part.dft_range, part.pred_range, color=COLORS[candidate], alpha=0.7, s=25, label=candidate.replace("matgpt_e13_", ""))
    axes[0].axhline(1, color="black", linestyle="--", label="colapso: rango predicho < 1 meV/átomo")
    axes[0].set_xticks(range(len(CANDIDATES)), [c.replace("matgpt_e13_", "") for c in CANDIDATES], rotation=20, ha="right")
    axes[0].set_ylabel("Rango predicho (meV/átomo)"); axes[0].set_title("Rango por composición y candidato"); axes[0].legend(fontsize=7, frameon=False); axes[0].grid(alpha=0.2)
    axes[1].axhline(1, color="black", linestyle="--", label="umbral de colapso 1 meV/átomo")
    axes[1].axvline(1, color="#777777", linestyle=":", label="DFT rango 1 meV/átomo")
    axes[1].set_xlabel("Rango DFT (meV/átomo)"); axes[1].set_ylabel("Rango predicho (meV/átomo)"); axes[1].set_title("Rangos y grupos DFT casi degenerados"); axes[1].legend(fontsize=7, frameon=False); axes[1].grid(alpha=0.2)
    for i, candidate in enumerate(CANDIDATES):
        part = avg[avg.configuration.eq(candidate)]
        axes[2].scatter(np.full(len(part), i), part.predicted_tie_fraction, color=COLORS[candidate], alpha=0.7, s=25)
    axes[2].set_xticks(range(len(CANDIDATES)), [c.replace("matgpt_e13_", "") for c in CANDIDATES], rotation=20, ha="right")
    axes[2].set_ylim(-0.04, 1.04); axes[2].set_ylabel("Fracción de 3 semillas con empate ML")
    axes[2].set_title("Empate predicho por composición"); axes[2].grid(alpha=0.2)
    fig.suptitle("E1.3: colapso, empates y diferencias pequeñas", weight="bold")
    fig.text(0.5, 0.01, "Punto = composición; rangos promediados entre 3 semillas. Los empates de estado fundamental usan tolerancia 1 meV/átomo; la línea de rango <1 meV/átomo es un control de colapso. Test excluido.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(dest / "06_colapso_empates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _figure_separated_history(metrics_oof: pd.DataFrame, dest: Path) -> None:
    """Resume E1.3 sin combinarlo visualmente como una misma prueba que E1."""
    means = metrics_oof.groupby("configuration").agg(mae=("absolute_mae_ev_atom_macro", "mean"), delta=("delta_e_mae_mev_atom_macro", "mean"), gs=("ground_state_accuracy", "mean"), pair=("pairwise_ordering_accuracy", "mean"))
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.axis("off")
    ax.set_title("Resumen de evidencia MatGPT: particiones distintas, no combinables", fontsize=14, weight="bold", pad=18)
    lines = ["E1.3 — GroupKFold OOF en desarrollo (210 estructuras / 70 composiciones; 5 folds × 3 semillas; test histórico excluido)", "Métricas macro por composición, promediadas por semilla:"]
    for candidate, row in means.iterrows():
        lines.append(f"  {candidate}: MAE {row.mae:.4f} eV/átomo | MAE ΔE {row.delta:.1f} meV/átomo | GS {row.gs:.3f} | pares {row.pair:.3f}")
    lines += ["", "E1.1/E1.2 validation histórico: 8 composiciones usadas durante preselección; describe el protocolo original.", "E1 test histórico: observado anteriormente y NO incluido ni consultado en E1.3; no constituye test ciego.", "Interpretación: la CV E1.3 mide sensibilidad/robustez; la selección previa en esas ocho fórmulas mantiene sesgo de selección."]
    ax.text(0.03, 0.9, "\n".join(lines), va="top", ha="left", fontsize=10, linespacing=1.6, family="monospace")
    fig.savefig(dest / "07_resumen_particiones_separadas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(out: Path, fold_seed: pd.DataFrame, oof_seed: pd.DataFrame, composition: pd.DataFrame,
                 paired_ci: dict[str, Any], folds: pd.DataFrame) -> None:
    """Escribe resumen científico con reglas interpretativas predeclaradas."""
    fold_stats = fold_seed.groupby("configuration").agg(mae_fold_mean=("absolute_mae_ev_atom_macro", "mean"), mae_fold_sd=("absolute_mae_ev_atom_macro", "std"),
        delta_fold_mean=("delta_e_mae_mev_atom_macro", "mean"), delta_fold_sd=("delta_e_mae_mev_atom_macro", "std"),
        gs_fold_mean=("ground_state_accuracy", "mean"), gs_fold_sd=("ground_state_accuracy", "std"), pair_fold_mean=("pairwise_ordering_accuracy", "mean"), pair_fold_sd=("pairwise_ordering_accuracy", "std"))
    seed_stats = oof_seed.groupby("configuration").agg(mae_seed_mean=("absolute_mae_ev_atom_macro", "mean"), mae_seed_sd=("absolute_mae_ev_atom_macro", "std"),
        delta_seed_mean=("delta_e_mae_mev_atom_macro", "mean"), delta_seed_sd=("delta_e_mae_mev_atom_macro", "std"),
        gs_seed_mean=("ground_state_accuracy", "mean"), gs_seed_sd=("ground_state_accuracy", "std"), pair_seed_mean=("pairwise_ordering_accuracy", "mean"), pair_seed_sd=("pairwise_ordering_accuracy", "std"))
    summary = fold_stats.join(seed_stats, rsuffix="_by_seed")
    summary.to_csv(out / "summary_by_candidate.csv")
    report_lines = ["# E1.3 — Validación cruzada agrupada por composición", "",
        "## Estado y alcance", "", "E1.3 evalúa 3 cabezas congeladas con GroupKFold de 5 pliegues sobre 210 estructuras / 70 composiciones de desarrollo. Se excluyeron completamente las 24 estructuras / 8 composiciones del test histórico. Se completaron 45 ejecuciones (3 candidatos × 5 folds × 3 semillas).",
        "", "Las configuraciones A/B/C fueron preseleccionadas usando las mismas 8 composiciones de validation histórica E1.1/E1.2. Esta CV mide sensibilidad y robustez; no elimina el sesgo de selección. Para estimación independiente se requiere un holdout externo nuevo o nested CV completo. El test histórico fue observado en E1, no es ciego, y no se consultó aquí.",
        "", "E1.2 incluyó análisis geométricos exploratorios de los 234 embeddings; cualquier correlación que incluyera test está registrada como exploratoria y no guió folds ni selección en E1.3.", "",
        "## Protocolo congelado", "", "GroupKFold asigna grupos por fórmula química completa; cada composición aparece una vez en outer-validation. En cada outer-training se separaron 11 de 56 grupos (20%, 33 estructuras) para inner-validation determinista (semilla base 13072026 + índice fold). Las 45 ejecuciones optimizan pesos solo con inner-train. Objetivo/checkpoint se elige por mínimo objetivo normalizado de inner-validation, con paciencia 20 y máximo 200 épocas. Escalas de energía y diferencias de pares se ajustan exclusivamente con inner-train. MatGPT permanece congelado; solo se entrenan las cabezas. Las predicciones outer se calculan una vez al final de cada ejecución.",
        "", "## Resultados", "", "MAE absoluta pooled se pondera por estructura; MAE macro y ΔE MAE se promedian por composición. GS excluye grupos con empate DFT o predicho a 1 meV/átomo. Exactitud por pares omite pares DFT empatados bajo la misma tolerancia; las predicciones ML empatadas no aciertan un orden. Las semillas se resumen por separado y no cuentan como composiciones independientes.", "",
        "| Candidato | MAE macro fold media ± DE (eV/átomo) | ΔE MAE fold media ± DE (meV/átomo) | GS fold media ± DE | Pares fold media ± DE | MAE OOF entre semillas media ± DE | ΔE MAE OOF entre semillas media ± DE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for candidate, row in summary.iterrows():
        report_lines.append(f"| {candidate} | {row.mae_fold_mean:.4f} ± {row.mae_fold_sd:.4f} | {row.delta_fold_mean:.2f} ± {row.delta_fold_sd:.2f} | {row.gs_fold_mean:.3f} ± {row.gs_fold_sd:.3f} | {row.pair_fold_mean:.3f} ± {row.pair_fold_sd:.3f} | {row.mae_seed_mean:.4f} ± {row.mae_seed_sd:.4f} | {row.delta_seed_mean:.2f} ± {row.delta_seed_sd:.2f} |")
    report_lines += ["", "Medianas y outliers por composición (tabla `metrics_composition.csv`, valores macro entre seeds):", "",
        "| Candidato | Mediana MAE comp. (eV/átomo) | Mediana MAE ΔE (meV/átomo) | Outliers MAE Tukey (comp×seed) | Colapsos <1 meV/átomo (comp×seed) |", "|---|---:|---:|---:|---:|"]
    for candidate in CANDIDATES:
        part = composition[composition.configuration.eq(candidate)]
        report_lines.append(f"| {candidate} | {part.absolute_mae_ev_atom.median():.4f} | {part.delta_e_mae_mev_atom.median():.2f} | {int(part.absolute_mae_outlier_tukey.sum())} | {int(part.collapsed_lt_1mev.sum())} |")
    report_lines += ["", "## Incertidumbre pareada y outliers", "", f"Bootstrap pareado por composición: {BOOTSTRAP_REPLICATES:,} réplicas; semilla {BOOTSTRAP_SEED}; cada remuestra toma grupos completos con sus tres polimorfos. Intervalos porcentiles exploratorios, no garantía de superioridad.",
        "", "| Comparación (izquierda − derecha) | Métrica | Diferencia media pareada | IC bootstrap 95% |", "|---|---|---:|---:|"]
    for comparison, metrics in paired_ci["comparisons"].items():
        for metric in ("delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "absolute_range_error_mev_atom"):
            item = metrics[metric]
            ci = item["ci95_percentile"]
            point = "ND" if item["mean_paired_difference_left_minus_right"] is None else f"{item['mean_paired_difference_left_minus_right']:.4g}"
            interval = "ND" if ci[0] is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
            report_lines.append(f"| {comparison} | {metric} | {point} | {interval} |")
    report_lines += ["", "Consultar también `paired_bootstrap_ci.json`, `paired_comparisons.csv` y `metrics_composition.csv`; outliers absolutos marcados con regla Tukey Q3 + 1.5×IQR.",
        "", "## Criterios interpretativos predeclarados", "", "- A favorece masked mean para mínimos solo si supera de forma consistente a B en GS/orden por pares, no presenta colapso y los intervalos pareados respaldan la diferencia.", "- B favorece last-token para energías relativas si reduce consistentemente ΔE MAE/error de rango.", "- C puede tener menor MAE absoluta y peor ranking; se conservan como objetivos distintos.", "- Cambios de signo entre folds o intervalos amplios implican evidencia insuficiente; no se fuerza un ganador.", "- Los errores no se atribuyen únicamente a la representación: también dependen de cabeza, loss, protocolo y tamaño del conjunto.", "", "## Artefactos y limitaciones", "", "Figuras PNG: `figures/`. Métricas: `metrics_per_fold_seed.csv`, `metrics_oof_per_seed.csv`, `metrics_composition.csv`; predicciones: `oof_predictions.csv`. La etiqueta es energía total estática GGA por átomo, no energía de formación. No se ejecutó E2.", ""]
    (out / "E1_3_cross_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run(out: Path) -> None:
    """Calcula métricas, bootstrap pareado, auditoría y siete PNG sin leer test."""
    pred_path = out / "oof_predictions.csv"
    if not pred_path.is_file():
        raise E13Error(f"Falta OOF predictions: {pred_path}; ejecuta entrenamiento primero.")
    oof = pd.read_csv(pred_path)
    if not set(oof.historical_split.astype(str)).issubset({"train", "val"}):
        raise E13Error("OOF contiene filas del test histórico; se detiene sin calcular métricas.")
    expected_configs, expected_seeds = set(CANDIDATES), set(SEEDS)
    if set(oof.configuration) != expected_configs or set(oof.seed.astype(int)) != expected_seeds:
        raise E13Error("OOF no contiene exactamente los 3 candidatos y 3 semillas fijados.")
    counts = oof.groupby(["configuration", "seed", "structure_id"]).size()
    if len(counts) != 3*3*210 or not counts.eq(1).all():
        raise E13Error("Cada (configuración, semilla, structure_id) debe aparecer una vez en OOF.")
    assignment_path = out / "fold_assignments.csv"
    assignments = pd.read_csv(assignment_path)
    expected_dev = set(assignments.loc[assignments.historical_split.isin(["train", "val"]), "structure_id"].astype(str))
    if set(oof.structure_id.astype(str)) != expected_dev:
        raise E13Error("OOF no coincide exactamente con los 210 IDs de desarrollo.")
    group_fold = assignments[assignments.outer_role.eq("outer_validation")][["structure_id", "composition_group", "outer_fold"]].rename(columns={"outer_fold": "fold"})
    fold_counts = group_fold.groupby("fold").agg(structures=("structure_id", "size"), compositions=("composition_group", "nunique"))
    if len(fold_counts) != 5 or not fold_counts.structures.eq(42).all() or not fold_counts.compositions.eq(14).all():
        raise E13Error("Cada outer fold debe evaluar 42 estructuras de 14 composiciones.")
    oof = oof.merge(group_fold, on=["structure_id", "composition_group", "fold"], how="left", validate="many_to_one")
    if oof.fold.isna().any() or not np.isfinite(oof.energy_pred_ev_atom).all() or not np.isfinite(oof.energy_dft_ev_atom).all():
        raise E13Error("Predicciones OOF incompletas/no finitas o sin fold.")
    if oof.groupby(["configuration", "seed", "composition_group"]).fold.nunique().gt(1).any():
        raise E13Error("Un grupo composición aparece en más de un fold OOF.")
    fold_seed, oof_seed, composition = metric_summary(oof, group_fold)
    fold_seed.to_csv(out / "metrics_per_fold_seed.csv", index=False)
    oof_seed.to_csv(out / "metrics_oof_per_seed.csv", index=False)
    composition.to_csv(out / "metrics_composition.csv", index=False)
    family_metrics(oof).to_csv(out / "metrics_by_family.csv", index=False)
    paired, ci = paired_by_composition(composition)
    paired.to_csv(out / "paired_comparisons.csv", index=False)
    write_json(out / "paired_bootstrap_ci.json", ci)
    figure_dir = out / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    _figure_fold_metrics(fold_seed, figure_dir)
    _figure_paired_compositions(composition, figure_dir)
    _figure_ranges(composition, figure_dir)
    _figure_delta_parity(oof, figure_dir)
    _figure_seeds(fold_seed, figure_dir)
    _figure_collapse(composition, figure_dir)
    _figure_separated_history(oof_seed, figure_dir)
    write_report(out, fold_seed, oof_seed, composition, ci, assignments)
    write_json(out / "evaluation_manifest.json", {"status": "PASS", "test_historical_rows_used": 0,
        "oof_rows": int(len(oof)), "fold_seed_rows": int(len(fold_seed)), "oof_seed_rows": int(len(oof_seed)),
        "composition_metric_rows": int(len(composition)), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "tie_tolerance_ev_atom": TIE_TOLERANCE_EV,
        "oof_sha256": sha256_file(pred_path), "evaluator_script_sha256": sha256_file(Path(__file__)),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")), "formats": ["PNG"]})
    log(LOGGER, logging.INFO, "e13_evaluation_complete", oof_rows=len(oof), bootstrap=BOOTSTRAP_REPLICATES,
        figures=len(list(figure_dir.glob("*.png"))), test_used=False, out_dir=out.resolve())


def main() -> None:
    """Valida argumentos y evalúa únicamente las predicciones OOF de desarrollo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except (E13Error, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "e13_evaluation_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
