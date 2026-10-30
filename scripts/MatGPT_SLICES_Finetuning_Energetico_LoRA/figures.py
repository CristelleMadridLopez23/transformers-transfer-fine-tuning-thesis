"""Genera doce figuras PNG científicas del experimento C completado."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment as ex
from evaluate import ols

COLORS = {"frozen": "#2673A6", "lora": "#C05A2D"}
NAMES = {"frozen": "MatGPT congelado", "lora": "MatGPT con LoRA"}
LOG = logging.getLogger("matgpt_lora_figures")


def save(fig: plt.Figure, path: Path, note: str) -> None:
    """Inserta nota contextual bajo paneles y exporta solo PNG."""
    fig.text(0.04, 0.025, note + " MAE = error absoluto medio; GS = estado fundamental; "
             "OLS = mínimos cuadrados ordinarios.", fontsize=8.5, color="#425166", va="bottom", wrap=True)
    fig.tight_layout(rect=(0.02, 0.18, 0.98, 0.94))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def parity(pred: pd.DataFrame, field_y: str, field_p: str, title: str,
           filename: str, units: str, note: str, path: Path) -> None:
    """Dibuja paridad OLS con ejes iguales y una semilla física por estructura."""
    selection = pred[pred.seed == 42]
    pooled = selection[[field_y, field_p]].to_numpy(float)
    bound_min = float(np.min(pooled)); bound_max = float(np.max(pooled))
    pad = 0.05 * (bound_max - bound_min)
    bounds = (bound_min - pad, bound_max + pad)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax, condition in zip(axes, ("frozen", "lora")):
        panel = selection[selection.condition == condition]
        x = panel[field_y].to_numpy(float); y = panel[field_p].to_numpy(float)
        stats = ols(x, y)
        ax.scatter(x, y, s=18, alpha=0.6, color=COLORS[condition], label="Estructura DFT estática")
        ax.plot(bounds, bounds, "--", color="#687B90", lw=1.3, label="Ideal: y = x")
        ax.plot(bounds, np.array(bounds) * stats["slope"] + stats["intercept"],
                "-.", color="#A424A1", lw=1.5, label="Ajuste OLS")
        ax.set(xlim=bounds, ylim=bounds, xlabel=f"DFT ({units})", ylabel=f"Predicción ({units})",
               title=NAMES[condition])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.97, f"N = {len(panel)}\nm = {stats['slope']:.3f}\n"
                f"b = {stats['intercept']:.3f} {units}\nR² OLS = {stats['r2_ols']:.3f}\n"
                f"MAE = {np.abs(y-x).mean():.2f} {units}", transform=ax.transAxes,
                va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .86, "edgecolor": "none"})
        ax.legend(loc="lower right", fontsize=8)
    save(fig, path / filename, note)


def make_figures(output: Path) -> None:
    """Lee predicciones OOF completas y produce las doce figuras solicitadas."""
    pred = pd.read_csv(output / "predicciones_outer_por_estructura.csv")
    summary = pd.read_csv(output / "metricas_por_semilla.csv")
    formula = pd.read_csv(output / "metricas_por_composicion.csv")
    pairs = pd.read_csv(output / "diferencias_por_par.csv")
    bins = pd.read_csv(output / "intervalos_gap_dft.csv")
    boot = pd.read_csv(output / "bootstrap_apareado.csv")
    drift = pd.read_csv(output / "cambio_embeddings_outer_semilla42.csv")
    drift_formula = pd.read_csv(output / "distancias_embeddings_intraformula_semilla42.csv")
    drift_summary = json.loads((output / "resumen_rango_y_varianza_embeddings_semilla42.json").read_text())
    if len(pred) != 1260 or len(pairs) != 1260:
        raise ValueError("Tablas incompletas: no generar figuras")
    dest = output / "figures"; dest.mkdir(exist_ok=True)
    note = ("Validación externa de los cinco pliegues de desarrollo; 70 fórmulas, 210 estructuras físicas "
            "por condición y semilla; 3 semillas son repeticiones de entrenamiento, no materiales independientes. "
            "El test histórico permanece cerrado.")
    # 1: comparación general, cada barra resume primero las tres semillas.
    metrics = [("mae_per_atom_mev", "MAE E/N\n(meV/átomo)"),
               ("mae_residual_mev", "MAE residual\n(meV/átomo)"),
               ("mae_pair_mev", "MAE de pares\n(meV/átomo)"),
               ("gs_eligible_accuracy", "Estado fundamental\nelegible (fracción)"),
               ("pairwise_accuracy_gt5", "Pares elegibles\n(fracción)")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
    fig.suptitle("ABS₃: MatGPT congelado frente a LoRA en energía y estabilidad", fontweight="bold")
    for ax, (column, label) in zip(axes, metrics):
        means = summary.groupby("condition")[column].mean()
        ax.bar([0, 1], [means["frozen"], means["lora"]],
               color=[COLORS["frozen"], COLORS["lora"]])
        ax.set_xticks([0, 1], ["Congelado", "LoRA"]); ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    save(fig, dest / "comparacion_finetuning_vs_congelado_metricas_energeticas_y_ranking.png", note)
    # 2: incertidumbre apareada por composición.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ABS₃: efecto apareado de LoRA, intervalo bootstrap del 95 %", fontweight="bold")
    metrics = [
        (axes[0], ["absolute_mae_mev_atom", "residual_mae_mev_atom", "pair_mae_mev_atom"],
         ["MAE energía E/N", "MAE residuo intrafórmula", "MAE de diferencias ΔE"],
         "LoRA − congelado (meV/átomo)"),
        (axes[1], ["gs_exact_argmin", "pairwise_accuracy_gt5"],
         ["Selección del estado fundamental", "Orden de pares |ΔE DFT| > 5"],
         "LoRA − congelado (fracción)"),
    ]
    for ax, keys, labels, xlabel in metrics:
        selected = boot.set_index("metric").loc[keys]
        center = selected.difference_lora_minus_frozen.to_numpy(float)
        low = selected.ci95_lower.to_numpy(float); high = selected.ci95_upper.to_numpy(float)
        y = np.arange(len(keys))
        ax.errorbar(center, y, xerr=[center-low, high-center], fmt="o",
                    color=COLORS["lora"], capsize=4, label="Diferencia e IC del 95 %")
        ax.set_yticks(y, labels); ax.axvline(0, color="#555", ls="--", label="Sin diferencia")
        ax.set_xlabel(xlabel); ax.set_ylim(len(keys)-.5, -.5)
        ax.grid(axis="x", alpha=.2); ax.legend(fontsize=8, loc="best")
    save(fig, dest / "efecto_del_finetuning_intervalos_confianza_bootstrap.png",
         note + " Intervalos por remuestreo de 70 fórmulas tras promediar semillas; GS/orden son diferencias de fracción indicadas en texto.")
    # 3: E/N absoluto.
    parity(pred, "energy_per_atom_dft_ev", "energy_per_atom_pred_ev",
           "ABS₃: energía DFT por átomo y predicción", "paridad_energia_dft_por_atomo_frozen_vs_lora.png",
           "eV/átomo", note + " Paridad mostrada para semilla 42; cada punto es una estructura.", dest)
    # 4: residuos centrados por fórmula.
    centered = pred.copy()
    for field in ("energy_per_atom_dft_ev", "energy_per_atom_pred_ev"):
        centered[field.replace("energy_per_atom", "residual")] = 1000 * (
            centered[field] - centered.groupby(["condition", "seed", "composition_id"])[field].transform("mean"))
    parity(centered, "residual_dft_ev", "residual_pred_ev",
           "ABS₃: residuo energético respecto a la media de cada fórmula",
           "paridad_residuos_energeticos_intraformula_frozen_vs_lora.png", "meV/átomo",
           note + " Residuos = 1000·[E/N − media de los tres polimorfos]; semilla 42.", dest)
    # 5: diferencias firmadas por pares.
    parity(pairs, "delta_dft_mev_atom", "delta_pred_mev_atom",
           "ABS₃: diferencias energéticas firmadas entre polimorfos",
           "paridad_diferencias_energeticas_pares_frozen_vs_lora.png", "meV/átomo",
           note + " 210 pares por condición y semilla; ΔE = 1000·(e_B − e_A), familias en orden perovskita, hexagonal, aguja.", dest)
    # 6: MAE por magnitud de gap.
    gap_labels = ["0–5", "5–10", "10–25", "25–50", "50–100", ">100"]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("ABS₃: error de ΔE por magnitud del gap DFT", fontweight="bold")
    x = np.arange(6); width = .27
    for offset, condition in ((-width, "frozen"), (0, "lora")):
        sub = bins[bins.condition == condition].groupby("gap_min_mev", sort=True).mean(numeric_only=True)
        ax.bar(x + offset, sub.mae_pair_mev, width, label=NAMES[condition], color=COLORS[condition])
    zero = bins[bins.condition == "frozen"].groupby("gap_min_mev", sort=True).mean(numeric_only=True)
    ax.bar(x + width, zero.mae_zero_mev, width, color="#8A8F99", label="ΔE predicho = 0")
    ax.set_xticks(x, gap_labels); ax.set_xlabel("|ΔE DFT| (meV/átomo)"); ax.set_ylabel("MAE de pares (meV/átomo)")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    save(fig, dest / "error_diferencias_energeticas_segun_gap_dft_frozen_vs_lora.png",
         note + " Cada par se asigna según |ΔE DFT|; barras = promedio de tres semillas, cada una reúne los cinco pliegues y los 210 pares físicos. En 0–5 meV/átomo no hay pares elegibles para ranking. El predictor cero no adjudica estado fundamental.")
    # 7: ranking por gap.
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("ABS₃: acierto de orden polimórfico según gap DFT", fontweight="bold")
    for condition in ("frozen", "lora"):
        sub = bins[bins.condition == condition].groupby("gap_min_mev", sort=True).mean(numeric_only=True)
        ax.plot(x[1:], sub.pair_accuracy.to_numpy()[1:], "o-", color=COLORS[condition], label=NAMES[condition])
    ax.axhline(.5, ls="--", color="#8A8F99", label="Azar: 50 %")
    ax.set_xticks(x[1:], gap_labels[1:]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("|ΔE DFT| (meV/átomo)"); ax.set_ylabel("Acierto entre pares elegibles")
    ax.legend(); ax.grid(alpha=.2)
    save(fig, dest / "exactitud_ordenamiento_polimorfos_por_gap_frozen_vs_lora.png",
         note + " Solo pares |ΔE DFT| >5 meV/átomo; predicciones con |ΔE| ≤1 cuentan como abstenciones incorrectas.")
    # 8: curvas de aprendizaje fold0/seed42, sin mezclar épocas de corridas.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle("ABS₃: pérdida residual normalizada durante el entrenamiento", fontweight="bold")
    for ax, condition in zip(axes, ("frozen", "lora")):
        run = output / "runs" / f"{condition}_fold0_seed42"
        hist = pd.read_csv(run / "history.csv")
        best = json.loads((run / "summary.json").read_text())["best_epoch"]
        ax.plot(hist.epoch, hist.train_loss, label="Entrenamiento por fórmula", color="#338A74")
        ax.plot(hist.epoch, hist.inner_validation_loss, label="Validación interna", color=COLORS[condition])
        ax.axvline(best, color="#444", ls="--", label=f"Seleccionado: época {best}")
        ax.set(xlabel="Época", ylabel="SmoothL1 normalizada", title=NAMES[condition])
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, dest / "evolucion_perdida_residual_entrenamiento_y_validacion.png",
         "Fold 0, semilla 42: 45 fórmulas train y 11 de validación interna; selección por mínima SmoothL1 residual, sin Ridge.")
    # 9: drift de representación de las 210 estructuras únicas para semilla 42.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("ABS₃: cambio de FINAL_END tras adaptar Q/V con LoRA", fontweight="bold")
    flat = axes.ravel()
    flat[0].hist(1 - drift.cosine_similarity, bins=25, color=COLORS["lora"])
    flat[0].set(xlabel="1 − coseno(original, adaptado)", ylabel="Estructuras", title="Cambio por estructura")
    flat[1].hist(drift.normalized_euclidean_distance, bins=25, color=COLORS["lora"])
    flat[1].set(xlabel="Distancia euclidiana / norma original", ylabel="Estructuras", title="Distancia de cambio")
    labels = ["Original", "LoRA"]
    colors = [COLORS["frozen"], COLORS["lora"]]
    for axis, key_a, key_b, title, ylabel in (
        (flat[2], "mean_intraformula_cosine_distance_original", "mean_intraformula_cosine_distance_lora", "Separación intrafórmula", "Distancia coseno media"),
        (flat[3], "mean_intraformula_normalized_euclidean_original", "mean_intraformula_normalized_euclidean_lora", "Separación intrafórmula", "Distancia euclidiana normalizada"),
        (flat[4], "variance_across_structures_original", "variance_across_structures_lora", "Variabilidad entre estructuras", "Varianza media por dimensión"),
        (flat[5], "effective_rank_original", "effective_rank_lora", "Rango espectral efectivo", "Rango efectivo (exp(entropía))")):
        values = [drift_summary[key_a], drift_summary[key_b]]
        axis.bar(labels, values, color=colors)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=.2)
    save(fig, dest / "cambio_representaciones_finales_tras_finetuning.png",
         note + f" Semilla 42: {drift_summary['n_structures']} estructuras/70 fórmulas; rango efectivo = exp(entropía de autovalores de covarianza). Distancia y rango describen embeddings, no demuestran utilidad predictiva.")
    # 10: errores firmados y absolutos de pares.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("ABS₃: distribución de errores de diferencias polimórficas", fontweight="bold")
    for condition in ("frozen", "lora"):
        sub = pairs[(pairs.condition == condition) & (pairs.seed == 42)]
        error = sub.delta_pred_mev_atom - sub.delta_dft_mev_atom
        axes[0].hist(error, bins=30, alpha=.55, color=COLORS[condition], label=NAMES[condition])
        axes[1].hist(np.abs(error), bins=30, alpha=.55, color=COLORS[condition], label=NAMES[condition])
    axes[0].set(xlabel="Error firmado ΔE pred − DFT (meV/átomo)", ylabel="Pares")
    axes[1].set(xlabel="|Error ΔE| (meV/átomo)", ylabel="Pares")
    for ax in axes: ax.legend(fontsize=8)
    save(fig, dest / "distribucion_errores_diferencias_energeticas_frozen_vs_lora.png",
         note + " Semilla 42, 210 pares por condición; histogramas con conteos, sin recortar colas.")
    # 11: error absoluto por familia de estructura; no atribuir ranking a una familia.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ABS₃: error y orden polimórfico por familia y tipo de par", fontweight="bold")
    family_names = ["Perovskita distorsionada", "Hexagonal", "Aguja"]
    for j, condition in enumerate(("frozen", "lora")):
        sub = pred[pred.condition == condition].copy()
        sub["absolute_error"] = 1000 * abs(sub.energy_per_atom_pred_ev - sub.energy_per_atom_dft_ev)
        vals = sub.groupby("polymorph_family").absolute_error.mean().reindex(ex.FAMILIES)
        axes[0].bar(np.arange(3) + (j-.5)*.35, vals, width=.35, color=COLORS[condition], label=NAMES[condition])
        subpairs = pairs[pairs.condition == condition]
        for i, (a, b) in enumerate(((ex.FAMILIES[0], ex.FAMILIES[1]),
                                    (ex.FAMILIES[0], ex.FAMILIES[2]),
                                    (ex.FAMILIES[1], ex.FAMILIES[2]))):
            selected = subpairs[(subpairs.family_a == a) & (subpairs.family_b == b)]
            eligible = int(selected.eligible_gt5.sum())
            accuracy = selected.correct_gt5.sum() / eligible if eligible else np.nan
            axes[1].bar(i + (j-.5)*.35, accuracy, width=.35, color=COLORS[condition],
                        label=NAMES[condition] if i == 0 else None)
            if eligible:
                axes[1].text(i + (j-.5)*.35, accuracy + .025, f"n={eligible}", ha="center", fontsize=7)
    axes[0].set_xticks(np.arange(3), family_names); axes[0].set_ylabel("MAE E/N (meV/átomo)")
    axes[0].set_title("Error por estructura"); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.2)
    axes[1].set_xticks(np.arange(3), ["Perovskita–hexagonal", "Perovskita–aguja", "Hexagonal–aguja"], rotation=15)
    axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("Acierto de orden entre pares elegibles")
    axes[1].set_title("Ranking por tipo de par"); axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.2)
    save(fig, dest / "errores_y_ordenamiento_por_familia_estructural.png",
         note + " Error resume estructuras por familia; ranking se agrupa por los dos tipos que forman el par. n sobre |ΔE DFT| >5 meV/átomo.")
    # 12: mejora por fórmula, promediando semillas.
    wide = formula.pivot_table(index="composition_id", columns="condition", values="residual_mae_mev_atom", aggfunc="mean")
    improvement = (wide.lora - wide.frozen).sort_values()
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("ABS₃: cambio de MAE residual por composición", fontweight="bold")
    ax.bar(range(len(improvement)), improvement.to_numpy(), color=np.where(improvement < 0, "#338A74", "#C05A2D"))
    ax.axhline(0, color="#555", lw=1)
    ax.set(xlabel="70 fórmulas ordenadas por cambio", ylabel="LoRA − congelado (meV/átomo)")
    ax.grid(axis="y", alpha=.2)
    save(fig, dest / "mejora_residual_y_ranking_por_composicion.png",
         note + " Cada barra es una fórmula con 3 polimorfos; se promedian las tres semillas antes de restar. No implica significación individual.")


def main() -> None:
    """Valida ruta de reportes y genera los PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = args.output.resolve()
    make_figures(output)
    LOG.info("%s", {"event": "figures_complete", "directory": str(output), "png_count": 12})


if __name__ == "__main__":
    main()
