"""Genera figuras científicas OOF de las métricas F0/F4 en ABS₃.

Las figuras separan energía absoluta, diferencias intrafórmula, ranking,
dependencia del gap y evolución de entrenamiento/validación. No reajustan modelos.
"""
from __future__ import annotations

import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LOG = logging.getLogger("abs3.f4.metrics_figures")
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71"
IMG = ROOT / "images/abs3_matgpt_mattergpt_original_f4_71"
COLORS = {"MatGPT": "#2878B5", "MatterGPT": "#D66AA8"}
STYLES = {"F0": "", "F4": "//"}
CONDITIONS = ("F0", "F4")
MODELS = ("MatGPT", "MatterGPT")


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y valida artefactos canónicos del experimento.

    Returns:
        Métricas OOF, predicciones por estructura/par y los historiales de pérdida.
    """
    metrics = json.loads((EXP / "metrics.json").read_text(encoding="utf-8"))
    structures = pd.read_csv(EXP / "predicciones_oof_por_estructura.csv")
    pairs = pd.read_csv(EXP / "predicciones_oof_por_par.csv")
    history = pd.read_csv(EXP / "train_history.csv")
    gap_history = pd.read_csv(EXP / "diagnostics/mae_curve_replay/training_mae_per_epoch.csv")
    expected = {(model, cond) for model in MODELS for cond in CONDITIONS}
    actual = set(zip(structures.model, structures.condition))
    if actual != expected:
        raise ValueError(f"Condiciones de predicción incompletas: {actual ^ expected}")
    if structures.structure_id.nunique() != 210 or structures.composition_id.nunique() != 70:
        raise ValueError("OOF de estructura no corresponde a 210 estructuras/70 composiciones")
    if len(pairs) != 2520 or not np.isfinite(pairs[["delta_dft_mev_atom", "delta_pred_mev_atom"]]).all().all():
        raise ValueError("Predicciones OOF por par incompletas o no finitas")
    return metrics, structures, pairs, history, gap_history


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ajusta OLS con intercepto y devuelve pendiente, intercepto y R²."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(r2_score(y, slope * x + intercept))


def plot_metric_summary(metrics: dict[str, Any]) -> None:
    """Grafica MAE absoluta, MAE residual y MAE ΔE de los cuatro modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    metric_specs = [
        ("mae_energy_ev_atom", "MAE de energía absoluta", 1000.0),
        ("mae_residual_mev_atom", "MAE del residuo intrafórmula", 1.0),
        ("mae_delta_signed_est_a_mev_atom", "MAE ΔE firmado (estimando A)", 1.0),
    ]
    x = np.arange(len(MODELS)); width = 0.34
    for ax, (key, title, scale) in zip(axes, metric_specs):
        for offset, cond in enumerate(CONDITIONS):
            values = [metrics[f"{model}_{cond}"][key] * scale for model in MODELS]
            bars = ax.bar(x + (offset - .5) * width, values, width, label=cond,
                          color=[COLORS[m] for m in MODELS], edgecolor="#222222",
                          hatch=STYLES[cond], alpha=.95)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, MODELS)
        ax.set_ylabel("meV/átomo")
        ax.grid(axis="y", alpha=.22)
    fig.legend(handles=[Patch(facecolor="#dddddd", edgecolor="#222222", label="F0 congelado"),
                        Patch(facecolor="#dddddd", edgecolor="#222222", hatch="//", label="F4: bloques 8–11")],
               loc="upper center", bbox_to_anchor=(.5, .89), ncol=2, frameon=False)
    fig.suptitle("ABS₃ OOF: energía absoluta, residuo y diferencias polimórficas")
    fig.text(.5, .015,
             "Color: MatGPT azul, MatterGPT magenta; relleno sólido = F0 y trama = F4. 70 composiciones · 210 estructuras · 5 pliegues × 3 semillas. "
             "Se promedian semillas dentro de composición y después las 70 composiciones. "
             "ΔE firmado = Eⱼ−Eᵢ para los tres pares intrafórmula; estimando A: media de errores de pares por fórmula y macro-media. "
             "F0 = backbone congelado; F4 = bloques 8–11 ajustados. Celdas relajadas por DFT.",
             ha="center", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0, .09, 1, .84))
    fig.savefig(IMG / "metricas_energia_f0_f4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bootstrap(metrics: dict[str, Any]) -> None:
    """Grafica cambio F4−F0 de MAE ΔE con IC bootstrap apareado."""
    rows = [r for r in metrics["bootstrap_contrasts"] if "F4 −" in r["contrast"] and "F0" in r["contrast"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(rows))
    est = np.array([r["estimate_mev_atom"] for r in rows])
    low = np.array([r["ci95_low"] for r in rows]); high = np.array([r["ci95_high"] for r in rows])
    ax.errorbar(est, y, xerr=[est - low, high - est], fmt="o", color="#384B63", capsize=4, lw=1.8)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y, [r["contrast"].replace("_", " ") for r in rows])
    ax.set_xlabel("F4 − F0: cambio en MAE ΔE (meV/átomo; negativo favorece F4)")
    ax.set_title("ABS₃: diferencia apareada de MAE ΔE")
    ax.grid(axis="x", alpha=.25)
    ax.text(.01, -.30, "Bootstrap por 70 composiciones completas; 10 000 réplicas, semilla 20261103; IC95% percentil. "
            "OOF de desarrollo, no prueba externa. Las semillas se agregan dentro de fórmula.",
            transform=ax.transAxes, fontsize=8.4, wrap=True)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(IMG / "bootstrap_apareado_deltaE_f4_vs_f0.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_energy_parity(structures: pd.DataFrame) -> None:
    """Grafica paridad de energía DFT/modelo con ajuste OLS por panel."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    limits = [min(structures.energy_dft_ev_atom.min(), structures.energy_pred_ev_atom.min()) * 1000,
              max(structures.energy_dft_ev_atom.max(), structures.energy_pred_ev_atom.max()) * 1000]
    for ax, (model, cond) in zip(axes.flat, [(m, c) for m in MODELS for c in CONDITIONS]):
        q = structures[(structures.model == model) & (structures.condition == cond)]
        x = q.energy_dft_ev_atom.to_numpy() * 1000
        y = q.energy_pred_ev_atom.to_numpy() * 1000
        m, b, r2 = _ols(x, y); mae = float(np.mean(np.abs(y - x)))
        ax.scatter(x, y, s=10, alpha=.22, color=COLORS[model], edgecolors="none")
        grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
        ax.plot(grid, grid, color="#222222", ls="--", lw=1, label="Ideal: y=x")
        ax.plot(grid, m * grid + b, color="#D97706", lw=1.5, label="OLS: y=mx+b")
        ax.set_title(f"{model} — {cond}")
        ax.text(.03, .97, f"N={len(q)} estructura×semilla\nm={m:.3f}; b={b:.1f} meV/átomo\nR²={r2:.3f}; MAE={mae:.1f} meV/átomo",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .86, "edgecolor": "#cccccc"})
        ax.grid(alpha=.2); ax.legend(loc="lower right", fontsize=7, frameon=True)
    for ax in axes.flat:
        ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1, :]: ax.set_xlabel("Energía DFT por átomo (meV/átomo)")
    for ax in axes[:, 0]: ax.set_ylabel("Energía predicha por átomo (meV/átomo)")
    fig.suptitle("ABS₃ OOF: paridad de energía absoluta por estructura")
    fig.text(.5, .015, "Punto = estructura×semilla; 210 estructuras en 70 composiciones, predichas OOF en 5 pliegues × 3 semillas (N=630 por panel). "
             "OLS e indicadores se calculan con todos los puntos del panel; las repeticiones por semilla no son independientes. "
             "La energía predicha reconstruye μ_Ridge-M2 + residuo centrado.", ha="center", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(IMG / "paridad_energia_absoluta_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_parity(pairs: pd.DataFrame) -> None:
    """Grafica paridad de ΔE firmado con límites comunes y OLS por panel."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    lim = float(max(pairs.delta_dft_mev_atom.abs().max(), pairs.delta_pred_mev_atom.abs().max()))
    for ax, (model, cond) in zip(axes.flat, [(m, c) for m in MODELS for c in CONDITIONS]):
        q = pairs[(pairs.model == model) & (pairs.condition == cond)]
        x = q.delta_dft_mev_atom.to_numpy(); y = q.delta_pred_mev_atom.to_numpy()
        m, b, r2 = _ols(x, y); mae = float(q.absolute_error_mev_atom.mean())
        ax.scatter(x, y, s=9, alpha=.18, color=COLORS[model], edgecolors="none")
        grid = np.linspace(-lim, lim, 100)
        ax.plot(grid, grid, color="#222222", ls="--", lw=1, label="Ideal: y=x")
        ax.plot(grid, m * grid + b, color="#D97706", lw=1.5, label="OLS: y=mx+b")
        ax.set_title(f"{model} — {cond}")
        ax.text(.03, .97, f"N={len(q)} par×semilla\nm={m:.3f}; b={b:.1f} meV/átomo\nR²={r2:.3f}; MAE={mae:.1f} meV/átomo",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .86, "edgecolor": "#cccccc"})
        ax.grid(alpha=.2); ax.legend(loc="lower right", fontsize=7, frameon=True)
    for ax in axes.flat:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1, :]: ax.set_xlabel("ΔE DFT firmada, Eⱼ−Eᵢ (meV/átomo)")
    for ax in axes[:, 0]: ax.set_ylabel("ΔE predicha firmada, Eⱼ−Eᵢ (meV/átomo)")
    fig.suptitle("ABS₃ OOF: paridad de diferencias energéticas entre polimorfos")
    fig.text(.5, .015, "Punto = par intrafórmula×semilla; 210 pares únicos (70 fórmulas×3) repetidos con 3 semillas (N=630 por panel). "
             "ΔE = Eⱼ−Eᵢ; signo negativo/positivo indica orden relativo del par. OLS e indicadores se calculan con todos los puntos mostrados; "
             "las repeticiones por fórmula y semilla no son independientes.", ha="center", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(IMG / "paridad_deltaE_firmada_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(metrics: dict[str, Any]) -> None:
    """Compara pairwise, pairwise resuelto, ground state y no resueltos."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    keys = [("pairwise_accuracy_eligible_including_unresolved", "Ranking pairwise elegible", "591 pares elegibles; no resueltos cuentan como error"),
            ("pairwise_resolved_accuracy", "Pairwise entre predicciones resueltas", "N resueltos: MatGPT 584/583; MatterGPT 563/562 (F0/F4)"),
            ("ground_state_accuracy", "Selección del estado fundamental", "201 casos fórmula-semilla elegibles por condición"),
            ("unresolved_fraction_eligible_pairs", "Fracción pairwise no resuelta", "Denominador: 591 pares elegibles; |ΔE pred| ≤ 1 meV/átomo")]
    x = np.arange(2); width = .34
    for ax, (key, title, denominator) in zip(axes.flat, keys):
        for offset, cond in enumerate(CONDITIONS):
            vals = [metrics[f"{model}_{cond}"][key] for model in MODELS]
            bars = ax.bar(x + (offset - .5) * width, vals, width, label=cond,
                          color=[COLORS[m] for m in MODELS], edgecolor="#222222", hatch=STYLES[cond])
            for bar, value in zip(bars, vals):
                ypos = value - .045 if value >= .12 else value + .018
                color = "white" if value >= .12 else "#222222"
                ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{value:.1%}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")
        ax.set_title(title); ax.set_xticks(x, MODELS); ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=.22); ax.text(.02, .93, denominator, transform=ax.transAxes, fontsize=7.4, va="top",
                                               bbox={"facecolor":"white","alpha":.86,"edgecolor":"none"})
        if key == "pairwise_accuracy_eligible_including_unresolved":
            ax.axhline(.9, color="#A33", ls="--", lw=1)
            ax.text(.98, .905, "Meta 90%", transform=ax.transAxes, ha="right", va="bottom", color="#922", fontsize=8,
                    bbox={"facecolor":"white","alpha":.88,"edgecolor":"none"})
    fig.legend(handles=[Patch(facecolor="#dddddd", edgecolor="#222222", label="F0 congelado"),
                        Patch(facecolor="#dddddd", edgecolor="#222222", hatch="//", label="F4: bloques 8–11")],
               loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.suptitle("ABS₃ OOF: ranking polimórfico y selección del mínimo DFT")
    fig.text(.5, .012, "Pairwise: correcto si coincide el signo para |ΔE_DFT|>5 meV/átomo; no resuelto (|ΔE_pred|≤1) cuenta incorrecto. "
             "Ground state solo incluye gap DFT al segundo mínimo >1 meV/átomo; empates predichos no aciertan. "
             "Se agregan semillas dentro de fórmula. F0 backbone congelado; F4 bloques 8–11 entrenables.",
             ha="center", fontsize=8.1, wrap=True)
    fig.tight_layout(rect=(0, .08, 1, .86))
    fig.savefig(IMG / "ranking_groundstate_f0_f4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gap_errors(pairs: pd.DataFrame) -> pd.DataFrame:
    """Grafica MAE de ΔE por magnitud de gap y devuelve tabla con denominadores."""
    bins = [0, 5, 10, 25, 50, 100, np.inf]
    labels = ["0–5", "5–10", "10–25", "25–50", "50–100", ">100"]
    df = pairs.copy()
    df["intervalo_gap_mev_atom"] = pd.cut(df.delta_dft_mev_atom.abs(), bins=bins, labels=labels,
                                          right=False, include_lowest=True)
    grouped = (df.groupby(["model", "condition", "intervalo_gap_mev_atom"], observed=False)
              .agg(n_pairs_seed=("absolute_error_mev_atom", "size"),
                   mae_deltaE_mev_atom=("absolute_error_mev_atom", "mean"))
              .reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(labels)); width = .34
    for ax, model in zip(axes, MODELS):
        for offset, cond in enumerate(CONDITIONS):
            q = grouped[(grouped.model == model) & (grouped.condition == cond)].set_index("intervalo_gap_mev_atom").reindex(labels)
            vals = q.mae_deltaE_mev_atom.to_numpy(float)
            bars = ax.bar(x + (offset - .5) * width, vals, width, color=COLORS[model],
                          edgecolor="#222222", hatch=STYLES[cond], label=cond, alpha=.9)
            for bar, n in zip(bars, q.n_pairs_seed.fillna(0).astype(int)):
                if np.isfinite(bar.get_height()):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"n={n}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_title(model); ax.set_xticks(x, labels); ax.set_xlabel("|ΔE DFT| (meV/átomo)")
        ax.grid(axis="y", alpha=.22); ax.legend(title="Condición", frameon=False)
    axes[0].set_ylabel("MAE de ΔE firmada (meV/átomo)")
    fig.suptitle("ABS₃ OOF: error de diferencias energéticas según el gap DFT")
    fig.text(.5, .01, "Cada barra pondera por pares×semilla; n indica ese denominador. 210 pares únicos se repiten con tres semillas; "
             "los registros de una fórmula no son independientes. F0/F4 usan las mismas composiciones, folds y targets.",
             ha="center", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0, .08, 1, .92))
    fig.savefig(IMG / "error_deltaE_por_magnitud_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return grouped


def plot_training(history: pd.DataFrame) -> None:
    """Grafica pérdida de minilotes y validación completa por época."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        part = history[history.model.eq(model)]
        count_ax = ax.twinx()
        for cond, color in [("F0", "#2878B5"), ("F4", "#E76F51")]:
            q = part[part.condition.eq(cond)]
            for column, style, label in [("train_minibatch_smoothl1", "-", f"{cond}: train minilotes"),
                                         ("inner_validation_full_smoothl1", "--", f"{cond}: validación completa")]:
                stat = q.groupby("epoch")[column].agg(["mean", "std"])
                x = stat.index.to_numpy(float); mean = stat["mean"].to_numpy(float)
                sd = stat["std"].fillna(0).to_numpy(float)
                ax.plot(x, mean, color=color, ls=style, label=label)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.10)
            n = q.groupby("epoch")[["fold", "seed"]].apply(lambda z: z.drop_duplicates().shape[0])
            count_ax.step(n.index.to_numpy(float), n.to_numpy(float), where="mid", color=color,
                          ls=":", alpha=.68, linewidth=1.2)
        count_ax.set_ylim(0, 16); count_ax.set_yticks([0, 5, 10, 15])
        count_ax.set_ylabel("Corridas activas n (eje derecho)", fontsize=8, color="#555555")
        count_ax.tick_params(axis="y", labelsize=8, colors="#555555")
        ax.set_title(model); ax.set_xlabel("Época local tras warmup compartido")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("SmoothL1 del residuo normalizado")
    fig.legend(handles=[Line2D([0], [0], color="#2878B5", ls="-", label="F0: train minilotes"),
                        Line2D([0], [0], color="#2878B5", ls="--", label="F0: validación completa"),
                        Line2D([0], [0], color="#E76F51", ls="-", label="F4: train minilotes"),
                        Line2D([0], [0], color="#E76F51", ls="--", label="F4: validación completa"),
                        Line2D([0], [0], color="#2878B5", ls=":", label="Corridas activas F0"),
                        Line2D([0], [0], color="#E76F51", ls=":", label="Corridas activas F4")],
               loc="upper center", bbox_to_anchor=(.5, .9), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("ABS₃: aprendizaje de MLP8 — F0 congelado frente a F4")
    fig.text(.5, .012, "15 corridas fold-semilla por condición; train = promedio de minilotes y validación = evaluación completa de 11 fórmulas. "
             "Bandas = ±1 DE entre corridas activas (no IC); el número activo cae por early stopping. Checkpoint elegido por inner-validation; "
             "45 fórmulas inner-train. No se muestran resultados outer en esta figura.", ha="center", fontsize=8.1, wrap=True)
    fig.tight_layout(rect=(0, .08, 1, .82))
    fig.savefig(IMG / "curvas_perdida_f0_f4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Valida datos, genera PNGs y registra tiempo y memoria empleados."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.perf_counter(); cpu = time.process_time()
    IMG.mkdir(parents=True, exist_ok=True)
    metrics, structures, pairs, history, _gap_history = load_inputs()
    plot_metric_summary(metrics); plot_delta_bootstrap(metrics); plot_energy_parity(structures)
    plot_delta_parity(pairs); plot_ranking(metrics); by_gap = plot_gap_errors(pairs); plot_training(history)
    by_gap.to_csv(EXP / "metricas_deltaE_por_intervalo_gap.csv", index=False)
    files = sorted(p.name for p in IMG.glob("*.png"))
    manifest = {"experiment": "abs3_matgpt_mattergpt_original_f4_71", "figures": files,
                "metric_sources": ["metrics.json", "predicciones_oof_por_estructura.csv", "predicciones_oof_por_par.csv", "train_history.csv"],
                "oof": {"compositions": 70, "structures": 210, "folds": 5, "seeds": [42, 123, 2026]},
                "test_accessed": False, "models_modified": False}
    (EXP / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXP / "figures_resources.json").write_text(json.dumps({"wall_seconds": time.perf_counter()-start,
       "process_cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
       "device": "cpu", "threads": 1, "outputs": len(files), "test_accessed": False}, indent=2) + "\n", encoding="utf-8")
    LOG.info("Generadas 7 figuras; OOF filas estructura=%d, pares=%d", len(structures), len(pairs))


if __name__ == "__main__":
    main()
