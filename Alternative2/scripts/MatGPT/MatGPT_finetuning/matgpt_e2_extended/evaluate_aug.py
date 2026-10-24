#!/usr/bin/env python3
"""Evaluate augmentation OOF predictions against E2-A with composition-paired metrics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
REPORTS = ALT / "reports/MatGPT_finetuning"
BASELINE = REPORTS / "matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv"
LOG = logging.getLogger("matgpt_e2_aug.evaluate")
SEED = 13072026


def composition_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Calculate macro-per-composition absolute and polymorph-ranking metrics.

    Args:
        frame: OOF rows from one candidate and a matched fold/seed set.
        model: Candidate label written to the result.
    Returns:
        One row per composition with MAE, delta-E, GS, pairwise, range, and collapse fields.
    """
    rows: list[dict[str, Any]] = []
    for comp, group in frame.groupby("composition_group", sort=True):
        y = group.energy_per_atom_ev.to_numpy(float)
        p = group.energy_pred_ev_atom.to_numpy(float)
        if len(group) != 3:
            raise RuntimeError(f"La composición {comp} tiene {len(group)} polimorfos OOF, se esperaban tres.")
        dft_delta = (y - y.min()) * 1000.0
        pred_delta = (p - p.min()) * 1000.0
        dft_gap = float(np.partition(y, 1)[1] - np.min(y)) * 1000.0
        pred_gap = float(np.partition(p, 1)[1] - np.min(p)) * 1000.0
        gs_defined = dft_gap > 1.0
        gs = bool(np.argmin(y) == np.argmin(p)) if gs_defined and pred_gap > 1.0 else False
        dft_sign = np.sign(y[:, None] - y[None, :])
        pred_sign = np.sign(p[:, None] - p[None, :])
        upper = np.triu(np.ones((3, 3), dtype=bool), k=1)
        eligible = upper & (np.abs((y[:, None] - y[None, :]) * 1000.0) > 1.0)
        n_pairs = int(eligible.sum())
        pair_accuracy = float(np.mean(dft_sign[eligible] == pred_sign[eligible])) if n_pairs else np.nan
        rows.append({
            "configuration": model,
            "composition_group": comp,
            "n_structures": int(len(group)),
            "absolute_mae_mev_atom": float(np.mean(np.abs(p - y)) * 1000.0),
            "delta_e_mae_mev_atom": float(np.mean(np.abs(pred_delta - dft_delta))),
            "ground_state_accuracy": float(gs) if gs_defined else np.nan,
            "ground_state_eligible": gs_defined,
            "pairwise_accuracy": pair_accuracy,
            "n_pairs_eligible": n_pairs,
            "range_error_mev_atom": abs(float(np.ptp(p - p.min())) - float(np.ptp(y - y.min()))) * 1000.0,
            "dft_range_mev_atom": float(np.ptp(y) * 1000.0),
            "pred_range_mev_atom": float(np.ptp(p) * 1000.0),
            "collapsed": bool(np.ptp(p) * 1000.0 < 1.0 or pred_gap <= 1.0),
            "dft_near_tie": bool(dft_gap <= 1.0),
            "n_comparisons": n_pairs,
        })
    return pd.DataFrame(rows)


def fold_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Aggregate OOF results per fold and seed, preserving equal-size structure weights."""
    rows: list[dict[str, Any]] = []
    for (fold, seed), group in frame.groupby(["fold", "seed"], sort=True):
        comp = composition_metrics(group, model)
        rows.append({
            "configuration": model, "fold": int(fold), "seed": int(seed),
            "n_structures": int(len(group)), "n_compositions": int(len(comp)),
            "absolute_mae_mev_atom_pooled": float(np.mean(np.abs(group.energy_pred_ev_atom - group.energy_per_atom_ev)) * 1000.0),
            "absolute_mae_mev_atom_macro": float(comp.absolute_mae_mev_atom.mean()),
            "delta_e_mae_mev_atom": float(comp.delta_e_mae_mev_atom.mean()),
            "ground_state_accuracy": float(comp.ground_state_accuracy.mean()),
            "n_ground_state_compositions": int(comp.ground_state_eligible.sum()),
            "pairwise_accuracy": float(np.average(comp.dropna(subset=["pairwise_accuracy"]).pairwise_accuracy,
                                                  weights=comp.dropna(subset=["pairwise_accuracy"]).n_pairs_eligible)) if comp.pairwise_accuracy.notna().any() else np.nan,
            "pairwise_accuracy_macro": float(comp.pairwise_accuracy.mean()),
            "n_pairs_eligible": int(comp.n_pairs_eligible.sum()),
            "range_error_mev_atom": float(comp.range_error_mev_atom.mean()),
            "mean_predicted_range_mev_atom": float(comp.pred_range_mev_atom.mean()),
            "collapsed_compositions": int(comp.collapsed.sum()),
        })
    return pd.DataFrame(rows)


def bootstrap_pair(left: pd.DataFrame, right: pd.DataFrame, name: str, n_boot: int = 2000) -> dict[str, Any]:
    """Bootstrap paired model differences by resampling complete composition groups."""
    metrics = ["absolute_mae_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "range_error_mev_atom"]
    joined = left.merge(right, on="composition_group", suffixes=("_left", "_right"), validate="one_to_one")
    rng = np.random.default_rng(SEED)
    output: dict[str, Any] = {"comparison": name, "n_compositions": int(len(joined)), "bootstrap_replicates": n_boot, "seed": SEED, "intervals": {}}
    for metric in metrics:
        a = joined[f"{metric}_left"].to_numpy(float)
        b = joined[f"{metric}_right"].to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        diffs = a[valid] - b[valid]
        if len(diffs) == 0:
            continue
        samples = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        boot = diffs[samples].mean(axis=1)
        output["intervals"][metric] = {
            "difference_left_minus_right_mean": float(diffs.mean()),
            "ci95_percentile": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "includes_zero": bool(np.quantile(boot, 0.025) <= 0 <= np.quantile(boot, 0.975)),
            "n_compositions": int(len(diffs)),
        }
    return output


def load_cv(experiment_dir: Path) -> pd.DataFrame:
    """Read fold prediction fragments and validate one outer prediction per expected row."""
    path = experiment_dir / "development_only_v2" / "cv" / "cpu" / "oof_predictions.csv"
    if not path.is_file():
        path = experiment_dir / "cv" / "auto" / "oof_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Faltan predicciones CV completas: {path}")
    frame = pd.read_csv(path)
    if frame.duplicated(["fold", "seed", "structure_id"]).any():
        raise RuntimeError(f"Predicciones OOF duplicadas en {path}")
    return frame


def render_figures(per_fold: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    """Write Spanish PNG summaries with partition, denominators, and interpretation notes."""
    output.mkdir(parents=True, exist_ok=True)
    models = list(per_fold.configuration.unique())
    colors = {name: color for name, color in zip(models, ("#2878B5", "#E07A24", "#2A9D70", "#8055A6"))}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    metrics = [("absolute_mae_mev_atom_macro", "MAE absoluta macro (meV/átomo)"),
               ("delta_e_mae_mev_atom", "MAE de ΔE (meV/átomo)"),
               ("ground_state_accuracy", "Exactitud de estado fundamental (fracción)")]
    for ax, (column, label) in zip(axes, metrics):
        for pos, model in enumerate(models):
            vals = per_fold.loc[per_fold.configuration.eq(model), column].dropna().to_numpy()
            if len(vals):
                jitter = np.linspace(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals, color=colors[model], label=model, alpha=0.85, s=45)
                ax.plot([pos - .12, pos + .12], [np.mean(vals)] * 2, color=colors[model], lw=2)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.grid(True, alpha=.25)
    fig.suptitle("MatGPT E2: métricas outer por fold y semilla", y=0.985)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=max(1, len(models)), frameon=False)
    fig.text(.01, .01, "Cada punto = un fold×semilla (15 por candidato); promedio por composición salvo MAE pooled.\n"
             "OOF de desarrollo, 210 estructuras/70 composiciones; test histórico excluido. Las semillas no son grupos independientes.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .84))
    fig.savefig(output / "metricas_fold_semilla.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if len(paired):
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        names = sorted(set(paired.configuration_left) | set(paired.configuration_right))
        for color, (key, rows) in zip(("#2878B5", "#E07A24", "#2A9D70", "#8055A6", "#BC4B51", "#4C956C"), paired.groupby(["configuration_left", "configuration_right"])):
            ax.scatter(rows.delta_e_mae_mev_atom_right, rows.delta_e_mae_mev_atom_left,
                       label=f"{key[0]} − {key[1]}", alpha=.8, s=38, color=color)
        finite = paired[["delta_e_mae_mev_atom_left", "delta_e_mae_mev_atom_right"]].to_numpy(float)
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        ax.plot([lo, hi], [lo, hi], "k--", label="Igualdad: y = x")
        ax.set(xlabel="MAE ΔE del comparador (meV/átomo)", ylabel="MAE ΔE del modelo (meV/átomo)",
               title="Comparación pareada por composición")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=.25)
        fig.text(.02, .01, f"Cada marcador = una composición emparejada ({len(paired)} en la tabla); métricas OOF macro.\n"
                 "Línea y=x indica igualdad; puntos bajo la línea favorecen al modelo en MAE ΔE.", fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, 1))
        fig.savefig(output / "comparacion_pareada_deltae.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def evaluate(experiment_dirs: list[Path], output: Path, include_baseline: bool) -> None:
    """Aggregate candidate OOFs, compare matched composition metrics, bootstrap, and plot."""
    sources: dict[str, pd.DataFrame] = {}
    for directory in experiment_dirs:
        name = directory.name
        frame = load_cv(directory)
        if frame.seed.nunique() != 3 or frame.fold.nunique() != 5:
            raise RuntimeError(f"CV incompleta para {name}: folds={frame.fold.nunique()}, seeds={frame.seed.nunique()}")
        if len(frame) != 630:
            raise RuntimeError(f"Se esperaban 5×3×42=630 predicciones outer para {name}, hay {len(frame)}.")
        sources[name] = frame
    if include_baseline:
        if not BASELINE.is_file():
            raise FileNotFoundError(BASELINE)
        base = pd.read_csv(BASELINE)
        base = base[base.configuration.eq("partial_ft")].copy()
        base = base.rename(columns={"energy_pred_ev_atom": "energy_pred_ev_atom"})
        if base.duplicated(["fold", "seed", "structure_id"]).any():
            raise RuntimeError("OOF baseline E2-A partial_ft contiene filas duplicadas.")
        if len(base) != 630:
            raise RuntimeError(f"Baseline E2-A partial_ft debe tener 630 predicciones, tiene {len(base)}.")
        sources["E2-A-partial_ft"] = base
    frames: list[pd.DataFrame] = []
    per_comp: list[pd.DataFrame] = []
    per_fold: list[pd.DataFrame] = []
    for name, frame in sources.items():
        frame = frame.copy()
        comp_parts = []
        for (fold, seed), part in frame.groupby(["fold", "seed"], sort=True):
            c = composition_metrics(part, name)
            c["fold"] = int(fold)
            c["seed"] = int(seed)
            comp_parts.append(c)
        comp = pd.concat(comp_parts, ignore_index=True)
        per_comp.append(comp)
        per_fold.append(fold_metrics(frame, name))
        frame["comparison_model"] = name
        frames.append(frame)
    all_comp = pd.concat(per_comp, ignore_index=True)
    all_fold = pd.concat(per_fold, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    all_fold.to_csv(output / "metrics_per_fold_seed.csv", index=False)
    all_comp.to_csv(output / "metrics_composition_fold_seed.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / "oof_predictions_comparison.csv", index=False)
    by_family_rows = []
    for model, source in sources.items():
        for family, group in source.groupby("polymorph_family", sort=True):
            by_family_rows.append({
                "configuration": model, "polymorph_family": family, "n_oof_rows_structure_seed": int(len(group)),
                "absolute_mae_mev_atom": float(np.mean(np.abs(group.energy_pred_ev_atom - group.energy_per_atom_ev)) * 1000.0),
                "signed_bias_mev_atom": float(np.mean(group.energy_pred_ev_atom - group.energy_per_atom_ev) * 1000.0),
            })
    pd.DataFrame(by_family_rows).to_csv(output / "metrics_by_polymorph_family.csv", index=False)

    # Summarize each seed without treating the 70 groups × folds as independent.
    per_seed = all_fold.groupby(["configuration", "seed"], as_index=False).agg(
        absolute_mae_mev_atom=("absolute_mae_mev_atom_macro", "mean"),
        delta_e_mae_mev_atom=("delta_e_mae_mev_atom", "mean"),
        ground_state_accuracy=("ground_state_accuracy", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
        range_error_mev_atom=("range_error_mev_atom", "mean"),
        collapsed_compositions=("collapsed_compositions", "sum"),
    )
    per_seed.to_csv(output / "metrics_per_seed.csv", index=False)
    paired_bootstrap: list[dict[str, Any]] = []
    pair_rows: list[pd.DataFrame] = []
    for left_name, right_name in [(a, b) for i, a in enumerate(sources) for b in list(sources)[i + 1:]]:
        left = all_comp[all_comp.configuration.eq(left_name)].groupby("composition_group").mean(numeric_only=True).reset_index()
        right = all_comp[all_comp.configuration.eq(right_name)].groupby("composition_group").mean(numeric_only=True).reset_index()
        paired_bootstrap.append(bootstrap_pair(left, right, f"{left_name} - {right_name}"))
        merged = left.merge(right, on="composition_group", suffixes=("_left", "_right"), validate="one_to_one")
        merged["configuration_left"], merged["configuration_right"] = left_name, right_name
        merged["delta_e_mae_mev_atom"] = merged.delta_e_mae_mev_atom_left - merged.delta_e_mae_mev_atom_right
        pair_rows.append(merged)
    paired = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    (output / "paired_bootstrap_ci.json").write_text(json.dumps({"interpretation": "IC exploratorios pareados por composición completa; no garantizan superioridad.", "comparisons": paired_bootstrap}, indent=2) + "\n")
    render_figures(all_fold, paired, output / "figures")
    summary = {
        "status": "PASS", "models": list(sources), "n_folds": 5, "n_seeds": 3,
        "oof_predictions_per_candidate": 630, "test_historical_used": False,
        "bootstrap_replicates": 2000, "bootstrap_seed": SEED,
        "metrics_macro_by_composition_except_absolute_pooled_field": True,
        "limitations": ["E2-A model candidates were previously selected using historical validation compositions.", "Bootstrap intervals are exploratory; seeds are summarized separately."],
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fold_summary = all_fold.groupby("configuration")[["absolute_mae_mev_atom_macro", "delta_e_mae_mev_atom",
                                                        "ground_state_accuracy", "pairwise_accuracy_macro"]].agg(["mean", "std"])
    report_lines = [
        "# E2-Aug — comparación OOF por composición", "",
        "Estado: PASS para la agregación de predicciones disponibles. El test histórico no se consultó. "
        "Las ocho composiciones de validation participaron en preselección histórica; estos IC no eliminan ese sesgo.", "",
        "| Configuración | MAE absoluta macro (meV/átomo), media ± DE fold | MAE ΔE (meV/átomo), media ± DE fold | GS, media ± DE fold | Pairwise macro, media ± DE fold |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sources:
        row = fold_summary.loc[model]
        def fmt(metric: str) -> str:
            return f"{row[(metric, 'mean')]:.3f} ± {row[(metric, 'std')]:.3f}"
        report_lines.append(f"| {model} | {fmt('absolute_mae_mev_atom_macro')} | {fmt('delta_e_mae_mev_atom')} | {fmt('ground_state_accuracy')} | {fmt('pairwise_accuracy_macro')} |")
    report_lines += ["", "Bootstrap pareado: 2,000 remuestras de composiciones completas, semilla 13072026; IC porcentiles exploratorios. "
                     "Consulte `paired_bootstrap_ci.json`, `metrics_per_fold_seed.csv`, `metrics_composition_fold_seed.csv` y las figuras PNG.", "",
                     "MAE ΔE se calcula dentro de composición, GS excluye gaps DFT ≤1 meV/átomo y considera predicciones empatadas a ≤1 como no discriminativas; pairwise omite pares DFT con diferencia ≤1 meV/átomo.", ""]
    (output / "E2_Aug_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOG.info("augmentation_evaluation_complete candidates=%d rows=%d out=%s", len(sources), len(pd.concat(frames)), output)


def main() -> None:
    """Parse paths, validate selected experiments, and create comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, action="append", required=True,
                        help="Ruta E2-Aug1 o E2-Aug2 con subcarpeta cv/cpu.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args.experiment_dir, args.out, not args.no_baseline)


if __name__ == "__main__":
    main()
