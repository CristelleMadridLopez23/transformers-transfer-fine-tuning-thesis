#!/usr/bin/env python3
"""Evaluate E1 heads, polymorph rankings, and export thesis-ready PNG figures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.common import ALT, DATASET, E1Error, REPORTS, configure_logging, sha256_file, verify_backbone_files, write_json
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.train_regressor import LinearProbe, MatGPTRegressor, metrics

LOGGER = logging.getLogger("matgpt_e1")
COLORS = {"train": "#3569a8", "val": "#e07a25", "test": "#34845a", "mean": "#777777", "linear": "#9a6ab0", "mlp": "#c84630"}


def load_predictor(path: Path, kind: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an E1 head and its train-only target normalization.

    Args:
        path: Serialized head checkpoint.
        kind: mlp or linear.
        device: Evaluation device.

    Returns:
        Head in eval mode and its checkpoint payload.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model: torch.nn.Module
    if kind == "mlp":
        model = MatGPTRegressor(int(payload["input_dim"]), int(payload["training_config"]["regressor"]["hidden_dim"]), float(payload["training_config"]["regressor"]["dropout"]))
    else:
        model = LinearProbe(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def composition_metrics(group: pd.DataFrame, tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate relative energy, ground-state, pairwise, and rank metrics.

    Args:
        group: Rows from one data split with DFT and prediction columns.
        tolerance: DFT energy tie tolerance in eV/atom.

    Returns:
        Summary metrics, per-composition records, and pairwise comparison rows.
    """
    relative_errors: list[float] = []
    ground_correct = 0
    ground_total = 0
    ground_tied_predictions = 0
    pair_correct = 0
    pair_total = 0
    dft_ties = 0
    ml_ties = 0
    near_tie_excluded = 0
    pair_rows: list[dict[str, Any]] = []
    formula_rows: list[dict[str, Any]] = []
    spearman_values: list[float] = []
    for formula, comp in group.groupby("composition", sort=True):
        if len(comp) < 2:
            continue
        dft = comp["energy_dft_ev_atom"].to_numpy(dtype=float)
        pred = comp["energy_pred_ev_atom"].to_numpy(dtype=float)
        dft_relative = dft - dft.min()
        pred_relative = pred - pred.min()
        relative_errors.extend(np.abs(pred_relative - dft_relative).tolist())
        true_min = np.flatnonzero(np.isclose(dft, dft.min(), atol=tolerance, rtol=0.0))
        pred_min = np.flatnonzero(np.isclose(pred, pred.min(), atol=tolerance, rtol=0.0))
        pred_tied = len(pred_min) > 1
        correct = None if pred_tied else bool(int(pred_min[0]) in set(true_min.tolist()))
        if pred_tied:
            ground_tied_predictions += 1
        else:
            ground_correct += int(correct)
            ground_total += 1
        comp_spearman = None
        if len(comp) >= 3 and np.ptp(dft) > 0.0 and np.ptp(pred) > 0.0:
            value = float(spearmanr(dft, pred).statistic)
            if np.isfinite(value):
                spearman_values.append(value)
                comp_spearman = value
        formula_rows.append({"composition": formula, "n_polymorphs": int(len(comp)), "ground_state_correct": correct,
                             "prediction_ground_state_tied_within_tolerance": pred_tied,
                             "dft_minimum_structure_ids": comp.iloc[true_min]["structure_id"].astype(str).tolist(),
                             "predicted_minimum_structure_ids": comp.iloc[pred_min]["structure_id"].astype(str).tolist(),
                             "relative_energy_mae_mev_atom": float(np.mean(np.abs(pred_relative - dft_relative)) * 1000),
                             "spearman_rho": comp_spearman})
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                dft_diff = float(dft[i] - dft[j])
                pred_diff = float(pred[i] - pred[j])
                dft_tie = abs(dft_diff) <= tolerance
                pred_tie = abs(pred_diff) <= tolerance
                dft_ties += int(dft_tie)
                ml_ties += int(pred_tie)
                excluded = dft_tie
                near_tie_excluded += int(excluded)
                correct_pair = None if excluded or pred_tie else bool(np.sign(dft_diff) == np.sign(pred_diff))
                if correct_pair is not None:
                    pair_correct += int(correct_pair)
                    pair_total += 1
                pair_rows.append({"composition": formula, "structure_id_i": str(comp.iloc[i]["structure_id"]),
                                  "structure_id_j": str(comp.iloc[j]["structure_id"]), "dft_delta_e_ev_atom": dft_diff,
                                  "predicted_delta_e_ev_atom": pred_diff, "dft_tie_within_tolerance": dft_tie,
                                  "prediction_tie_within_tolerance": pred_tie, "included_pairwise_accuracy": correct_pair})
    summary = {
        "eligible_compositions": int(len(formula_rows)),
        "ground_state_evaluable_compositions": ground_total,
        "ground_state_tied_prediction_compositions_excluded": ground_tied_predictions,
        "relative_energy_mae_ev_per_atom": float(np.mean(relative_errors)) if relative_errors else None,
        "relative_energy_mae_mev_per_atom": float(np.mean(relative_errors) * 1000) if relative_errors else None,
        "ground_state_correct_compositions": ground_correct,
        "ground_state_accuracy": ground_correct / ground_total if ground_total else None,
        "pairwise_correct": pair_correct, "pairwise_evaluable_pairs": pair_total,
        "pairwise_ordering_accuracy": pair_correct / pair_total if pair_total else None,
        "pairwise_dft_ties_excluded": near_tie_excluded, "pairwise_prediction_ties": ml_ties,
        "pairwise_dft_tie_tolerance_ev_per_atom": tolerance,
        "spearman_compositions_n_ge_3": len(spearman_values),
        "mean_composition_spearman_rho": float(np.mean(spearman_values)) if spearman_values else None,
    }
    return summary, formula_rows, pair_rows


def save_figures(predictions: pd.DataFrame, history: pd.DataFrame, polymorph: pd.DataFrame,
                 pairs: pd.DataFrame, metrics_by_split: dict[str, Any], figure_dir: Path) -> None:
    """Create E1 learning, parity, composition, and polymorph PNG figures.

    Args:
        predictions: Full structure-level predictions table.
        history: Training history for both heads.
        polymorph: Per-composition test ranking table.
        pairs: Pairwise test comparisons.
        metrics_by_split: Energy metrics by method and split.
        figure_dir: PNG output directory.

    Returns:
        None.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    note_style = {"fontsize": 8, "color": "#444444"}
    if "head" not in history.columns or "epoch" not in history.columns:
        raise E1Error("training_history.csv debe incluir columnas explícitas 'head' y 'epoch'.")
    mlp_history = history.loc[history["head"].astype(str).eq("mlp")].copy()
    if mlp_history.empty:
        raise E1Error("training_history.csv no contiene historial de la MLP.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(mlp_history.epoch, mlp_history.train_loss_smooth_l1_normalized, label="Train", color=COLORS["train"])
    axes[0].plot(mlp_history.epoch, mlp_history.val_loss_smooth_l1_normalized, label="Validation", color=COLORS["val"])
    axes[0].set(xlabel="Epoch (stage-local, 1-based)", ylabel="SmoothL1 loss (train-normalized target)", title="MLP loss")
    axes[0].legend()
    axes[1].plot(mlp_history.epoch, mlp_history.train_mae_ev_per_atom, label="Train", color=COLORS["train"])
    axes[1].plot(mlp_history.epoch, mlp_history.val_mae_ev_per_atom, label="Validation", color=COLORS["val"])
    best = mlp_history.loc[mlp_history.val_mae_ev_per_atom.idxmin()]
    axes[1].axvline(best.epoch, color="#222222", linestyle="--", label=f"Selected epoch {int(best.epoch)}")
    axes[1].set(xlabel="Epoch (stage-local, 1-based)", ylabel="MAE (eV/atom)", title="MLP energy error")
    axes[1].legend()
    fig.suptitle("MatGPT frozen embeddings: training and validation")
    fig.text(.5, .01, f"Train N={len(predictions[predictions.split == 'train'])}; validation N={len(predictions[predictions.split == 'val'])}. Historical split is grouped by composition; target is static DFT total energy per atom.", ha="center", **note_style)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(figure_dir / "01_learning_curves.png", dpi=300)
    plt.close(fig)

    test = predictions[predictions.split == "test"]
    actual = test.energy_dft_ev_atom.to_numpy(float)
    pred = test.energy_pred_ev_atom.to_numpy(float)
    slope, intercept = np.polyfit(actual, pred, 1)
    fitted = slope * actual + intercept
    r2 = float(1.0 - np.sum((pred - fitted) ** 2) / np.sum((pred - pred.mean()) ** 2)) if len(pred) > 1 and np.var(pred) > 0 else None
    limits = [float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(test.energy_dft_ev_atom, test.energy_pred_ev_atom, label=f"Held-out test (N={len(test)})", color=COLORS["test"], alpha=.8, s=30)
    ax.plot(limits, limits, color="#222222", linestyle="--", label="Ideal agreement: y = x")
    ax.plot(limits, slope * np.asarray(limits) + intercept, color="#c84630", linewidth=1.5, label="Test free-intercept OLS fit")
    ax.set(xlim=limits, ylim=limits, xlabel="DFT total energy per atom (eV/atom)", ylabel="MLP predicted energy per atom (eV/atom)", title="DFT versus MatGPT E1 predictions")
    stats = metrics(actual, pred)
    ols_stats = {"n": int(len(actual)), "slope_dimensionless": float(slope), "intercept_ev_per_atom": float(intercept),
                 "r2": r2, "mae_ev_per_atom": float(stats["mae_ev_per_atom"]),
                 "x_definition": "DFT total energy per atom (eV/atom)",
                 "y_definition": "MLP predicted total energy per atom (eV/atom)"}
    r2_text = f"{r2:.3f}" if r2 is not None else "undefined"
    annotation = f"Test N={len(test)}; slope={slope:.3f}; intercept={intercept:.4f} eV/atom; R²={r2_text}; MAE={stats['mae_ev_per_atom']:.4f} eV/atom"
    ax.text(.03, .97, annotation, transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    ax.legend(loc="lower right", fontsize=8)
    fig.text(.5, .01, "Each point is one static ABS3 test structure; held-out compositions were not used for selection. OLS uses every displayed observation.", ha="center", **note_style)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(figure_dir / "02_dft_vs_mlp.png", dpi=300)
    plt.close(fig)

    comp_error = test.assign(abs_error_mev=(test.energy_pred_ev_atom - test.energy_dft_ev_atom).abs() * 1000).groupby("composition").abs_error_mev.mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(comp_error.index, comp_error.values, color=COLORS["test"])
    ax.set(xlabel="Composition (formula)", ylabel="Mean absolute error (meV/atom)", title="MLP absolute error by held-out composition")
    ax.tick_params(axis="x", rotation=65)
    fig.text(.5, .01, f"Test partition only; {len(comp_error)} compositions, {len(test)} structures. Each bar is an unweighted mean over that composition's displayed structures.", ha="center", **note_style)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(figure_dir / "03_error_by_composition.png", dpi=300)
    plt.close(fig)

    relative = test.copy()
    relative["dft_delta_mev"] = relative.energy_dft_ev_atom - relative.groupby("composition").energy_dft_ev_atom.transform("min")
    relative["pred_delta_mev"] = relative.energy_pred_ev_atom - relative.groupby("composition").energy_pred_ev_atom.transform("min")
    relative[["dft_delta_mev", "pred_delta_mev"]] *= 1000
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(relative.dft_delta_mev, relative.pred_delta_mev, color=COLORS["test"], alpha=.8, s=30, label=f"Test structures (N={len(relative)})")
    lim = [0, float(max(relative.dft_delta_mev.max(), relative.pred_delta_mev.max(), 1e-8))]
    ax.plot(lim, lim, "k--", label="Ideal relative-energy agreement: y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="DFT relative energy ΔE (meV/atom)", ylabel="MLP relative energy ΔE (meV/atom)", title="Relative energies within held-out compositions")
    ax.legend()
    fig.text(.5, .01, "Each ΔE is measured from the minimum within the same composition and split; no cross-composition pairs are formed.", ha="center", **note_style)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(figure_dir / "04_relative_energies.png", dpi=300)
    plt.close(fig)

    if len(polymorph):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = polymorph.sort_values("composition")
        statuses = ["Tied" if pd.isna(value) else "Correct" if bool(value) else "Incorrect" for value in ordered.ground_state_correct]
        positions = [{"Incorrect": 0, "Correct": 1, "Tied": 2}[value] for value in statuses]
        colors = [{"Incorrect": COLORS["mlp"], "Correct": COLORS["test"], "Tied": COLORS["mean"]}[value] for value in statuses]
        from matplotlib.patches import Patch
        ax.bar(ordered.composition, positions, color=colors)
        ax.legend(handles=[Patch(color=COLORS["test"], label="Correct"), Patch(color=COLORS["mlp"], label="Incorrect"), Patch(color=COLORS["mean"], label="Prediction tie")], loc="upper right")
        ax.set(ylim=(-.1, 2.2), yticks=[0, 1, 2], yticklabels=["Incorrect", "Correct", "Prediction tie"], xlabel="Composition (formula)", ylabel="Lowest-energy selection", title="Held-out ground-state selection")
        ax.tick_params(axis="x", rotation=65)
        evaluable = ordered.ground_state_correct.notna()
        correct_count = int(ordered.loc[evaluable, "ground_state_correct"].astype(bool).sum())
        fig.text(.5, .01, f"Test: {correct_count}/{int(evaluable.sum())} evaluable compositions correct; {int((~evaluable).sum())} predicted ties shown separately. DFT minima within 1 meV/atom count as tied.", ha="center", **note_style)
        fig.tight_layout(rect=(0, .08, 1, 1))
        fig.savefig(figure_dir / "05_ground_state_selection.png", dpi=300)
        plt.close(fig)

    valid_pairs = pairs[pairs.included_pairwise_accuracy.notna()].copy()
    if len(valid_pairs):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ok = valid_pairs.included_pairwise_accuracy.astype(bool)
        ax.scatter(valid_pairs.loc[ok, "dft_delta_e_ev_atom"] * 1000, valid_pairs.loc[ok, "predicted_delta_e_ev_atom"] * 1000,
                   color=COLORS["test"], label=f"Correct order (N={int(ok.sum())})", alpha=.8)
        ax.scatter(valid_pairs.loc[~ok, "dft_delta_e_ev_atom"] * 1000, valid_pairs.loc[~ok, "predicted_delta_e_ev_atom"] * 1000,
                   color=COLORS["mlp"], label=f"Wrong order (N={int((~ok).sum())})", alpha=.8)
        vals = np.r_[valid_pairs.dft_delta_e_ev_atom.to_numpy(), valid_pairs.predicted_delta_e_ev_atom.to_numpy()] * 1000
        bounds = [float(vals.min()), float(vals.max())]
        ax.plot(bounds, bounds, "k--", label="Equal pair difference: y = x")
        ax.axhline(0, color="#888888", linewidth=.7)
        ax.axvline(0, color="#888888", linewidth=.7)
        ax.set(xlabel="DFT pair difference ΔEᵢⱼ (meV/atom)", ylabel="MLP pair difference ΔEᵢⱼ (meV/atom)", title="Pairwise ordering within held-out compositions")
        ax.legend(fontsize=8)
        excluded = int(pairs.included_pairwise_accuracy.isna().sum())
        fig.text(.5, .01, f"Test pairs from same composition only; {excluded} pairs excluded for DFT or prediction ties within 1 meV/atom. Reference line is y = x.", ha="center", **note_style)
        fig.tight_layout(rect=(0, .06, 1, 1))
        fig.savefig(figure_dir / "06_pairwise_ordering.png", dpi=300)
        plt.close(fig)

    if "polymorph_family" in test:
        family = test.assign(abs_error_mev=(test.energy_pred_ev_atom - test.energy_dft_ev_atom).abs() * 1000).groupby("polymorph_family").agg(mean_mae_mev_atom=("abs_error_mev", "mean"), n=("structure_id", "size"))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(family.index.astype(str), family.mean_mae_mev_atom, color=["#3569a8", "#e07a25", "#34845a"][:len(family)])
        ax.set(xlabel="Source polymorph family", ylabel="Mean absolute error (meV/atom)", title="MLP error by verified family label")
        for pos, (_, row) in enumerate(family.iterrows()):
            ax.text(pos, row.mean_mae_mev_atom, f"N={int(row.n)}", ha="center", va="bottom", fontsize=8)
        fig.text(.5, .01, "Test partition only. Bars are unweighted means over structures in each source-labeled family.", ha="center", **note_style)
        fig.tight_layout(rect=(0, .06, 1, 1))
        fig.savefig(figure_dir / "07_error_by_family.png", dpi=300)
        plt.close(fig)


def write_e1_report(report_dir: Path, predictions: pd.DataFrame, summary: dict[str, Any],
                    polymorph_summary: dict[str, Any], mlp_payload: dict[str, Any],
                    linear_payload: dict[str, Any], embedding_dim: int, tie_tolerance: float) -> None:
    """Write the final report from saved metrics and checkpoints.

    Args:
        report_dir: Completed evaluation output directory.
        predictions: Saved structure-level predictions.
        summary: Saved aggregate evaluation metrics.
        polymorph_summary: Saved MLP polymorph metrics.
        mlp_payload: MLP checkpoint metadata.
        linear_payload: Linear checkpoint metadata.
        embedding_dim: MatGPT embedding width.
        tie_tolerance: Tie tolerance in eV/atom.

    Returns:
        None.
    """
    energy_metrics = summary["energy_metrics_by_split"]
    test = predictions.loc[predictions["split"].astype(str).eq("test")]
    report_text = "\n".join([
        "# MatGPT E1: backbone congelado + regresión energética", "",
        "## Estado", "", "`PARTIAL`: entrenamiento y evaluación se completaron, pero la procedencia local no especifica el funcional DFT ni sus correcciones; verificarlo antes de interpretar energías absolutas.", "",
        "## Datos", "", f"- Estructuras aceptadas: {len(predictions)}; test: {len(test)} estructuras en {test.composition.nunique()} composiciones.",
        "- Target: energía total DFT estática dividida entre número de átomos (eV/átomo); no es energía de formación.",
        "- Etiqueta de origen: `gga-static`; procedencia exacta de XC/correcciones no presente en archivos locales.",
        "- Split histórico por composición, heredado del dataset; no se dividió al azar por fila.", "",
        "## Modelo y evaluación", "", f"- MatGPT unconditional congelado; pooling en token `<end>` posterior a `ln_f`, dimensión {embedding_dim}.",
        f"- MLP seleccionada en época {mlp_payload['best_epoch']} mediante MAE de validation.",
        f"- Test MLP: MAE {energy_metrics['test']['mlp']['mae_ev_per_atom']:.6f} eV/átomo; RMSE {energy_metrics['test']['mlp']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Test baseline media: MAE {energy_metrics['test']['train_mean']['mae_ev_per_atom']:.6f}; RMSE {energy_metrics['test']['train_mean']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Test linear probe: MAE {energy_metrics['test']['linear_probe']['mae_ev_per_atom']:.6f}; RMSE {energy_metrics['test']['linear_probe']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Ground-state accuracy: {polymorph_summary['ground_state_correct_compositions']}/{polymorph_summary['ground_state_evaluable_compositions']} composiciones evaluables; {polymorph_summary['ground_state_tied_prediction_compositions_excluded']} con empate predicho quedan sin selección evaluable.",
        f"- Pairwise ordering accuracy: {polymorph_summary['pairwise_correct']}/{polymorph_summary['pairwise_evaluable_pairs']} pares; empates DFT o predichos de hasta {tie_tolerance*1000:g} meV/átomo excluidos.", "",
        "## Límites", "", "- La selección de estado base compara únicamente las estructuras incluidas de la misma composición; no mide estabilidad frente al convex hull ni estabilidad experimental.",
        "- No se verificó solapamiento entre las estructuras de evaluación y el corpus de preentrenamiento Alex-20.",
        "- Pooling causal en un único token; resultados sujetos al tamaño y composición del split.",
        "- El estado permanece PARTIAL hasta verificar la procedencia DFT y sus convenciones energéticas.", "",
        "## Artefactos", "", "- `dataset_audit.json`, `split_manifest.json`, `embedding_audit.json`, `training_history.csv`, `metrics_by_split.csv`.",
        "- `predictions.csv`, `polymorph_metrics.csv`, `pairwise_comparisons.csv`, `best_mlp.pt`, `best_linear.pt`.",
        "- Figuras científicas PNG en `figures/`; no se exportan PDF ni SVG.", "",
    ])
    (report_dir / "E1_report.md").write_text(report_text, encoding="utf-8")


def finalize_existing_evaluation(dataset_path: Path, embeddings_path: Path, model_dir: Path,
                                 report_dir: Path) -> dict[str, Any]:
    """Complete figures and report from saved metrics without evaluating test again.

    Args:
        dataset_path: Prepared dataset used in the prior evaluation.
        embeddings_path: Original embedding cache used in the prior evaluation.
        model_dir: Directory containing the selected head checkpoints and history.
        report_dir: Directory with saved predictions and partial metrics.

    Returns:
        The existing metrics summary, unchanged by this finalization.
    """
    import json

    required = [report_dir / name for name in ("metrics.json", "predictions.csv", "metrics_by_split.csv", "polymorph_metrics.csv", "pairwise_comparisons.csv")]
    required.extend([dataset_path, embeddings_path, model_dir / "best_mlp.pt", model_dir / "best_linear.pt", model_dir / "training_history.csv"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise E1Error(f"No se pueden recuperar resultados parciales; faltan archivos: {missing}")
    if (report_dir / "E1_report.md").exists():
        raise E1Error(f"Ya existe un E1_report.md en {report_dir}; no se sobrescribirá.")
    existing_figures = list((report_dir / "figures").glob("*.png")) if (report_dir / "figures").is_dir() else []
    if existing_figures:
        raise E1Error("Ya hay PNGs en la carpeta parcial; usa un directorio nuevo para preservar los resultados.")
    summary = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary.get("test_evaluated_once") is not True:
        raise E1Error("El archivo metrics.json no demuestra una evaluación previa de test.")
    hashes = verify_backbone_files()
    if summary.get("checkpoint_sha256") != hashes["checkpoint_sha256"] or summary.get("dataset_sha256") != sha256_file(dataset_path) or summary.get("embedding_cache_sha256") != sha256_file(embeddings_path):
        raise E1Error("Los hashes de los resultados parciales no coinciden con dataset, cache o backbone.")
    predictions = pd.read_csv(report_dir / "predictions.csv", low_memory=False)
    saved_test_count = int(predictions["split"].astype(str).eq("test").sum())
    if saved_test_count != int(summary.get("test_structures", -1)):
        raise E1Error("El conteo de predicciones test no coincide con metrics.json.")
    mlp_payload = torch.load(model_dir / "best_mlp.pt", map_location="cpu", weights_only=True)
    linear_payload = torch.load(model_dir / "best_linear.pt", map_location="cpu", weights_only=True)
    if mlp_payload.get("backbone_checkpoint_sha256") != hashes["checkpoint_sha256"] or linear_payload.get("backbone_checkpoint_sha256") != hashes["checkpoint_sha256"]:
        raise E1Error("Los checkpoints de cabeza no coinciden con el backbone validado.")
    embedding_audit = json.loads(embeddings_path.with_name("embedding_audit.json").read_text(encoding="utf-8"))
    history = pd.read_csv(model_dir / "training_history.csv")
    polymorph = pd.read_csv(report_dir / "polymorph_metrics.csv", low_memory=False)
    pairs = pd.read_csv(report_dir / "pairwise_comparisons.csv", low_memory=False)
    tolerance = float(yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))["evaluation"]["near_tie_tolerance_ev_per_atom"])
    save_figures(predictions, history, polymorph, pairs, summary["energy_metrics_by_split"], report_dir / "figures")
    write_e1_report(report_dir, predictions, summary, summary["test_polymorph_metrics_by_method"]["mlp"],
                    mlp_payload, linear_payload, int(embedding_audit["dimension"]), tolerance)
    LOGGER.info("partial_evaluation_finalized_without_test_rerun", extra={"report_dir": str(report_dir), "test_structures": summary["test_structures"]})
    return summary


def run(dataset_path: Path, embeddings_path: Path, model_dir: Path, report_dir: Path, device_name: str) -> dict[str, Any]:
    """Evaluate train/validation/test once and create reports and PNG figures.

    Args:
        dataset_path: Prepared dataset CSV.
        embeddings_path: Frozen MatGPT NPZ cache.
        model_dir: Directory containing the train/validation-selected heads and training history.
        report_dir: New evaluation report destination.
        device_name: cpu, cuda, mps, or auto.

    Returns:
        Energy and polymorph metrics.
    """
    if (report_dir / "predictions.csv").exists() or (report_dir / "metrics.json").exists():
        raise E1Error(f"La evaluación ya existe en {report_dir}; elige un nuevo --report-dir para conservar resultados.")
    report_dir.mkdir(parents=True, exist_ok=True)
    hashes = verify_backbone_files()
    cache_path = embeddings_path
    metadata_path = embeddings_path.with_name("embedding_metadata.csv")
    audit_path = embeddings_path.with_name("embedding_audit.json")
    for path in (dataset_path, cache_path, metadata_path, audit_path, model_dir / "best_mlp.pt", model_dir / "best_linear.pt"):
        if not path.is_file():
            raise E1Error(f"Falta entrada necesaria para evaluación: {path}")
    audit = __import__("json").loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("checkpoint_sha256") != hashes["checkpoint_sha256"] or audit.get("dataset_sha256") != sha256_file(dataset_path):
        raise E1Error("Los hashes del embedding cache no coinciden con checkpoint/dataset.")
    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings = cache["embeddings"].astype(np.float32, copy=True)
    data = pd.read_csv(dataset_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if data.structure_id.astype(str).tolist() != metadata.structure_id.astype(str).tolist() or len(data) != len(embeddings):
        raise E1Error("IDs y embeddings no tienen correspondencia exacta.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise E1Error("CUDA no está disponible.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise E1Error("MPS no está disponible.")
    mlp, mlp_payload = load_predictor(model_dir / "best_mlp.pt", "mlp", device)
    linear, linear_payload = load_predictor(model_dir / "best_linear.pt", "linear", device)
    if mlp_payload["backbone_checkpoint_sha256"] != hashes["checkpoint_sha256"] or linear_payload["backbone_checkpoint_sha256"] != hashes["checkpoint_sha256"]:
        raise E1Error("Los checkpoints de cabeza no corresponden al backbone actual.")
    x = torch.from_numpy(embeddings).to(device)
    with torch.inference_mode():
        mlp_pred_norm = mlp(x).cpu().numpy()
        linear_pred_norm = linear(x).cpu().numpy()
    y_mean = float(mlp_payload["target_mean_train_ev_per_atom"])
    y_std = float(mlp_payload["target_std_train_ev_per_atom"])
    mlp_prediction = mlp_pred_norm * y_std + y_mean
    linear_prediction = linear_pred_norm * float(linear_payload["target_std_train_ev_per_atom"]) + float(linear_payload["target_mean_train_ev_per_atom"])
    mean_prediction = np.full(len(data), y_mean)
    predictions = data[["structure_id", "reduced_formula", "polymorph_family", "split", "energy_total_ev", "num_atoms", "energy_per_atom_ev"]].copy()
    predictions = predictions.rename(columns={"reduced_formula": "composition", "energy_per_atom_ev": "energy_dft_ev_atom"})
    predictions["energy_pred_ev_atom"] = mlp_prediction
    predictions["energy_pred_linear_ev_atom"] = linear_prediction
    predictions["energy_pred_train_mean_ev_atom"] = mean_prediction
    predictions["absolute_error_ev_atom"] = (predictions.energy_pred_ev_atom - predictions.energy_dft_ev_atom).abs()
    if not np.isfinite(predictions[["energy_pred_ev_atom", "energy_pred_linear_ev_atom", "energy_pred_train_mean_ev_atom"]].to_numpy()).all():
        raise E1Error("Las predicciones contienen NaN o Inf.")
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    energy_metrics: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for split, group in predictions.groupby("split", sort=False):
        y_true = group.energy_dft_ev_atom.to_numpy(float)
        for method, column in (("train_mean", "energy_pred_train_mean_ev_atom"), ("linear_probe", "energy_pred_linear_ev_atom"), ("mlp", "energy_pred_ev_atom")):
            values = metrics(y_true, group[column].to_numpy(float))
            energy_metrics.setdefault(str(split), {})[method] = values
            metric_rows.append({"split": split, "method": method, **values, "n_structures": len(group)})
    pd.DataFrame(metric_rows).to_csv(report_dir / "metrics_by_split.csv", index=False)
    test = predictions[predictions.split == "test"].copy()
    tie_tolerance = float(yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())["evaluation"]["near_tie_tolerance_ev_per_atom"])
    polymorph_summary, formula_metrics, pair_rows = composition_metrics(test, tie_tolerance)
    pd.DataFrame(formula_metrics).to_csv(report_dir / "polymorph_metrics.csv", index=False)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(report_dir / "pairwise_comparisons.csv", index=False)
    polymorph_by_method: dict[str, Any] = {"mlp": polymorph_summary}
    for method, column in (("linear_probe", "energy_pred_linear_ev_atom"), ("train_mean", "energy_pred_train_mean_ev_atom")):
        alternate = test.copy()
        alternate["energy_pred_ev_atom"] = alternate[column]
        polymorph_by_method[method] = composition_metrics(alternate, tie_tolerance)[0]
    parity_x = test["energy_dft_ev_atom"].to_numpy(dtype=float)
    parity_y = test["energy_pred_ev_atom"].to_numpy(dtype=float)
    if len(parity_x) < 2 or np.ptp(parity_x) <= 0.0:
        raise E1Error("No se puede ajustar la paridad OLS: test tiene menos de dos valores DFT distintos.")
    ols_slope, ols_intercept = np.polyfit(parity_x, parity_y, 1)
    ols_fitted = ols_slope * parity_x + ols_intercept
    ols_denominator = float(np.sum((parity_y - parity_y.mean()) ** 2))
    ols_r2 = None if ols_denominator <= 0.0 else float(1.0 - np.sum((parity_y - ols_fitted) ** 2) / ols_denominator)
    ols_stats = {
        "n": int(len(parity_x)), "slope_dimensionless": float(ols_slope),
        "intercept_ev_per_atom": float(ols_intercept), "r2": ols_r2,
        "mae_ev_per_atom": float(metrics(parity_x, parity_y)["mae_ev_per_atom"]),
        "x_definition": "DFT total energy per atom (eV/atom)",
        "y_definition": "MLP predicted total energy per atom (eV/atom)",
    }
    summary = {
        "status": "PARTIAL_METHOD_PROVENANCE_UNVERIFIED", "test_evaluated_once": True,
        "energy_metrics_by_split": energy_metrics, "test_polymorph_metrics_by_method": polymorph_by_method,
        "test_parity_ols": ols_stats,
        "target": "static DFT total energy per atom", "units": "eV/atom",
        "functional_source_label": "gga-static; exact functional/corrections are not specified in local provenance",
        "best_mlp_epoch": int(mlp_payload["best_epoch"]), "best_linear_epoch": int(linear_payload["best_epoch"]),
        "test_compositions": sorted(test.composition.unique().tolist()), "test_structures": int(len(test)),
        "checkpoint_sha256": hashes["checkpoint_sha256"], "dataset_sha256": sha256_file(dataset_path),
        "embedding_cache_sha256": sha256_file(cache_path), "device": str(device),
        "interpretation_limits": ["Do not call total energy per atom formation energy.",
                                  "Exact exchange-correlation functional and corrections remain unverified from local metadata.",
                                  "Ground-state selection is among the supplied structures per composition, not a convex-hull or experimental stability claim.",
                                  "Possible overlap with MatGPT pretraining corpus Alex-20 has not been checked."],
    }
    history_path = model_dir / "training_history.csv"
    if not history_path.is_file():
        raise E1Error(f"Falta historial de entrenamiento para construir las curvas: {history_path}")
    history = pd.read_csv(history_path)
    save_figures(predictions, history, pd.DataFrame(formula_metrics), pair_frame, energy_metrics, report_dir / "figures")
    write_json(report_dir / "metrics.json", summary)
    report_text = "\n".join([
        "# MatGPT E1: backbone congelado + regresión energética", "",
        "## Estado", "", "`PARTIAL`: entrenamiento y evaluación se completaron, pero la procedencia local no especifica el funcional DFT ni sus correcciones; verificarlo antes de interpretar energías absolutas.", "",
        "## Datos", "", f"- Estructuras aceptadas: {len(predictions)}; test: {len(test)} estructuras en {test.composition.nunique()} composiciones.",
        "- Target: energía total DFT estática dividida entre número de átomos (eV/átomo); no es energía de formación.",
        "- Etiqueta de origen: `gga-static`; procedencia exacta de XC/correcciones no presente en archivos locales.",
        "- Split histórico por composición, heredado del dataset; no se dividió al azar por fila.", "",
        "## Modelo y evaluación", "", f"- MatGPT unconditional congelado; pooling en token `<end>` posterior a `ln_f`, dimensión {embeddings.shape[1]}.",
        f"- MLP seleccionada en época {mlp_payload['best_epoch']} mediante MAE de validation.",
        f"- Test MLP: MAE {energy_metrics['test']['mlp']['mae_ev_per_atom']:.6f} eV/átomo; RMSE {energy_metrics['test']['mlp']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Test baseline media: MAE {energy_metrics['test']['train_mean']['mae_ev_per_atom']:.6f}; RMSE {energy_metrics['test']['train_mean']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Test linear probe: MAE {energy_metrics['test']['linear_probe']['mae_ev_per_atom']:.6f}; RMSE {energy_metrics['test']['linear_probe']['rmse_ev_per_atom']:.6f} eV/átomo.",
        f"- Ground-state accuracy: {polymorph_summary['ground_state_correct_compositions']}/{polymorph_summary['ground_state_evaluable_compositions']} composiciones evaluables; {polymorph_summary['ground_state_tied_prediction_compositions_excluded']} con empate predicho quedan sin selección evaluable.",
        f"- Pairwise ordering accuracy: {polymorph_summary['pairwise_correct']}/{polymorph_summary['pairwise_evaluable_pairs']} pares; empates DFT de hasta {tie_tolerance*1000:g} meV/átomo excluidos.", "",
        "## Límites", "", "- La selección de estado base compara únicamente las estructuras incluidas de la misma composición; no mide estabilidad frente al convex hull ni estabilidad experimental.",
        "- No se verificó solapamiento entre las estructuras de evaluación y el corpus de preentrenamiento Alex-20.",
        "- Pooling causal en un único token; resultados sujetos al tamaño y composición del split.",
        "- El estado permanece PARTIAL hasta verificar la procedencia DFT y sus convenciones energéticas.", "",
        "## Artefactos", "", "- `dataset_audit.json`, `split_manifest.json`, `embedding_audit.json`, `training_history.csv`, `metrics_by_split.csv`.",
        "- `predictions.csv`, `polymorph_metrics.csv`, `pairwise_comparisons.csv`, `best_mlp.pt`, `best_linear.pt`.",
        "- Figuras científicas PNG en `figures/`; no se exportan PDF ni SVG.", "",
    ])
    (report_dir / "E1_report.md").write_text(report_text, encoding="utf-8")
    LOGGER.info("evaluation_complete", extra={"status": summary["status"], "test_structures": len(test), "ground_state_accuracy": polymorph_summary["ground_state_accuracy"]})
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for final evaluation.

    Returns:
        CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--embeddings", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "matgpt_uncond_last_token.npz")
    parser.add_argument("--model-dir", type=Path, default=REPORTS)
    parser.add_argument("--report-dir", type=Path, default=REPORTS.with_name("matgpt_e1_v2_eval_v3"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--finalize-existing", action="store_true", help="Genera figuras e informe desde outputs parciales sin volver a evaluar test.")
    return parser.parse_args()


def main() -> None:
    """Run final evaluation and emit structured errors."""
    logger = configure_logging()
    args = parse_args()
    try:
        if args.finalize_existing:
            finalize_existing_evaluation(args.dataset, args.embeddings, args.model_dir, args.report_dir)
        else:
            run(args.dataset, args.embeddings, args.model_dir, args.report_dir, args.device)
    except (E1Error, OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("evaluation_failed", extra={"error": str(exc)})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
