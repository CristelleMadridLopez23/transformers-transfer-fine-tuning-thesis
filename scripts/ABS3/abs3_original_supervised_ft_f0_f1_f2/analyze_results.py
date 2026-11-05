"""Resume OOF, calcula métricas científicas y produce figuras F0/F1/F2.

El remuestreo es apareado por composición y conserva semillas/p pares dentro de
cada fórmula. El análisis no lee ni abre el test histórico independiente.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/abs3_original_supervised_ft_f0_f1_f2"
IMAGE_DIR = ROOT / "images/abs3_original_supervised_ft_f0_f1_f2"
REPORT_DIR = ROOT / "reports/ABS3/abs3_original_supervised_ft_f0_f1_f2"
LOGGER = logging.getLogger("abs3_original_supervised_ft_f0_f1_f2.analysis")
MODEL_ORDER = ("MatGPT", "MatterGPT")
MODALITY_ORDER = ("SLICES", "SLICES_GEOMETRY")
POLICY_ORDER = ("F0", "F1", "F2")
POLYMORPH_LABELS = {"dist_perovskite": "Perovskita", "hexagonal": "Hexagonal", "needle": "Tipo aguja"}
BOOTSTRAP_SEED = 24062026
BOOTSTRAP_N = 10000


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 en bloques.

    Args:
        path: Archivo.
    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Escribe JSON mediante reemplazo atómico.

    Args:
        path: Archivo destino.
        data: Contenido serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_helper() -> Any:
    """Carga validadores y extractores históricos sin ejecutar su `main`."""
    path = ROOT / "scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py"
    spec = importlib.util.spec_from_file_location("abs3_f4_analysis_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga CSV OOF por corrida y valida claves, cohortes y finitud.

    Returns:
        Tablas de estructuras y pares.
    """
    structure_paths = sorted((EXP / "predictions/structures").glob("*.csv"))
    pair_paths = sorted((EXP / "predictions/pairs").glob("*.csv"))
    if len(structure_paths) != 180 or len(pair_paths) != 180:
        raise ValueError(f"Se esperan 180 corridas OOF; halladas estructuras={len(structure_paths)} pares={len(pair_paths)}")
    structures = pd.concat([pd.read_csv(path) for path in structure_paths], ignore_index=True)
    pairs = pd.concat([pd.read_csv(path) for path in pair_paths], ignore_index=True)
    s_key = ["condition", "fold", "seed", "formula", "structure_id"]
    p_key = ["condition", "fold", "seed", "formula", "structure_id_i", "structure_id_j"]
    if structures.duplicated(s_key).any() or pairs.duplicated(p_key).any():
        raise ValueError("Predicciones OOF duplicadas")
    if len(structures) != 7560 or len(pairs) != 7560:
        raise ValueError(f"Cohorte OOF incorrecta: {len(structures)} estructuras, {len(pairs)} pares")
    numeric_s = ["E_DFT_eV_atom", "mu_M2_eV_atom", "raw_mlp_score", "centered_residual_eV_atom", "E_pred_eV_atom"]
    numeric_p = ["DeltaE_DFT_meV_atom", "DeltaE_pred_meV_atom", "DeltaE_abs_error_meV_atom"]
    if not np.isfinite(structures[numeric_s].to_numpy(float)).all() or not np.isfinite(pairs[numeric_p].to_numpy(float)).all():
        raise ValueError("Predicciones OOF contienen NaN/Inf")
    counts = structures.groupby(["condition", "fold", "seed", "formula"]).size()
    pair_counts = pairs.groupby(["condition", "fold", "seed", "formula"]).size()
    if counts.ne(3).any() or pair_counts.ne(3).any() or set(counts.index) != set(pair_counts.index):
        raise ValueError("Tripletes/p pares OOF incompletos")
    return structures, pairs


def per_run_metrics(structures: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Calcula Energy, ΔE, ranking y ground state por composición/semilla.

    Args:
        structures: Predicción por estructura.
        pairs: Diferencias firmadas.
    Returns:
        Tabla con unidad fórmula-semilla.
    """
    pair_groups = {(str(c), int(f), int(s), str(formula)): group
                   for (c, f, s, formula), group in pairs.groupby(["condition", "fold", "seed", "formula"])}
    records: list[dict[str, Any]] = []
    for (condition, fold, seed, formula), group in structures.groupby(["condition", "fold", "seed", "formula"], sort=True):
        group = group.sort_values("polymorph", key=lambda col: col.map({"dist_perovskite": 0, "hexagonal": 1, "needle": 2}))
        pg = pair_groups[(str(condition), int(fold), int(seed), str(formula))]
        truth = group.E_DFT_eV_atom.to_numpy(float) * 1000
        pred = group.E_pred_eV_atom.to_numpy(float) * 1000
        energy_error = pred - truth
        order_truth = np.argsort(truth, kind="stable")
        order_pred = np.argsort(pred, kind="stable")
        ground_eligible = bool(truth[order_truth[1]] - truth[order_truth[0]] > 1)
        ground_resolved = bool(pred[order_pred[1]] - pred[order_pred[0]] > 1)
        ground_correct = bool(ground_eligible and ground_resolved and order_truth[0] == order_pred[0])
        dft_diffs = [abs(truth[j] - truth[i]) for i, j in ((0, 1), (0, 2), (1, 2))]
        full_eligible = bool(all(value > 1 for value in dft_diffs))
        pred_diffs = [abs(pred[j] - pred[i]) for i, j in ((0, 1), (0, 2), (1, 2))]
        full_resolved = bool(all(value > 1 for value in pred_diffs))
        full_correct = bool(full_eligible and full_resolved and np.array_equal(order_truth, order_pred))
        eligible = pg.pairwise_eligible.astype(bool).to_numpy()
        correct = pg.pairwise_correct.astype(bool).to_numpy()
        unresolved = pg.unresolved.astype(bool).to_numpy()
        rho = spearmanr(truth, pred).statistic
        tau = kendalltau(truth, pred).statistic
        residual_true = (group.E_DFT_eV_atom.to_numpy(float) - group.mu_M2_eV_atom.to_numpy(float)) * 1000
        residual_pred = group.centered_residual_eV_atom.to_numpy(float) * 1000
        records.append({"condition": condition, "model": group.model.iloc[0], "modality": group.modality.iloc[0],
            "freeze_policy": group.freeze_policy.iloc[0], "fold": int(fold), "seed": int(seed), "formula": formula,
            "n_structures": 3, "energy_mae_mev_atom": float(np.mean(np.abs(energy_error))),
            "energy_rmse_global_mev_atom": float(np.sqrt(np.mean(energy_error ** 2))),
            "energy_bias_mev_atom": float(np.mean(energy_error)),
            "residual_mae_mev_atom": float(np.mean(np.abs(residual_pred - residual_true))),
            "delta_mae_mev_atom": float(pg.DeltaE_abs_error_mev_atom.mean()),
            "delta_rmse_mev_atom": float(np.sqrt(np.mean((pg.DeltaE_pred_mev_atom - pg.DeltaE_DFT_mev_atom) ** 2))),
            "eligible_pairs": int(eligible.sum()), "pair_correct": int(correct[eligible].sum()),
            "unresolved_pairs": int(unresolved[eligible].sum()),
            "ground_eligible": int(ground_eligible), "ground_correct": int(ground_correct),
            "full_triplet_eligible": int(full_eligible), "full_triplet_correct": int(full_correct),
            "spearman": float(rho) if np.isfinite(rho) else np.nan,
            "kendall": float(tau) if np.isfinite(tau) else np.nan,
        })
    return pd.DataFrame(records)


def aggregate_metrics(per_run: pd.DataFrame, pairs: pd.DataFrame, structures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Agrega semillas dentro de fórmula y calcula métricas por condición.

    Args:
        per_run: Métricas fórmula-semilla.
        pairs: Predicciones OOF por pares.
        structures: Predicciones OOF por estructura.
    Returns:
        (por fórmula, por fold, aggregate por condición).
    """
    keys = ["condition", "model", "modality", "freeze_policy", "formula"]
    metric_cols = ["energy_mae_mev_atom", "energy_bias_mev_atom", "residual_mae_mev_atom",
                   "delta_mae_mev_atom", "delta_rmse_mev_atom", "spearman", "kendall"]
    comp = per_run.groupby(keys, as_index=False).agg({**{c: "mean" for c in metric_cols},
        "eligible_pairs": "sum", "pair_correct": "sum", "unresolved_pairs": "sum",
        "ground_eligible": "sum", "ground_correct": "sum",
        "full_triplet_eligible": "sum", "full_triplet_correct": "sum"})
    fold = per_run.groupby(["condition", "model", "modality", "freeze_policy", "fold"], as_index=False).agg(
        energy_mae_mev_atom=("energy_mae_mev_atom", "mean"), delta_mae_mev_atom=("delta_mae_mev_atom", "mean"),
        eligible_pairs=("eligible_pairs", "sum"), pair_correct=("pair_correct", "sum"),
        ground_eligible=("ground_eligible", "sum"), ground_correct=("ground_correct", "sum"),
        full_triplet_eligible=("full_triplet_eligible", "sum"), full_triplet_correct=("full_triplet_correct", "sum"))
    rows: list[dict[str, Any]] = []
    for condition, group in per_run.groupby("condition", sort=True):
        all_pairs = pairs[pairs.condition.eq(condition)]
        all_structures = structures[structures.condition.eq(condition)]
        energy_errors = (all_structures.E_pred_eV_atom.to_numpy(float) -
                         all_structures.E_DFT_eV_atom.to_numpy(float)) * 1000.0
        x = all_pairs.DeltaE_DFT_meV_atom.to_numpy(float)
        y = all_pairs.DeltaE_pred_meV_atom.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
        elig = all_pairs.pairwise_eligible.astype(bool)
        denom = int(elig.sum())
        ground_den = int(group.ground_eligible.sum())
        full_den = int(group.full_triplet_eligible.sum())
        rows.append({"condition": condition, "model": group.model.iloc[0], "modality": group.modality.iloc[0],
            "freeze_policy": group.freeze_policy.iloc[0], "n_compositions": int(group.formula.nunique()),
            "n_structures_oof": int(len(group) * 3), "n_formula_seed_cases": int(len(group)),
            "energy_mae_mev_atom_estimand_A": float(group.groupby("formula").energy_mae_mev_atom.mean().mean()),
            "energy_rmse_global_mev_atom": float(np.sqrt(np.mean(energy_errors ** 2))),
            "energy_bias_mev_atom": float(group.groupby("formula").energy_bias_mev_atom.mean().mean()),
            "residual_mae_mev_atom": float(group.groupby("formula").residual_mae_mev_atom.mean().mean()),
            "delta_mae_mev_atom_estimand_A": float(group.groupby("formula").delta_mae_mev_atom.mean().mean()),
            "delta_rmse_global_mev_atom": float(np.sqrt(np.mean((y - x) ** 2))),
            "pairwise_accuracy_eligible": float(all_pairs.loc[elig, "pairwise_correct"].astype(bool).mean()),
            "pairwise_denominator": denom,
            "unresolved_fraction_eligible": float(all_pairs.loc[elig, "unresolved"].astype(bool).mean()),
            "ground_state_accuracy": float(group.ground_correct.sum() / ground_den) if ground_den else np.nan,
            "ground_state_denominator": ground_den,
            "full_triplet_accuracy": float(group.full_triplet_correct.sum() / full_den) if full_den else np.nan,
            "full_triplet_denominator": full_den,
            "spearman_mean_formula": float(group.groupby("formula").spearman.mean().mean()),
            "kendall_mean_formula": float(group.groupby("formula").kendall.mean().mean()),
            "delta_ols_slope": float(slope), "delta_ols_intercept_mev_atom": float(intercept),
            "delta_ols_r2": float(r2), "delta_predictive_r2": float(1 - np.sum((y-x)**2)/np.sum((x-x.mean())**2)),
        })
    aggregate = pd.DataFrame(rows)
    return comp, fold, aggregate


def bootstrap_contrasts(comp: pd.DataFrame, per_run: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap por fórmula para mejoras apareadas, con seeds agrupadas.

    Args:
        comp: Métricas agregadas por fórmula.
        per_run: Métricas por fórmula/semilla para denominadores de exactitud.
        pairs: Predicciones por pares.
    Returns:
        Tabla de IC95% percentiles sin ajuste multiplicidad.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, Any]] = []
    metrics = ("energy_mae_mev_atom", "delta_mae_mev_atom")
    for model in MODEL_ORDER:
        for modality in MODALITY_ORDER:
            for contrast in (("F1", "F0"), ("F2", "F0"), ("F2", "F1")):
                left_policy, right_policy = contrast
                left = comp[(comp.model == model) & (comp.modality == modality) & (comp.freeze_policy == left_policy)].set_index("formula")
                right = comp[(comp.model == model) & (comp.modality == modality) & (comp.freeze_policy == right_policy)].set_index("formula")
                formulas = sorted(set(left.index) & set(right.index))
                if len(formulas) != 70:
                    raise ValueError(f"Bootstrap requiere 70 fórmulas, recibió {len(formulas)}")
                for metric in metrics:
                    a, b = left.loc[formulas, metric].to_numpy(float), right.loc[formulas, metric].to_numpy(float)
                    draw = rng.integers(0, len(formulas), size=(BOOTSTRAP_N, len(formulas)))
                    deltas = (a[draw] - b[draw]).mean(axis=1)
                    records.append({"contrast": f"{model}_{modality}_{left_policy}-{right_policy}",
                        "metric": metric, "estimate_left_minus_right": float((a-b).mean()),
                        "ci95_low": float(np.quantile(deltas, .025)), "ci95_high": float(np.quantile(deltas, .975)),
                        "bootstrap_replicates": BOOTSTRAP_N, "unit": "composition; seeds averaged within composition",
                        "interval_adjustment": "none; percentile paired bootstrap"})
                # Para exactitudes, remuestrear fórmulas completas y recalcular
                # cada razón con sus denominadores elegibles en la réplica.
                rate_specs = (
                    ("pairwise_accuracy_eligible", "pair_correct", "eligible_pairs"),
                    ("ground_state_accuracy", "ground_correct", "ground_eligible"),
                    ("full_triplet_accuracy", "full_triplet_correct", "full_triplet_eligible"),
                )
                for rate_name, numerator, denominator in rate_specs:
                    left_counts = comp[(comp.model == model) & (comp.modality == modality) &
                                       (comp.freeze_policy == left_policy)].set_index("formula").loc[formulas]
                    right_counts = comp[(comp.model == model) & (comp.modality == modality) &
                                        (comp.freeze_policy == right_policy)].set_index("formula").loc[formulas]
                    draws = rng.integers(0, len(formulas), size=(BOOTSTRAP_N, len(formulas)))
                    a_num = left_counts[numerator].to_numpy(float)[draws].sum(axis=1)
                    a_den = left_counts[denominator].to_numpy(float)[draws].sum(axis=1)
                    b_num = right_counts[numerator].to_numpy(float)[draws].sum(axis=1)
                    b_den = right_counts[denominator].to_numpy(float)[draws].sum(axis=1)
                    valid = (a_den > 0) & (b_den > 0)
                    delta_rate = a_num[valid] / a_den[valid] - b_num[valid] / b_den[valid]
                    estimate = float(left_counts[numerator].sum() / left_counts[denominator].sum() -
                                     right_counts[numerator].sum() / right_counts[denominator].sum())
                    records.append({"contrast": f"{model}_{modality}_{left_policy}-{right_policy}",
                        "metric": rate_name, "estimate_left_minus_right": estimate,
                        "ci95_low": float(np.quantile(delta_rate, .025)),
                        "ci95_high": float(np.quantile(delta_rate, .975)),
                        "bootstrap_replicates": BOOTSTRAP_N,
                        "unit": "composiciones; razones recalculadas con denominadores por réplica; semillas agrupadas",
                        "interval_adjustment": "none; percentile paired bootstrap"})
    # Geometry × fine-tuning difference-in-differences for energy and ΔE MAE.
    for model in MODEL_ORDER:
        for policy in ("F1", "F2"):
            for metric in metrics:
                table = comp[(comp.model.eq(model)) & (comp.freeze_policy.eq(policy))].pivot(
                    index="formula", columns="modality", values=metric)
                f0 = comp[(comp.model.eq(model)) & (comp.freeze_policy.eq("F0"))].pivot(
                    index="formula", columns="modality", values=metric)
                delta = (table["SLICES_GEOMETRY"] - f0["SLICES_GEOMETRY"]) - (table["SLICES"] - f0["SLICES"])
                vals = delta.to_numpy(float)
                draws = rng.integers(0, len(vals), size=(BOOTSTRAP_N, len(vals)))
                samples = vals[draws].mean(axis=1)
                records.append({"contrast": f"{model}_{policy}_geometry_interaction",
                    "metric": metric, "estimate_left_minus_right": float(vals.mean()),
                    "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)),
                    "bootstrap_replicates": BOOTSTRAP_N, "unit": "composition; seeds averaged within composition",
                    "interval_adjustment": "none; percentile paired bootstrap"})
    return pd.DataFrame(records)


def make_figures(structures: pd.DataFrame, pairs: pd.DataFrame, run_metrics: pd.DataFrame,
                 comp: pd.DataFrame, aggregate: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    """Genera diez figuras científicas con unidades, N y definiciones en español.

    Args:
        structures: Predicciones OOF por estructura.
        pairs: Predicciones OOF por par.
        run_metrics: Métricas fórmula/semilla.
        comp: Métricas fórmula.
        aggregate: Métricas por condición.
        bootstrap: Contrastes bootstrap.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"MatGPT": "#1f77b4", "MatterGPT": "#d95f02"}
    markers = {"SLICES": "o", "SLICES_GEOMETRY": "s"}
    conds = [f"{m}_{mod}_{p}" for m in MODEL_ORDER for mod in MODALITY_ORDER for p in POLICY_ORDER]
    labels = [f"{m}\n{'SLICES' if mod == 'SLICES' else 'SLICES + geometría'}\n{p}" for m in MODEL_ORDER for mod in MODALITY_ORDER for p in POLICY_ORDER]

    # 1. Architecture illustration.
    fig, ax = plt.subplots(figsize=(12, 4.5)); ax.axis("off")
    boxes = [(0.03,.47,.16,.24,"SLICES\n[tokenizador propio]"),(0.25,.47,.18,.24,"Transformer\n12 bloques"),
        (0.50,.72,.16,.18,"F0: backbone\ncongelado"),(0.50,.47,.16,.18,"F1: bloque 11\nentrenable"),
        (0.50,.22,.16,.18,"F2: bloques 10–11\nentrenables"),(0.72,.47,.12,.24,"FINAL_END\n768"),
        (0.72,.14,.12,.18,"Geometría\n7 (opcional)"),(0.88,.47,.10,.24,"MLP8\n→ residuo")]
    for x,y,w,h,label in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#edf2f7",edgecolor="#36536b",lw=1.5)); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=10)
    for a,b in [((.19,.59),(.25,.59)),((.43,.59),(.50,.81)),((.43,.59),(.50,.56)),((.43,.59),(.50,.31)),((.66,.81),(.72,.59)),((.84,.59),(.88,.59)),((.84,.23),(.90,.47))]:
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":1.4,"color":"#333333"})
    ax.text(.93,.28,"Ridge M2 fijo\n17 fracciones → μc",ha="center",va="center",fontsize=9)
    ax.text(.50,.04,"MLP8 entrenable en F0/F1/F2 · geometría entra después del Transformer · test independiente cerrado",ha="center",fontsize=10)
    fig.suptitle("ABS₃: transferencia supervisada parcial de MatGPT y MatterGPT",fontsize=14)
    fig.savefig(IMAGE_DIR/"f0_f1_f2_arquitectura.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # 2. Learning curves with active-run count visible.
    hist_path=EXP/"training_history_full.csv"
    hist=pd.read_csv(hist_path)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,(model,modality) in zip(axes.flat,[(m,mo) for m in MODEL_ORDER for mo in MODALITY_ORDER]):
        subset=hist[(hist.model==model)&(hist.modality==modality)&(hist.execution_phase=="full")]
        for policy,color in zip(POLICY_ORDER,("#555555","#1f77b4","#d95f02")):
            q=subset[subset.condition==policy]
            if q.empty: continue
            summary=q.groupby("epoch").agg(train=("train_minibatch_smoothl1","mean"),val=("inner_validation_smoothl1","mean"),n=("seed","count"))
            ax.plot(summary.index,summary.train,color=color,ls="--",alpha=.9,label=f"{policy} train minibatches")
            ax.plot(summary.index,summary.val,color=color,ls="-",label=f"{policy} validación completa")
        ax.set_title(f"{model} · {'SLICES-only' if modality=='SLICES' else 'SLICES + geometría'}")
        ax.set_ylabel("SmoothL1 residual normalizada (adimensional)");ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    for ax in axes[-1]: ax.set_xlabel("Época local de continuación (máximo 71)")
    fig.suptitle("ABS₃: historial de entrenamiento y validación; promedios sobre corridas disponibles por época")
    fig.text(.5,.01,"Train = promedio de minilotes inner-train; validación = conjunto completo inner-val. N activo por época se conserva en training_history_full.csv. Selección por MAE ΔE signed de inner-val.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(IMAGE_DIR/"f0_f1_f2_curvas_entrenamiento.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 3/4 metrics bars.
    for metric,title,ylabel,file in [
        ("energy_mae_mev_atom_estimand_A","ABS₃: MAE de energía DFT por átomo","MAE de energía (meV/átomo)","f0_f1_f2_energy_mae.png"),
        ("delta_mae_mev_atom_estimand_A","ABS₃: MAE de diferencias energéticas firmadas","MAE ΔE firmado (meV/átomo)","f0_f1_f2_deltaE_mae.png")]:
        fig,ax=plt.subplots(figsize=(14,5));x=np.arange(len(aggregate));vals=aggregate.set_index("condition").loc[conds,metric].to_numpy(float)
        ax.bar(x,vals,color=[colors[m] for m in [m for m in MODEL_ORDER for _ in range(6)]],alpha=.85)
        ax.set_xticks(x,labels,fontsize=8);ax.set_ylabel(ylabel);ax.set_title(title+" · OOF por composición")
        ax.grid(axis="y",alpha=.25);ax.text(.01,.98,"Unidad de análisis: 70 fórmulas; 3 semillas promediadas dentro de fórmula; 210 estructuras por semilla; 5 folds agrupados.",transform=ax.transAxes,va="top",fontsize=8)
        fig.tight_layout();fig.savefig(IMAGE_DIR/file,dpi=300,bbox_inches="tight");plt.close(fig)

    # 5. Pairwise accuracy.
    fig,ax=plt.subplots(figsize=(14,5));tab=aggregate.set_index("condition").loc[conds];v=tab.pairwise_accuracy_eligible.to_numpy(float)*100
    ax.bar(np.arange(12),v,color=[colors[m] for m in [m for m in MODEL_ORDER for _ in range(6)]])
    ax.axhline(90,color="#a00000",ls="--",label="Meta aspiracional 90%")
    ax.set_xticks(np.arange(12),labels,fontsize=8);ax.set_ylim(0,100);ax.set_ylabel("Exactitud pairwise (%)");ax.set_title("ABS₃: ranking pairwise en outer OOF")
    for i,(val,n) in enumerate(zip(v,tab.pairwise_denominator)):ax.text(i,val+1,f"{val:.1f}%\nn={int(n)}",ha="center",fontsize=7)
    ax.legend();ax.grid(axis="y",alpha=.25);fig.text(.5,.01,"Par elegible si |ΔE_DFT|>5 meV/átomo; predicción no resuelta |ΔE_pred|≤1 cuenta incorrecta. Unidad bootstrap: composición.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(IMAGE_DIR/"f0_f1_f2_pairwise.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 6. Ground-state and full-triplet ranking.
    fig,ax=plt.subplots(figsize=(14,5));tab=aggregate.set_index("condition").loc[conds]
    x=np.arange(12);ax.bar(x-.18,tab.ground_state_accuracy*100,.36,label="Estado fundamental",color="#4c78a8")
    ax.bar(x+.18,tab.full_triplet_accuracy*100,.36,label="Orden completo triplete",color="#f58518")
    ax.set_xticks(x,labels,fontsize=8);ax.set_ylabel("Exactitud (%)");ax.set_ylim(0,100);ax.set_title("ABS₃: selección del ground state y orden completo")
    ax.legend();ax.grid(axis="y",alpha=.25);fig.text(.5,.01,"Ground state elegible si el gap DFT al segundo mínimo >1 meV/átomo; predicción empatada a ≤1 es incorrecta. Tripletes con gaps DFT ≤1 se excluyen.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(IMAGE_DIR/"f0_f1_f2_groundstate_triplet.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 7. DeltaE parity panels by policy.
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),sharex=True,sharey=True)
    for ax,policy in zip(axes,POLICY_ORDER):
        q=pairs[pairs.freeze_policy.eq(policy)];x=q.DeltaE_DFT_meV_atom.to_numpy(float);y=q.DeltaE_pred_meV_atom.to_numpy(float)
        ax.scatter(x,y,s=7,alpha=.22,c=[colors[m] for m in q.model],marker="o")
        lim=max(float(np.max(np.abs(x))),float(np.max(np.abs(y))))*1.05;ax.plot([-lim,lim],[-lim,lim],color="#222",ls="--",label="Ideal y=x")
        slope,inter=np.polyfit(x,y,1);ax.plot([-lim,lim],slope*np.array([-lim,lim])+inter,color="#7b3294",label="OLS todos los pares")
        r2=1-np.sum((y-(slope*x+inter))**2)/np.sum((y-y.mean())**2)
        ax.text(.03,.97,f"N={len(x)} pares\nm={slope:.3f}\nb={inter:.1f} meV/átomo\nR²={r2:.3f}\nMAE={np.mean(abs(y-x)):.1f} meV/átomo",transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax.set_title(policy);ax.set_xlabel("ΔE DFT firmado (meV/átomo)");ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel("ΔE predicho firmado (meV/átomo)");fig.suptitle("ABS₃: paridad de diferencias energéticas firmadas; puntos = pares outer OOF")
    fig.text(.5,.01,"Cada punto es un par intrafórmula; cinco folds agrupados y tres semillas. Colores: MatGPT azul, MatterGPT naranja. Datos de SLICES y SLICES+geometría combinados por panel.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(IMAGE_DIR/"f0_f1_f2_deltaE_parity.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 8. Bootstrap forest plot.
    q=bootstrap[bootstrap.metric.isin(["energy_mae_mev_atom","delta_mae_mev_atom"]) & bootstrap.contrast.str.contains("F[12]-F0")]
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    for ax,(metric,title) in zip(axes,[("energy_mae_mev_atom","Cambio en MAE de energía"),("delta_mae_mev_atom","Cambio en MAE ΔE firmado")]):
        z=q[q.metric.eq(metric)].reset_index(drop=True);y=np.arange(len(z));ax.errorbar(z.estimate_left_minus_right,y,xerr=[z.estimate_left_minus_right-z.ci95_low,z.ci95_high-z.estimate_left_minus_right],fmt="o",capsize=3,color="#36536b");ax.axvline(0,color="#777",ls="--");ax.set_yticks(y,z.contrast,fontsize=7);ax.set_xlabel("Diferencia F1/F2 − F0 (meV/átomo)");ax.set_title(title);ax.grid(axis="x",alpha=.2)
    fig.suptitle("ABS₃: bootstrap apareado por composición, IC95% percentil")
    fig.text(.5,.01,"70 fórmulas; semillas agregadas dentro de fórmula; 10 000 réplicas. Intervalos no ajustados por multiplicidad; negativo favorece fine-tuning.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(IMAGE_DIR/"f0_f1_f2_bootstrap.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 9. Geometry interaction plot.
    interactions=bootstrap[bootstrap.contrast.str.contains("geometry_interaction")]
    fig,ax=plt.subplots(figsize=(10,5));names=interactions.contrast+" · "+interactions.metric;yy=np.arange(len(interactions));ax.errorbar(interactions.estimate_left_minus_right,yy,xerr=[interactions.estimate_left_minus_right-interactions.ci95_low,interactions.ci95_high-interactions.estimate_left_minus_right],fmt="o",capsize=3);ax.axvline(0,color="#777",ls="--");ax.set_yticks(yy,names,fontsize=8);ax.set_xlabel("Interacción (geometría − SLICES) del cambio de MAE (meV/átomo)");ax.set_title("ABS₃: interacción entre geometría y fine-tuning");ax.grid(axis="x",alpha=.2);fig.text(.5,.01,"Interacción = (FT−F0) en SLICES+geometría − (FT−F0) en SLICES-only; bootstrap por fórmula, IC95% no ajustado.",ha="center",fontsize=8);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(IMAGE_DIR/"f0_f1_f2_geometria_interaccion.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 10. Block drift; embedding drift is added by optional representation audit.
    drift_path=EXP/"diagnostics/weight_drift.csv"
    drift=pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(10,5))
    if not drift.empty:
        drift.groupby(["model","freeze_policy","block"],as_index=False).relative_weight_drift.mean().plot(x="block",y="relative_weight_drift",kind="bar",ax=ax,legend=True)
    else: ax.text(.5,.5,"Deriva pendiente de la auditoría de checkpoints",ha="center",va="center",transform=ax.transAxes)
    ax.set_ylabel("Deriva relativa de pesos (adimensional)");ax.set_title("ABS₃: deriva de los bloques actualizados respecto al checkpoint original");ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(IMAGE_DIR/"f0_f1_f2_deriva_pesos.png",dpi=300,bbox_inches="tight");plt.close(fig)


def representation_drift(run_records: pd.DataFrame, structures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mide deriva de pesos y embeddings en estructuras outer retenidas.

    Args:
        run_records: Checkpoints ganadores por condición/fold/seed.
        structures: Predicciones outer OOF.
    Returns:
        Tablas con deriva por bloque y por estructura.
    """
    helper = import_helper()
    data, _ = helper.load_data()
    ids = data.structure_id.astype(str).to_numpy()
    device = torch.device("cpu")
    torch.set_num_threads(2)
    extractor = helper.import_extractor()
    weight_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        cache = helper.load_embedding_cache(model_name, ids)
        original, loaded, _ = helper.load_model_base(model_name, extractor)
        tokenizer = loaded["stoi"]
        encoded = helper.encode_sequences(data, model_name, tokenizer, loaded["context"])
        stack_orig = helper.stack_for(original, model_name)
        subset = run_records[(run_records.model == model_name) & (run_records.freeze_policy.isin(("F1", "F2")))]
        for run in subset.itertuples():
            path = ROOT / run.checkpoint
            state = torch.load(path, map_location="cpu", weights_only=True)
            saved_blocks = state["blocks"]
            for block, block_state in saved_blocks.items():
                base_state = stack_orig[int(block)].state_dict()
                num = math.sqrt(sum(float((block_state[key].float() - base_state[key].float()).pow(2).sum()) for key in block_state))
                den = math.sqrt(sum(float(base_state[key].float().pow(2).sum()) for key in block_state))
                weight_rows.append({"condition": run.condition, "model": model_name, "modality": run.modality,
                    "freeze_policy": run.freeze_policy, "fold": int(run.fold), "seed": int(run.seed),
                    "block": int(block), "relative_weight_drift": num / max(den, 1e-12),
                    "absolute_weight_drift_l2": num})
            original.eval()
            finetuned, _, _ = helper.load_model_base(model_name, extractor)
            active = helper.configure_backbone(finetuned, model_name, False)
            stack_ft = helper.stack_for(finetuned, model_name)
            for block, block_state in saved_blocks.items():
                stack_ft[int(block)].load_state_dict(block_state, strict=True)
            finetuned.to(device).eval()
            subset_structures = structures[(structures.model == model_name) & (structures.modality == run.modality)
                & (structures.freeze_policy == run.freeze_policy) & (structures.fold == int(run.fold))
                & (structures.seed == int(run.seed))].copy()
            positions = {str(sid): idx for idx, sid in enumerate(ids)}
            for formula, group in subset_structures.groupby("formula"):
                batch_ids = group.sort_values("polymorph", key=lambda col: col.map({"dist_perovskite":0,"hexagonal":1,"needle":2})).structure_id.astype(str).tolist()
                with torch.no_grad():
                    fresh = helper.live_embeddings(finetuned, model_name, batch_ids, encoded, tokenizer, device).cpu().numpy()
                old = np.stack([cache[positions[sid]] for sid in batch_ids])
                cos = np.sum(fresh * old, axis=1) / np.maximum(np.linalg.norm(fresh,axis=1)*np.linalg.norm(old,axis=1),1e-12)
                rel = np.linalg.norm(fresh-old,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
                for idx,sid in enumerate(batch_ids):
                    embedding_rows.append({"condition":run.condition,"model":model_name,"modality":run.modality,
                        "freeze_policy":run.freeze_policy,"fold":int(run.fold),"seed":int(run.seed),"formula":formula,
                        "structure_id":sid,"cosine_original_finetuned":float(cos[idx]),"relative_l2_drift":float(rel[idx])})
            del finetuned
        del original
    return pd.DataFrame(weight_rows), pd.DataFrame(embedding_rows)


def write_report(aggregate: pd.DataFrame, bootstrap: pd.DataFrame, resources: dict[str, Any]) -> None:
    """Escribe reporte científico con resultados observados y límites.

    Args:
        aggregate: Métricas por condición.
        bootstrap: Contrastes apareados.
        resources: Recursos medidos.
    """
    lines=["# Fine-tuning supervisado F0/F1/F2 de MatGPT y MatterGPT en ABS₃","",
        "## Hechos verificados","",
        "Los dos checkpoints originales pasaron carga estricta. La extracción FINAL_END reprodujo las cachés 210×768 (error máximo MatGPT 2.86e-6; MatterGPT 0). Se conservaron folds agrupados 45/11/14, Ridge M2 y test independiente cerrado.","",
        "## Resultados experimentales","",
        "| Modelo | Entrada | Política | Energy MAE (meV/átomo) | ΔE MAE firmado (meV/átomo) | Pairwise | Ground state | Orden triplete | Pendiente ΔE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.itertuples():
        lines.append(f"| {row.model} | {row.modality} | {row.freeze_policy} | {row.energy_mae_mev_atom_estimand_A:.2f} | {row.delta_mae_mev_atom_estimand_A:.2f} | {row.pairwise_accuracy_eligible:.1%} ({row.pairwise_denominator}) | {row.ground_state_accuracy:.1%} ({row.ground_state_denominator}) | {row.full_triplet_accuracy:.1%} ({row.full_triplet_denominator}) | {row.delta_ols_slope:.3f} |")
    lines += ["","Métrica primaria: MAE de tres ΔE firmados por fórmula, con semillas agrupadas dentro de fórmula y macro-promedio de 70 composiciones. Los pares elegibles cumplen |ΔE_DFT|>5 meV/átomo; |ΔE_pred|≤1 cuenta incorrecto. Ground state requiere gap DFT>1 meV/átomo y un mínimo predicho resuelto; los tripletes con cualquier gap DFT≤1 se excluyen.","",
        "## Contrastes bootstrap","","Los IC95% percentil se remuestrearon en parejas por fórmula (10 000 réplicas); no se ajustaron por multiplicidad. Intervalos descriptivos de validación cruzada de desarrollo, no evidencia de generalización a test externo.","",
        "| Contraste | Métrica | Diferencia | IC95% |","|---|---|---:|---:|"]
    for row in bootstrap.itertuples():
        lines.append(f"| {row.contrast} | {row.metric} | {row.estimate_left_minus_right:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] |")
    lines += ["","## Inferencias y limitaciones","",
        "F0/F1/F2 parten del mismo checkpoint original y cabeza warmup por fold/semilla/modalidad. La selección usa exclusivamente MAE ΔE de inner-validation. Las diferencias MatGPT–MatterGPT no aíslan causalmente la arquitectura: difieren en tokenización, preentrenamiento y condicionamiento. La geometría procede de celdas relajadas por DFT; no es predicción de energía sin estructura relajada.","",
        "Se evaluaron 70 composiciones de desarrollo; ‘no vista’ significa no usada en el ajuste supervisado del fold. No se demostró ausencia en los corpus generativos originales. El test independiente permaneció cerrado. F4 se conserva como referencia de protocolo distinto.","",
        "## Recursos","",
        f"Dispositivo: {resources.get('device')}; hilos CPU: {resources.get('torch_threads')}; tiempo: {resources.get('wall_seconds'):.1f} s; CPU: {resources.get('process_cpu_seconds'):.1f} s; RSS máximo: {resources.get('peak_rss_bytes')} bytes. MPS disponible: {resources.get('mps_available')}.","",
        "## Referencias metodológicas","",
        "- Kumar et al. (ICLR 2022), Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution. https://arxiv.org/abs/2202.10054",
        "- Howard & Ruder (ACL 2018), Universal Language Model Fine-tuning for Text Classification. https://aclanthology.org/P18-1031/",
        "- Chen & Ong (npj Computational Materials 2021), AtomSets. https://www.nature.com/articles/s41524-021-00639-w",
        "- Gururangan et al. (ACL 2020), Don't Stop Pretraining. https://aclanthology.org/2020.acl-main.740/","",
        "El protocolo es fine-tuning supervisado específico de tarea. Estas referencias motivan el diseño, no prueban eficacia directa en estabilidad polimórfica ABS₃.",""]
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    (REPORT_DIR/"reporte_cientifico.md").write_text("\n".join(lines),encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Calcula métricas, bootstrap, deriva y figuras desde OOF completas.

    Args:
        args: Opciones de análisis.
    """
    started=time.monotonic();cpu_start=time.process_time()
    structures,pairs=load_predictions()
    per_run=per_run_metrics(structures,pairs)
    comp,fold,aggregate=aggregate_metrics(per_run,pairs,structures)
    bootstrap=bootstrap_contrasts(comp,per_run,pairs)
    out=EXP/"metrics";out.mkdir(parents=True,exist_ok=True)
    structures.to_csv(EXP/"oof_structure_predictions.csv",index=False)
    pairs.to_csv(EXP/"oof_pair_predictions.csv",index=False)
    per_run.to_csv(out/"per_run.csv",index=False);comp.to_csv(out/"per_composition.csv",index=False)
    fold.to_csv(out/"per_fold.csv",index=False);aggregate.to_csv(out/"aggregate.csv",index=False)
    bootstrap.to_csv(out/"bootstrap_paired.csv",index=False)
    drift_weight=pd.DataFrame();drift_embedding=pd.DataFrame()
    if args.embedding_drift:
        run_records=pd.read_csv(EXP/"run_records_full.csv")
        run_records=run_records[run_records.execution_phase.eq("full")]
        drift_weight,drift_embedding=representation_drift(run_records,structures)
        (EXP/"diagnostics").mkdir(exist_ok=True)
        drift_weight.to_csv(EXP/"diagnostics/weight_drift.csv",index=False)
        drift_embedding.to_csv(EXP/"diagnostics/embedding_drift.csv",index=False)
    make_figures(structures,pairs,per_run,comp,aggregate,bootstrap)
    resources={"phase":"analysis", "wall_seconds":time.monotonic()-started,
        "process_cpu_seconds":time.process_time()-cpu_start,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024)),
        "cpu_count_logical":os.cpu_count(),"device":"CPU","torch_threads":torch.get_num_threads(),
        "platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
        "structures":len(structures),"pairs":len(pairs),"historical_test_accessed":False}
    atomic_json(EXP/"resources_analysis.json",resources)
    summary={"status":"COMPLETE","experiment":"abs3_original_supervised_ft_f0_f1_f2",
        "estimand":"A: semillas promediadas dentro de fórmula, macro promedio de 70 composiciones",
        "n_compositions":70,"n_structures_per_condition_seed":210,"n_pairs_per_condition_seed":210,
        "historical_test_accessed":False,"metrics":aggregate.to_dict(orient="records"),
        "bootstrap_replicates":BOOTSTRAP_N,"bootstrap_seed":BOOTSTRAP_SEED}
    atomic_json(EXP/"metrics.json",summary);atomic_json(EXP/"summary.json",summary)
    write_report(aggregate,bootstrap,resources)
    atomic_json(EXP/"analysis_status.json",{"status":"COMPLETE","figures":10,"resources":resources})


def main() -> None:
    """Parsea opciones y ejecuta el análisis sin acceder al test histórico."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-drift",action="store_true",help="Calcula deriva post hoc de pesos/embeddings (más CPU).")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
