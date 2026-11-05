"""Evalúa y grafica el fine-tuning supervisado F0/F1/F2 sobre ABS3.

Carga únicamente predicciones OOF, historiales de entrenamiento y registros de
selección interna ya guardados. No carga modelos ni vuelve a entrenar redes.
Produce tablas métricas y figuras científicas PNG para MatGPT y MatterGPT.
"""
from __future__ import annotations

import csv
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

LOGGER = logging.getLogger("abs3.f0f1f2.figures")
MODELS = ("MatGPT", "MatterGPT")
MODALITIES = ("SLICES", "SLICES_GEOMETRY")
POLICIES = ("F0", "F1", "F2")
COLORS = {"MatGPT": "#2878B5", "MatterGPT": "#C65B78"}
POLICY_STYLES = {"F0": "-", "F1": "--", "F2": "-."}
POLICY_HATCH = {"F0": "", "F1": "//", "F2": "xx"}
GAP_EDGES = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, np.inf)
GAP_LABELS = ("0–5", "5–10", "10–25", "25–50", "50–100", ">100")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261104
PAIRWISE_DFT_THRESHOLD = 5.0
PAIRWISE_PRED_THRESHOLD = 1.0
GROUND_STATE_GAP_THRESHOLD = 1.0


def project_root() -> Path:
    """Localiza la raíz del proyecto buscando AGENTS.md."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    raise FileNotFoundError("No se encontró AGENTS.md en los ancestros del script")


ROOT = project_root()
EXP = ROOT / "experiments/02_complementarios/abs3_original_supervised_ft_f0_f1_f2"
IMG = ROOT / "images/abs3_original_supervised_ft_f0_f1_f2"


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    """Valida columnas obligatorias y reporta el archivo de origen."""
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: faltan columnas requeridas: {sorted(missing)}")


def read_csvs(folder: Path, prefix: str) -> pd.DataFrame:
    """Carga y concatena CSVs de predicción con un prefijo de modelo."""
    files = sorted(folder.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay predicciones {prefix} en {folder}")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga artefactos OOF completos y valida cobertura, claves y valores finitos."""
    status_path = EXP / "training_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed_path = EXP / "completed_runs.csv"
    with completed_path.open(newline="", encoding="utf-8") as handle:
        completed = list(csv.DictReader(handle))
    if status.get("status") != "COMPLETE" or len(completed) != 180:
        raise ValueError("La fase completa debe registrar 180/180 corridas antes de evaluar")
    structures = read_csvs(EXP / "predictions/structures", "MatGPT")
    structures = pd.concat([structures, read_csvs(EXP / "predictions/structures", "MatterGPT")],
                           ignore_index=True)
    pairs = read_csvs(EXP / "predictions/pairs", "MatGPT")
    pairs = pd.concat([pairs, read_csvs(EXP / "predictions/pairs", "MatterGPT")],
                      ignore_index=True)
    history_path = EXP / "training_history_full.csv"
    history = pd.read_csv(history_path)
    records_path = EXP / "run_records_full.csv"
    records = pd.read_csv(records_path)
    require_columns(structures, {"condition", "model", "modality", "freeze_policy", "fold", "seed",
                                 "formula", "structure_id", "polymorph", "E_DFT_eV_atom",
                                 "mu_M2_eV_atom", "centered_residual_eV_atom", "E_pred_eV_atom"},
                    EXP / "predictions/structures")
    require_columns(pairs, {"condition", "model", "modality", "freeze_policy", "fold", "seed",
                            "formula", "polymorph_i", "polymorph_j", "DeltaE_DFT_meV_atom",
                            "DeltaE_pred_meV_atom", "pairwise_eligible", "pairwise_correct",
                            "unresolved"}, EXP / "predictions/pairs")
    require_columns(history, {"model", "modality", "condition", "fold", "seed", "backbone_lr",
                              "epoch", "train_minibatch_smoothl1", "inner_validation_smoothl1",
                              "inner_validation_delta_mae_mev_atom"}, history_path)
    require_columns(records, {"model", "modality", "policy", "fold", "seed", "backbone_lr",
                              "best_epoch", "best_validation_delta_mae_mev_atom"}, records_path)
    expected_conditions = {f"{m}_{mod}_{pol}" for m in MODELS for mod in MODALITIES for pol in POLICIES}
    for frame, name, per_run in ((structures, "structures", 42), (pairs, "pairs", 42)):
        actual = set(frame.condition.astype(str).unique())
        if actual != expected_conditions:
            raise ValueError(f"{name}: condiciones faltantes/extra: {actual ^ expected_conditions}")
        if frame.duplicated(["condition", "fold", "seed", "structure_id"] if name == "structures"
                            else ["condition", "fold", "seed", "formula", "polymorph_i", "polymorph_j"]).any():
            raise ValueError(f"{name}: hay claves OOF duplicadas")
    if len(structures) != 12 * 630 or len(pairs) != 12 * 630:
        raise ValueError(f"OOF incompleto: {len(structures)} estructuras y {len(pairs)} pares")
    if structures.groupby("condition").structure_id.nunique().ne(210).any():
        raise ValueError("Cada condición debe cubrir las mismas 210 estructuras")
    if structures.groupby("condition").formula.nunique().ne(70).any():
        raise ValueError("Cada condición debe cubrir las mismas 70 fórmulas")
    numeric_s = ["E_DFT_eV_atom", "mu_M2_eV_atom", "E_pred_eV_atom", "centered_residual_eV_atom"]
    numeric_p = ["DeltaE_DFT_meV_atom", "DeltaE_pred_meV_atom"]
    if not np.isfinite(structures[numeric_s].to_numpy(float)).all():
        raise ValueError("Predicciones/targets de energía contienen NaN o Inf")
    if not np.isfinite(pairs[numeric_p].to_numpy(float)).all():
        raise ValueError("Predicciones/targets ΔE contienen NaN o Inf")
    return structures, pairs, history, records


def identity_parts(condition: str) -> tuple[str, str, str]:
    """Separa modelo, modalidad y política del identificador canónico."""
    for model in MODELS:
        prefix = model + "_"
        if condition.startswith(prefix):
            rest = condition[len(prefix):]
            for modality in sorted(MODALITIES, key=len, reverse=True):
                mod_prefix = modality + "_"
                if rest.startswith(mod_prefix):
                    return model, modality, rest[len(mod_prefix):]
    raise ValueError(f"Identificador de condición desconocido: {condition}")


def modality_label(modality: str, concise: bool = True) -> str:
    """Devuelve el nombre español de la modalidad de entrada."""
    if modality == "SLICES":
        return "SLICES"
    if modality == "SLICES_GEOMETRY":
        return "SLICES + geom." if concise else "SLICES + geometría"
    raise ValueError(f"Modalidad desconocida: {modality}")


def boolean_values(series: pd.Series) -> np.ndarray:
    """Convierte columnas CSV booleanas a un vector booleano seguro."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True,
                                               "0": False}).fillna(False).to_numpy(bool)


def ols_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calcula OLS con intercepto libre, R² y MAE sobre los puntos mostrados."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "mae": float(np.mean(np.abs(y - x)))}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - float(np.square(y - fitted).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "mae": float(np.mean(np.abs(y - x)))}


def calculate_metrics(structures: pd.DataFrame, pairs: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula métricas por fórmula, condición, gaps y polimorfo."""
    comp_rows: list[dict[str, Any]] = []
    gs_rows: list[dict[str, Any]] = []
    for (condition, fold, seed, formula), group in structures.groupby(
            ["condition", "fold", "seed", "formula"], sort=True):
        group = group.sort_values("polymorph").copy()
        y = group.E_DFT_eV_atom.to_numpy(float)
        pred = group.E_pred_eV_atom.to_numpy(float)
        mu = group.mu_M2_eV_atom.to_numpy(float)
        if len(group) != 3 or not np.allclose(mu, mu[0], rtol=0, atol=1e-12):
            raise ValueError(f"Triplete/Ridge M2 inválido: {condition}, {formula}, seed={seed}")
        energy_error_mev = (pred - y) * 1000.0
        residual_pred = group.centered_residual_eV_atom.to_numpy(float) * 1000.0
        residual_target = (y - mu) * 1000.0
        comp_rows.append({
            "condition": condition, "model": group.model.iloc[0], "modality": group.modality.iloc[0],
            "policy": group.freeze_policy.iloc[0], "fold": int(fold), "seed": int(seed),
            "formula": formula, "n_structures": len(group),
            "energy_mae_mev_atom": float(np.mean(np.abs(energy_error_mev))),
            "energy_rmse_mev_atom": float(np.sqrt(np.mean(energy_error_mev ** 2))),
            "energy_bias_mev_atom": float(np.mean(energy_error_mev)),
            "residual_mae_mev_atom": float(np.mean(np.abs(residual_pred - residual_target))),
            "rho_spearman": float(spearmanr(y, pred).statistic),
            "tau_kendall": float(kendalltau(y, pred).statistic),
        })
        sorted_dft = np.sort(y)
        gap_gs_mev = float((sorted_dft[1] - sorted_dft[0]) * 1000.0)
        gs_eligible = gap_gs_mev > GROUND_STATE_GAP_THRESHOLD
        min_pred = float(np.min(pred))
        pred_min_idx = np.flatnonzero(np.isclose(pred, min_pred, rtol=0.0, atol=1e-12))
        unique_pred_min = len(pred_min_idx) == 1
        gs_correct = bool(unique_pred_min and np.isclose(y[pred_min_idx[0]], sorted_dft[0],
                                                         rtol=0.0, atol=1e-12))
        dft_gaps = np.diff(np.sort(y))
        full_eligible = bool(np.all(dft_gaps * 1000.0 > GROUND_STATE_GAP_THRESHOLD))
        pred_gaps = np.diff(np.sort(pred))
        full_correct = bool(full_eligible and np.all(pred_gaps > 1e-12) and
                            np.array_equal(np.argsort(y, kind="stable"),
                                           np.argsort(pred, kind="stable")))
        gs_rows.append({"condition": condition, "model": group.model.iloc[0],
                        "modality": group.modality.iloc[0], "policy": group.freeze_policy.iloc[0],
                        "fold": int(fold), "seed": int(seed), "formula": formula,
                        "ground_state_eligible": gs_eligible, "ground_state_correct": gs_correct,
                        "ground_state_gap_mev_atom": gap_gs_mev,
                        "full_triplet_eligible": full_eligible, "full_triplet_correct": full_correct})

    comp = pd.DataFrame(comp_rows)
    gs = pd.DataFrame(gs_rows)
    p = pairs.copy()
    p["absolute_error_mev_atom"] = np.abs(p.DeltaE_pred_meV_atom - p.DeltaE_DFT_meV_atom)
    p["eligible_bool"] = boolean_values(p.pairwise_eligible)
    p["correct_bool"] = boolean_values(p.pairwise_correct)
    p["unresolved_bool"] = boolean_values(p.unresolved)
    p["dft_gap_abs_mev_atom"] = np.abs(p.DeltaE_DFT_meV_atom)
    pair_mae = (p.groupby(["condition", "fold", "seed", "formula"], as_index=False)
                .absolute_error_mev_atom.mean()
                .rename(columns={"absolute_error_mev_atom": "mae_delta_signed_estimand_A_mev_atom"}))
    comp = comp.merge(pair_mae, on=["condition", "fold", "seed", "formula"],
                      how="left", validate="one_to_one")

    summary_rows: list[dict[str, Any]] = []
    for condition in sorted(comp.condition.unique()):
        c = comp[comp.condition.eq(condition)]
        q = p[p.condition.eq(condition)]
        g = gs[gs.condition.eq(condition)]
        # Promedio de semillas dentro de fórmula; después, macro-promedio de 70 fórmulas.
        formula = c.groupby("formula", as_index=False).agg(
            energy_mae=("energy_mae_mev_atom", "mean"),
            residual_mae=("residual_mae_mev_atom", "mean"),
            bias=("energy_bias_mev_atom", "mean"),
            spearman=("rho_spearman", "mean"), kendall=("tau_kendall", "mean"))
        pair_formula_seed = q.groupby(["formula", "seed"], as_index=False).absolute_error_mev_atom.mean()
        de_formula = pair_formula_seed.groupby("formula").absolute_error_mev_atom.mean()
        energy_sq = c.groupby("formula").energy_rmse_mev_atom.apply(lambda x: np.mean(x ** 2))
        all_pairs = q[["DeltaE_DFT_meV_atom", "DeltaE_pred_meV_atom"]].to_numpy(float)
        delta_ols = ols_summary(all_pairs[:, 0], all_pairs[:, 1])
        eligible = q[q.eligible_bool]
        resolved = eligible[~eligible.unresolved_bool]
        gs_eligible = g[g.ground_state_eligible]
        full = g[g.full_triplet_eligible]
        model, modality, policy = identity_parts(condition)
        summary_rows.append({
            "condition": condition, "model": model, "modality": modality, "policy": policy,
            "n_formulas": int(c.formula.nunique()), "n_structures_formula_seed": int(len(c) * 3),
            "n_structure_seed_points": int(len(c) * 3), "n_pair_seed_points": int(len(q)),
            "energy_mae_mev_atom": float(formula.energy_mae.mean()),
            "energy_rmse_global_mev_atom": float(np.sqrt(np.mean(energy_sq.to_numpy(float)))),
            "energy_bias_mev_atom": float(formula.bias.mean()),
            "residual_mae_mev_atom": float(formula.residual_mae.mean()),
            "deltaE_mae_signed_estimand_A_mev_atom": float(de_formula.mean()),
            "deltaE_rmse_global_mev_atom": float(np.sqrt(np.mean(q.absolute_error_mev_atom.to_numpy(float) ** 2))),
            "pairwise_correct": int(eligible.correct_bool.sum()),
            "pairwise_eligible": int(len(eligible)),
            "pairwise_accuracy_eligible": float(eligible.correct_bool.mean()),
            "pairwise_resolved_correct": int(resolved.correct_bool.sum()),
            "pairwise_resolved": int(len(resolved)),
            "pairwise_accuracy_resolved": float(resolved.correct_bool.mean()) if len(resolved) else float("nan"),
            "unresolved_eligible": int(eligible.unresolved_bool.sum()),
            "unresolved_fraction_eligible": float(eligible.unresolved_bool.mean()),
            "groundstate_correct": int(gs_eligible.ground_state_correct.sum()),
            "groundstate_eligible": int(len(gs_eligible)),
            "groundstate_accuracy": float(gs_eligible.ground_state_correct.mean()),
            "full_triplet_correct": int(full.full_triplet_correct.sum()),
            "full_triplet_eligible": int(len(full)),
            "full_triplet_accuracy": float(full.full_triplet_correct.mean()) if len(full) else float("nan"),
            "spearman_mean_by_formula": float(formula.spearman.mean()),
            "kendall_mean_by_formula": float(formula.kendall.mean()),
            "ols_deltaE_slope": delta_ols["slope"],
            "ols_deltaE_intercept_mev_atom": delta_ols["intercept"],
            "ols_deltaE_r2": delta_ols["r2"],
        })
    summary = pd.DataFrame(summary_rows)
    # Error por intervalo del gap: promedia primero dentro de fórmula y semilla.
    p["gap_bin"] = pd.cut(p.dft_gap_abs_mev_atom, bins=GAP_EDGES, labels=GAP_LABELS,
                          right=False, include_lowest=True)
    gap_formula = (p.groupby(["condition", "formula", "seed", "gap_bin"], observed=False)
                   .absolute_error_mev_atom.mean().rename("formula_seed_mae").reset_index())
    gap_table = (gap_formula.groupby(["condition", "formula", "gap_bin"], observed=False)
                 .formula_seed_mae.mean().groupby(["condition", "gap_bin"], observed=False)
                 .agg(mae_deltaE_mev_atom="mean", n_formulas="count").reset_index())
    pair_counts = (p.groupby(["condition", "gap_bin"], observed=False).size()
                   .rename("n_pair_seed_observations").reset_index())
    gap_table = gap_table.merge(pair_counts, on=["condition", "gap_bin"], how="left")

    # Error energético por tipo de polimorfo, promediando semillas dentro de fórmula.
    rows: list[dict[str, Any]] = []
    energy_error = structures.copy()
    energy_error["abs_energy_error_mev_atom"] = np.abs(
        (energy_error.E_pred_eV_atom - energy_error.E_DFT_eV_atom) * 1000.0)
    for (condition, formula_name, seed, polymorph), group in energy_error.groupby(
            ["condition", "formula", "seed", "polymorph"]):
        rows.append({"condition": condition, "formula": formula_name, "seed": seed,
                     "polymorph": polymorph,
                     "abs_energy_error_mev_atom": float(group.abs_energy_error_mev_atom.mean())})
    polymorph_seed = pd.DataFrame(rows)
    polymorph = (polymorph_seed.groupby(["condition", "formula", "polymorph"], as_index=False)
                 .abs_energy_error_mev_atom.mean()
                 .groupby(["condition", "polymorph"], as_index=False)
                 .agg(mae_energy_mev_atom=("abs_energy_error_mev_atom", "mean"),
                      n_formulas=("formula", "nunique")))
    return comp.merge(gs, on=["condition", "model", "modality", "policy", "fold", "seed", "formula"],
                      how="left"), summary, gap_table, polymorph


def selected_history(history: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Filtra curvas al LR seleccionado por validación para cada fold-semilla."""
    selected = records.copy()
    selected["backbone_lr_key"] = selected.backbone_lr.fillna("").astype(str)
    h = history.copy()
    h["backbone_lr_key"] = h.backbone_lr.fillna("").astype(str)
    h["policy"] = h.condition.astype(str)
    keys = ["model", "modality", "policy", "fold", "seed", "backbone_lr_key"]
    chosen = selected[keys].drop_duplicates()
    out = h.merge(chosen, on=keys, how="inner", validate="many_to_one")
    if out.empty:
        raise ValueError("No se pudieron emparejar historiales con los LR seleccionados")
    return out


def condition_order() -> list[str]:
    """Devuelve orden fijo modelo, modalidad, política para todas las figuras."""
    return [f"{m}_{mod}_{pol}" for m in MODELS for mod in MODALITIES for pol in POLICIES]


def annotate_figure(fig: plt.Figure, note: str, bottom: float = 0.10) -> None:
    """Añade nota metodológica legible y reserva espacio inferior."""
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))


def plot_metric_summary(summary: pd.DataFrame) -> None:
    """Grafica MAE de energía, residuo, ΔE y RMSE global por condición."""
    order = condition_order()
    labels = [f"{m}\n{'SLICES' if mod == 'SLICES' else 'SLICES + geom.'} {pol}"
              for m in MODELS for mod in MODALITIES for pol in POLICIES]
    specs = [
        ("energy_mae_mev_atom", "MAE de energía absoluta por átomo"),
        ("residual_mae_mev_atom", "MAE del residuo E−μ_Ridge-M2"),
        ("deltaE_mae_signed_estimand_A_mev_atom", "MAE ΔE firmado (estimando A)"),
        ("deltaE_rmse_global_mev_atom", "RMSE global de ΔE firmado"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, (metric, title) in zip(axes.flat, specs):
        q = summary.set_index("condition").reindex(order)
        values = q[metric].to_numpy(float)
        colors = [COLORS[identity_parts(c)[0]] for c in order]
        bars = ax.bar(np.arange(len(order)), values, color=colors, edgecolor="#333333",
                      hatch=[POLICY_HATCH[identity_parts(c)[2]] for c in order], linewidth=.6)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("meV/átomo")
        ax.grid(axis="y", alpha=.22)
        ax.set_xticks(np.arange(len(order)), labels, rotation=35, ha="right", fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLORS["MatGPT"], label="MatGPT"),
                        Patch(facecolor=COLORS["MatterGPT"], label="MatterGPT"),
                        Patch(facecolor="white", edgecolor="#333", hatch="", label="F0: backbone congelado"),
                        Patch(facecolor="white", edgecolor="#333", hatch="//", label="F1: último bloque"),
                        Patch(facecolor="white", edgecolor="#333", hatch="xx", label="F2: dos bloques finales")],
               loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("ABS₃: métricas energéticas OOF por estructura y diferencias intrafórmula", y=1.04)
    annotate_figure(fig,
        "OOF de desarrollo: 70 fórmulas, 210 estructuras, 5 pliegues externos × 3 semillas. "
        "Cada fórmula pesa igual tras promediar semillas. ΔE = Eⱼ−Eᵢ en los tres pares intrafórmula. "
        "E_pred = μ_Ridge-M2 + residuo centrado; por definición, el error del residuo coincide con el error energético. "
        "Las barras RMSE usan raíz de la media del error cuadrático global. No se abrió el test histórico.")
    fig.savefig(IMG / "metricas_energeticas_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(summary: pd.DataFrame) -> None:
    """Compara ranking pairwise, pares resueltos, unresolved y selección de mínimos."""
    order = condition_order()
    labels = [f"{identity_parts(c)[0]}\n{modality_label(identity_parts(c)[1])} {identity_parts(c)[2]}"
              for c in order]
    specs = [
        ("pairwise_accuracy_eligible", "Exactitud pairwise elegible", "pairwise_correct", "pairwise_eligible"),
        ("pairwise_accuracy_resolved", "Exactitud pairwise entre resueltos", "pairwise_resolved_correct", "pairwise_resolved"),
        ("unresolved_fraction_eligible", "Fracción de pares no resueltos", "unresolved_eligible", "pairwise_eligible"),
        ("groundstate_accuracy", "Selección del estado fundamental", "groundstate_correct", "groundstate_eligible"),
        ("full_triplet_accuracy", "Triplete completo correctamente ordenado", "full_triplet_correct", "full_triplet_eligible"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for ax, (metric, title, num, den) in zip(axes.flat, specs):
        q = summary.set_index("condition").reindex(order)
        vals = q[metric].to_numpy(float) * 100.0
        bars = ax.bar(np.arange(len(order)), vals,
                      color=[COLORS[identity_parts(c)[0]] for c in order],
                      edgecolor="#333", hatch=[POLICY_HATCH[identity_parts(c)[2]] for c in order])
        for bar, n, d, val in zip(bars, q[num], q[den], vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1.1, f"{val:.1f}%\n{int(n)}/{int(d)}",
                        ha="center", va="bottom", fontsize=6.2)
        ax.set_title(title); ax.set_ylim(0, 108); ax.set_ylabel("%")
        ax.grid(axis="y", alpha=.2)
        ax.set_xticks(np.arange(len(order)), labels, rotation=40, ha="right", fontsize=6.7)
        if metric == "pairwise_accuracy_eligible":
            ax.axhline(90, color="#A33", linestyle=":", linewidth=1.2)
            ax.text(.98, .86, "referencia 90%", color="#922", transform=ax.transAxes,
                    ha="right", fontsize=7)
    axes.flat[-1].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLORS["MatGPT"], label="MatGPT"),
                        Patch(facecolor=COLORS["MatterGPT"], label="MatterGPT")],
               loc="upper center", bbox_to_anchor=(.5, .99), ncol=2, frameon=False)
    fig.suptitle("ABS₃ OOF: ranking pairwise y selección del polimorfo estable", y=1.02)
    annotate_figure(fig,
        "Pairwise: elegible si |ΔE_DFT|>5 meV/átomo; |ΔE_pred|≤1 meV/átomo cuenta incorrecto. "
        "Ground state: gap DFT al segundo mínimo >1 meV/átomo; empate predicho cuenta incorrecto. "
        "Triplete completo: solo fórmulas con tres energías DFT separadas por >1 meV/átomo; requiere orden total correcto y sin empates predichos. "
        "Conteos acumulan fórmula×semilla; denominadores esperados: 591 pares elegibles y 201 casos ground-state por condición.")
    fig.savefig(IMG / "ranking_y_groundstate_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parity(structures: pd.DataFrame, pairs: pd.DataFrame, modality: str, kind: str) -> pd.DataFrame:
    """Grafica paridad con OLS por condición y exporta sus estadísticas."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5), sharex=True, sharey=True)
    stats_rows: list[dict[str, Any]] = []
    limit_values: list[float] = []
    source = structures if kind == "energy" else pairs
    for model in MODELS:
        for policy in POLICIES:
            cond = f"{model}_{modality}_{policy}"
            q = source[source.condition.eq(cond)]
            if kind == "energy":
                x = q.E_DFT_eV_atom.to_numpy(float) * 1000.0
                y = q.E_pred_eV_atom.to_numpy(float) * 1000.0
                xlabel = "E DFT por átomo (meV/átomo)"
                ylabel = "E predicha por átomo (meV/átomo)"
                title_name = "energía absoluta"
            else:
                x = q.DeltaE_DFT_meV_atom.to_numpy(float)
                y = q.DeltaE_pred_meV_atom.to_numpy(float)
                xlabel = "ΔE DFT firmada = Eⱼ−Eᵢ (meV/átomo)"
                ylabel = "ΔE predicha firmada = Eⱼ−Eᵢ (meV/átomo)"
                title_name = "diferencias firmadas"
            ax = axes[MODELS.index(model), POLICIES.index(policy)]
            m = ols_summary(x, y)
            stats_rows.append({"condition": cond, "quantity": kind, "n_points": int(len(q)),
                               "slope": m["slope"], "intercept": m["intercept"], "r2": m["r2"],
                               "mae_mev_atom": m["mae"]})
            limit_values.extend([float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))])
            ax.scatter(x, y, s=8, alpha=.2, color=COLORS[model], edgecolor="none")
            grid_min, grid_max = min(x.min(), y.min()), max(x.max(), y.max())
            grid = np.linspace(grid_min, grid_max, 150)
            ax.plot(grid, grid, color="#222", linestyle="--", linewidth=1, label="Ideal y=x")
            ax.plot(grid, m["slope"]*grid + m["intercept"], color="#E07A24",
                    linewidth=1.35, label="OLS y=mx+b")
            ax.set_title(f"{model} · {policy}", fontsize=10)
            ax.grid(alpha=.18)
            ax.text(.035, .96,
                    f"N={len(q)} estructura×semilla\nm={m['slope']:.3f}; b={m['intercept']:.1f} meV/átomo\n"
                    f"R²={m['r2']:.3f}; MAE={m['mae']:.1f} meV/átomo",
                    transform=ax.transAxes, va="top", fontsize=7.1,
                    bbox={"facecolor": "white", "alpha": .86, "edgecolor": "#bbb"})
    lim_min, lim_max = min(limit_values), max(limit_values)
    pad = (lim_max - lim_min)*.025
    for ax in axes.flat:
        ax.set_xlim(lim_min-pad, lim_max+pad); ax.set_ylim(lim_min-pad, lim_max+pad)
        ax.set_aspect("equal", adjustable="box")
    fig.legend(handles=axes[0,0].get_legend_handles_labels()[0],
               labels=axes[0,0].get_legend_handles_labels()[1],
               loc="upper center", bbox_to_anchor=(.5,.99), ncol=2, frameon=False)
    fig.suptitle(f"ABS₃ OOF: paridad de {title_name} · {modality.replace('_', ' + ').replace('SLICES + GEOMETRY','SLICES + geometría')}",
                 y=1.02, fontsize=14)
    fig.supxlabel(xlabel, y=.065); fig.supylabel(ylabel, x=.012)
    note = ("Punto = estructura×semilla; 210 estructuras×3 semillas=630 observaciones por panel. "
            "Los paneles comparten límites y usan los mismos 70 compuestos, cinco pliegues externos y targets. "
            "Línea discontinua: y=x. OLS y=m x+b, N, pendiente, intercepto, R² y MAE se calculan con todos los puntos del panel. "
            "La energía absoluta se reconstruye como μ_Ridge-M2 + residuo centrado."
            if kind == "energy" else
            "Punto = par intrafórmula×semilla; 210 pares×3 semillas=630 observaciones por panel. "
            "ΔE=Eⱼ−Eᵢ, siempre dentro de la misma fórmula; signo negativo indica que j está por debajo de i. "
            "Paneles comparten límites. Línea discontinua: y=x; OLS y=m x+b y estadísticas usan todos los puntos mostrados.")
    fig.text(.5, .012, note, ha="center", va="bottom", fontsize=7.8, wrap=True)
    fig.tight_layout(rect=(.04, .12, 1, .95))
    filename = "paridad_energia_absoluta" if kind == "energy" else "paridad_deltaE_firmada"
    fig.savefig(IMG / f"{filename}_{modality.lower()}_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def selected_learning_history(history: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Une las curvas con los learning rates seleccionados por validación interna."""
    h = history.copy()
    r = records.copy()
    h["policy"] = h.condition.astype(str)
    for frame in (h, r):
        frame["fold"] = frame.fold.astype(int)
        frame["seed"] = frame.seed.astype(int)
    h["lr_key"] = h.backbone_lr.fillna("").astype(str)
    r["lr_key"] = r.backbone_lr.fillna("").astype(str)
    selected_keys = r[["model", "modality", "policy", "fold", "seed", "lr_key"]].drop_duplicates()
    chosen = h.merge(selected_keys, on=["model", "modality", "policy", "fold", "seed", "lr_key"],
                     how="inner", validate="many_to_one")
    if chosen.empty:
        raise ValueError("Historial seleccionado vacío; revisar claves y LR")
    return chosen


def plot_learning_curves(history: pd.DataFrame) -> None:
    """Grafica MAE ΔE de validación por época, seleccionando LR interno."""
    chosen = selected_learning_history(history, pd.read_csv(EXP / "run_records_full.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    for ax, (model, modality) in zip(axes.flat,
            [(m, mod) for m in MODELS for mod in MODALITIES]):
        subset = chosen[(chosen.model == model) & (chosen.modality == modality)]
        for policy in POLICIES:
            q = subset[subset.policy.eq(policy)]
            stat = q.groupby("epoch").inner_validation_delta_mae_mev_atom.agg(["mean", "std", "count"])
            x = stat.index.to_numpy(float)
            mean = stat["mean"].to_numpy(float)
            sem = stat["std"].fillna(0).to_numpy(float) / np.sqrt(stat["count"].to_numpy(float))
            ax.plot(x, mean, color=COLORS[model], linestyle=POLICY_STYLES[policy],
                    linewidth=1.8, label=f"{policy}: n máx. 15")
            ax.fill_between(x, mean-sem, mean+sem, color=COLORS[model], alpha=.12)
        ax.set_title(f"{model} · {'SLICES' if modality == 'SLICES' else 'SLICES + geometría'}")
        ax.set_xlabel("Época local de continuación")
        ax.set_ylabel("MAE ΔE validation (meV/átomo)")
        ax.grid(alpha=.2); ax.legend(title="Política", frameon=False, fontsize=8)
    fig.suptitle("ABS₃: evolución del MAE ΔE de validación interna", y=.99)
    fig.text(.5, .012,
        "Media ± error estándar entre corridas fold-semilla activas en cada época (máx. n=15); no es un IC poblacional. "
        "Se muestra el learning rate seleccionado por inner-validation para cada fold/semilla; por ello las curvas son descriptivas y selección-condicionadas. "
        "Validación completa: 11 fórmulas; 45 fórmulas de inner-train. No se representan datos outer en la selección.",
        ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(IMG / "curva_validacion_mae_deltaE_por_epoca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: pd.DataFrame) -> None:
    """Grafica objetivo SmoothL1 de minilotes y validación completa."""
    chosen = selected_learning_history(history, pd.read_csv(EXP / "run_records_full.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True, sharey=True)
    for ax, (model, modality) in zip(axes.flat,
            [(m, mod) for m in MODELS for mod in MODALITIES]):
        subset = chosen[(chosen.model == model) & (chosen.modality == modality)]
        for policy in POLICIES:
            q = subset[subset.policy.eq(policy)]
            for col, ls, description in (
                ("train_minibatch_smoothl1", POLICY_STYLES[policy], "train minilotes"),
                ("inner_validation_smoothl1", POLICY_STYLES[policy], "validación completa"),
            ):
                stat = q.groupby("epoch")[col].agg(["mean", "std", "count"])
                x = stat.index.to_numpy(float)
                mean = stat["mean"].to_numpy(float)
                sd = stat["std"].fillna(0).to_numpy(float)
                ax.plot(x, mean, color=COLORS[model], linestyle=ls,
                        linewidth=1.6, alpha=.95 if col.startswith("inner") else .68,
                        label=f"{policy} · {description}")
                ax.fill_between(x, mean-sd, mean+sd, color=COLORS[model], alpha=.045)
        ax.set_title(f"{model} · {'SLICES' if modality == 'SLICES' else 'SLICES + geometría'}")
        ax.set_xlabel("Época local de continuación")
        ax.set_ylabel("SmoothL1 del residuo normalizado (adimensional)")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=6.8, ncol=2)
    fig.suptitle("ABS₃: curvas de entrenamiento y validación de MLP8", y=.99)
    fig.text(.5, .012,
        "Train = promedio de minilotes registrado durante optimización; validation = evaluación completa en 11 fórmulas. "
        "Bandas ±1 desviación estándar entre corridas disponibles (no IC); el número de corridas cae por early stopping. "
        "Curvas F1/F2 usan LR seleccionado en inner-validation y comienzan tras warmup común; selección externa no se muestra.",
        ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(IMG / "curvas_train_validation_smoothl1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gap_error(gap: pd.DataFrame) -> None:
    """Grafica MAE ΔE según la magnitud absoluta del gap DFT."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    x = np.arange(len(GAP_LABELS))
    for ax, model in zip(axes, MODELS):
        sub = gap[gap.condition.str.startswith(model + "_")]
        for offset, modality in enumerate(MODALITIES):
            for policy in POLICIES:
                cond = f"{model}_{modality}_{policy}"
                q = sub[sub.condition.eq(cond)].set_index("gap_bin").reindex(GAP_LABELS)
                off = (MODALITIES.index(modality)*3 + POLICIES.index(policy) - 2.5)*.085
                vals = q.mae_deltaE_mev_atom.to_numpy(float)
                ax.plot(x+off, vals, marker=("o" if modality=="SLICES" else "^"),
                        color=COLORS[model], linestyle=POLICY_STYLES[policy],
                        linewidth=1.35, markersize=4, label=f"{modality_label(modality, concise=False)} {policy}")
        ax.set_title(model); ax.set_xticks(x, GAP_LABELS); ax.set_xlabel("|ΔE DFT| (meV/átomo)")
        ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, fontsize=6.8, ncol=2)
    axes[0].set_ylabel("MAE ΔE en el intervalo (meV/átomo)")
    fig.suptitle("ABS₃ OOF: error de diferencias energéticas por magnitud del gap", y=.98)
    fig.text(.5, .012,
        "Bins: [0,5), [5,10), [10,25), [25,50), [50,100), [100,∞) meV/átomo. "
        "Hay 13, 17, 41, 46, 56 y 37 pares DFT únicos en esos bins, respectivamente (replicados con 3 semillas); "
        "cada fórmula aporta el promedio de sus pares y luego se promedian semillas y fórmulas. "
        "Los conteos por bin y fórmulas presentes se guardan en metricas_deltaE_por_gap.csv. Pares siempre intrafórmula; OOF de desarrollo.",
        ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0,.09,1,.93))
    fig.savefig(IMG / "error_deltaE_por_magnitud_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_polymorph_errors(polymorph: pd.DataFrame) -> None:
    """Compara error absoluto de energía por familia de polimorfo."""
    order = condition_order()
    labels = [f"{identity_parts(c)[0]}\n{modality_label(identity_parts(c)[1])} {identity_parts(c)[2]}"
              for c in order]
    types = sorted(polymorph.polymorph.unique())
    fig, axes = plt.subplots(1, len(types), figsize=(17, 5.8), sharey=True)
    for ax, kind in zip(axes, types):
        q = polymorph[polymorph.polymorph.eq(kind)].set_index("condition").reindex(order)
        values = q.mae_energy_mev_atom.to_numpy(float)
        bars = ax.bar(np.arange(len(order)), values,
                      color=[COLORS[identity_parts(c)[0]] for c in order],
                      edgecolor="#333", hatch=[POLICY_HATCH[identity_parts(c)[2]] for c in order])
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)
        ax.set_title(kind.replace("_", " "))
        ax.set_xticks(np.arange(len(order)), labels, rotation=55, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("MAE energía por átomo (meV/átomo)")
    fig.suptitle("ABS₃ OOF: error energético por familia de polimorfo", y=.98)
    fig.text(.5, .012,
        "Cada punto de partida es la estructura×semilla; se promedian semillas dentro de cada fórmula y después las 70 fórmulas. "
        "Tres polimorfos por fórmula; 210 estructuras en cinco pliegues externos. Energía reconstruida con Ridge M2 + residuo centrado.",
        ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0,.09,1,.92))
    fig.savefig(IMG / "error_energia_por_polimorfo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bootstrap_delta_contrasts(comp: pd.DataFrame) -> pd.DataFrame:
    """Calcula IC bootstrap apareados por fórmula para F1/F2 frente a F0."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for model in MODELS:
        for modality in MODALITIES:
            for policy in ("F1", "F2"):
                cond = f"{model}_{modality}_{policy}"
                base_cond = f"{model}_{modality}_F0"
                a = comp[comp.condition.eq(cond)].groupby("formula").mae_delta_signed_estimand_A_mev_atom.mean()
                b = comp[comp.condition.eq(base_cond)].groupby("formula").mae_delta_signed_estimand_A_mev_atom.mean()
                joined = pd.concat([a.rename("candidate"), b.rename("baseline")], axis=1).dropna()
                diff = (joined.candidate - joined.baseline).to_numpy(float)
                indices = rng.integers(0, len(diff), size=(BOOTSTRAP_REPLICATES, len(diff)))
                boot = diff[indices].mean(axis=1)
                rows.append({"contrast": f"{cond} - {base_cond}", "model": model, "modality": modality,
                             "policy": policy, "metric": "MAE ΔE estimando A",
                             "estimate_mev_atom": float(diff.mean()),
                             "ci95_low_mev_atom": float(np.quantile(boot, .025)),
                             "ci95_high_mev_atom": float(np.quantile(boot, .975)),
                             "n_formulas": int(len(diff)), "replicates": BOOTSTRAP_REPLICATES,
                             "seed": BOOTSTRAP_SEED})
    return pd.DataFrame(rows)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    """Visualiza cambios apareados de MAE ΔE con intervalos percentiles."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    q = bootstrap.reset_index(drop=True)
    y = np.arange(len(q))
    est = q.estimate_mev_atom.to_numpy(float)
    lo = q.ci95_low_mev_atom.to_numpy(float)
    hi = q.ci95_high_mev_atom.to_numpy(float)
    colors = [COLORS[m] for m in q.model]
    for i in range(len(q)):
        ax.errorbar(est[i], y[i], xerr=[[est[i]-lo[i]], [hi[i]-est[i]]],
                    fmt="o", color=colors[i], capsize=3, linewidth=1.5)
    ax.axvline(0, color="#222", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{r.model} · {'SLICES' if r.modality=='SLICES' else 'SLICES + geometría'}: {r.policy}−F0"
                      for r in q.itertuples()])
    ax.invert_yaxis(); ax.set_xlabel("Cambio de MAE ΔE vs F0 (meV/átomo; negativo favorece F1/F2)")
    ax.set_title("ABS₃: bootstrap apareado por fórmula para cambios F1/F2")
    ax.grid(axis="x", alpha=.22)
    fig.text(.5, .012,
        f"{BOOTSTRAP_REPLICATES:,} réplicas percentiles; 70 fórmulas remuestreadas con reemplazo; "
        "las tres semillas se promedian dentro de fórmula. IC95% sin ajuste por multiplicidad; análisis de desarrollo OOF, "
        "contrastes definidos después de observar el conjunto de condiciones, no prueba externa independiente.",
        ha="center", fontsize=8.1, wrap=True)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(IMG / "bootstrap_apareado_deltaE_f1_f2_vs_f0.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    """Convierte valores NumPy y valores no finitos a JSON estándar."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    """Valida datos completos, calcula métricas, genera PNG y registra recursos."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    cpu_started = time.process_time()
    IMG.mkdir(parents=True, exist_ok=True)
    structures, pairs, history, records = validate_inputs()
    comp, summary, gap, polymorph = calculate_metrics(structures, pairs)
    bootstrap = bootstrap_delta_contrasts(comp)
    comp.to_csv(EXP / "metricas_por_composicion_final.csv", index=False)
    summary.to_csv(EXP / "metricas_oof_finales.csv", index=False)
    gap.to_csv(EXP / "metricas_deltaE_por_gap.csv", index=False)
    polymorph.to_csv(EXP / "metricas_por_polimorfo.csv", index=False)
    bootstrap.to_csv(EXP / "bootstrap_apareado_final.csv", index=False)
    ols_energy = plot_parity(structures, pairs, "SLICES", "energy")
    ols_energy_geom = plot_parity(structures, pairs, "SLICES_GEOMETRY", "energy")
    ols_delta = plot_parity(structures, pairs, "SLICES", "delta")
    ols_delta_geom = plot_parity(structures, pairs, "SLICES_GEOMETRY", "delta")
    ols = pd.concat([ols_energy, ols_energy_geom, ols_delta, ols_delta_geom], ignore_index=True)
    ols.to_csv(EXP / "estadisticas_ols_paridad.csv", index=False)
    plot_metric_summary(summary)
    plot_ranking(summary)
    plot_learning_curves(history)
    plot_loss_curves(history)
    plot_gap_error(gap)
    plot_polymorph_errors(polymorph)
    plot_bootstrap(bootstrap)
    metrics = {r.condition: json_safe(r._asdict()) for r in summary.itertuples(index=False)}
    (EXP / "metrics.json").write_text(json.dumps({
        "status": "COMPLETE", "primary_metric": "MAE ΔE firmado intrafórmula, estimando A, meV/átomo",
        "evaluation": "OOF development; 70 formulas; five grouped outer folds; three seeds aggregated within formula",
        "historical_test_accessed": False, "models_modified": False,
        "pairwise_threshold_dft_mev_atom": PAIRWISE_DFT_THRESHOLD,
        "pairwise_unresolved_threshold_pred_mev_atom": PAIRWISE_PRED_THRESHOLD,
        "ground_state_eligibility_gap_mev_atom": GROUND_STATE_GAP_THRESHOLD,
        "metrics": metrics, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status = json.loads((EXP / "training_status.json").read_text(encoding="utf-8"))
    training_resources = status.get("resources", {})
    generated = sorted(p.name for p in IMG.glob("*.png"))
    figures_manifest = {
        "experiment": "abs3_original_supervised_ft_f0_f1_f2",
        "figures": generated,
        "source_predictions": "predictions/{structures,pairs}/*.csv",
        "source_training_history": "training_history_full.csv",
        "conditions": [f"{m}_{mod}_{pol}" for m in MODELS for mod in MODALITIES for pol in POLICIES],
        "oof": {"formulas": 70, "structures": 210, "outer_folds": 5, "seeds": [42, 123, 2026]},
        "test_accessed": False, "models_modified": False,
        "note": "F0=backbone congelado; F1=último bloque; F2=dos bloques finales. Sin Alexandria."}
    (EXP / "figures_manifest.json").write_text(
        json.dumps(figures_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    process_resources = {
        "analysis_status": "COMPLETE", "analysis_wall_seconds": time.perf_counter()-started,
        "analysis_process_cpu_seconds": time.process_time()-cpu_started,
        "analysis_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "analysis_device": "CPU", "analysis_threads": 1, "figure_count_png": len(generated),
        "training_run_resources": training_resources,
        "historical_test_accessed": False, "backbones_modified": False}
    (EXP / "figures_resources.json").write_text(
        json.dumps(json_safe(process_resources), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Figuras=%d; condiciones=%d; estructuras=%d; pares=%d; bootstrap=%d",
                len(generated), summary.shape[0], len(structures), len(pairs), BOOTSTRAP_REPLICATES)


if __name__ == "__main__":
    main()
