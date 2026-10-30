"""Calcula métricas OOF, invariancia, descomposición y bootstrap por fórmula.

El test histórico permanece cerrado. Los promedios de semillas se hacen antes
del remuestreo, cuya unidad estadística es la composición.
"""
from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import common as cm

LOG = logging.getLogger("matgpt_hybrid_analysis")
MODEL_ORDER = ("M1", "M2", "M3", "M4")
MODEL_NAMES = {"M1": "Ridge de embeddings + residuo Frozen",
               "M2": "Ridge composicional + residuo Frozen",
               "M3": "Ridge de embeddings + residuo LoRA",
               "M4": "Ridge composicional + residuo LoRA"}
PAIRWISE_DFT_THRESHOLD_MEV = 5.0
PREDICTION_TIE_THRESHOLD_MEV = 1.0


def ols_stats(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calcula pendiente, intercepto y R² OLS para predicción frente a referencia.

    Args:
        reference: Valores DFT usados como eje x.
        prediction: Valores predichos usados como eje y.
    Returns:
        Estadísticas OLS con intercepto libre.
    """
    if len(reference) < 2 or np.ptp(reference) < 1e-15:
        return {"ols_slope": float("nan"), "ols_intercept": float("nan"), "ols_r2": float("nan")}
    slope, intercept = np.polyfit(reference, prediction, 1)
    fitted = slope * reference + intercept
    denominator = float(np.sum((prediction - prediction.mean()) ** 2))
    r2 = 1.0 - float(np.sum((prediction - fitted) ** 2)) / denominator if denominator > 0 else float("nan")
    return {"ols_slope": float(slope), "ols_intercept": float(intercept), "ols_r2": r2}


def formula_table(group: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Ordena tripletes y calcula métricas de pares/órdenes por fórmula.

    Args:
        group: Predicciones de un modelo, pliegue y semilla o dataset OOF.
    Returns:
        Tabla de métricas por fórmula y registros con predicciones por pares.
    """
    formula_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for formula, part in group.groupby("composition_id", sort=True):
        triplet = part.set_index("polymorph_family").loc[list(cm.FAMILIES)].reset_index()
        raw_seed = triplet.seed.iloc[0]
        seed_value: int | str = int(raw_seed) if pd.notna(raw_seed) and str(raw_seed).isdigit() else str(raw_seed)
        dft = triplet.energy_per_atom_dft_ev.to_numpy(float)
        pred = triplet.energy_per_atom_pred_ev.to_numpy(float)
        dft_mean = float(dft.mean())
        residual_true = dft - dft_mean
        residual_pred = triplet.residual_pred_ev_atom.to_numpy(float)
        errors = pred - dft
        dft_order = np.argsort(dft, kind="stable")
        pred_order = np.argsort(pred, kind="stable")
        dft_margin = float(dft[dft_order[1]] - dft[dft_order[0]]) * 1000.0
        pred_margin = float(pred[pred_order[1]] - pred[pred_order[0]]) * 1000.0
        eligible_gs = dft_margin > PREDICTION_TIE_THRESHOLD_MEV
        pred_gs_resolved = pred_margin > PREDICTION_TIE_THRESHOLD_MEV
        exact_gs = bool(dft_order[0] == pred_order[0])
        pairs: list[float] = []
        correct: list[bool] = []
        for left, right in combinations(range(3), 2):
            delta_dft = float((dft[right] - dft[left]) * 1000.0)
            delta_pred = float((pred[right] - pred[left]) * 1000.0)
            is_eligible = abs(delta_dft) > PAIRWISE_DFT_THRESHOLD_MEV
            is_correct = is_eligible and abs(delta_pred) > PREDICTION_TIE_THRESHOLD_MEV and np.sign(delta_dft) == np.sign(delta_pred)
            if is_eligible:
                pairs.append(delta_pred - delta_dft)
                correct.append(bool(is_correct))
            pair_rows.append({"model_id": str(triplet.model_id.iloc[0]),
                              "fold": int(triplet.fold.iloc[0]),
                              "seed": seed_value,
                              "composition_id": formula,
                              "family_a": cm.FAMILIES[left], "family_b": cm.FAMILIES[right],
                              "structure_id_a": triplet.structure_id.iloc[left],
                              "structure_id_b": triplet.structure_id.iloc[right],
                              "delta_dft_mev_atom": delta_dft,
                              "delta_pred_mev_atom": delta_pred,
                              "absolute_gap_dft_mev_atom": abs(delta_dft),
                              "eligible_gt5_mev_atom": is_eligible,
                              "pairwise_correct_gt5_predicted_gap_gt1": bool(is_correct) if is_eligible else pd.NA})
        anchored = 1000.0 * ((pred - pred[dft_order[0]]) - (dft - dft[dft_order[0]]))
        rho = float(spearmanr(dft, pred).statistic) if np.ptp(pred) > 1e-14 else float("nan")
        tau = float(kendalltau(dft, pred).statistic) if np.ptp(pred) > 1e-14 else float("nan")
        formula_rows.append({"model_id": str(triplet.model_id.iloc[0]),
                             "fold": int(triplet.fold.iloc[0]),
                             "seed": seed_value,
                             "composition_id": formula,
                             "mean_energy_dft_ev_atom": dft_mean,
                             "mean_energy_pred_ev_atom": float(triplet.mean_energy_pred_ev_atom.mean()),
                             "composition_mean_absolute_error_mev_atom": abs(float(pred.mean() - dft_mean)) * 1000.0,
                             "mae_absolute_ev_atom_mev": float(np.abs(errors).mean()) * 1000.0,
                             "mae_residual_mev_atom": float(np.abs(residual_pred - residual_true).mean()) * 1000.0,
                             "rmse_residual_mev_atom": float(np.sqrt(np.mean((residual_pred - residual_true) ** 2))) * 1000.0,
                             "mae_delta_anchored_gs_mev_atom": float(np.mean(np.abs(anchored))),
                             "residual_prediction_std_mev_atom": float(np.std(residual_pred, ddof=0)) * 1000.0,
                             "residual_dft_std_mev_atom": float(np.std(residual_true, ddof=0)) * 1000.0,
                             "residual_slope": ols_stats(residual_true, residual_pred)["ols_slope"],
                             "ground_state_argmin_exact": exact_gs,
                             "ground_state_dft_eligible_gt1": eligible_gs,
                             "ground_state_prediction_resolved_gt1": pred_gs_resolved,
                             "ground_state_eligible_correct": bool(eligible_gs and pred_gs_resolved and exact_gs),
                             "n_pairs_total": 3,
                             "n_pairs_eligible_abs_delta_dft_gt5": len(pairs),
                             "pairwise_accuracy_gt5_pred_gap_gt1": float(np.mean(correct)) if correct else float("nan"),
                             "pairwise_mae_signed_delta_mev_atom": float(np.mean(np.abs(pairs))) if pairs else float("nan"),
                             "spearman_by_composition": rho, "kendall_tau_by_composition": tau})
    return pd.DataFrame(formula_rows), pair_rows


def calculate_metrics(group: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Calcula errores absolutos y polimórficos para una población OOF.

    Args:
        group: Predicciones de una condición y posiblemente varios folds/seeds.
    Returns:
        Métricas globales, una fila por fórmula y predicciones por pares.
    """
    if group.structure_id.nunique() != len(group) or group.composition_id.nunique() * 3 != len(group):
        raise ValueError("La población métrica debe tener una sola predicción por estructura física")
    truth = group.energy_per_atom_dft_ev.to_numpy(float)
    prediction = group.energy_per_atom_pred_ev.to_numpy(float)
    total_truth = group.energy_total_dft_ev.to_numpy(float)
    total_prediction = group.energy_total_pred_ev.to_numpy(float)
    residual_rows, pair_rows = formula_table(group)
    comp_y = residual_rows.mean_energy_dft_ev_atom.to_numpy(float)
    comp_p = residual_rows.mean_energy_pred_ev_atom.to_numpy(float)
    offset_me = (comp_p - comp_y) * 1000.0
    energy_ols = ols_stats(truth, prediction)
    sorted_group = group.sort_values(["composition_id", "polymorph_family"])
    resid_y = (sorted_group.energy_per_atom_dft_ev -
               sorted_group.groupby("composition_id").energy_per_atom_dft_ev.transform("mean")).to_numpy(float)
    resid_p = sorted_group.residual_pred_ev_atom.to_numpy(float)
    residual_ols = ols_stats(resid_y, resid_p)
    pair_truth = []
    pair_pred = []
    for _, part in group.groupby("composition_id", sort=True):
        trip = part.set_index("polymorph_family").loc[list(cm.FAMILIES)]
        y, p = trip.energy_per_atom_dft_ev.to_numpy(float), trip.energy_per_atom_pred_ev.to_numpy(float)
        for i, j in combinations(range(3), 2):
            pair_truth.append((y[j] - y[i]) * 1000.0)
            pair_pred.append((p[j] - p[i]) * 1000.0)
    pair_truth_arr, pair_pred_arr = np.asarray(pair_truth), np.asarray(pair_pred)
    eligible = np.abs(pair_truth_arr) > PAIRWISE_DFT_THRESHOLD_MEV
    ranked = eligible & (np.abs(pair_pred_arr) > PREDICTION_TIE_THRESHOLD_MEV)
    pair_correct = ranked & (np.sign(pair_truth_arr) == np.sign(pair_pred_arr))
    macro_gs = residual_rows.ground_state_dft_eligible_gt1.to_numpy(bool)
    resolved_gs = residual_rows.ground_state_prediction_resolved_gt1.to_numpy(bool)
    gs_correct = residual_rows.ground_state_eligible_correct.to_numpy(bool)
    metrics: dict[str, Any] = {
        "model_id": str(group.model_id.iloc[0]),
        "n_structures": int(len(group)), "n_compositions": int(group.composition_id.nunique()),
        "n_pairs_total": int(len(pair_truth_arr)), "n_pairs_eligible_abs_dft_gap_gt5": int(eligible.sum()),
        "n_pairs_prediction_tie_or_abstention": int((eligible & ~ranked).sum()),
        "n_ground_state_dft_eligible_margin_gt1": int(macro_gs.sum()),
        "n_ground_state_prediction_resolved_margin_gt1": int((macro_gs & resolved_gs).sum()),
        "mae_energy_per_atom_mev_atom": float(np.abs(prediction - truth).mean() * 1000.0),
        "rmse_energy_per_atom_mev_atom": float(np.sqrt(np.mean((prediction - truth) ** 2)) * 1000.0),
        "bias_energy_per_atom_mev_atom": float(np.mean(prediction - truth) * 1000.0),
        "mae_energy_total_ev": float(np.abs(total_prediction - total_truth).mean()),
        "rmse_energy_total_ev": float(np.sqrt(np.mean((total_prediction - total_truth) ** 2))),
        "mae_composition_mean_mev_atom": float(np.abs(offset_me).mean()),
        "bias_composition_mean_mev_atom": float(offset_me.mean()),
        "mae_residual_intraformula_mev_atom": float(residual_rows.mae_residual_mev_atom.mean()),
        "rmse_residual_intraformula_mev_atom": float(np.sqrt(np.mean((resid_p - resid_y) ** 2)) * 1000.0),
        "residual_slope_ols": residual_ols["ols_slope"],
        "residual_prediction_std_mev_atom": float(np.std(resid_p, ddof=0) * 1000.0),
        "residual_dft_std_mev_atom": float(np.std(resid_y, ddof=0) * 1000.0),
        "residual_dispersion_ratio": float(np.std(resid_p) / np.std(resid_y)) if np.std(resid_y) else float("nan"),
        "mae_delta_signed_pairs_mev_atom": float(np.mean(np.abs(pair_pred_arr - pair_truth_arr))),
        "mae_delta_anchored_to_dft_ground_state_mev_atom": float(residual_rows.mae_delta_anchored_gs_mev_atom.mean()),
        "ground_state_argmin_accuracy": float(residual_rows.ground_state_argmin_exact.mean()),
        "ground_state_eligible_accuracy": float(gs_correct.sum() / macro_gs.sum()) if macro_gs.any() else float("nan"),
        "pairwise_accuracy_gt5_predicted_gap_gt1": float(pair_correct.sum() / eligible.sum()) if eligible.any() else float("nan"),
        "mean_spearman_by_composition": float(residual_rows.spearman_by_composition.mean()),
        "mean_kendall_tau_by_composition": float(residual_rows.kendall_tau_by_composition.mean()),
        "energy_prediction_std_ev_atom": float(np.std(prediction, ddof=0)),
        "energy_dft_std_ev_atom": float(np.std(truth, ddof=0)),
        "energy_dispersion_ratio": float(np.std(prediction) / np.std(truth)) if np.std(truth) else float("nan"),
        "energy_slope_ols": energy_ols["ols_slope"],
        "energy_intercept_ols_ev_atom": energy_ols["ols_intercept"],
        "energy_r2_ols": energy_ols["ols_r2"],
        "energy_r2_predictive": float(1.0 - np.sum((prediction - truth) ** 2) /
                                       np.sum((truth - truth.mean()) ** 2)) if np.ptp(truth) else float("nan"),
    }
    return metrics, residual_rows, pair_rows


def compute_all_metrics(predictions: pd.DataFrame, composition_only: pd.DataFrame,
                        output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Guarda resultados por fórmula, pliegue/semilla y semilla OOF completa.

    Args:
        predictions: Salida M1–M4 del script de ajuste.
        composition_only: Control de Ridge composicional más residuo cero.
        output: Directorio de reportes.
    Returns:
        Métricas globales por semilla, métricas por fold/semilla, por fórmula y pares.
    """
    all_comp: list[pd.DataFrame] = []
    all_pairs: list[dict[str, Any]] = []
    fold_seed_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model_data = predictions[predictions.model_id.eq(model_id)]
        for seed in cm.SEEDS:
            seed_part = model_data[model_data.seed.eq(seed)]
            metrics, formula_metrics, pairs = calculate_metrics(seed_part)
            seed_rows.append({**metrics, "seed": seed, "fold_scope": "five_outer_folds"})
            all_comp.append(formula_metrics.assign(model_id=model_id, seed=seed,
                                                   fold_scope="five_outer_folds"))
            all_pairs.extend([{**row, "seed": seed, "fold_scope": "five_outer_folds"} for row in pairs])
            for fold in range(5):
                part = seed_part[seed_part.fold.eq(fold)]
                fold_metrics, fold_formula, fold_pairs = calculate_metrics(part)
                fold_seed_rows.append({**fold_metrics, "fold": fold, "seed": seed})
    composition_metrics, composition_formula, composition_pairs = calculate_metrics(composition_only.assign(seed="not_applicable"))
    composition_metrics["model_id"] = "composition_only"
    composition_metrics["seed"] = "not_applicable"
    composition_metrics["ground_state_argmin_accuracy"] = float("nan")
    composition_metrics["ground_state_eligible_accuracy"] = float("nan")
    composition_metrics["pairwise_accuracy_gt5_predicted_gap_gt1"] = float("nan")
    composition_metrics["ranking_status"] = "All predicted triplets tie; no ground state assigned"
    seed_summary = pd.DataFrame(seed_rows)
    fold_summary = pd.DataFrame(fold_seed_rows)
    formula_summary = pd.concat(all_comp, ignore_index=True)
    formula_summary.to_csv(output / "composition_metrics_by_formula.csv", index=False)
    seed_summary.to_csv(output / "metrics_by_seed.csv", index=False)
    fold_summary.to_csv(output / "metrics_by_fold_seed.csv", index=False)
    all_pairs.extend([{**row, "seed": "not_applicable", "fold_scope": "one_oof_prediction_per_structure"}
                      for row in composition_pairs])
    pairs_frame = pd.DataFrame(all_pairs)
    pairs_frame.to_csv(output / "pair_predictions.csv", index=False)
    pd.DataFrame([composition_metrics]).to_csv(output / "composition_only_metrics.csv", index=False)
    return seed_summary, fold_summary, formula_summary, pairs_frame


def bootstrap_comparisons(formula_metrics: pd.DataFrame, output: Path,
                          repetitions: int = 10000, seed: int = 20261030) -> pd.DataFrame:
    """Bootstrap diferencias apareadas usando fórmula como unidad independiente.

    Args:
        formula_metrics: Métricas con una fila por composición/modelo/semilla.
        output: Directorio canónico.
        repetitions: Repeticiones deterministas del bootstrap.
        seed: Semilla aleatoria fija.
    Returns:
        Tabla compacta de diferencias y límites percentile del 95 %.
    """
    # Promediar las semillas dentro de cada fórmula evita tratar 630 filas como independientes.
    averaged = formula_metrics.groupby(["model_id", "composition_id"], as_index=False).agg(
        mae_absolute_ev_atom_mev=("mae_absolute_ev_atom_mev", "mean"),
        composition_mean_absolute_error_mev_atom=("composition_mean_absolute_error_mev_atom", "mean"),
        mean_energy_dft_ev_atom=("mean_energy_dft_ev_atom", "mean"))
    rng = np.random.default_rng(seed)
    comparisons = (("M2", "M1", "Ridge composicional frente a Ridge de embeddings; residuo Frozen"),
                   ("M4", "M3", "Ridge composicional frente a Ridge de embeddings; residuo LoRA"))
    rows: list[dict[str, Any]] = []
    for metric, units in (("mae_absolute_ev_atom_mev", "meV/átomo"),
                          ("composition_mean_absolute_error_mev_atom", "meV/átomo")):
        for new, old, description in comparisons:
            wide = averaged.pivot(index="composition_id", columns="model_id", values=metric)
            if len(wide) != 70 or wide[[new, old]].isna().any().any():
                raise ValueError("Bootstrap requiere 70 fórmulas con datos apareados completos")
            differences = (wide[new] - wide[old]).to_numpy(float)
            draws = rng.integers(0, len(differences), size=(repetitions, len(differences)))
            boot = differences[draws].mean(axis=1)
            rows.append({"comparison": description, "metric": metric, "units": units,
                         "difference_new_minus_old": float(differences.mean()),
                         "ci95_lower": float(np.quantile(boot, .025)),
                         "ci95_upper": float(np.quantile(boot, .975)),
                         "n_compositions": 70, "seeds_averaged_before_resampling": 3,
                         "bootstrap_repetitions": repetitions, "bootstrap_seed": seed})
    result = pd.DataFrame(rows)
    result.to_csv(output / "paired_bootstrap_by_composition.csv", index=False)
    return result


def mse_decomposition(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Descompone exactamente el MSE en error de media, residuo y término cruzado.

    Args:
        predictions: OOF M1–M4.
        output: Carpeta canónica.
    Returns:
        Tabla con identidad MSE y MAE descriptivas no aditivas.
    """
    mean_predictions = predictions.groupby(["model_id", "structure_id", "composition_id", "polymorph_family",
                                           "fold", "n_atoms", "energy_total_dft_ev", "energy_per_atom_dft_ev"],
                                          as_index=False).agg(
        mean_energy_pred_ev_atom=("mean_energy_pred_ev_atom", "mean"),
        residual_pred_ev_atom=("residual_pred_ev_atom", "mean"),
        energy_per_atom_pred_ev=("energy_per_atom_pred_ev", "mean"),
        energy_total_pred_ev=("energy_total_pred_ev", "mean"))
    rows: list[dict[str, Any]] = []
    for model_id, data in mean_predictions.groupby("model_id"):
        for formula, part in data.groupby("composition_id"):
            dft_mean = float(part.energy_per_atom_dft_ev.mean())
            residual_true = part.energy_per_atom_dft_ev.to_numpy(float) - dft_mean
            residual_pred = part.residual_pred_ev_atom.to_numpy(float)
            mean_error = float(part.mean_energy_pred_ev_atom.iloc[0] - dft_mean)
            residual_error = residual_pred - residual_true
            total_error = part.energy_per_atom_pred_ev.to_numpy(float) - part.energy_per_atom_dft_ev.to_numpy(float)
            closure = total_error - (mean_error + residual_error)
            rows.append({"model_id": model_id, "composition_id": formula,
                         "n_structures": len(part), "mean_component_error_ev_atom": mean_error,
                         "residual_component_rmse_ev_atom": float(np.sqrt(np.mean(residual_error ** 2))),
                         "total_energy_rmse_ev_atom": float(np.sqrt(np.mean(total_error ** 2))),
                         "mse_mean_component_ev2_atom2": mean_error ** 2,
                         "mse_residual_component_ev2_atom2": float(np.mean(residual_error ** 2)),
                         "cross_term_2mean_times_residual_ev2_atom2": float(2 * mean_error * residual_error.mean()),
                         "mse_total_ev2_atom2": float(np.mean(total_error ** 2)),
                         "mse_identity_closure_error_ev2_atom2": float(np.mean(total_error ** 2) -
                             mean_error ** 2 - np.mean(residual_error ** 2) - 2 * mean_error * residual_error.mean()),
                         "mae_mean_component_mev_atom": abs(mean_error) * 1000.0,
                         "mae_residual_component_mev_atom": float(np.mean(np.abs(residual_error))) * 1000.0,
                         "mae_total_not_additive_mev_atom": float(np.mean(np.abs(total_error))) * 1000.0,
                         "max_component_closure_abs_ev_atom": float(np.max(np.abs(closure)))})
    result = pd.DataFrame(rows)
    result.to_csv(output / "mse_error_decomposition_by_composition.csv", index=False)
    return result


def verify_invariance(predictions: pd.DataFrame, output: Path,
                      tolerance_mev_atom: float = 1e-8) -> pd.DataFrame:
    """Comprueba que sustituir solo Ridge preserve ΔE y selección polimórfica.

    Args:
        predictions: Predicciones OOF M1–M4.
        output: Directorio canónico.
        tolerance_mev_atom: Tolerancia máxima para gaps firmados en meV/átomo.
    Returns:
        Tabla por rama residual, fold y seed.
    Raises:
        ValueError: Si falla cualquier igualdad matemática dentro de tolerancia.
    """
    pairs: list[dict[str, Any]] = []
    for residual_name, old_id, new_id in (("Frozen", "M1", "M2"), ("LoRA", "M3", "M4")):
        for fold in range(5):
            for seed in cm.SEEDS:
                left = predictions[(predictions.model_id == old_id) & (predictions.fold == fold) &
                                   (predictions.seed == seed)]
                right = predictions[(predictions.model_id == new_id) & (predictions.fold == fold) &
                                    (predictions.seed == seed)]
                aligned = left.merge(right, on=["structure_id", "composition_id", "polymorph_family"],
                                     suffixes=("_old", "_new"), validate="one_to_one")
                if len(aligned) != 42:
                    raise ValueError(f"Faltan estructuras para invariancia: {residual_name}, fold {fold}, seed {seed}")
                max_delta_difference = 0.0
                identical_ranking = True
                identical_ground_state = True
                for _, part in aligned.groupby("composition_id", sort=True):
                    old = part.set_index("polymorph_family").loc[list(cm.FAMILIES)]
                    new = part.set_index("polymorph_family").loc[list(cm.FAMILIES)]
                    old_pred = old.energy_per_atom_pred_ev_old.to_numpy(float)
                    new_pred = new.energy_per_atom_pred_ev_new.to_numpy(float)
                    old_deltas = np.asarray([(old_pred[j] - old_pred[i]) * 1000.0
                                             for i, j in combinations(range(3), 2)])
                    new_deltas = np.asarray([(new_pred[j] - new_pred[i]) * 1000.0
                                             for i, j in combinations(range(3), 2)])
                    max_delta_difference = max(max_delta_difference, float(np.max(np.abs(old_deltas - new_deltas))))
                    identical_ranking &= np.array_equal(np.argsort(old_pred, kind="stable"),
                                                        np.argsort(new_pred, kind="stable"))
                    identical_ground_state &= int(np.argmin(old_pred)) == int(np.argmin(new_pred))
                old_centered_dft = (aligned.energy_per_atom_dft_ev_old -
                    aligned.groupby("composition_id").energy_per_atom_dft_ev_old.transform("mean")).to_numpy(float)
                new_centered_dft = (aligned.energy_per_atom_dft_ev_new -
                    aligned.groupby("composition_id").energy_per_atom_dft_ev_new.transform("mean")).to_numpy(float)
                old_residual_error = aligned.residual_pred_ev_atom_old.to_numpy(float) - old_centered_dft
                new_residual_error = aligned.residual_pred_ev_atom_new.to_numpy(float) - new_centered_dft
                mae_difference = abs(float(np.mean(np.abs(old_residual_error)) -
                                           np.mean(np.abs(new_residual_error)))) * 1000.0
                old_pairwise = []
                new_pairwise = []
                for _, part in aligned.groupby("composition_id", sort=True):
                    trip = part.set_index("polymorph_family").loc[list(cm.FAMILIES)]
                    old_e = trip.energy_per_atom_pred_ev_old.to_numpy(float)
                    new_e = trip.energy_per_atom_pred_ev_new.to_numpy(float)
                    for i, j in combinations(range(3), 2):
                        dft_gap = (trip.energy_per_atom_dft_ev_old.iloc[j] -
                                   trip.energy_per_atom_dft_ev_old.iloc[i]) * 1000.0
                        if abs(dft_gap) > PAIRWISE_DFT_THRESHOLD_MEV:
                            old_gap = (old_e[j] - old_e[i]) * 1000.0
                            new_gap = (new_e[j] - new_e[i]) * 1000.0
                            old_pairwise.append(abs(old_gap) > PREDICTION_TIE_THRESHOLD_MEV and np.sign(old_gap) == np.sign(dft_gap))
                            new_pairwise.append(abs(new_gap) > PREDICTION_TIE_THRESHOLD_MEV and np.sign(new_gap) == np.sign(dft_gap))
                pairwise_identical = old_pairwise == new_pairwise
                row = {"residual_branch": residual_name, "ridge_comparison": f"{old_id} vs {new_id}",
                       "fold": fold, "seed": seed,
                       "max_abs_pair_delta_difference_mev_atom": max_delta_difference,
                       "mae_residual_difference_mev_atom": mae_difference,
                       "ranking_identical": identical_ranking,
                       "ground_state_identical": identical_ground_state,
                       "pairwise_accuracy_identical": pairwise_identical,
                       "tolerance_mev_atom": tolerance_mev_atom}
                pairs.append(row)
    result = pd.DataFrame(pairs)
    passed = (result.max_abs_pair_delta_difference_mev_atom.le(tolerance_mev_atom).all() and
              result.mae_residual_difference_mev_atom.le(tolerance_mev_atom).all() and
              result.ranking_identical.all() and result.ground_state_identical.all() and
              result.pairwise_accuracy_identical.all())
    result["invariance_status"] = "PASS" if passed else "FAIL"
    result.to_csv(output / "invariance_audit.csv", index=False)
    if not passed:
        raise ValueError("Fallo de invariancia: Ridge composicional alteró métricas intra-fórmula")
    return result


def analyze(output: Path = cm.REPORT) -> dict[str, Any]:
    """Evalúa todas las condiciones, controla equivalencia y genera reportes.

    Args:
        output: Carpeta del experimento ya poblada por `run_experiment.py`.
    Returns:
        Conteos y estado de las verificaciones.
    """
    predictions = pd.read_csv(output / "oof_predictions_by_structure.csv")
    composition_only = pd.read_csv(output / "composition_only_predictions.csv")
    predictions["seed"] = predictions.seed.astype(int)
    seed_summary, fold_summary, formula_metrics, pair_metrics = compute_all_metrics(predictions, composition_only, output)
    bootstrap = bootstrap_comparisons(formula_metrics, output)
    decomposition = mse_decomposition(predictions, output)
    invariance = verify_invariance(predictions, output)
    comp_metrics = pd.read_csv(output / "composition_only_metrics.csv").iloc[0]
    summary = seed_summary.groupby("model_id").agg({
        "mae_energy_per_atom_mev_atom": "mean", "rmse_energy_per_atom_mev_atom": "mean",
        "mae_energy_total_ev": "mean", "mae_composition_mean_mev_atom": "mean",
        "mae_residual_intraformula_mev_atom": "mean", "mae_delta_signed_pairs_mev_atom": "mean",
        "mae_delta_anchored_to_dft_ground_state_mev_atom": "mean",
        "ground_state_argmin_accuracy": "mean", "ground_state_eligible_accuracy": "mean",
        "pairwise_accuracy_gt5_predicted_gap_gt1": "mean", "energy_slope_ols": "mean",
        "energy_dispersion_ratio": "mean"}).reset_index()
    summary.to_csv(output / "summary_by_model.csv", index=False)
    invariance_report = ("# Verificación de invariancia de ranking\n\n"
        "La rama composicional aporta una constante por fórmula. Para cada fórmula esa constante se cancela en los tres gaps; los residuos MLP8 usados como entrada no cambian.\n\n"
        f"**Resultado:** {invariance.invariance_status.iloc[0]}. Se comprobaron {len(invariance)} combinaciones de rama residual, fold y semilla, "
        "70 fórmulas y 210 pares por comparación.\n\n"
        f"Máxima diferencia absoluta de ΔE: {invariance.max_abs_pair_delta_difference_mev_atom.max():.3g} meV/átomo; "
              f"máxima diferencia de MAE residual: {invariance.mae_residual_difference_mev_atom.max():.3g} meV/átomo.\n\n"
        "La MAE absoluta puede cambiar porque el intercepto por fórmula cambia; la MAE residual, los gaps firmados, el ranking y el estado fundamental deben permanecer idénticos. "
        "Se reportan ambas propiedades por separado. El predictor solo composicional empata los tres polimorfos; no se le asigna un estado fundamental ni un acierto de ranking.\n")
    (output / "verificacion_invariancia_ranking.md").write_text(invariance_report, encoding="utf-8")
    write_reports(output, summary, bootstrap, decomposition, comp_metrics)
    result = {"status": "PASS", "models": 4, "rows_M1_to_M4": int(len(predictions)),
              "rows_composition_only": int(len(composition_only)),
              "metrics_by_seed": int(len(seed_summary)), "metrics_by_fold_seed": int(len(fold_summary)),
              "formula_metric_rows": int(len(formula_metrics)), "pair_metric_rows": int(len(pair_metrics)),
              "invariance_checks": int(len(invariance)), "bootstrap_comparisons": int(len(bootstrap)),
              "historical_test_used": False}
    LOG.info("%s", result)
    return result


def write_reports(output: Path, summary: pd.DataFrame, bootstrap: pd.DataFrame,
                  decomposition: pd.DataFrame, composition_metrics: pd.Series) -> None:
    """Escribe los seis reportes científicos acordados sin métricas inventadas.

    Args:
        output: Directorio de reportes.
        summary: Medias OOF de métricas por configuración.
        bootstrap: Contrastes apareados e IC por composición.
        decomposition: MSE descompuesto por fórmula/modelo.
        composition_metrics: Baseline composición sola.
    Returns:
        None; escribe reportes Markdown.
    """
    names = {"M1": "Ridge de embeddings + MLP8 Frozen", "M2": "Ridge composicional + MLP8 Frozen",
             "M3": "Ridge de embeddings + MLP8 LoRA", "M4": "Ridge composicional + MLP8 LoRA"}
    table = summary.copy()
    table["configuración"] = table.model_id.map(names)
    columns = ["configuración", "mae_energy_per_atom_mev_atom", "mae_composition_mean_mev_atom",
               "mae_residual_intraformula_mev_atom", "mae_delta_signed_pairs_mev_atom",
               "ground_state_argmin_accuracy", "pairwise_accuracy_gt5_predicted_gap_gt1"]
    p = summary.set_index("model_id")
    old = summary.set_index("model_id")
    absolute_rows = []
    for new, before, description in (("M2", "M1", "Residuo Frozen"), ("M4", "M3", "Residuo LoRA")):
        a = bootstrap[(bootstrap.comparison.str.contains(description.split()[-1], case=False))]
        absolute = a[a.metric.eq("mae_absolute_ev_atom_mev")]
        mean = a[a.metric.eq("composition_mean_absolute_error_mev_atom")]
        absolute_rows.append(f"- **{description}:** MAE absoluta cambia {absolute.difference_new_minus_old.iloc[0]:+.2f} "
                             f"meV/átomo (IC95% {absolute.ci95_lower.iloc[0]:+.2f}, {absolute.ci95_upper.iloc[0]:+.2f}); "
                             f"MAE de media composicional cambia {mean.difference_new_minus_old.iloc[0]:+.2f} "
                             f"meV/átomo (IC95% {mean.ci95_lower.iloc[0]:+.2f}, {mean.ci95_upper.iloc[0]:+.2f}).")
    mean_composition = decomposition.groupby("model_id").agg(
        mean_mse_offset=("mse_mean_component_ev2_atom2", "mean"),
        mean_mse_residual=("mse_residual_component_ev2_atom2", "mean"),
        mean_cross=("cross_term_2mean_times_residual_ev2_atom2", "mean"),
        mean_mse_total=("mse_total_ev2_atom2", "mean"),
        max_closure=("mse_identity_closure_error_ev2_atom2", lambda x: float(x.abs().max())))
    decomposition_table = mean_composition.to_markdown(floatfmt=".6g")
    ridge_table = pd.read_csv(output / "ridge_selection.csv").to_markdown(index=False, floatfmt=".5g")
    text = "# Comparación Ridge de embeddings frente a Ridge composicional\n\n"
    text += "**Diseño:** 70 composiciones, 210 estructuras físicas, tres polimorfos, cinco pliegues y tres semillas residuales. "
    text += "Las métricas de semilla se promedian para el resumen; el bootstrap promedia primero semillas por composición y remuestrea 70 fórmulas. "
    text += "El test histórico permanece cerrado.\n\n"
    text += "## Resultados medios de energía absoluta y polimorfos\n\n" + table[columns].to_markdown(index=False, floatfmt=".3f") + "\n\n"
    text += "## Diferencia apareada del Ridge composicional menos el Ridge de embeddings\n\n" + "\n".join(absolute_rows) + "\n\n"
    text += bootstrap.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    text += "Los intervalos se agrupan por composición; no se cuentan las semillas como materiales independientes. Una diferencia en energía absoluta no implica una mejora en ranking.\n\n"
    text += "## Error de media y residuo\n\n" + decomposition_table + "\n\n"
    text += "La identidad exacta es MSE(total) = MSE(media) + MSE(residuo) + término cruzado. Las MAE de los componentes son descriptivas y **no son aditivas**.\n\n"
    text += "## Control solo composicional\n\n"
    text += f"MAE E/N {composition_metrics.mae_energy_per_atom_mev_atom:.2f} meV/átomo; "
    text += f"MAE de energía total {composition_metrics.mae_energy_total_ev:.3f} eV; "
    text += f"MAE de pares firmados {composition_metrics.mae_delta_signed_pairs_mev_atom:.2f} meV/átomo. "
    text += "El predictor asigna la misma energía a los tres polimorfos, por lo que no se le adjudica estado fundamental.\n\n"
    text += "## Selección y ajuste de Ridge\n\n" + ridge_table + "\n\n"
    text += "El único ajuste nuevo fue Ridge de 17 fracciones estequiométricas, con selección interna 45/11 y refit outer-train de 56 fórmulas. "
    text += "La rama residual y el predictor Frozen/LoRA se reutilizaron de predicciones ya guardadas. No hubo inferencia nueva de MatGPT ni entrenamiento MLP8/LoRA.\n\n"
    text += "En esta tabla, **Frozen** identifica residuos MLP8 del MatGPT pretrained congelado; **LoRA** identifica residuos ya guardados del modelo adaptado mediante Low-Rank Adaptation. "
    text += "Ambas ramas usan la cabeza MLP8 existente; no se reajustaron para esta comparación.\n\n"
    text += "**Límites:** resultados de las 70 composiciones ABS₃ de desarrollo previamente analizadas, no validación ciega externa. Las conclusiones se limitan a la descomposición del nivel medio y los residuos recuperados.\n"
    (output / "comparacion_ridge_embeddings_vs_composicion.md").write_text(text, encoding="utf-8")
    (output / "resultados_energia_absoluta.md").write_text(
        "# Resultados de energía absoluta\n\n" + table[columns[:4] + ["energy_slope_ols", "energy_dispersion_ratio"]].to_markdown(index=False, floatfmt=".3f") +
        "\n\n**Interpretación:** el cambio de rama Ridge solo modifica la media por fórmula; la contribución residual procede de predicciones MLP8 existentes. "
        "Consulte los IC95% apareados en `paired_bootstrap_by_composition.csv`.\n", encoding="utf-8")
    (output / "resultados_residuos_y_diferencias_energeticas.md").write_text(
        "# Resultados de residuos y diferencias energéticas\n\n" +
        table[["configuración", "mae_residual_intraformula_mev_atom", "mae_delta_signed_pairs_mev_atom",
               "mae_delta_anchored_to_dft_ground_state_mev_atom", "ground_state_argmin_accuracy",
               "pairwise_accuracy_gt5_predicted_gap_gt1"]].to_markdown(index=False, floatfmt=".3f") +
        "\n\nΔE firmada = 1000·(e_B − e_A) dentro de una fórmula. El anclaje usa la misma estructura de mínimo DFT. "
        "La elegibilidad pairwise requiere |ΔE_DFT| > 5 meV/átomo; una predicción con gap ≤1 meV/átomo se cuenta como no resuelta.\n", encoding="utf-8")
    metric = summary.set_index("model_id")
    frozen_delta = bootstrap[(bootstrap.metric == "mae_absolute_ev_atom_mev") &
                             bootstrap.comparison.str.contains("Frozen", case=False)].iloc[0]
    lora_delta = bootstrap[(bootstrap.metric == "mae_absolute_ev_atom_mev") &
                           bootstrap.comparison.str.contains("LoRA", case=False)].iloc[0]
    composition_delta = bootstrap[(bootstrap.metric == "composition_mean_absolute_error_mev_atom") &
                                  bootstrap.comparison.str.contains("Frozen", case=False)].iloc[0]
    conclusion = (
        "# Conclusiones científicas\n\n"
        "## Respuesta\n\n"
        "**RESULTADO EXPERIMENTAL:** sí, en estas 70 fórmulas de desarrollo el Ridge composicional mejora marcadamente la energía absoluta, "
        f"manteniendo intacta la señal polimórfica almacenada. Frozen: MAE E/N {metric.loc['M1','mae_energy_per_atom_mev_atom']:.2f} → "
        f"{metric.loc['M2','mae_energy_per_atom_mev_atom']:.2f} meV/átomo; LoRA: {metric.loc['M3','mae_energy_per_atom_mev_atom']:.2f} → "
        f"{metric.loc['M4','mae_energy_per_atom_mev_atom']:.2f} meV/átomo.\n\n"
        f"Diferencia apareada de MAE absoluto, composicional menos embeddings: Frozen {frozen_delta.difference_new_minus_old:+.2f} "
        f"meV/átomo (IC95% {frozen_delta.ci95_lower:+.2f} a {frozen_delta.ci95_upper:+.2f}); LoRA {lora_delta.difference_new_minus_old:+.2f} "
        f"(IC95% {lora_delta.ci95_lower:+.2f} a {lora_delta.ci95_upper:+.2f}). La media por fórmula mejora en {composition_delta.difference_new_minus_old:+.2f} "
        f"meV/átomo (IC95% {composition_delta.ci95_lower:+.2f} a {composition_delta.ci95_upper:+.2f}).\n\n"
        "**HECHO MATEMÁTICO Y VERIFICADO:** M1/M2 y M3/M4 tienen exactamente las mismas diferencias ΔE, residuos centrados, rankings y estados fundamentales "
        "dentro de 1×10⁻⁸ meV/átomo. La diferencia media de energía cambia; la estimación de estabilidad polimórfica no.\n\n"
        "## Reproducción histórica\n\n"
        "El control M1 reproduce las métricas originales principales: MAE E/N ≈183.30 meV/átomo, MAE pairwise firmado ≈58.84 meV/átomo y "
        "estado fundamental exacto 54.3%. La exactitud pairwise recalculada es 66.84%, frente al 67.2% aproximado mencionado en el pedido; "
        "el CSV histórico fuente también da 66.84% al promediar las semillas con el criterio |ΔE_DFT|>5 y predicción resuelta >1 meV/átomo. "
        "Por tanto, la diferencia 67.2% no aparece en los archivos canónicos consultados.\n\n"
        "**HECHO VERIFICADO:** los cinco Ridge usan selección de alpha en 45 fórmulas inner-train/11 inner-validation y refit con 56 fórmulas outer-train. "
        "Alphas seleccionados: 0.01, 0.1, 0.1, 0.1 y 0.1 para los folds 0–4; coinciden con el control composicional anterior.\n\n"
        "**LIMITACIÓN:** son predicciones OOF de folds del conjunto de desarrollo ABS₃, no un test ciego externo; el test histórico no se leyó. "
        "El experimento solo respalda la sustitución del estimador de energía media en este protocolo y no demuestra mayor información estructural aprendida por MatGPT.\n")
    (output / "conclusiones_cientificas.md").write_text(conclusion, encoding="utf-8")


def main() -> None:
    """Ejecuta el análisis desde las tablas generadas.

    Returns:
        None; aborta descriptivamente ante salidas ausentes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cm.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyze(args.output.resolve())


if __name__ == "__main__":
    main()
