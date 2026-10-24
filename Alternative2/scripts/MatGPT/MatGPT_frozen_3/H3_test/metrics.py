#!/usr/bin/env python3
"""Métricas energéticas y polimórficas para el ensemble H3-TEST bloqueado."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import FAMILIES, H3TestError, TAU_MEV


def _classify_difference(value_mev: float, tolerance: float = TAU_MEV) -> str:
    """Clasifica E_i−E_j con orientación canónica i_lower/tie/j_lower."""
    if value_mev < -tolerance:
        return "i_lower"
    if value_mev > tolerance:
        return "j_lower"
    return "tie"


def _predicted_gs(values: np.ndarray, tolerance: float = TAU_MEV) -> str:
    """Devuelve familia mínima o tie/abstain si hay predicciones casi iguales."""
    minimum = float(values.min())
    near = np.flatnonzero((values - minimum) * 1000.0 <= tolerance)
    return FAMILIES[int(near[0])] if len(near) == 1 else "tie/abstain"


def _r2_predict(y: np.ndarray, prediction: np.ndarray) -> float:
    """Calcula R² predictivo contra y=x, sin ajustar una regresión."""
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator) if denominator > 0 else float("nan")


def _ols(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    """Ajusta OLS descriptivo libre y=m*x+b sobre puntos mostrados."""
    if len(y) < 2 or np.ptp(y) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(y, prediction, 1)
    fitted = slope * y + intercept
    denominator = float(np.sum((prediction - prediction.mean()) ** 2))
    r2 = float(1.0 - np.sum((prediction - fitted) ** 2) / denominator) if denominator > 0 else float("nan")
    return float(slope), float(intercept), r2


def composition_metrics(group: pd.DataFrame, prediction_column: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calcula errores y denominadores de una composición completa.

    Args:
        group: Tres filas de una fórmula, ordenadas por familia.
        prediction_column: Columna de energía predicha en eV/átomo.
    Returns:
        Métricas por fórmula y tres diagnósticos de pareja orientada.
    """
    group = group.set_index("polymorph_family").loc[list(FAMILIES)].reset_index()
    y = group.E_DFT_eV_atom.to_numpy(float)
    pred = group[prediction_column].to_numpy(float)
    formula = str(group.composition_id.iloc[0])
    if len(y) != 3 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise H3TestError(f"Energías incompletas/no finitas para {formula}.")
    mu_y, mu_pred = float(y.mean()), float(pred.mean())
    dft_resid, pred_resid = y - mu_y, pred - mu_pred
    dft_order = np.argsort(y, kind="stable")
    dft_gap_mev = float((y[dft_order[1]] - y[dft_order[0]]) * 1000.0)
    dft_gs = FAMILIES[int(dft_order[0])]
    pred_gs = _predicted_gs(pred)
    eligible_gs = dft_gap_mev > TAU_MEV
    dft_anchor = (y - y[dft_order[0]]) * 1000.0
    pred_anchor = (pred - pred[dft_order[0]]) * 1000.0
    dft_own = (y - y.min()) * 1000.0
    pred_own = (pred - pred.min()) * 1000.0
    pairs = []
    pair_correct = 0
    pair_eligible = 0
    pair_signed_errors = []
    for i, j in combinations(range(3), 2):
        diff_dft = float((y[i] - y[j]) * 1000.0)
        diff_pred = float((pred[i] - pred[j]) * 1000.0)
        class_dft = _classify_difference(diff_dft)
        class_pred = _classify_difference(diff_pred)
        eligible = class_dft != "tie"
        correct = bool(eligible and class_pred == class_dft)
        pair_correct += int(correct)
        pair_eligible += int(eligible)
        if eligible:
            pair_signed_errors.append(diff_pred - diff_dft)
        record = {"composition_id": formula, "family_i": FAMILIES[i], "family_j": FAMILIES[j],
                  "DFT_difference_Ei_minus_Ej_mev_atom": diff_dft,
                  "pred_difference_Ei_minus_Ej_mev_atom": diff_pred,
                  "DFT_class": class_dft, "predicted_class": class_pred,
                  "DFT_pair_eligible": eligible, "pair_correct": correct}
        pairs.append(record)
    m_total = float(np.mean((pred - y) ** 2) * 1e6)
    m_mu = float((mu_pred - mu_y) ** 2 * 1e6)
    m_resid = float(np.mean((pred_resid - dft_resid) ** 2) * 1e6)
    row = {"composition_id": formula, "MAE_absolute_mev_atom": float(np.mean(np.abs(pred-y))*1000),
           "RMSE_absolute_mev_atom": float(np.sqrt(np.mean((pred-y)**2))*1000),
           "bias_mev_atom": float(np.mean(pred-y)*1000), "MAE_mu_mev_atom": abs(mu_pred-mu_y)*1000,
           "MAE_delta_centered_mev_atom": float(np.mean(np.abs(pred_resid-dft_resid))*1000),
           "MAE_delta_GS_anchored_mev_atom": float(np.mean(np.abs(pred_anchor-dft_anchor))),
           "MAE_delta_own_min_mev_atom": float(np.mean(np.abs(pred_own-dft_own))),
           "MSE_total_mev2_atom2": m_total, "MSE_mu_mev2_atom2": m_mu,
           "MSE_residual_mev2_atom2": m_resid, "MSE_identity_abs_error": abs(m_total-m_mu-m_resid),
           "DFT_GS_family": dft_gs, "predicted_GS_family": pred_gs,
           "DFT_GS_gap_mev_atom": dft_gap_mev, "GS_eligible": eligible_gs,
           "GS_correct": bool(eligible_gs and pred_gs == dft_gs),
           "GS_predicted_abstention": pred_gs == "tie/abstain",
           "pairwise_correct_count": pair_correct, "pairwise_eligible_count": pair_eligible,
           "pairwise_accuracy": pair_correct/pair_eligible if pair_eligible else float("nan"),
           "pairwise_abs_signed_error_mev_atom": float(np.mean(np.abs(pair_signed_errors))) if pair_signed_errors else float("nan"),
           "pairwise_mean_signed_error_mev_atom": float(np.mean(pair_signed_errors)) if pair_signed_errors else float("nan"),
           "DFT_range_mev_atom": float((y.max()-y.min())*1000),
           "predicted_range_mev_atom": float((pred.max()-pred.min())*1000),
           "absolute_range_error_mev_atom": abs(float((pred.max()-pred.min())-(y.max()-y.min())))*1000}
    return row, pairs


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Calcula intervalo Wilson 95% para un conteo binomial pequeño.

    Args:
        successes: Aciertos observados.
        total: Denominador elegible.
        z: Cuantil normal correspondiente a 95%.
    Returns:
        Límites inferior y superior en [0,1], o NaN si no hay casos.
    """
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denom = 1 + z*z/total
    center = (proportion + z*z/(2*total)) / denom
    half = z * np.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total)) / denom
    return float(max(0, center-half)), float(min(1, center+half))


def summarize_model(frame: pd.DataFrame, prediction_column: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Agrega métricas físicas, macro por fórmula y matriz pairwise.

    Args:
        frame: 24 energías DFT y predicciones de un único modelo.
        prediction_column: Nombre de predicción.
    Returns:
        Resumen global, tabla por composición y tabla con 24 posibles parejas.
    """
    rows, pairs = [], []
    for _, group in frame.groupby("composition_id", sort=True):
        if len(group) != 3:
            raise H3TestError("Se necesita triplete por fórmula para métricas.")
        row, pair_rows = composition_metrics(group, prediction_column)
        rows.append(row); pairs.extend(pair_rows)
    comp = pd.DataFrame(rows)
    pair_frame = pd.DataFrame(pairs)
    y = frame.E_DFT_eV_atom.to_numpy(float)
    pred = frame[prediction_column].to_numpy(float)
    slope, intercept, r2_ols = _ols(y*1000, pred*1000)
    gs_eligible = comp.loc[comp.GS_eligible]
    pair_eligible = pair_frame.loc[pair_frame.DFT_pair_eligible]
    gs_correct, n_gs = int(gs_eligible.GS_correct.sum()), len(gs_eligible)
    pair_correct, n_pairs = int(pair_eligible.pair_correct.sum()), len(pair_eligible)
    gs_ci = wilson_interval(gs_correct, n_gs)
    pair_ci = wilson_interval(pair_correct, n_pairs)
    # Multiclass diagnostics include DFT-tied GS as a class only when such a tie occurs.
    gs_labels, gs_preds = [], []
    for row in comp.itertuples(index=False):
        if row.GS_eligible:
            gs_labels.append(row.DFT_GS_family); gs_preds.append(row.predicted_GS_family)
        else:
            gs_labels.append("DFT_tie_excluded"); gs_preds.append(row.predicted_GS_family)
    gs_classes = list(FAMILIES) + ["tie/abstain"]
    gs_confusion = [[int(sum((truth == family) and (predicted == column) for truth, predicted in zip(gs_labels, gs_preds)))
                     for column in gs_classes] for family in FAMILIES]
    pair_classes = ["i_lower", "tie", "j_lower"]
    pair_confusion = [[int(((pair_frame.DFT_class == truth) & (pair_frame.predicted_class == pred_class)).sum())
                       for pred_class in pair_classes] for truth in pair_classes]
    r2_pred = _r2_predict(y*1000, pred*1000)
    total_mse = float(comp.MSE_total_mev2_atom2.mean())
    mu_mse = float(comp.MSE_mu_mev2_atom2.mean())
    resid_mse = float(comp.MSE_residual_mev2_atom2.mean())
    summary = {"model": prediction_column.removeprefix("E_pred_").removesuffix("_eV_atom"),
        "N_structures": len(frame), "N_compositions": int(frame.composition_id.nunique()),
        "MAE_absolute_mev_atom_pooled": float(np.mean(np.abs(pred-y))*1000),
        "MAE_absolute_mev_atom_macro_composition": float(comp.MAE_absolute_mev_atom.mean()),
        "RMSE_absolute_mev_atom": float(np.sqrt(np.mean((pred-y)**2))*1000),
        "bias_mev_atom": float(np.mean(pred-y)*1000), "R2_predictive_absolute": r2_pred,
        "R2_OLS_descriptive": r2_ols, "OLS_slope_dimensionless": slope,
        "OLS_intercept_mev_atom": intercept,
        "MAE_mu_mev_atom_macro": float(comp.MAE_mu_mev_atom.mean()),
        "MAE_delta_centered_mev_atom_macro": float(comp.MAE_delta_centered_mev_atom.mean()),
        "MAE_delta_DFT_GS_anchored_mev_atom_macro": float(comp.MAE_delta_GS_anchored_mev_atom.mean()),
        "MAE_delta_own_min_mev_atom_macro": float(comp.MAE_delta_own_min_mev_atom.mean()),
        "GS_correct": gs_correct, "GS_eligible": n_gs, "GS_excluded_DFT_gap_le_1mev": int((~comp.GS_eligible).sum()),
        "GS_predicted_abstentions_eligible": int(gs_eligible.GS_predicted_abstention.sum()),
        "GS_accuracy": gs_correct/n_gs if n_gs else float("nan"), "GS_Wilson_95_low": gs_ci[0], "GS_Wilson_95_high": gs_ci[1],
        "pairwise_correct": pair_correct, "pairwise_eligible": n_pairs,
        "pairwise_DFT_ties_excluded": int((~pair_frame.DFT_pair_eligible).sum()),
        "pairwise_predicted_ties_on_eligible": int((pair_eligible.predicted_class == "tie").sum()),
        "pairwise_accuracy": pair_correct/n_pairs if n_pairs else float("nan"),
        "pairwise_Wilson_95_low": pair_ci[0], "pairwise_Wilson_95_high": pair_ci[1],
        "SignedPairwiseMAE_mev_atom": float(pair_eligible.pred_difference_Ei_minus_Ej_mev_atom.sub(
            pair_eligible.DFT_difference_Ei_minus_Ej_mev_atom).abs().mean()) if n_pairs else float("nan"),
        "pairwise_mean_signed_error_mev_atom": float(pair_eligible.pred_difference_Ei_minus_Ej_mev_atom.sub(
            pair_eligible.DFT_difference_Ei_minus_Ej_mev_atom).mean()) if n_pairs else float("nan"),
        "mean_absolute_range_error_mev_atom": float(comp.absolute_range_error_mev_atom.mean()),
        "MSE_total_mev2_atom2": total_mse, "MSE_mu_mev2_atom2": mu_mse,
        "MSE_residual_mev2_atom2": resid_mse,
        "MSE_decomposition_max_abs_error": float(comp.MSE_identity_abs_error.max()),
        "GS_confusion_rows_DFT_P_H_N_columns_pred_P_H_N_abstain": gs_confusion,
        "pairwise_confusion_rows_DFT_i_tie_j_columns_pred_i_tie_j": pair_confusion}
    return summary, comp, pair_frame


def bootstrap_paired(frame: pd.DataFrame, prediction_a: str, prediction_b: str,
                     replicates: int = 20000, seed: int = 13072026) -> dict[str, Any]:
    """Bootstrap pareado por composición, remuestreando tripletes completos.

    Args:
        frame: Test etiquetado con ambas predicciones.
        prediction_a: H3-A.
        prediction_b: H2.
        replicates: Remuestras de fórmulas.
        seed: Semilla del generador.
    Returns:
        Diferencias pareadas e intervalos percentiles descriptivos.
    """
    formulas = sorted(frame.composition_id.unique().tolist())
    by_formula = {formula: frame.loc[frame.composition_id.eq(formula)] for formula in formulas}
    observed: dict[str, list[float]] = {"MAE_absolute": [], "MAE_mu": [], "MAE_delta_centered": [],
        "MAE_delta_GS_anchored": [], "MAE_delta_own_min": [], "GS_accuracy": [], "pairwise_accuracy": []}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        metric_rows = {name: {prediction_a: [], prediction_b: []} for name in observed}
        for formula in sampled:
            group = by_formula[str(formula)]
            for model, column in ((prediction_a, prediction_a), (prediction_b, prediction_b)):
                row, _ = composition_metrics(group, column)
                vals = {"MAE_absolute": row["MAE_absolute_mev_atom"], "MAE_mu": row["MAE_mu_mev_atom"],
                    "MAE_delta_centered": row["MAE_delta_centered_mev_atom"],
                    "MAE_delta_GS_anchored": row["MAE_delta_GS_anchored_mev_atom"],
                    "MAE_delta_own_min": row["MAE_delta_own_min_mev_atom"]}
                for name, value in vals.items():
                    metric_rows[name][model].append(value)
                metric_rows["GS_accuracy"][model].append((int(row["GS_correct"]), int(row["GS_eligible"])))
                metric_rows["pairwise_accuracy"][model].append((row["pairwise_correct_count"], row["pairwise_eligible_count"]))
        for name in observed:
            if name in {"GS_accuracy", "pairwise_accuracy"}:
                numerator_a = sum(v[0] for v in metric_rows[name][prediction_a]); denom_a = sum(v[1] for v in metric_rows[name][prediction_a])
                numerator_b = sum(v[0] for v in metric_rows[name][prediction_b]); denom_b = sum(v[1] for v in metric_rows[name][prediction_b])
                if denom_a and denom_b:
                    observed[name].append(numerator_a/denom_a - numerator_b/denom_b)
            else:
                observed[name].append(float(np.mean(metric_rows[name][prediction_a]) - np.mean(metric_rows[name][prediction_b])))
    result = {}
    for name, values in observed.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[name] = {"difference_H3A_minus_H2": float(np.mean(finite)) if len(finite) else float("nan"),
                        "bootstrap_95_ci_low": float(np.percentile(finite, 2.5)) if len(finite) else float("nan"),
                        "bootstrap_95_ci_high": float(np.percentile(finite, 97.5)) if len(finite) else float("nan"),
                        "valid_resamples": int(len(finite))}
    return {"status": "PASS", "replicates": replicates, "seed": seed,
            "sampling_unit": "eight test compositions, each triplet kept together",
            "interpretation": "Descriptive, high uncertainty with N=8; not a robust significance claim.",
            "comparisons": result}
