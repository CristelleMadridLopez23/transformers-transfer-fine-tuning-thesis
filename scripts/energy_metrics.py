"""Encoder-independent metrics for grouped per-atom energy predictions.

Inputs are one row per physical structure. Energy columns use eV/atom; outputs
report both eV/atom and meV/atom where useful. This module does not fit models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def composition_metrics(frame: pd.DataFrame, truth: str, prediction: str,
                        group: str = "composition_id", tolerance_mev: float = 1.0) -> dict[str, Any]:
    """Compute absolute, relative, and ranking metrics by physical composition.

    Args:
        frame: Rows containing group, truth, prediction, and polymorph_family.
        truth: Column with DFT energies in eV/atom.
        prediction: Column with predicted energies in eV/atom.
        group: Composition identifier column.
        tolerance_mev: Tie/gap threshold in meV/atom.

    Returns:
        Dictionary of metrics and denominators. Predicted ties are abstentions.
    """
    required = {group, truth, prediction, "polymorph_family"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[[truth, prediction]].to_numpy(float)).all():
        raise ValueError("Datos vacíos o energías no finitas")
    mae = float(np.mean(np.abs(frame[prediction] - frame[truth])) * 1000.0)
    rmse = float(np.sqrt(np.mean((frame[prediction] - frame[truth]) ** 2)) * 1000.0)
    gs_correct = gs_eligible = gs_abstain = pair_correct = pair_eligible = 0
    delta_abs: list[float] = []
    delta_own_abs: list[float] = []
    mean_abs: list[float] = []
    residual_abs: list[float] = []
    pair_abs: list[float] = []
    range_abs: list[float] = []
    range_ratios: list[float] = []
    err_sq: list[float] = []
    mean_err_sq: list[float] = []
    residual_err_sq: list[float] = []
    abs_comp_macro: list[float] = []
    delta_comp_macro: list[float] = []
    own_delta_comp_macro: list[float] = []
    gs_comp_scores: list[float] = []
    pair_comp_scores: list[float] = []
    pair_truth_classes: list[int] = []
    pair_pred_classes: list[int] = []
    for _, rows in frame.groupby(group, sort=False):
        if len(rows) != 3 or rows["polymorph_family"].duplicated().any():
            raise ValueError("Cada composición debe tener exactamente tres familias distintas")
        y = rows[truth].to_numpy(float)
        p = rows[prediction].to_numpy(float)
        gap = (np.partition(y, 1)[1] - np.min(y)) * 1000.0
        gy = int(np.argmin(y))
        gp = int(np.argmin(p))
        mu_y, mu_p = float(y.mean()), float(p.mean())
        dy_center, dp_center = y-mu_y, p-mu_p
        error = p-y; mean_error=float(mu_p-mu_y); residual_error=dp_center-dy_center
        abs_comp_macro.append(float(np.mean(np.abs(error))*1000.0))
        mean_abs.append(abs(mean_error)*1000.0)
        residual_abs.extend(np.abs(residual_error).tolist())
        err_sq.extend((error**2).tolist()); mean_err_sq.extend([mean_error**2]*len(y)); residual_err_sq.extend((residual_error**2).tolist())
        dft_range=float(np.ptp(y)); pred_range=float(np.ptp(p))
        range_abs.append(abs(pred_range-dft_range)*1000.0)
        if dft_range > 0: range_ratios.append(pred_range/dft_range)
        if gap > tolerance_mev:
            gs_eligible += 1
            is_tie = (np.partition(p, 1)[1] - np.min(p)) * 1000.0 <= tolerance_mev
            gs_abstain += int(is_tie)
            gs_correct += int(gp == gy and not is_tie)
        dft_delta = (y - y[gy]) * 1000.0
        pred_delta = (p - p[gy]) * 1000.0
        delta_abs.extend(np.abs(pred_delta - dft_delta).tolist())
        own_dft=(y-y.min())*1000.0; own_pred=(p-p.min())*1000.0
        delta_own_abs.extend(np.abs(own_pred-own_dft).tolist())
        delta_comp_macro.append(float(np.mean(np.abs(pred_delta-dft_delta))))
        own_delta_comp_macro.append(float(np.mean(np.abs(own_pred-own_dft))))
        for i, j in ((0, 1), (0, 2), (1, 2)):
            dy = (y[i] - y[j]) * 1000.0
            dp = (p[i] - p[j]) * 1000.0
            pair_abs.append(abs(dp-dy))
            if abs(dy) > tolerance_mev:
                pair_eligible += 1
                correct=int(np.sign(dy) == np.sign(dp) and abs(dp) > tolerance_mev)
                pair_correct += correct
                pair_truth_classes.append(int(np.sign(dy)))
                pair_pred_classes.append(int(np.sign(dp)) if abs(dp)>tolerance_mev else 0)
        if gap > tolerance_mev:
            gs_comp_scores.append(float(gp == gy and (np.partition(p,1)[1]-np.min(p))*1000.0 > tolerance_mev))
        eligible_pairs=[(y[i]-y[j],p[i]-p[j]) for i,j in ((0,1),(0,2),(1,2)) if abs((y[i]-y[j])*1000.0)>tolerance_mev]
        if eligible_pairs:
            pair_comp_scores.append(float(np.mean([np.sign(dy)==np.sign(dp) and abs(dp)*1000.0>tolerance_mev for dy,dp in eligible_pairs])))
    return {
        "n_structures": int(len(frame)), "n_compositions": int(frame[group].nunique()),
        "mae_absolute_mev_atom": mae, "rmse_absolute_mev_atom": rmse,
        "mae_absolute_macro_by_composition_mev_atom": float(np.mean(abs_comp_macro)),
        "mae_delta_dft_gs_anchored_mev_atom": float(np.mean(delta_abs)),
        "mae_delta_dft_gs_anchored_macro_by_composition_mev_atom": float(np.mean(delta_comp_macro)),
        "mae_delta_own_min_mev_atom": float(np.mean(delta_own_abs)),
        "mae_delta_own_min_macro_by_composition_mev_atom": float(np.mean(own_delta_comp_macro)),
        "mae_composition_mean_mev_atom": float(np.mean(mean_abs)),
        "mae_centered_residual_mev_atom": float(np.mean(residual_abs)*1000.0),
        "signed_pairwise_mae_mev_atom": float(np.mean(pair_abs)),
        "mean_absolute_range_error_mev_atom": float(np.mean(range_abs)),
        "mean_predicted_to_dft_range_ratio": float(np.mean(range_ratios)) if range_ratios else None,
        "mse_total_ev2_atom2": float(np.mean(err_sq)),
        "mse_mean_component_ev2_atom2": float(np.mean(mean_err_sq)),
        "mse_centered_residual_component_ev2_atom2": float(np.mean(residual_err_sq)),
        "mse_decomposition_abs_error": float(abs(np.mean(err_sq)-np.mean(mean_err_sq)-np.mean(residual_err_sq))),
        "gs_correct": gs_correct, "gs_eligible": gs_eligible, "gs_abstentions": gs_abstain,
        "gs_accuracy": gs_correct / gs_eligible if gs_eligible else None,
        "gs_accuracy_macro_by_composition": float(np.mean(gs_comp_scores)) if gs_comp_scores else None,
        "pairwise_correct": pair_correct, "pairwise_eligible": pair_eligible,
        "pairwise_accuracy": pair_correct / pair_eligible if pair_eligible else None,
        "pairwise_accuracy_macro_by_composition": float(np.mean(pair_comp_scores)) if pair_comp_scores else None,
        "pairwise_balanced_accuracy": float(np.mean([np.mean(np.asarray(pair_pred_classes)[np.asarray(pair_truth_classes)==label] == label) for label in (-1,1) if np.any(np.asarray(pair_truth_classes)==label)])) if pair_truth_classes else None,
        "pairwise_macro_f1": float(f1_score(pair_truth_classes,pair_pred_classes,labels=[-1,1],average="macro",zero_division=0)) if pair_truth_classes else None,
        "spearman_rho": float(spearmanr(frame[truth].to_numpy(float),frame[prediction].to_numpy(float)).statistic) if frame[truth].nunique()>1 and frame[prediction].nunique()>1 else None,
    }
