#!/usr/bin/env python3
"""Métricas OOF homogéneas para energía absoluta y selección polimórfica H3."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import FAMILIES


def _argmin_class(values: np.ndarray, tau_ev: float = 0.001) -> tuple[int | None, bool]:
    """Devuelve clase mínima o abstención cuando el menor par está empatado."""
    order = np.argsort(values, kind="stable")
    tied = float(values[order[1]] - values[order[0]]) <= tau_ev
    return (None if tied else int(order[0])), bool(tied)


def per_composition(group: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calcula métricas de triplete, ΔE anclada y las tres parejas canónicas.

    Args:
        group: Tres filas de un modelo/composición/fold/semilla.
    Returns:
        Métricas por composición y filas de parejas orientadas P-H, P-N, H-N.
    """
    g = group.set_index("polymorph_family").loc[list(FAMILIES)].reset_index()
    y = g.E_DFT_eV_atom.to_numpy(float)
    pred = g.E_pred_eV_atom.to_numpy(float)
    dft_mu, pred_mu = float(y.mean()), float(pred.mean())
    dft_delta, pred_delta = y - dft_mu, pred - pred_mu
    dft_g, dft_tied = _argmin_class(y)
    pred_g, pred_tied = _argmin_class(pred)
    dft_gs_index = int(np.argmin(y))
    dft_anchor = (y - y[dft_gs_index]) * 1000
    pred_anchor = (pred - pred[dft_gs_index]) * 1000
    dft_own, pred_own = (y - y.min()) * 1000, (pred - pred.min()) * 1000
    error = (pred - y) * 1000
    error_mu = float(error.mean())
    error_residual = error - error_mu
    dft_gap = float(np.partition(y, 1)[1] - np.min(y)) * 1000
    gs_correct = bool(not dft_tied and not pred_tied and dft_g == pred_g)
    pairs: list[dict[str, Any]] = []
    correct = eligible = 0
    signed_abs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = float((y[i] - y[j]) * 1000)
        p = float((pred[i] - pred[j]) * 1000)
        true_class = "i_lower" if d < -1 else "j_lower" if d > 1 else "tie"
        pred_class = "i_lower" if p < -1 else "j_lower" if p > 1 else "tie"
        use = true_class != "tie"
        good = use and pred_class == true_class
        eligible += int(use); correct += int(good); signed_abs.append(abs(p - d))
        pairs.append({"composition_id": g.composition_id.iloc[0], "family_i": FAMILIES[i], "family_j": FAMILIES[j],
                      "diff_DFT_mev_atom": d, "diff_pred_mev_atom": p, "DFT_class": true_class,
                      "pred_class": pred_class, "eligible_primary": use, "correct_primary": good})
    row = {
        "model": str(g.model.iloc[0]), "composition_id": str(g.composition_id.iloc[0]),
        "fold": int(g.fold.iloc[0]), "seed": int(g.seed.iloc[0]),
        "MAE_absolute_mev_atom": float(np.mean(np.abs(error))),
        "RMSE_absolute_mev_atom": float(np.sqrt(np.mean(error ** 2))),
        "R2_predictive_absolute": float(r2_score(y, pred)) if np.var(y) > 0 else np.nan,
        "MAE_mu_mev_atom": abs(error_mu), "MAE_delta_centered_mev_atom": float(np.mean(np.abs(pred_delta - dft_delta)) * 1000),
        "MAE_delta_GS_anchored_mev_atom": float(np.mean(np.abs(pred_anchor - dft_anchor))),
        "MAE_delta_own_min_legacy_mev_atom": float(np.mean(np.abs(pred_own - dft_own))),
        "MSE_total_mev2_atom2": float(np.mean(error ** 2)), "MSE_mu_mev2_atom2": error_mu ** 2,
        "MSE_residual_mev2_atom2": float(np.mean(error_residual ** 2)),
        "MSE_identity_abs_error": abs(float(np.mean(error ** 2)) - error_mu ** 2 - float(np.mean(error_residual ** 2))),
        "GS_correct": gs_correct, "GS_eligible": not dft_tied, "GS_DFT_tie": dft_tied,
        "GS_predicted_abstention": pred_tied, "GS_predicted_class": "tie/abstain" if pred_tied else FAMILIES[pred_g],
        "GS_DFT_class": FAMILIES[int(np.argmin(y))], "DFT_GS_gap_mev_atom": dft_gap,
        "pairwise_correct_count": correct, "pairwise_eligible_count": eligible,
        "pairwise_accuracy": correct / eligible if eligible else np.nan,
        "pairwise_signed_MAE_mev_atom": float(np.mean(signed_abs)),
        "DFT_range_mev_atom": float((y.max() - y.min()) * 1000),
        "predicted_range_mev_atom": float((pred.max() - pred.min()) * 1000),
        "absolute_range_error_mev_atom": float(abs((pred.max() - pred.min()) - (y.max() - y.min())) * 1000),
        "energy_collapse": bool((pred.max() - pred.min()) * 1000 < 1),
        "E_DFT_mu_eV_atom": dft_mu, "E_pred_mu_eV_atom": pred_mu,
    }
    return row, pairs


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula métricas por composición y matrices de confusión sin inflar N.

    Args:
        predictions: OOF de desarrollo, una fila por estructura/modelo/fold/seed.
    Returns:
        (métricas por composición, parejas por composición, resumen fold/semilla).
    """
    rows, pairs = [], []
    for (_, _, _, _), group in predictions.groupby(["model", "composition_id", "fold", "seed"], sort=False):
        metric, pair_rows = per_composition(group)
        rows.append(metric)
        pairs.extend([{**pair, "model": metric["model"], "fold": metric["fold"], "seed": metric["seed"]} for pair in pair_rows])
    comp = pd.DataFrame(rows)
    pair_frame = pd.DataFrame(pairs)
    summaries = []
    for (model, fold, seed), group in comp.groupby(["model", "fold", "seed"], sort=False):
        summaries.append({"model": model, "fold": int(fold), "seed": int(seed), "compositions": len(group),
                          "MAE_absolute_mev_atom": group.MAE_absolute_mev_atom.mean(),
                          "RMSE_absolute_mev_atom": float(np.sqrt(np.mean(group.MSE_total_mev2_atom2))),
                          "MAE_mu_mev_atom": group.MAE_mu_mev_atom.mean(),
                          "MAE_delta_centered_mev_atom": group.MAE_delta_centered_mev_atom.mean(),
                          "MAE_delta_GS_anchored_mev_atom": group.MAE_delta_GS_anchored_mev_atom.mean(),
                          "MAE_delta_own_min_legacy_mev_atom": group.MAE_delta_own_min_legacy_mev_atom.mean(),
                          "GS_accuracy": group.loc[group.GS_eligible, "GS_correct"].mean(),
                          "GS_eligible_compositions": int(group.GS_eligible.sum()),
                          "GS_abstentions": int(group.GS_predicted_abstention.sum()),
                          "pairwise_accuracy": group.pairwise_correct_count.sum() / group.pairwise_eligible_count.sum(),
                          "pairwise_eligible_pairs": int(group.pairwise_eligible_count.sum()),
                          "pairwise_signed_MAE_mev_atom": group.pairwise_signed_MAE_mev_atom.mean(),
                          "absolute_range_error_mev_atom": group.absolute_range_error_mev_atom.mean(),
                          "balanced_accuracy": _balanced(group), "macro_f1": _macro_f1(group)})
    return comp, pair_frame, pd.DataFrame(summaries)


def _balanced(group: pd.DataFrame) -> float:
    """Balanced accuracy excluyendo DFT ties; las abstenciones cuentan como miss."""
    use = group.loc[group.GS_eligible]
    if use.empty:
        return float("nan")
    pred = use.GS_predicted_class.replace("tie/abstain", "__abstain__")
    recalls = []
    for label in FAMILIES:
        actual = use.GS_DFT_class.eq(label)
        if actual.any():
            recalls.append(float(pred.loc[actual].eq(label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _macro_f1(group: pd.DataFrame) -> float:
    """F1 macro de GS con etiquetas fijas y abstención como clase incorrecta."""
    use = group.loc[group.GS_eligible]
    if use.empty:
        return float("nan")
    pred = use.GS_predicted_class.replace("tie/abstain", "__abstain__")
    return float(f1_score(use.GS_DFT_class, pred, labels=list(FAMILIES), average="macro", zero_division=0))
