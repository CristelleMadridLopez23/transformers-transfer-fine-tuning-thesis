#!/usr/bin/env python3
"""Métricas energéticas y de orden polimórfico con denominadores auditables."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import FAMILIES, TAU_MEV
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.data_and_folds import ordered_triplet


def classify_pair(diff: float, tau: float = TAU_MEV) -> str:
    """Clasifica E_i-E_j en i_lower, tie o j_lower con orientación fija."""
    return "i_lower" if diff < -tau else "j_lower" if diff > tau else "tie"


def triplet_metrics(group: pd.DataFrame, tau: float = TAU_MEV) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calcula métricas por composición, anclaje DFT y los tres pares canónicos."""
    g = ordered_triplet(group)
    y = g.E_DFT_eV_atom.to_numpy(float) * 1000.0
    pred = g.E_pred_eV_atom.to_numpy(float) * 1000.0
    dft_order = np.argsort(y, kind="stable")
    pred_order = np.argsort(pred, kind="stable")
    dft_gap = float(y[dft_order[1]] - y[dft_order[0]])
    pred_gap = float(pred[pred_order[1]] - pred[pred_order[0]])
    dft_tie = dft_gap <= tau
    pred_tie = pred_gap <= tau
    dft_min, pred_min = int(dft_order[0]), int(pred_order[0])
    mu_y, mu_pred = float(y.mean()), float(pred.mean())
    delta_y, delta_pred = y - mu_y, pred - float(pred.mean())
    error = pred - y
    err_mu = float(error.mean())
    err_resid = error - err_mu
    mse_total = float(np.mean(error**2))
    mse_mean = float(err_mu**2)
    mse_resid = float(np.mean(err_resid**2))
    mae_abs = float(np.mean(np.abs(error)))
    dft_anchor = y - y[dft_min]
    pred_anchor = pred - pred[dft_min]
    dft_own = y - y.min()
    pred_own = pred - pred.min()
    pred_tied_families = [FAMILIES[i] for i in range(3) if abs(pred[i] - pred[pred_min]) <= tau]
    gs_correct = bool(not dft_tie and not pred_tie and dft_min == pred_min)
    per_pairs: list[dict[str, Any]] = []
    correct = eligible = 0
    signed_error = []
    family_pairs = ((0, 1), (0, 2), (1, 2))
    for i, j in family_pairs:
        diff_dft, diff_pred = float(y[i] - y[j]), float(pred[i] - pred[j])
        truth, guessed = classify_pair(diff_dft, tau), classify_pair(diff_pred, tau)
        pair_eligible = truth != "tie"
        pair_correct = pair_eligible and guessed == truth
        correct += int(pair_correct)
        eligible += int(pair_eligible)
        signed_error.append(abs(diff_pred - diff_dft))
        per_pairs.append({"composition_id": str(g.composition_id.iloc[0]), "family_i": FAMILIES[i], "family_j": FAMILIES[j],
            "diff_dft_mev_atom": diff_dft, "diff_pred_mev_atom": diff_pred, "dft_class": truth,
            "pred_class": guessed, "eligible_primary": pair_eligible, "correct_primary": pair_correct})
    # The equality is algebraic for MSE after centering within triplet; MAE is not additive.
    return ({"composition_id": str(g.composition_id.iloc[0]), "fold": int(g.fold.iloc[0]) if "fold" in g else -1,
        "seed": int(g.seed.iloc[0]) if "seed" in g and pd.notna(g.seed.iloc[0]) else -1,
        "MAE_absolute_mev_atom": mae_abs, "MAE_mu_mev_atom": abs(err_mu),
        "MAE_delta_centered_mev_atom": float(np.mean(np.abs(delta_pred - delta_y))),
        "MAE_delta_GS_anchored_mev_atom": float(np.mean(np.abs(pred_anchor - dft_anchor))),
        "MAE_delta_own_min_legacy_mev_atom": float(np.mean(np.abs(pred_own - dft_own))),
        "MSE_total_mev2_atom2": mse_total, "MSE_mean_mev2_atom2": mse_mean,
        "MSE_residual_mev2_atom2": mse_resid, "MSE_identity_abs_error": abs(mse_total - mse_mean - mse_resid),
        "GS_correct": gs_correct, "GS_eligible": not dft_tie, "GS_DFT_near_degenerate": dft_tie,
        "GS_predicted_abstention": pred_tie, "GS_predicted_family": "tie/abstain" if pred_tie else FAMILIES[pred_min],
        "GS_DFT_family": FAMILIES[dft_min], "GS_DFT_gap_mev_atom": dft_gap,
        "pairwise_correct_count": correct, "pairwise_eligible_count": eligible,
        "pairwise_accuracy": correct / eligible if eligible else np.nan,
        "pairwise_signed_MAE_mev_atom": float(np.mean(signed_error)),
        "DFT_range_mev_atom": float(y.max() - y.min()), "predicted_range_mev_atom": float(pred.max() - pred.min()),
        "absolute_range_error_mev_atom": float(abs((pred.max() - pred.min()) - (y.max() - y.min()))),
        "predicted_near_collapse": bool(pred.max() - pred.min() < tau), "E_DFT_mu_mev_atom": mu_y,
        "E_pred_mu_mev_atom": float(pred.mean()), "polymorph_families": list(FAMILIES),
        "predicted_tied_families": pred_tied_families}, per_pairs)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula tabla por composición y tabla canónica de tres pares."""
    summaries, pairs = [], []
    # Raw OOF rows contain one triplet per fold×seed; ensemble rows contain one
    # triplet per fold. Keep these evaluation units separate rather than
    # accidentally concatenating the three seed triplets into a nine-row group.
    keys = ["model", "composition_id"]
    keys.extend(column for column in ("fold", "seed") if column in predictions.columns)
    for key, group in predictions.groupby(keys, sort=False, dropna=False):
        row, pair_rows = triplet_metrics(group)
        key_values = key if isinstance(key, tuple) else (key,)
        row.update(dict(zip(keys, key_values)))
        summaries.append(row)
        for pair in pair_rows:
            pair.update(dict(zip(keys, key_values)))
            pairs.append(pair)
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def classification_metrics(confusion: pd.DataFrame) -> dict[str, float]:
    """Calcula balanced accuracy/F1 macro sin convertir abstenciones en clases.

    Args:
        confusion: Filas con `true_class` y `predicted_class`, DFT tie ya excluido.
    Returns:
        Balanced accuracy, macro-F1 y N; empates predichos son errores/abstenciones.
    """
    valid = confusion.copy()
    labels = ["dist_perovskite", "hexagonal", "needle"]
    y = valid.true_class.to_numpy()
    pred = valid.predicted_class.to_numpy()
    if not len(y):
        return {"balanced_accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}
    recalls = [float(np.mean(pred[y == label] == label)) for label in labels if np.any(y == label)]
    return {"balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), "n": len(y)}
