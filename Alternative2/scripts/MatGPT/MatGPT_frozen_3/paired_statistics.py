#!/usr/bin/env python3
"""Bootstrap pareado agrupado por fórmula para OOF ya agregadas por semilla."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def paired_bootstrap(metrics: pd.DataFrame, replicates: int = 2000, seed: int = 13072026,
                     reference: str = "H2") -> dict[str, Any]:
    """Remuestrea fórmulas, manteniendo juntos tripletes y métricas derivadas.

    Args:
        metrics: Un renglón por modelo y composición física.
        replicates: Número de remuestras pareadas.
        seed: Semilla reproducible.
        reference: Modelo base comparativo.
    Returns:
        Medias y percentiles 2.5/97.5 de diferencias candidate-reference.
    """
    required = {"model", "composition_id", "MAE_absolute_mev_atom", "MAE_delta_GS_anchored_mev_atom",
        "MAE_delta_own_min_legacy_mev_atom", "MAE_mu_mev_atom", "MAE_delta_centered_mev_atom",
        "GS_correct", "GS_eligible", "pairwise_correct_count", "pairwise_eligible_count"}
    if required - set(metrics.columns):
        raise ValueError(f"Faltan columnas bootstrap: {sorted(required - set(metrics.columns))}")
    rng = np.random.default_rng(seed)
    groups = sorted(metrics.composition_id.unique())
    comparison: dict[str, Any] = {}
    model_names = list(metrics.model.unique())
    metrics_numeric = ("MAE_absolute_mev_atom", "MAE_delta_GS_anchored_mev_atom", "MAE_delta_own_min_legacy_mev_atom",
                       "MAE_mu_mev_atom", "MAE_delta_centered_mev_atom", "GS_accuracy", "pairwise_accuracy")
    indexed = {model: metrics.loc[metrics.model.eq(model)].set_index("composition_id").reindex(groups)
               for model in model_names}
    for candidate in model_names:
        if candidate == reference or reference not in indexed:
            continue
        item: dict[str, Any] = {}
        for metric in metrics_numeric:
            if metric in {"GS_accuracy", "pairwise_accuracy"} and not bool(indexed[candidate].get("ranking_discriminative", pd.Series([True])).fillna(True).all()):
                item[metric] = {"difference_mean_candidate_minus_reference": None, "ci95_low": None, "ci95_high": None,
                    "candidate_mean": None, "reference_mean": None, "status": "not_applicable_no_polymorph_discrimination"}
                continue
            if metric in indexed[candidate] and not np.isfinite(indexed[candidate][metric].to_numpy(dtype=float)).any():
                item[metric] = {"difference_mean_candidate_minus_reference": None, "ci95_low": None, "ci95_high": None,
                    "candidate_mean": None, "reference_mean": None, "status": "not_applicable_for_component_only_target"}
                continue
            draws = []
            for _ in range(replicates):
                sampled = rng.integers(0, len(groups), len(groups))
                left, right = indexed[candidate].iloc[sampled], indexed[reference].iloc[sampled]
                if metric == "GS_accuracy":
                    lmask, rmask = left.GS_eligible.astype(bool), right.GS_eligible.astype(bool)
                    lv = left.loc[lmask, "GS_correct"].astype(float).mean()
                    rv = right.loc[rmask, "GS_correct"].astype(float).mean()
                elif metric == "pairwise_accuracy":
                    ln, rn = left.pairwise_eligible_count.sum(), right.pairwise_eligible_count.sum()
                    lv = left.pairwise_correct_count.sum() / ln if ln else np.nan
                    rv = right.pairwise_correct_count.sum() / rn if rn else np.nan
                else:
                    lv, rv = left[metric].mean(), right[metric].mean()
                draws.append(float(lv - rv))
            full_left, full_right = indexed[candidate], indexed[reference]
            if metric == "GS_accuracy":
                mean_l = full_left.loc[full_left.GS_eligible.astype(bool), "GS_correct"].astype(float).mean()
                mean_r = full_right.loc[full_right.GS_eligible.astype(bool), "GS_correct"].astype(float).mean()
            elif metric == "pairwise_accuracy":
                mean_l = full_left.pairwise_correct_count.sum() / full_left.pairwise_eligible_count.sum()
                mean_r = full_right.pairwise_correct_count.sum() / full_right.pairwise_eligible_count.sum()
            else:
                mean_l, mean_r = full_left[metric].mean(), full_right[metric].mean()
            finite_draws = np.asarray(draws, dtype=float)
            finite_draws = finite_draws[np.isfinite(finite_draws)]
            if not len(finite_draws) or not np.isfinite(mean_l - mean_r):
                item[metric] = {"difference_mean_candidate_minus_reference": None, "ci95_low": None, "ci95_high": None,
                    "candidate_mean": None, "reference_mean": None, "status": "not_estimable"}
            else:
                item[metric] = {"difference_mean_candidate_minus_reference": float(mean_l - mean_r),
                    "ci95_low": float(np.quantile(finite_draws, .025)), "ci95_high": float(np.quantile(finite_draws, .975)),
                    "candidate_mean": float(mean_l), "reference_mean": float(mean_r)}
        comparison[f"{candidate}_minus_{reference}"] = item
    return {"method": "paired bootstrap by physical composition; predictions averaged across seeds per structure before metrics",
        "groups": len(groups), "replicates": replicates, "seed": seed,
        "interpretation": "exploratory; not an independent confirmation because historical validation informed prior selection",
        "comparisons_vs": reference, "comparisons": comparison}
