#!/usr/bin/env python3
"""Asocia exposición candidata con errores OOF de solo las 70 composiciones dev."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import ALT, REPORT_DIR, AuditError, write_json
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.audit_domain_overlap import parse_formula

COMPARISONS = ALT / "reports/MatGPT_frozen_3/run_20261023_v2/comparisons"
MODELS = ["H0", "H2", "R3-DROP30", "R4-SELECTED-ABSOLUTE"]
SEED = 20261023
BOOTSTRAPS = 2000


def run_analysis() -> dict[str, Any]:
    """Calcula asociaciones descriptivas y bootstrap por composición.

    Returns:
        Correlaciones y grupos exploratorios con incertidumbre bootstrap.
    """
    overlap_path = REPORT_DIR / "domain_overlap" / "abs3_composition_overlap.csv"
    metrics_path = COMPARISONS / "metrics_by_composition.csv"
    if not overlap_path.is_file() or not metrics_path.is_file():
        raise AuditError("Faltan solapamiento de fórmulas o métricas R1–R4 ya existentes.")
    exposure = pd.read_csv(overlap_path)
    # Este manifiesto entrega fórmula y split; no se leen etiquetas DFT del test.
    exposure = exposure[exposure.historical_split.isin(["train", "val"])].copy()
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics.model.isin(MODELS) & metrics.target_kind.eq("absolute")].copy()
    metrics = metrics[metrics.composition_id.map(lambda formula: parse_formula(str(formula)) in
                         {parse_formula(str(value)) for value in exposure.target_formula})]
    exposure["formula_key"] = exposure.target_formula.map(lambda formula: parse_formula(str(formula)))
    metrics["formula_key"] = metrics.composition_id.map(lambda formula: parse_formula(str(formula)))
    joined = metrics.merge(exposure, on="formula_key", how="inner", validate="many_to_one")
    if joined.composition_id.nunique() != 70 or set(joined.model.unique()) != set(MODELS):
        raise AuditError(f"OOF no cubre exactamente 70 composiciones dev y 4 modelos: {joined.groupby('model').composition_id.nunique().to_dict()}")
    joined.to_csv(REPORT_DIR / "error_correlations" / "development_exposure_error_rows.csv", index=False)
    rng = np.random.default_rng(SEED)
    output: list[dict[str, Any]] = []
    for model in MODELS:
        data = joined[joined.model.eq(model)].drop_duplicates("composition_id").copy()
        predictors = ["train_candidate_rows_exact_formula", "full_corpus_rows_exact_formula"]
        outcomes = ["MAE_absolute_mev_atom", "MAE_delta_GS_anchored_mev_atom", "GS_correct", "pairwise_accuracy", "absolute_range_error_mev_atom"]
        for predictor in predictors:
            for outcome in outcomes:
                pairs = data[[predictor, outcome]].dropna().to_numpy(float)
                if len(pairs) < 4 or np.unique(pairs[:, 0]).size < 2 or np.unique(pairs[:, 1]).size < 2:
                    continue
                observed = float(spearmanr(pairs[:, 0], pairs[:, 1]).statistic)
                boot: list[float] = []
                for _ in range(BOOTSTRAPS):
                    sample = pairs[rng.integers(0, len(pairs), len(pairs))]
                    if np.unique(sample[:, 0]).size > 1 and np.unique(sample[:, 1]).size > 1:
                        boot.append(float(spearmanr(sample[:, 0], sample[:, 1]).statistic))
                lo, hi = np.quantile(boot, [0.025, 0.975]).tolist() if boot else [None, None]
                output.append({"model": model, "exposure_measure": predictor, "outcome": outcome,
                               "n_compositions": len(pairs), "spearman_rho": observed,
                               "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi,
                               "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": SEED,
                               "interpretation": "exploratory association, not causal; R1–R4 selection used part of development"})
    out = REPORT_DIR / "error_correlations"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(out / "spearman_bootstrap.csv", index=False)
    group_rows: list[dict[str, Any]] = []
    for model in MODELS:
        data = joined[joined.model.eq(model)].drop_duplicates("composition_id")
        seen = data[data.train_candidate_rows_exact_formula.gt(0)]
        unseen = data[data.train_candidate_rows_exact_formula.eq(0)]
        for metric in ["MAE_absolute_mev_atom", "MAE_delta_GS_anchored_mev_atom", "GS_correct", "pairwise_accuracy"]:
            a = seen[metric].dropna().to_numpy(float)
            b = unseen[metric].dropna().to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                continue
            boot_diff = np.array([rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
                                  for _ in range(BOOTSTRAPS)])
            group_rows.append({"model": model, "metric": metric, "seen_n": len(a), "unseen_n": len(b),
                               "seen_mean": float(a.mean()), "unseen_mean": float(b.mean()),
                               "mean_difference_seen_minus_unseen": float(a.mean() - b.mean()),
                               "bootstrap_95_ci_low": float(np.quantile(boot_diff, .025)),
                               "bootstrap_95_ci_high": float(np.quantile(boot_diff, .975)),
                               "bootstrap_seed": SEED,
                               "interpretation": "descriptive; only 11 development formulas have exact train-candidate matches"})
    pd.DataFrame(group_rows).to_csv(out / "seen_vs_unseen_descriptive.csv", index=False)
    summary = {"status": "EXPLORATORY_ONLY", "development_compositions": 70,
               "exact_train_candidate_seen_compositions": int(joined.drop_duplicates("composition_id").train_candidate_rows_exact_formula.gt(0).sum()),
               "models": MODELS, "metrics_file": str(metrics_path), "bootstrap_replicates": BOOTSTRAPS,
               "bootstrap_seed": SEED, "test_metric_labels_loaded": False,
               "limitations": ["The reconstructed split is not proven to match the checkpoint's training CSV byte-for-byte.",
                               "R1–R4 used development validation compositions for selection; these associations are exploratory.",
                               "Group sizes are small and correlations do not show causation."]}
    write_json(out / "analysis_summary.json", summary)
    return summary

def main() -> None:
    """Ejecuta correlaciones exploratorias development-only."""
    try:
        print(json.dumps(run_analysis(), indent=2, ensure_ascii=False))
    except (AuditError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc

if __name__ == "__main__":
    main()
