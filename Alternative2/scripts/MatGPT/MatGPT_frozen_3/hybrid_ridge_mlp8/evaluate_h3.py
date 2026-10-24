#!/usr/bin/env python3
"""Agrega OOF H3 y referencias; calcula métricas, matrices y bootstrap pareado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import H2_ROOT, H3Error, REPORT_DIR, R4_ROOT, SEEDS, write_json
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.metrics import metric_tables

PRED_DIRS = {
    "H3-A": REPORT_DIR / "H3_A_recombination/oof_predictions.csv",
    "H3-B": REPORT_DIR / "H3_B_ridge_fixed_delta/oof_predictions.csv",
    "H3-C": REPORT_DIR / "H3_C_ridge_retrained_delta/oof_predictions.csv",
}
PAIRWISE = (("dist_perovskite", "hexagonal"), ("dist_perovskite", "needle"), ("hexagonal", "needle"))
COMPARISONS = (("H3-A", "H2"), ("H3-B", "H2"), ("H3-C", "H2"),
               ("H3-B", "H3-A"), ("H3-C", "H3-B"), ("H3-C", "R4"))
METRICS = ("MAE_absolute_mev_atom", "MAE_mu_mev_atom", "MAE_delta_centered_mev_atom",
           "MAE_delta_GS_anchored_mev_atom", "MAE_delta_own_min_legacy_mev_atom",
           "GS_correct", "pairwise_accuracy", "absolute_range_error_mev_atom")


def _baselines() -> pd.DataFrame:
    """Carga solo OOF guardados del conjunto de desarrollo."""
    enriched = pd.read_csv(H2_ROOT / "oof_predictions_enriched.csv")
    pieces = []
    for model in ("H0", "H2"):
        part = enriched.loc[enriched.model.eq(model)].copy()
        pieces.append(part[["structure_id", "composition_id", "polymorph_family", "fold", "seed", "model",
                            "E_DFT_eV_atom", "E_pred_eV_atom"]])
    r4all = pd.read_csv(R4_ROOT / "oof_predictions.csv")
    r4 = r4all.loc[r4all.model.eq("R4-SELECTED-ABSOLUTE")].copy()
    r4["model"] = "R4"
    pieces.append(r4[["structure_id", "composition_id", "polymorph_family", "fold", "seed", "model",
                      "E_DFT_eV_atom", "E_pred_eV_atom"]])
    d1 = pd.read_csv(H2_ROOT / "D1_ridge_composition_only_oof.csv").copy()
    d1["seed"] = -1; d1["model"] = "D1"
    pieces.append(d1[["structure_id", "composition_id", "polymorph_family", "fold", "seed", "model",
                      "E_DFT_eV_atom", "E_pred_eV_atom"]])
    return pd.concat(pieces, ignore_index=True)


def _load_all_predictions() -> pd.DataFrame:
    """Combina bases y H3-A/B/C tras validar cobertura/duplicados."""
    pieces = [_baselines()]
    for model, path in PRED_DIRS.items():
        if not path.is_file():
            raise H3Error(f"Faltan predicciones {model}: {path}")
        part = pd.read_csv(path)
        if "E_DFT_eV_atom" not in part or "E_pred_eV_atom" not in part:
            raise H3Error(f"Esquema de predicciones {model} inválido.")
        pieces.append(part[["structure_id", "composition_id", "polymorph_family", "fold", "seed", "model",
                            "E_DFT_eV_atom", "E_pred_eV_atom"]])
    predictions = pd.concat(pieces, ignore_index=True)
    if predictions.duplicated(["model", "structure_id", "fold", "seed"]).any():
        raise H3Error("Hay claves OOF repetidas entre predicciones.")
    if not np.isfinite(predictions[["E_DFT_eV_atom", "E_pred_eV_atom"]].to_numpy()).all():
        raise H3Error("Predicciones/etiquetas contienen NaN/Inf.")
    if (predictions.groupby(["model", "fold", "seed"]).size().ne(42)).any():
        raise H3Error("Cada evaluación OOF debe cubrir 14 tripletes / 42 estructuras.")
    predictions.to_csv(REPORT_DIR / "comparative_analysis/oof_predictions_all_models.csv", index=False)
    return predictions


def _ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    """Promedia semillas por estructura antes de recomputar métricas físicas."""
    return predictions.groupby(["model", "structure_id", "composition_id", "polymorph_family", "fold"], as_index=False).agg(
        E_DFT_eV_atom=("E_DFT_eV_atom", "first"), E_pred_eV_atom=("E_pred_eV_atom", "mean")).assign(seed=-1)


def _macro_summary(ensemble_metrics: pd.DataFrame, ensemble_predictions: pd.DataFrame) -> pd.DataFrame:
    """Resumen macro por composición y denominadores para modelos deterministas/ensemble."""
    output = []
    for model, group in ensemble_metrics.groupby("model", sort=False):
        point_predictions = ensemble_predictions.loc[ensemble_predictions.model.eq(model)]
        eligible = group.loc[group.GS_eligible]
        pair_eligible = group.loc[group.pairwise_eligible_count.gt(0)]
        output.append({"model": model, "N_compositions": group.composition_id.nunique(),
                       "MAE_absolute_mev_atom_macro": group.MAE_absolute_mev_atom.mean(),
                       "RMSE_absolute_mev_atom_pooled": float(np.sqrt(group.MSE_total_mev2_atom2.mean())),
                       "R2_predictive_absolute_pooled": float(r2_score(point_predictions.E_DFT_eV_atom, point_predictions.E_pred_eV_atom)),
                       "MAE_mu_mev_atom_macro": group.MAE_mu_mev_atom.mean(),
                       "MAE_delta_centered_mev_atom_macro": group.MAE_delta_centered_mev_atom.mean(),
                       "MAE_delta_GS_anchored_mev_atom_macro": group.MAE_delta_GS_anchored_mev_atom.mean(),
                       "MAE_delta_own_min_legacy_mev_atom_macro": group.MAE_delta_own_min_legacy_mev_atom.mean(),
                       "GS_accuracy": eligible.GS_correct.mean(), "GS_eligible_compositions": len(eligible),
                       "GS_DFT_ties_excluded": int((~group.GS_eligible).sum()),
                       "GS_predicted_abstentions": int(group.GS_predicted_abstention.sum()),
                       "GS_balanced_accuracy": _balanced(eligible), "GS_macro_f1": _f1(eligible),
                       "Pairwise_accuracy_macro_composition": pair_eligible.pairwise_accuracy.mean(),
                       "Pairwise_accuracy_pooled": pair_eligible.pairwise_correct_count.sum() / pair_eligible.pairwise_eligible_count.sum(),
                       "Pairwise_signed_MAE_mev_atom_macro": group.pairwise_signed_MAE_mev_atom.mean(),
                       "Pairwise_eligible_pairs": int(group.pairwise_eligible_count.sum()),
                       "Mean_absolute_range_error_mev_atom": group.absolute_range_error_mev_atom.mean(),
                       "Energy_collapse_compositions": int(group.energy_collapse.sum())})
    return pd.DataFrame(output)


def _balanced(group: pd.DataFrame) -> float:
    """Balanced accuracy de familias ganadoras elegibles; abstenciones cuentan como fallos."""
    if group.empty:
        return float("nan")
    labels = ["dist_perovskite", "hexagonal", "needle"]
    values = []
    for label in labels:
        subset = group.loc[group.GS_DFT_class.eq(label)]
        if len(subset):
            values.append(float(subset.GS_predicted_class.eq(label).mean()))
    return float(np.mean(values)) if values else float("nan")


def _f1(group: pd.DataFrame) -> float:
    """Macro-F1 de clases GS; una predicción empatada no obtiene acierto."""
    if group.empty:
        return float("nan")
    scores = []
    for label in ("dist_perovskite", "hexagonal", "needle"):
        tp = int((group.GS_DFT_class.eq(label) & group.GS_predicted_class.eq(label)).sum())
        fp = int((~group.GS_DFT_class.eq(label) & group.GS_predicted_class.eq(label)).sum())
        fn = int((group.GS_DFT_class.eq(label) & ~group.GS_predicted_class.eq(label)).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def _confusions(ensemble: pd.DataFrame, comp: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Exporta conteos de confusión de GS/pairwise para observaciones físicas únicas."""
    gs_rows = []
    for row in comp.itertuples(index=False):
        if not row.GS_eligible:
            continue
        gs_rows.append({"model": row.model, "composition_id": row.composition_id,
                        "true_class": row.GS_DFT_class, "predicted_class": row.GS_predicted_class})
    pd.DataFrame(gs_rows).to_csv(REPORT_DIR / "comparative_analysis/gs_confusion_observations.csv", index=False)
    gs_counts = pd.DataFrame(gs_rows).groupby(["model", "true_class", "predicted_class"], dropna=False).size().rename("count").reset_index()
    gs_counts.to_csv(REPORT_DIR / "comparative_analysis/gs_confusion_counts.csv", index=False)
    pair_rows = []
    for row in pairs.itertuples(index=False):
        pair_rows.append({"model": row.model, "composition_id": row.composition_id,
                          "family_i": row.family_i, "family_j": row.family_j,
                          "DFT_class": row.DFT_class, "pred_class": row.pred_class,
                          "eligible_primary": row.eligible_primary})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(REPORT_DIR / "comparative_analysis/pairwise_confusion_observations.csv", index=False)
    pair_counts = pair_df.groupby(["model", "DFT_class", "pred_class"], dropna=False).size().rename("count").reset_index()
    pair_counts.to_csv(REPORT_DIR / "comparative_analysis/pairwise_confusion_counts.csv", index=False)


def _paired_bootstrap(comp: pd.DataFrame) -> dict[str, Any]:
    """Bootstrap diferencias pareadas por fórmula, manteniendo tripletes juntos."""
    rng = np.random.default_rng(13072026)
    nboot = 2000
    output: dict[str, Any] = {"status": "PASS", "replicates": nboot, "seed": 13072026,
                              "sampling_unit": "composition; its three polymorphs remain together",
                              "interval_type": "percentile 95%; exploratory, not a guarantee of superiority",
                              "comparisons": {}}
    for candidate, reference in COMPARISONS:
        a = comp.loc[comp.model.eq(candidate)].set_index("composition_id")
        b = comp.loc[comp.model.eq(reference)].set_index("composition_id")
        common = sorted(set(a.index) & set(b.index))
        comparison: dict[str, Any] = {"candidate": candidate, "reference": reference, "N_compositions": len(common), "metrics": {}}
        for metric in METRICS:
            aa, bb = a.loc[common, metric].to_numpy(float), b.loc[common, metric].to_numpy(float)
            valid = np.isfinite(aa) & np.isfinite(bb)
            aa, bb = aa[valid], bb[valid]
            if metric == "GS_correct":
                eligible = a.loc[np.asarray(common)[valid], "GS_eligible"].to_numpy(bool) & b.loc[np.asarray(common)[valid], "GS_eligible"].to_numpy(bool)
                aa, bb = aa[eligible], bb[eligible]
            if metric == "pairwise_accuracy":
                eligible = (a.loc[np.asarray(common)[valid], "pairwise_eligible_count"].to_numpy(float) > 0) & (b.loc[np.asarray(common)[valid], "pairwise_eligible_count"].to_numpy(float) > 0)
                aa, bb = aa[eligible], bb[eligible]
            if not len(aa):
                comparison["metrics"][metric] = {"n_paired": 0, "mean_difference_candidate_minus_reference": None}
                continue
            diff = aa - bb
            draws = np.empty(nboot, dtype=float)
            for index in range(nboot):
                draws[index] = float(np.mean(diff[rng.integers(0, len(diff), len(diff))]))
            comparison["metrics"][metric] = {"n_paired": len(diff), "mean_difference_candidate_minus_reference": float(diff.mean()),
                "bootstrap_95_ci_low": float(np.quantile(draws, .025)), "bootstrap_95_ci_high": float(np.quantile(draws, .975))}
        output["comparisons"][f"{candidate}_vs_{reference}"] = comparison
    return output


def evaluate() -> dict[str, Any]:
    """Produce CSVs, comparaciones pareadas, intervalos y pruebas algebraicas."""
    out = REPORT_DIR / "comparative_analysis"; out.mkdir(parents=True, exist_ok=True)
    predictions = _load_all_predictions()
    raw_comp, raw_pairs, per_run = metric_tables(predictions)
    raw_comp.to_csv(out / "metrics_by_composition_seed.csv", index=False)
    raw_pairs.to_csv(out / "pairwise_observations_by_seed.csv", index=False)
    per_run.to_csv(out / "metrics_per_fold_seed.csv", index=False)
    ensemble = _ensemble(predictions)
    ens_comp, ens_pairs, _ = metric_tables(ensemble)
    ens_comp.to_csv(out / "metrics_by_composition_ensemble.csv", index=False)
    ens_pairs.to_csv(out / "pairwise_observations_ensemble.csv", index=False)
    summary = _macro_summary(ens_comp, ensemble)
    summary.to_csv(out / "comparative_metrics.csv", index=False)
    stability_rows = []
    for model, group in per_run.groupby("model", sort=False):
        for metric in ("MAE_absolute_mev_atom", "MAE_mu_mev_atom", "MAE_delta_centered_mev_atom",
                       "MAE_delta_GS_anchored_mev_atom", "GS_accuracy", "pairwise_accuracy"):
            fold_means = group.groupby("fold")[metric].mean().dropna()
            seed_means = group.loc[group.seed.ne(-1)].groupby("seed")[metric].mean().dropna()
            stability_rows.append({"model": model, "metric": metric,
                "mean_over_fold_seed_runs": float(group[metric].mean()),
                "sd_over_fold_seed_runs": float(group[metric].std(ddof=1)),
                "mean_of_fold_means": float(fold_means.mean()),
                "sd_between_fold_means_after_seed_average": float(fold_means.std(ddof=1)),
                "mean_of_seed_means": float(seed_means.mean()) if len(seed_means) else np.nan,
                "sd_between_seed_means_after_fold_average": float(seed_means.std(ddof=1)) if len(seed_means)>1 else np.nan,
                "fold_points": len(fold_means), "seed_points": len(seed_means)})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out / "metrics_stability_fold_seed.csv", index=False)
    _confusions(ensemble, ens_comp, ens_pairs)
    bootstrap = _paired_bootstrap(ens_comp)
    write_json(out / "paired_bootstrap_ci.json", bootstrap)
    paired_rows=[]
    for comparison_name, detail in bootstrap["comparisons"].items():
        for metric, estimate in detail["metrics"].items():
            paired_rows.append({"comparison":comparison_name,"candidate":detail["candidate"],"reference":detail["reference"],
                "metric":metric,**estimate})
    pd.DataFrame(paired_rows).to_csv(out / "paired_comparisons.csv", index=False)

    outlier_parts=[]
    for model, group in ens_comp.groupby("model", sort=False):
        part=group.nlargest(10,"MAE_absolute_mev_atom").copy();part["outlier_rank_within_model"]=np.arange(1,len(part)+1);outlier_parts.append(part)
    pd.concat(outlier_parts,ignore_index=True).to_csv(out / "largest_composition_errors.csv",index=False)

    # Algebraic and invariance checks: H3-A/B are exact delta-preserving recombinations of H2.
    checks = []
    h2 = predictions.loc[predictions.model.eq("H2")]
    for model in ("H3-A", "H3-B"):
        candidate = predictions.loc[predictions.model.eq(model)]
        joined = candidate.merge(h2, on=["structure_id", "composition_id", "polymorph_family", "fold", "seed"],
                                 suffixes=("_candidate", "_h2"), validate="one_to_one")
        cand_d = joined.groupby(["composition_id", "fold", "seed"]).apply(
            lambda g: g.set_index("polymorph_family").loc[list(("dist_perovskite", "hexagonal", "needle")), "E_pred_eV_atom_candidate"].to_numpy()
            - g.set_index("polymorph_family").loc[list(("dist_perovskite", "hexagonal", "needle")), "E_pred_eV_atom_candidate"].mean(), include_groups=False)
        h2_d = joined.groupby(["composition_id", "fold", "seed"]).apply(
            lambda g: g.set_index("polymorph_family").loc[list(("dist_perovskite", "hexagonal", "needle")), "E_pred_eV_atom_h2"].to_numpy()
            - g.set_index("polymorph_family").loc[list(("dist_perovskite", "hexagonal", "needle")), "E_pred_eV_atom_h2"].mean(), include_groups=False)
        maxdiff = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(cand_d, h2_d))
        if maxdiff > 1e-9:
            raise H3Error(f"{model}: recombinación cambió diferencias/ranking H2 ({maxdiff}).")
        checks.append({"model": model, "status": "PASS", "max_abs_centered_delta_difference_ev_atom": maxdiff,
                       "GS_pairwise_delta_invariant": True})
    write_json(out / "algebraic_checks.json", {"status": "PASS", "checks": checks,
        "ridge_cancels_from_pairwise_differences": True, "MSE_decomposition_checked_per_composition": True,
        "maximum_MSE_identity_abs_error": float(ens_comp.MSE_identity_abs_error.max()), "test_used": False})
    return {"status": "PASS", "models": summary.to_dict(orient="records"), "predictions_rows": len(predictions),
            "ensemble_structures_per_model": 210, "test_used": False}


def main() -> None:
    """Ejecuta evaluación después de que H3-A/B/C hayan producido sus OOF."""
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    try:
        print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
    except (H3Error, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"H3 evaluation FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
