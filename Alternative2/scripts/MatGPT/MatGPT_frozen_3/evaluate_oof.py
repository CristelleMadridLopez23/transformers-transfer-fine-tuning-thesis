#!/usr/bin/env python3
"""Recalcula resultados R1-R4 y H0-H2 desde predicciones OOF originales."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (FAMILIES, H2_REPORT, REPORT_DIR, ExperimentError,
    load_dev, log, sha256_file, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.metrics import classification_metrics, metric_tables
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.paired_statistics import paired_bootstrap


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga H0-H2 históricos y predicciones nuevas sin incluir IDs de test."""
    data = load_dev()
    dev_ids = set(data.structure_id)
    h2_path = H2_REPORT / "oof_predictions.csv"
    h2 = pd.read_csv(h2_path, low_memory=False)
    h2 = h2.loc[h2.model.isin(["H0", "H1", "H2"])].copy()
    h2["structure_id"] = h2.structure_id.astype(str)
    if set(h2.structure_id) != dev_ids or len(h2) != 1890:
        raise ExperimentError("Predicciones históricas H0-H2 deben contener solo los 210 IDs dev por seed/fold.")
    h2["target_kind"] = "absolute"
    h2["layer_pooling"] = h2.model.map({"H0": "FINAL_END", "H1": "FINAL_END triplet mean", "H2": "FINAL_MEAN common; FINAL_END residual"})
    parts = [h2]
    for ridge_path in (REPORT_DIR / "R1_linear_probes" / "oof_predictions.csv",
                       REPORT_DIR / "R4_layers_pooling" / "oof_predictions.csv"):
        if ridge_path.is_file():
            parts.append(pd.read_csv(ridge_path))
    for folder in ("R2_head_capacity", "R3_regularization"):
        root = REPORT_DIR / folder
        for path in root.glob("*/oof_predictions.csv"):
            frame = pd.read_csv(path)
            if len(frame):
                frame["layer_pooling"] = "FINAL_MEAN common; FINAL_END residual"
                frame["target_kind"] = "absolute"
                parts.append(frame)
    raw = pd.concat(parts, ignore_index=True, sort=False)
    if set(raw.structure_id.astype(str)) - dev_ids:
        raise ExperimentError("Se detectaron IDs ajenos a development en las predicciones.")
    if not np.isfinite(raw.E_pred_eV_atom.to_numpy(float)).all():
        raise ExperimentError("OOF contiene predicciones no finitas.")
    data_map = data.set_index("structure_id")
    for row in raw.itertuples(index=False):
        if not np.isclose(row.E_DFT_eV_atom, data_map.loc[str(row.structure_id), "E_DFT_eV_atom"], atol=1e-10, rtol=0):
            raise ExperimentError(f"Etiqueta OOF inconsistente para {row.structure_id}.")
    return raw, data


def aggregate_seed_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    """Promedia energías por estructura y modelo; conserva fold único y metadatos."""
    keys = ["model", "structure_id", "composition_id", "polymorph_family", "fold"]
    ensemble = raw.groupby(keys, as_index=False).agg(E_DFT_eV_atom=("E_DFT_eV_atom", "first"),
        E_pred_eV_atom=("E_pred_eV_atom", "mean"), seed_count=("seed", "nunique"),
        seed_values=("seed", lambda s: ";".join(str(v) for v in sorted(set(s)))),
        E_pred_mu_eV_atom=("E_pred_mu_eV_atom", "mean"), delta_pred_centered_eV_atom=("delta_pred_centered_eV_atom", "mean"),
        target_kind=("target_kind", "first"), layer_pooling=("layer_pooling", "first"),
        head_width=("head_width", "first"), dropout=("dropout", "first"), weight_decay=("weight_decay", "first"),
        alpha_mu=("alpha_mu", "first"), alpha_delta=("alpha_delta", "first"), historical_split=("historical_split", "first"))
    # Deterministic ridge rows carry seed=-1 and remain single observations.
    return ensemble


def aggregate_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por fold×semilla con exclusiones y denominadores."""
    out = []
    for (model, fold, seed), group in table.groupby(["model", "fold", "seed"], dropna=False):
        target_kind = str(group.target_kind.iloc[0]) if "target_kind" in group else "absolute"
        ranking_applicable = target_kind != "mu"
        eligible_gs = group.loc[group.GS_eligible.astype(bool)] if ranking_applicable else group.iloc[0:0]
        pair_correct = int(group.pairwise_correct_count.sum()) if ranking_applicable else 0
        pair_n = int(group.pairwise_eligible_count.sum()) if ranking_applicable else 0
        out.append({"model": model, "fold": fold, "seed": seed, "compositions": group.composition_id.nunique(),
            "MAE_absolute_macro_mev_atom": group.MAE_absolute_mev_atom.mean(),
            "MAE_mu_macro_mev_atom": group.MAE_mu_mev_atom.mean(),
            "MAE_delta_centered_macro_mev_atom": group.MAE_delta_centered_mev_atom.mean(),
            "MAE_delta_GS_anchored_macro_mev_atom": group.MAE_delta_GS_anchored_mev_atom.mean(),
            "MAE_delta_own_min_legacy_macro_mev_atom": group.MAE_delta_own_min_legacy_mev_atom.mean(),
            "GS_accuracy": eligible_gs.GS_correct.mean() if len(eligible_gs) else np.nan,
            "GS_n_eligible": len(eligible_gs), "GS_n_DFT_ties_excluded": int(group.GS_DFT_near_degenerate.sum()),
            "GS_predicted_abstentions": int(group.GS_predicted_abstention.sum()),
            "pairwise_accuracy": pair_correct / pair_n if pair_n else np.nan,
            "pairwise_correct": pair_correct, "pairwise_n_eligible": pair_n,
            "pairwise_signed_MAE_mev_atom": group.pairwise_signed_MAE_mev_atom.mean(),
            "mean_range_error_mev_atom": group.absolute_range_error_mev_atom.mean(),
            "collapsed_compositions": int(group.predicted_near_collapse.sum()), "MSE_identity_max_abs_error": group.MSE_identity_abs_error.max()})
    return pd.DataFrame(out)


def make_confusions(ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye observaciones físicas GS y pares canónicos una sola vez."""
    summaries, pairs = metric_tables(ensemble)
    gs = summaries[["model", "composition_id", "GS_DFT_family", "GS_predicted_family", "GS_eligible",
        "GS_DFT_near_degenerate", "GS_predicted_abstention", "GS_DFT_gap_mev_atom"]].copy()
    gs["true_class"] = gs.GS_DFT_family
    gs["predicted_class"] = gs.GS_predicted_family
    # Exactitud primaria excludes DFT near-ties; matrix retains them with their actual minimum for diagnosis.
    gs.loc[gs.GS_DFT_near_degenerate, "primary_denominator"] = False
    gs.loc[~gs.GS_DFT_near_degenerate, "primary_denominator"] = True
    gs["primary_denominator"] = gs.primary_denominator.fillna(False).astype(bool)
    gs_rows = []
    for model, group in gs.groupby("model"):
        for true in list(FAMILIES) + ["DFT near-tie"]:
            subset = group.loc[group.true_class.eq(true) & group.primary_denominator] if true != "DFT near-tie" else group.loc[~group.primary_denominator]
            for pred in list(FAMILIES) + ["tie/abstain"]:
                count = int(subset.predicted_class.eq(pred).sum())
                denom = len(subset)
                gs_rows.append({"model": model, "true_class": true, "predicted_class": pred, "count": count,
                    "row_denominator": denom, "row_percent": (100 * count / denom if denom else np.nan),
                    "primary_GS_row": true != "DFT near-tie"})
    pair_rows = []
    for model, group in pairs.groupby("model"):
        for true in ("i_lower", "tie", "j_lower"):
            sub = group.loc[group.dft_class.eq(true)]
            for pred in ("i_lower", "tie", "j_lower"):
                count = int(sub.pred_class.eq(pred).sum())
                pair_rows.append({"model": model, "true_class": true, "predicted_class": pred,
                    "count": count, "row_denominator": len(sub), "row_percent": 100 * count / len(sub) if len(sub) else np.nan,
                    "pair_orientation": "(perovskita,hexagonal),(perovskita,aguja),(hexagonal,aguja)"})
    return gs, pd.DataFrame(gs_rows), pd.DataFrame(pair_rows)


def run(bootstrap_replicates: int = 2000) -> dict[str, Any]:
    """Evalúa y guarda CSV/JSON solo dentro del árbol MatGPT_frozen_3."""
    if bootstrap_replicates < 2000:
        raise ExperimentError("Se requieren al menos 2,000 remuestras bootstrap.")
    raw, data = load_predictions()
    ensemble = aggregate_seed_predictions(raw)
    if ensemble.groupby("model").structure_id.nunique().lt(210).any():
        raise ExperimentError("Algún modelo no cubre las 210 estructuras OOF.")
    raw_path = REPORT_DIR / "comparisons" / "raw_predictions_all_models.csv"
    ensemble_path = REPORT_DIR / "comparisons" / "ensemble_predictions.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    ensemble.to_csv(ensemble_path, index=False)
    comp_metrics, pairs = metric_tables(ensemble)
    model_target = ensemble.groupby("model").target_kind.first().to_dict()
    comp_metrics["target_kind"] = comp_metrics.model.map(model_target)
    comp_metrics["absolute_metrics_applicable"] = comp_metrics.target_kind.isin(["absolute", "mu_plus_delta"])
    comp_metrics["mu_metrics_applicable"] = ~comp_metrics.target_kind.eq("delta")
    comp_metrics["relative_metrics_applicable"] = ~comp_metrics.target_kind.eq("mu")
    comp_metrics["ranking_discriminative"] = comp_metrics.target_kind.ne("mu")
    comp_metrics.loc[~comp_metrics.absolute_metrics_applicable, ["MAE_absolute_mev_atom", "MSE_total_mev2_atom2"]] = np.nan
    comp_metrics.loc[~comp_metrics.mu_metrics_applicable, ["MAE_mu_mev_atom", "MSE_mean_mev2_atom2"]] = np.nan
    comp_metrics.loc[~comp_metrics.relative_metrics_applicable, ["MAE_delta_GS_anchored_mev_atom",
        "MAE_delta_own_min_legacy_mev_atom", "MAE_delta_centered_mev_atom", "MSE_residual_mev2_atom2"]] = np.nan
    comp_metrics.loc[~comp_metrics.ranking_discriminative, ["GS_correct", "GS_eligible", "pairwise_accuracy",
        "pairwise_correct_count", "pairwise_eligible_count", "pairwise_signed_MAE_mev_atom"]] = np.nan
    comp_metrics.to_csv(REPORT_DIR / "comparisons" / "metrics_by_composition.csv", index=False)
    pairs.to_csv(REPORT_DIR / "comparisons" / "all_pairwise_observations.csv", index=False)
    per_seed, _ = metric_tables(raw)
    per_seed["target_kind"] = per_seed.model.map(model_target)
    per_seed.loc[per_seed.target_kind.eq("delta"), ["MAE_absolute_mev_atom", "MAE_mu_mev_atom",
        "MSE_total_mev2_atom2", "MSE_mean_mev2_atom2"]] = np.nan
    per_seed.loc[per_seed.target_kind.eq("mu"), ["MAE_delta_centered_mev_atom", "MAE_delta_GS_anchored_mev_atom",
        "MAE_delta_own_min_legacy_mev_atom", "MSE_residual_mev2_atom2", "GS_correct", "GS_eligible",
        "pairwise_accuracy", "pairwise_correct_count", "pairwise_eligible_count"]] = np.nan
    aggregate_metric_table(per_seed).to_csv(REPORT_DIR / "comparisons" / "metrics_per_fold_seed.csv", index=False)
    per_seed_summary = aggregate_metric_table(per_seed).groupby(["model", "seed"], as_index=False).mean(numeric_only=True)
    per_seed_summary.to_csv(REPORT_DIR / "comparisons" / "metrics_per_seed.csv", index=False)
    gs_obs, gs_matrix, pair_matrix = make_confusions(ensemble)
    summary = []
    for model, group in comp_metrics.groupby("model"):
        applicable = bool(group.ranking_discriminative.iloc[0])
        eligible = group.loc[group.GS_eligible.astype(bool)] if applicable else group.iloc[0:0]
        pairs_model = pairs.loc[pairs.model.eq(model)]
        correct, pair_n = int(pairs_model.loc[pairs_model.eligible_primary, "correct_primary"].sum()), int(pairs_model.eligible_primary.sum())
        if not applicable:
            correct, pair_n = 0, 0
        gs_valid = gs_obs.loc[gs_obs.model.eq(model) & gs_obs.primary_denominator,
            ["true_class", "predicted_class"]]
        cls = classification_metrics(gs_valid) if applicable else {"balanced_accuracy": np.nan, "macro_f1": np.nan, "n": 0}
        target_kind = str(ensemble.loc[ensemble.model.eq(model), "target_kind"].iloc[0])
        summary.append({"model": model, "target_kind": target_kind,
            "absolute_metrics_applicable": target_kind in ("absolute", "mu_plus_delta"),
            "mu_metrics_applicable": target_kind != "delta", "ranking_discriminative": target_kind != "mu",
            "layer_pooling": str(ensemble.loc[ensemble.model.eq(model), "layer_pooling"].iloc[0]),
            "structures": int(ensemble.loc[ensemble.model.eq(model), "structure_id"].nunique()),
            "compositions": int(group.composition_id.nunique()), "MAE_absolute_macro_mev_atom": group.MAE_absolute_mev_atom.mean(),
            "RMSE_absolute_pooled_mev_atom": (float(np.sqrt(np.mean((ensemble.loc[ensemble.model.eq(model), "E_pred_eV_atom"]-
                ensemble.loc[ensemble.model.eq(model), "E_DFT_eV_atom"])**2))*1000)
                if target_kind in ("absolute", "mu_plus_delta") else np.nan),
            "MAE_mu_mev_atom": group.MAE_mu_mev_atom.mean(), "MAE_delta_centered_mev_atom": group.MAE_delta_centered_mev_atom.mean(),
            "MAE_delta_GS_anchored_mev_atom": group.MAE_delta_GS_anchored_mev_atom.mean(),
            "MAE_delta_own_min_legacy_mev_atom": group.MAE_delta_own_min_legacy_mev_atom.mean(),
            "GS_accuracy": eligible.GS_correct.mean() if len(eligible) else np.nan,
            "GS_correct": int(eligible.GS_correct.sum()), "GS_n_eligible": len(eligible),
            "GS_n_DFT_ties_excluded": int(group.GS_DFT_near_degenerate.sum()),
            "GS_predicted_abstentions": int(group.GS_predicted_abstention.sum()),
            "pairwise_accuracy": correct/pair_n if pair_n else np.nan, "pairwise_correct": correct,
            "pairwise_n_eligible": pair_n, "pairwise_signed_MAE_mev_atom": (
                float(np.mean(np.abs(pairs_model.diff_pred_mev_atom - pairs_model.diff_dft_mev_atom)))
                if len(pairs_model) else np.nan),
            "mean_range_error_mev_atom": group.absolute_range_error_mev_atom.mean(),
            "DFT_range_mean_mev_atom": group.DFT_range_mev_atom.mean(),
            "predicted_range_mean_mev_atom": group.predicted_range_mev_atom.mean(),
            "near_collapse_fraction": group.predicted_near_collapse.mean(),
            "MSE_total_mev2_atom2": group.MSE_total_mev2_atom2.mean(),
            "MSE_mean_mev2_atom2": group.MSE_mean_mev2_atom2.mean(),
            "MSE_residual_mev2_atom2": group.MSE_residual_mev2_atom2.mean(),
            "MSE_identity_max_abs_error": group.MSE_identity_abs_error.max(),
            "GS_balanced_accuracy": cls["balanced_accuracy"], "GS_macro_F1": cls["macro_f1"],
            "GS_balance_N": cls["n"]})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(REPORT_DIR / "comparisons" / "comparative_metrics.csv", index=False)
    gs_obs.to_csv(REPORT_DIR / "comparisons" / "gs_observations.csv", index=False)
    gs_matrix.to_csv(REPORT_DIR / "comparisons" / "confusion_matrices.csv", index=False)
    pair_matrix.to_csv(REPORT_DIR / "comparisons" / "pairwise_confusion_matrices.csv", index=False)
    metrics_for_bootstrap = comp_metrics.rename(columns={"MAE_absolute_mev_atom": "MAE_absolute_mev_atom"})
    bootstrap = paired_bootstrap(metrics_for_bootstrap, bootstrap_replicates, 13072026, reference="H2")
    write_json(REPORT_DIR / "comparisons" / "paired_bootstrap.json", bootstrap)
    # D1 is joined only as an external energy reference, never used as a model component.
    d1_path = PROJECT / "Alternative2/reports/MatGPT_diagnostics/D1/v3/oof_predictions.csv"
    d1 = pd.read_csv(d1_path)
    d1 = d1.loc[d1.configuration.eq("D1-B Ridge composición")].copy()
    d1 = d1.rename(columns={"composition_group": "composition_id", "energy_per_atom_ev": "E_DFT_eV_atom",
        "energy_pred_ev_atom": "E_pred_eV_atom"})
    d1.structure_id = d1.structure_id.astype(str)
    if set(d1.structure_id) != set(data.structure_id) or not np.allclose(d1.E_DFT_eV_atom, data.set_index("structure_id").loc[d1.structure_id].E_DFT_eV_atom.to_numpy()):
        raise ExperimentError("D1 control externo no coincide con development.")
    d1_mae = float(np.mean(np.abs(d1.E_pred_eV_atom-d1.E_DFT_eV_atom))*1000)
    write_json(REPORT_DIR / "baseline_audit" / "d1_external_control.json", {"status": "PASS", "MAE_absolute_mev_atom": d1_mae,
        "same_210_development_ids": True, "composition_constant_prediction": all(g.E_pred_eV_atom.nunique()==1 for _,g in d1.groupby("composition_id")),
        "ranking": "indeterminate/abstention", "used_inside_R1_R4": False})
    # Model-specific provenance, source hashes, versions, test guardrail.
    write_json(REPORT_DIR / "manifests" / "evaluation_manifest.json", {"status": "PASS", "models": summary_frame.model.tolist(),
        "development_structures": 210, "development_compositions": 70, "outer_folds": 5,
        "test_used": False, "ensemble_rule": "average predictions by structure ID across seeds, then calculate each composition once",
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 13072026,
        "D1_mae_mev_atom": d1_mae, "evaluator_sha256": sha256_file(Path(__file__))})
    log("oof_evaluation_complete", models=len(summary_frame), rows=len(ensemble), test_used=False)
    return {"status": "PASS", "models": len(summary_frame), "rows": len(ensemble), "d1_mae": d1_mae}


def main() -> None:
    """Evalúa predicciones OOF con tolerancias configuradas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    try:
        if args.bootstrap < 2000:
            parser.error("--bootstrap debe ser >=2000")
        result = run(args.bootstrap)
        log("evaluation_complete", status=result["status"], models=result["models"], test_used=False)
    except (ExperimentError, OSError, ValueError, KeyError, TypeError) as exc:
        log("evaluation_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
