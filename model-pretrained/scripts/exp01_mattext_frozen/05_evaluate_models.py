"""Evaluate saved independent regressors once on held-out compositions.

Reads audited embeddings and fitted heads; writes real test predictions, paired
energy errors, composition/family metrics and training-only majority baseline.
"""
from __future__ import annotations

import importlib
import itertools
import os
from collections import Counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils import (LOGGER, audit_fingerprint, cli, load_config, load_embeddings,
                   read_json, require_gate, sha256, write_json)


def composition_metrics(frame: pd.DataFrame, tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one composition with true/predicted eV/atom; return metrics and pair rows.

    True minima within tolerance are accepted as quasi-degenerate. Predicted ties
    use lexicographic structure ID, without consulting true energies. Directional
    concordance excludes true quasi-degenerate pairs; predicted ties count 0.5.
    """
    if frame.reduced_formula.nunique() != 1 or len(frame) < 2:
        raise ValueError("Relative metrics require >=2 structures from one composition")
    if tolerance < 0 or not np.isfinite(frame[["energy_per_atom_ev", "prediction_ev_atom"]]).all().all():
        raise ValueError("Invalid tolerance or nonfinite predictions/references")
    group = frame.sort_values("structure_id")
    best_true = float(group.energy_per_atom_ev.min())
    chosen = group.loc[group.prediction_ev_atom.eq(group.prediction_ev_atom.min())].iloc[0]
    regret = float(chosen.energy_per_atom_ev - best_true)
    pairs, concordance = [], []
    for first, second in itertools.combinations(group.to_dict("records"), 2):
        true_delta = second["energy_per_atom_ev"] - first["energy_per_atom_ev"]
        predicted_delta = second["prediction_ev_atom"] - first["prediction_ev_atom"]
        correct = None
        if abs(true_delta) > tolerance:
            correct = 0.5 if abs(predicted_delta) <= tolerance else float(np.sign(true_delta) == np.sign(predicted_delta))
            concordance.append(correct)
        pairs.append({"first_id": first["structure_id"], "second_id": second["structure_id"],
                      "reduced_formula": first["reduced_formula"], "true_delta_ev_atom": true_delta,
                      "predicted_delta_ev_atom": predicted_delta,
                      "absolute_delta_error_ev_atom": abs(predicted_delta - true_delta),
                      "direction_concordance": correct})
    residuals = group.prediction_ev_atom - group.energy_per_atom_ev
    metrics = {"reduced_formula": group.reduced_formula.iloc[0], "samples": len(group),
               "mae_ev_atom": float(np.abs(residuals).mean()),
               "rmse_ev_atom": float(np.sqrt(np.square(residuals).mean())),
               "relative_energy_mae_ev_atom": float(np.mean([item["absolute_delta_error_ev_atom"] for item in pairs])),
               "minimum_selection_accuracy": float(regret <= tolerance),
               "selection_regret_mev_atom": regret * 1000,
               "pairwise_concordance": float(np.mean(concordance)) if concordance else None,
               "nondegenerate_pairs": len(concordance), "chosen_id": chosen.structure_id,
               "chosen_family": chosen.polymorph_family}
    return metrics, pairs


def frequent_family(training: pd.DataFrame, tolerance: float) -> str:
    """Find family most frequently among GGA minima in training compositions; return family."""
    counts: Counter[str] = Counter()
    for _, group in training.groupby("reduced_formula"):
        minima = group.loc[group.energy_per_atom_ev <= group.energy_per_atom_ev.min() + tolerance]
        for family in minima.polymorph_family.unique():
            counts[family] += 1 / minima.polymorph_family.nunique()
    if not counts:
        raise ValueError("No training GGA minima for prototype baseline")
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def evaluate(config: dict[str, Any]) -> None:
    """Evaluate saved models from config on test compositions; persist physically labeled metrics."""
    require_gate(config)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scientific evaluation must run inside the scheduled experiment")
    frame, arrays = load_embeddings(config)
    paths = config["paths"]
    artifacts = paths["experiment"] / "artifacts"
    trained = read_json(artifacts / "training_manifest.json")
    if trained["fingerprint"] != audit_fingerprint(config):
        raise RuntimeError("Model training provenance differs from audited inputs")
    tables, metrics_dir = paths["reports"] / "tables", paths["reports"] / "metrics"
    if (metrics_dir / "evaluation_manifest.json").exists():
        raise RuntimeError("Final test evaluation already exists; do not repeatedly inspect test for selection")
    predictor_module = importlib.import_module("04_train_regressors")
    test = frame.split.eq("test").to_numpy()
    development = frame.loc[~test]
    test_frame = frame.loc[test].copy()
    tolerance = config["evaluation"]["degeneracy_tolerance_ev_atom"]
    summary, by_composition, by_family, predictions, pair_rows = [], [], [], [], []
    for name, record in trained["models"].items():
        artifact = config["root"] / record["path"]
        if sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"Model artifact integrity failed: {name}")
        model = joblib.load(artifact)
        current = test_frame.copy()
        current["model"] = name
        current["prediction_ev_atom"] = predictor_module.predict(model, arrays[trained["common_pooling"]][test])
        current["residual_ev_atom"] = current.prediction_ev_atom - current.energy_per_atom_ev
        predictions.extend(current.to_dict("records"))
        local_metrics = []
        for _, group in current.groupby("reduced_formula"):
            values, pairs = composition_metrics(group, tolerance)
            by_composition.append({"model": name, **values})
            local_metrics.append(values)
            pair_rows.extend({"model": name, **pair} for pair in pairs)
        composition_table = pd.DataFrame(local_metrics)
        residual = current.residual_ev_atom.to_numpy()
        summary.append({"model": name, "mae_ev_atom": float(np.abs(residual).mean()),
                        "rmse_ev_atom": float(np.sqrt(np.square(residual).mean())),
                        "relative_energy_mae_ev_atom": float(composition_table.relative_energy_mae_ev_atom.mean()),
                        "minimum_selection_accuracy": float(composition_table.minimum_selection_accuracy.mean()),
                        "selection_regret_mev_atom": float(composition_table.selection_regret_mev_atom.mean()),
                        "pairwise_concordance": float(composition_table.pairwise_concordance.mean())
                        if composition_table.pairwise_concordance.notna().any() else None})
        for family, group in current.groupby("polymorph_family"):
            by_family.append({"model": name, "polymorph_family": family, "samples": len(group),
                              "mae_ev_atom": float(np.abs(group.residual_ev_atom).mean()),
                              "rmse_ev_atom": float(np.sqrt(np.square(group.residual_ev_atom).mean()))})
    family = frequent_family(development, tolerance)
    baseline = []
    for formula, group in test_frame.groupby("reduced_formula"):
        candidates = group.loc[group.polymorph_family.eq(family)].sort_values("structure_id")
        if candidates.empty:
            raise ValueError(f"Baseline's selected family {family} absent in {formula}")
        chosen = candidates.iloc[0]
        regret = float(chosen.energy_per_atom_ev - group.energy_per_atom_ev.min())
        row = {"model": "frequent_prototype", "reduced_formula": formula,
               "chosen_id": chosen.structure_id, "chosen_family": family,
               "minimum_selection_accuracy": float(regret <= tolerance),
               "selection_regret_mev_atom": regret * 1000}
        baseline.append(row)
        by_composition.append(row)
    summary.append({"model": "frequent_prototype", "minimum_selection_accuracy": float(np.mean(
        [row["minimum_selection_accuracy"] for row in baseline])),
        "selection_regret_mev_atom": float(np.mean([row["selection_regret_mev_atom"] for row in baseline]))})
    pd.DataFrame(summary).to_csv(tables / "metrics_summary.csv", index=False)
    pd.DataFrame(by_composition).to_csv(tables / "metrics_by_composition.csv", index=False)
    pd.DataFrame(by_family).to_csv(tables / "metrics_by_polymorph.csv", index=False)
    pd.DataFrame(predictions).to_csv(tables / "predictions_test.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(tables / "relative_energy_pairs.csv", index=False)
    pd.read_csv(artifacts / "cross_validation_results.csv").to_csv(tables / "cross_validation_results.csv", index=False)
    write_json(metrics_dir / "best_hyperparameters.json", read_json(artifacts / "best_hyperparameters.json"))
    write_json(metrics_dir / "evaluation_manifest.json", {"fingerprint": audit_fingerprint(config),
               "test_compositions": int(test_frame.reduced_formula.nunique()), "test_samples": len(test_frame),
               "degeneracy_tolerance_ev_atom": tolerance, "majority_family_from_training": family,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "model_ranking_by_test": "not performed; MLP remains the predeclared primary head"})
    LOGGER.info("evaluation_saved test_samples=%d test_compositions=%d", len(test_frame), test_frame.reduced_formula.nunique())


def main() -> None:
    """Run final test evaluation from CLI configuration; return None."""
    evaluate(load_config(cli(__doc__).parse_args().config))


if __name__ == "__main__":
    main()
