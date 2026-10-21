"""Evaluate one completed final fine-tuned model on untouched test compositions.

Report energy errors for all accepted test frames and polymorph ordering only for
complete static candidate sets. Reuse the original experiment's physical metrics;
never select a model, epoch or tokenizer with test labels.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from collections import Counter
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from finetuning import FrameDataset, collate, load_rows, model, pad_id
from workflow import (LOGGER, ROOT, cli, fingerprint, load_config, read_json,
                      require_audit, sha256, write_json)


def physical_metrics() -> Any:
    """Load existing documented polymorph metric functions by Path; return module without CLI execution."""
    directory = ROOT / "model-pretrained/scripts/exp01_mattext_frozen"
    spec = importlib.util.spec_from_file_location("original_polymorph_metrics", directory / "05_evaluate_models.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize finite true/predicted eV/atom in frame; return sample count, MAE and RMSE."""
    residual = frame.prediction_ev_atom.to_numpy() - frame.energy_per_atom_ev.to_numpy()
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Empty or nonfinite predictions/references")
    return {"samples": len(frame), "mae_ev_atom": float(np.abs(residual).mean()),
            "rmse_ev_atom": float(np.sqrt(np.square(residual).mean()))}


def evaluate(config: dict[str, Any]) -> None:
    """Evaluate gated final artifact once; write real test energy/polymorph tables and resource provenance."""
    require_audit(config)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Final evaluation must run in the allocated experiment job")
    paths = config["paths"]
    artifact = paths["experiment"] / "artifacts/final"
    manifest = read_json(artifact / "training_manifest.json")
    if manifest["fold"] != -1 or manifest["fingerprint"] != fingerprint(config):
        raise ValueError("Final training provenance is not current")
    checkpoint_path = artifact / "best.pt"
    if sha256(checkpoint_path) != manifest["artifact_sha256"]:
        raise ValueError("Supervised model artifact integrity failure")
    metrics_dir = paths["reports"] / "metrics"
    if (metrics_dir / "evaluation_manifest.json").exists():
        raise FileExistsError("Final test evaluation already exists; do not reuse test for selection")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables = paths["reports"] / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = load_rows(config)
    test = [row for row in rows if row["split"] == "test"]
    loader = DataLoader(FrameDataset(test), batch_size=config["training"]["batch_size"], shuffle=False,
                        collate_fn=partial(collate, pad_id=pad_id(config)))
    network = model(config)
    network.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            predictions.extend(network(input_ids=batch["input_ids"].to(device),
                                      attention_mask=batch["attention_mask"].to(device)).logits.squeeze(-1).cpu().tolist())
    frame = pd.DataFrame([{key: row[key] for key in ["structure_id", "reduced_formula", "polymorph_family",
                          "data_origin", "source_key", "frame_index", "energy_per_atom_ev"]} for row in test])
    frame["prediction_ev_atom"] = predictions
    frame.to_csv(tables / "predictions_test.csv", index=False)
    overall = error_metrics(frame)
    by_origin = [{"data_origin": name, **error_metrics(group)} for name, group in frame.groupby("data_origin")]
    pd.DataFrame(by_origin).to_csv(tables / "metrics_by_origin.csv", index=False)
    pd.DataFrame([{"polymorph_family": name, **error_metrics(group)} for name, group in
                  frame.groupby("polymorph_family")]).to_csv(tables / "metrics_by_polymorph.csv", index=False)
    pd.DataFrame([{"reduced_formula": name, **error_metrics(group)} for name, group in
                  frame.groupby("reduced_formula")]).to_csv(tables / "energy_metrics_by_composition.csv", index=False)
    metrics = physical_metrics()
    expected_families = {"dist_perovskite", "hexagonal", "needle"}
    training_static = pd.DataFrame([r for r in rows if r["split"] == "train" and r["data_origin"] == "static"])
    complete_training = [group for _, group in training_static.groupby("reduced_formula")
                         if len(group) == 3 and set(group.polymorph_family) == expected_families]
    if not complete_training:
        raise ValueError("No complete training compositions for frequent-prototype baseline")
    tolerance = config["training"]["evaluation_degeneracy_tolerance_ev_atom"]
    preferred = metrics.frequent_family(pd.concat(complete_training), tolerance)
    composition, pairs, skipped, baseline = [], [], [], []
    for formula, group in frame.loc[frame.data_origin.eq("static")].groupby("reduced_formula"):
        if len(group) != 3 or set(group.polymorph_family) != expected_families:
            skipped.append({"reduced_formula": formula, "reason": "Incomplete static candidate set; minimum selection not evaluated"})
            continue
        values, local_pairs = metrics.composition_metrics(group, tolerance)
        composition.append(values)
        pairs.extend(local_pairs)
        chosen = group.loc[group.polymorph_family.eq(preferred)].iloc[0]
        regret = float(chosen.energy_per_atom_ev - group.energy_per_atom_ev.min())
        baseline.append({"reduced_formula": formula, "preferred_family": preferred,
                         "minimum_selection_accuracy": float(regret <= tolerance),
                         "selection_regret_mev_atom": regret * 1000})
    pd.DataFrame(composition).to_csv(tables / "static_polymorph_metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(tables / "static_relative_energy_pairs.csv", index=False)
    pd.DataFrame(baseline).to_csv(tables / "frequent_prototype_baseline.csv", index=False)
    pd.DataFrame(skipped, columns=["reduced_formula", "reason"]).to_csv(tables / "unevaluated_static_compositions.csv", index=False)
    selection = pd.DataFrame(composition)
    summary = {**overall, "polymorph_compositions_evaluated": len(composition),
               "polymorph_compositions_incomplete": len(skipped), "baseline_preferred_family": preferred,
               "minimum_selection_accuracy": float(selection.minimum_selection_accuracy.mean()) if len(selection) else None,
               "selection_regret_mev_atom": float(selection.selection_regret_mev_atom.mean()) if len(selection) else None,
               "baseline_selection_accuracy": float(np.mean([r["minimum_selection_accuracy"] for r in baseline])) if baseline else None}
    write_json(metrics_dir / "metrics_summary.json", summary)
    write_json(metrics_dir / "evaluation_manifest.json", {"fingerprint": fingerprint(config),
        "test_compositions": sorted(set(frame.reduced_formula)), "test_used_for_selection": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "artifact_sha256": manifest["artifact_sha256"],
        "prediction_sha256": sha256(tables / "predictions_test.csv")})
    LOGGER.info("evaluation_complete frames=%d static_compositions=%d", len(frame), len(composition))


def main() -> None:
    """Run final evaluation from CLI settings; log descriptive failures and propagate exit status."""
    try:
        evaluate(load_config(cli(__doc__).parse_args().config))
    except Exception:
        LOGGER.exception("final_evaluation_failed")
        raise


if __name__ == "__main__":
    main()
