"""Table metrics for single-property and unconditional generation (Tables 1-3, ESI Sections S9-S10).

Run after sli_to_cif.py and the proxy script for the same config, e.g.
    python metrics.py --config-name form

Reads the reconstructed CSV (with the Predicted_* and Is_Novel columns added by
the proxy script) and reports the following for each target in sample.properties. As in the
paper's tables, the counts use the reconstructed structures that the proxy could evaluate
(finite prediction):
  Validity    evaluated structures / requested sequences (sample.number_sequences)
  Uniqueness  distinct SLICES strings (whitespace-normalized) / requested sequences
  Novelty     Is_Novel == 1, relative to the evaluated structures or the requested
              sequences (metrics.novelty_relative_to, as in each table caption)
  Errors      MAE, median absolute error, RMSE, mean signed error (prediction - target),
              MAPE and the hit rate within +/- metrics.hit_tolerance (skipped when null)

Unconditional samples have no target column: the same samples are evaluated against each
value in metrics.reference_targets, and the crystal system frequencies of the reconstructed
structures (Fig. 11) are saved next to the metrics table.
"""
import math
import os
from statistics import median

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig


def crystal_system_frequency(cif_paths):
    """Frequency of each crystal system over the structures whose crystal system could be assigned."""
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    systems, failed = [], 0
    for cif_path in cif_paths:
        try:
            systems.append(SpacegroupAnalyzer(Structure.from_file(cif_path)).get_crystal_system())
        except Exception as e:
            print(f"Could not assign a crystal system to {cif_path}: {e}")
            failed += 1
    if not systems:
        raise RuntimeError("No crystal system could be assigned; check that the CIF paths exist "
                           "(run metrics.py from the same folder as the other scripts).")
    print(f"Crystal system assigned for {len(systems)} structures, failed for {failed}.")
    counts = pd.Series(systems, dtype=object).value_counts()
    return pd.DataFrame({"crystal_system": counts.index, "count": counts.values,
                         "frequency_percent": 100.0 * counts.values / max(1, len(systems))})


@hydra.main(config_path="config", config_name="form", version_base=None)
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.proxy_model.csv_path)
    pred_col = cfg.metrics.prediction_column
    tolerance = cfg.metrics.get("hit_tolerance")
    requested = int(cfg.sample.number_sequences)
    relative_to = cfg.metrics.novelty_relative_to
    if relative_to not in ("reconstructed", "requested"):
        raise ValueError(f"metrics.novelty_relative_to must be 'reconstructed' or 'requested', not {relative_to!r}")
    if pred_col not in data.columns:
        raise SystemExit(f"Column '{pred_col}' not found in {cfg.proxy_model.csv_path}; run the proxy script for this config first.")
    has_novelty = "Is_Novel" in data.columns and pd.to_numeric(data["Is_Novel"], errors="coerce").notna().any()

    if cfg.training.selected_properties:
        target_values = pd.to_numeric(data[cfg.training.selected_properties[0]], errors="coerce")
        groups = [(float(t), data[np.isclose(target_values, float(t))]) for t in cfg.sample.properties]
    else:
        # Unconditional: one set of samples, evaluated against each reference target
        groups = [(float(t), data) for t in cfg.metrics.reference_targets]

    results = []
    for target, rows in groups:
        predictions = pd.to_numeric(rows[pred_col], errors="coerce")
        evaluated = rows[np.isfinite(predictions)]
        errors = [p - target for p in predictions[np.isfinite(predictions)]]
        n = len(errors)
        n_unique = len({" ".join(str(s).split()) for s in evaluated["SLICES"].dropna()})
        n_novel = int(pd.to_numeric(evaluated["Is_Novel"], errors="coerce").sum()) if has_novelty else None
        novelty_base = n if relative_to == "reconstructed" else requested
        if len(rows) == 0:
            print(f"Warning: no reconstructed structures for target {target}")

        results.append({
            "target": target,
            "requested": requested,
            "reconstructed": len(rows),
            "evaluated": n,
            "validity_percent": 100.0 * n / requested,
            "unique_slices": n_unique,
            "uniqueness_percent": 100.0 * n_unique / requested,
            "novel": n_novel,
            "novelty_percent": 100.0 * n_novel / novelty_base if n_novel is not None and novelty_base else math.nan,
            "mae": sum(abs(e) for e in errors) / n if n else math.nan,
            "median_absolute_error": median(abs(e) for e in errors) if n else math.nan,
            "rmse": math.sqrt(sum(e * e for e in errors) / n) if n else math.nan,
            "mean_signed_error": sum(errors) / n if n else math.nan,
            "mape_percent": 100.0 * sum(abs(e) for e in errors) / n / abs(target) if n and target != 0 else math.nan,
            # the small margin keeps errors that equal the tolerance (up to float rounding) inside it
            "hit_rate_percent": 100.0 * sum(abs(e) <= tolerance + 1e-9 for e in errors) / n if n and tolerance is not None else math.nan,
        })

    table = pd.DataFrame(results)
    os.makedirs(os.path.dirname(cfg.metrics.output_csv) or ".", exist_ok=True)
    table.to_csv(cfg.metrics.output_csv, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nNovelty is relative to the {relative_to} structures. Saved to {cfg.metrics.output_csv}")

    if not cfg.training.selected_properties:
        frequency = crystal_system_frequency(data["CIF Path"].dropna())
        frequency_csv = os.path.join(os.path.dirname(cfg.metrics.output_csv) or ".", "crystal_system_frequency.csv")
        frequency.to_csv(frequency_csv, index=False)
        print(frequency.round(2).to_string(index=False))
        print(f"Crystal system frequencies saved to {frequency_csv}")


if __name__ == "__main__":
    main()
