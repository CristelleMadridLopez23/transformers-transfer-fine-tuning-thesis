#!/usr/bin/env python3
"""Audit the raw crystal dataset and produce a compact table for notebooks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results_no_meta.json"
DEFAULT_OUTPUT = TRANSFERABILITY_ROOT / "data" / "preprocessing" / "records.csv"


def split_record_id(record_id: str, formula: str) -> tuple[str, str, str]:
    parts = record_id.split("--")
    if len(parts) != 4:
        raise ValueError(f"Unexpected record id: {record_id!r}")
    structure_name, magnetic_order, calculation = parts[1:]
    prefix = f"{formula}_"
    prototype = structure_name[len(prefix) :] if structure_name.startswith(prefix) else structure_name
    return prototype, magnetic_order, calculation


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)

    rows = []
    for record_id, payload in source.items():
        results = payload["results"]
        structure = payload["structure"]
        gap = payload.get("gap", {})
        formula = results["formula"]
        prototype, magnetic_order, calculation = split_record_id(record_id, formula)
        rows.append(
            {
                "record_id": record_id,
                "formula": formula,
                "prototype": prototype,
                "magnetic_order": magnetic_order,
                "calculation": calculation,
                "converged": results.get("convergence"),
                "n_sites": len(structure.get("sites", [])),
                "volume": structure.get("lattice", {}).get("volume"),
                "energy_per_atom": results.get("E_per_at"),
                "relaxed_energy_per_atom": results.get("E_relax"),
                "bandgap": gap.get("bandgap"),
                "is_direct_gap": gap.get("is_direct"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    formula_counts = Counter(row["formula"] for row in rows)
    summary = {
        "source": str(args.input.resolve()),
        "records_table": str(args.output.resolve()),
        "records": len(rows),
        "converged": sum(row["converged"] is True for row in rows),
        "unique_formulas": len(formula_counts),
        "records_per_formula": dict(sorted(Counter(formula_counts.values()).items())),
        "calculations": dict(sorted(Counter(row["calculation"] for row in rows).items())),
        "prototypes": dict(sorted(Counter(row["prototype"] for row in rows).items())),
        "magnetic_orders": dict(sorted(Counter(row["magnetic_order"] for row in rows).items())),
        "n_sites": dict(sorted(Counter(row["n_sites"] for row in rows).items())),
        "targets": {
            key: numeric_summary([float(row[key]) for row in rows])
            for key in ("energy_per_atom", "relaxed_energy_per_atom", "bandgap")
        },
    }
    summary_path = args.output.with_name("dataset_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
