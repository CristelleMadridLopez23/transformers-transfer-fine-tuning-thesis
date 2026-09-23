#!/usr/bin/env python3
"""Create a composition-only, SMILES-shaped baseline table from crystal JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from functools import reduce
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results_no_meta.json"
DEFAULT_OUTPUT = TRANSFERABILITY_ROOT / "data" / "SMILES" / "composition_smiles.csv"
ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse the flat integer formulas used by this dataset."""
    position = 0
    counts: dict[str, int] = {}
    for match in ELEMENT_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Unsupported formula syntax: {formula!r}")
        element, raw_count = match.groups()
        count = int(raw_count) if raw_count else 1
        if count < 1:
            raise ValueError(f"Non-positive atom count in {formula!r}")
        counts[element] = counts.get(element, 0) + count
        position = match.end()
    if not counts or position != len(formula):
        raise ValueError(f"Unsupported formula syntax: {formula!r}")
    return counts


def reduce_counts(counts: dict[str, int]) -> dict[str, int]:
    divisor = reduce(math.gcd, counts.values())
    return {element: count // divisor for element, count in counts.items()}


def canonical_formula(counts: dict[str, int]) -> str:
    return "".join(
        f"{element}{count if count != 1 else ''}"
        for element, count in sorted(counts.items())
    )


def composition_smiles(counts: dict[str, int]) -> str:
    """Encode reduced composition as disconnected, neutral bracket atoms.

    This is syntactically a SMILES string, but deliberately makes no claim about
    bonds, oxidation states, coordinates, symmetry, or periodicity.
    """
    atoms = [
        f"[{element}]"
        for element, count in sorted(counts.items())
        for _ in range(count)
    ]
    return ".".join(atoms)


def site_counts(structure: dict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for site in structure.get("sites", []):
        species = site.get("species", [])
        if len(species) != 1 or species[0].get("occu") != 1:
            raise ValueError("Only ordered, fully occupied sites are supported")
        counts[species[0]["element"]] += 1
    return dict(counts)


def same_ratio(left: dict[str, int], right: dict[str, int]) -> bool:
    if set(left) != set(right):
        return False
    ratios = {right[element] / left[element] for element in left}
    return len(ratios) == 1


def split_record_id(record_id: str, formula: str) -> tuple[str, str, str]:
    parts = record_id.split("--")
    if len(parts) != 4:
        raise ValueError(f"Unexpected record id: {record_id!r}")
    structure_name, magnetic_order, calculation = parts[1:]
    prefix = f"{formula}_"
    prototype = structure_name[len(prefix) :] if structure_name.startswith(prefix) else structure_name
    return prototype, magnetic_order, calculation


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
    formula_histogram: Counter[str] = Counter()
    smiles_histogram: Counter[str] = Counter()
    calculation_histogram: Counter[str] = Counter()
    prototype_histogram: Counter[str] = Counter()

    for record_id, payload in source.items():
        results = payload["results"]
        structure = payload["structure"]
        gap = payload.get("gap", {})
        formula = results["formula"]
        parsed = parse_formula(formula)
        reduced = reduce_counts(parsed)
        from_sites = site_counts(structure)
        if not same_ratio(reduced, from_sites):
            raise ValueError(
                f"Formula/sites mismatch for {record_id}: {reduced} vs {from_sites}"
            )

        prototype, magnetic_order, calculation = split_record_id(record_id, formula)
        smiles = composition_smiles(reduced)
        tight_sg = results.get("sg", {}).get("tight", {})
        loose_sg = results.get("sg", {}).get("loose", {})
        lattice = structure.get("lattice", {})
        rows.append(
            {
                "record_id": record_id,
                "formula": formula,
                "canonical_reduced_formula": canonical_formula(reduced),
                "composition_smiles": smiles,
                "representation_scope": "reduced_composition_only",
                "prototype": prototype,
                "magnetic_order": magnetic_order,
                "calculation": calculation,
                "converged": results.get("convergence"),
                "n_sites": len(structure.get("sites", [])),
                "spacegroup_number_tight": tight_sg.get("number"),
                "spacegroup_symbol_tight": tight_sg.get("symbol"),
                "spacegroup_number_loose": loose_sg.get("number"),
                "spacegroup_symbol_loose": loose_sg.get("symbol"),
                "volume": lattice.get("volume"),
                "energy_per_atom": results.get("E_per_at"),
                "relaxed_energy_per_atom": results.get("E_relax"),
                "bandgap": gap.get("bandgap"),
                "is_direct_gap": gap.get("is_direct"),
                "fermi_energy": gap.get("EF"),
            }
        )
        formula_histogram[formula] += 1
        smiles_histogram[smiles] += 1
        calculation_histogram[calculation] += 1
        prototype_histogram[prototype] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "records": len(rows),
        "unique_formulas": len(formula_histogram),
        "unique_composition_smiles": len(smiles_histogram),
        "records_per_formula": dict(sorted(Counter(formula_histogram.values()).items())),
        "calculations": dict(sorted(calculation_histogram.items())),
        "prototypes": dict(sorted(prototype_histogram.items())),
        "n_sites": dict(sorted(Counter(row["n_sites"] for row in rows).items())),
        "warning": (
            "composition_smiles is a composition-only control. It discards bonds, "
            "coordinates, lattice, symmetry, oxidation states, and periodicity."
        ),
    }
    report_path = args.output.with_suffix(".audit.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
