#!/usr/bin/env python3
"""Sanity-check invariance and supercell sensitivity of unit-cell SMILES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_unit_cell_smiles import graph_to_canonical_smiles, projected_crystalnn_edges


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results_no_meta.json"
DEFAULT_OUTPUT = (
    TRANSFERABILITY_ROOT / "data" / "unit_cell_SMILES" / "invariance_audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    try:
        from openbabel import openbabel as ob
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.core import Lattice, Structure
    except ImportError as exc:
        raise SystemExit(
            "Missing pymatgen/Open Babel. Install transferability/requirements.txt."
        ) from exc

    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)

    first_formula = next(iter(source.values()))["results"]["formula"]
    selected = [
        (record_id, payload)
        for record_id, payload in source.items()
        if payload["results"]["formula"] == first_formula
        and record_id.endswith("gga-static")
    ]

    def encode(structure):
        edges, _, _ = projected_crystalnn_edges(structure, CrystalNN)
        return graph_to_canonical_smiles(structure, edges, ob), len(edges)

    results = []
    for record_id, payload in selected:
        structure = Structure.from_dict(payload["structure"])
        original, original_edges = encode(structure)

        permuted = Structure.from_sites(list(reversed(structure.sites)))
        shifted = structure.copy()
        shifted.translate_sites(
            range(len(shifted)), [0.137, 0.271, 0.389], frac_coords=True, to_unit_cell=True
        )
        axes = [1, 2, 0]
        axis_permuted = Structure(
            Lattice(structure.lattice.matrix[axes]),
            structure.species,
            structure.frac_coords[:, axes],
            coords_are_cartesian=False,
        )
        atom_permutation, _ = encode(permuted)
        origin_shift, _ = encode(shifted)
        axis_permutation, _ = encode(axis_permuted)

        supercell = structure.copy()
        supercell.make_supercell([2, 1, 1])
        results.append(
            {
                "record_id": record_id,
                "prototype": record_id.split("--")[1].removeprefix(f"{first_formula}_"),
                "original_sites": len(structure),
                "original_edges": original_edges,
                "atom_permutation_invariant": atom_permutation == original,
                "origin_shift_invariant": origin_shift == original,
                "cyclic_axis_permutation_invariant": axis_permutation == original,
                "supercell_sites": len(supercell),
                "supercell_invariant_by_construction": False,
            }
        )

    report = {
        "formula_tested": first_formula,
        "transformations": {
            "atom_permutation": "reverse site order",
            "origin_shift_fractional": [0.137, 0.271, 0.389],
            "axis_permutation": "(a,b,c) -> (b,c,a)",
            "supercell": "2x1x1; not encoded because the node count necessarily doubles",
        },
        "results": results,
        "warning": (
            "This is a sanity check on one formula and three prototypes, not a proof "
            "of invariance for every possible equivalent cell transformation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
