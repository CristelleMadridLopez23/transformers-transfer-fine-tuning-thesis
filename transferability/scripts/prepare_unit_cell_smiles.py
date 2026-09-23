#!/usr/bin/env python3
"""Build a finite unit-cell connectivity SMILES approximation for each crystal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import warnings
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results_no_meta.json"
DEFAULT_OUTPUT = TRANSFERABILITY_ROOT / "data" / "unit_cell_SMILES" / "unit_cell_smiles.csv"
ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_formula(formula: str) -> dict[str, int]:
    position = 0
    counts: dict[str, int] = {}
    for match in ELEMENT_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Unsupported formula syntax: {formula!r}")
        element, raw_count = match.groups()
        count = int(raw_count) if raw_count else 1
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


def disconnected_smiles(counts: dict[str, int]) -> str:
    return ".".join(
        f"[{element}]"
        for element, count in sorted(counts.items())
        for _ in range(count)
    )


def split_record_id(record_id: str, formula: str) -> tuple[str, str, str, str]:
    parts = record_id.split("--")
    if len(parts) != 4:
        raise ValueError(f"Unexpected record id: {record_id!r}")
    structure_name, magnetic_order, calculation = parts[1:]
    prefix = f"{formula}_"
    prototype = structure_name[len(prefix) :] if structure_name.startswith(prefix) else structure_name
    structure_key = "--".join(parts[:3])
    return prototype, magnetic_order, calculation, structure_key


def projected_crystalnn_edges(structure, CrystalNN) -> tuple[list[tuple[int, int]], int, int]:
    """Project periodic CrystalNN edges onto unit-cell sites.

    Translation labels and parallel periodic edges are deliberately discarded,
    producing a finite simple graph that standard SMILES can serialize.
    """
    strategy = CrystalNN(weighted_cn=False, porous_adjustment=False)
    edges: set[tuple[int, int]] = set()
    periodic_neighbor_links = 0
    ignored_self_loops = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for source in range(len(structure)):
            for neighbor in strategy.get_nn_info(structure, source):
                target = int(neighbor["site_index"])
                image = tuple(int(round(float(value))) for value in neighbor["image"])
                if image != (0, 0, 0):
                    periodic_neighbor_links += 1
                if source == target:
                    ignored_self_loops += 1
                    continue
                edges.add(tuple(sorted((source, target))))
    return sorted(edges), periodic_neighbor_links, ignored_self_loops


def graph_to_canonical_smiles(structure, edges, ob) -> str:
    molecule = ob.OBMol()
    for site in structure:
        atom = molecule.NewAtom()
        atom.SetAtomicNum(int(site.specie.Z))
    for source, target in edges:
        molecule.AddBond(source + 1, target + 1, 1)

    conversion = ob.OBConversion()
    if not conversion.SetOutFormat("can"):
        raise RuntimeError("Open Babel canonical SMILES writer is unavailable")
    conversion.AddOption("i", ob.OBConversion.OUTOPTIONS)  # omit inferred chirality
    conversion.AddOption("n", ob.OBConversion.OUTOPTIONS)  # omit molecule title
    smiles = conversion.WriteString(molecule, True).strip().split("\t", 1)[0]
    if not smiles:
        raise ValueError("Open Babel produced an empty SMILES string")
    return smiles


def main() -> None:
    try:
        from openbabel import openbabel as ob
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.core import Structure
    except ImportError as exc:
        raise SystemExit(
            "Missing pymatgen/Open Babel. Install transferability/requirements.txt."
        ) from exc

    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)

    graph_cache: dict[str, dict] = {}
    rows = []
    formula_to_prototype_smiles: defaultdict[str, dict[str, str]] = defaultdict(dict)

    for record_id, payload in source.items():
        results = payload["results"]
        gap = payload.get("gap", {})
        formula = results["formula"]
        reduced = reduce_counts(parse_formula(formula))
        prototype, magnetic_order, calculation, structure_key = split_record_id(record_id, formula)

        if structure_key not in graph_cache:
            structure = Structure.from_dict(payload["structure"])
            cell_counts = Counter(site.specie.symbol for site in structure)
            edges, periodic_links, self_loops = projected_crystalnn_edges(structure, CrystalNN)
            fallback = disconnected_smiles(dict(cell_counts))
            graph_smiles = graph_to_canonical_smiles(structure, edges, ob) if edges else fallback
            graph_cache[structure_key] = {
                "unit_cell_disconnected_smiles": fallback,
                "unit_cell_smiles": graph_smiles,
                "n_sites": len(structure),
                "quotient_edges": len(edges),
                "periodic_neighbor_links": periodic_links,
                "ignored_periodic_self_loops": self_loops,
                "used_disconnected_fallback": not bool(edges),
            }

        representation = graph_cache[structure_key]
        lattice = payload["structure"].get("lattice", {})
        tight_sg = results.get("sg", {}).get("tight", {})
        rows.append(
            {
                "record_id": record_id,
                "formula": formula,
                "canonical_reduced_formula": canonical_formula(reduced),
                "composition_smiles": disconnected_smiles(reduced),
                **representation,
                "representation_scope": "unit_cell_quotient_graph_without_translation_labels",
                "neighbor_method": "pymatgen_CrystalNN",
                "serialization": "OpenBabel_canonical_SMILES_single_bonds_no_stereo",
                "prototype": prototype,
                "magnetic_order": magnetic_order,
                "calculation": calculation,
                "converged": results.get("convergence"),
                "spacegroup_number_tight": tight_sg.get("number"),
                "spacegroup_symbol_tight": tight_sg.get("symbol"),
                "volume": lattice.get("volume"),
                "energy_per_atom": results.get("E_per_at"),
                "relaxed_energy_per_atom": results.get("E_relax"),
                "bandgap": gap.get("bandgap"),
                "is_direct_gap": gap.get("is_direct"),
                "fermi_energy": gap.get("EF"),
            }
        )
        formula_to_prototype_smiles[formula][prototype] = representation["unit_cell_smiles"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    distinct_per_formula = Counter(
        len(set(prototype_map.values()))
        for prototype_map in formula_to_prototype_smiles.values()
    )
    edge_counts = [entry["quotient_edges"] for entry in graph_cache.values()]
    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "records": len(rows),
        "unique_crystal_graphs_computed": len(graph_cache),
        "unique_formulas": len(formula_to_prototype_smiles),
        "unique_composition_smiles": len({row["composition_smiles"] for row in rows}),
        "unique_unit_cell_disconnected_smiles": len(
            {row["unit_cell_disconnected_smiles"] for row in rows}
        ),
        "unique_unit_cell_smiles": len({row["unit_cell_smiles"] for row in rows}),
        "distinct_prototype_smiles_per_formula": dict(sorted(distinct_per_formula.items())),
        "formulas_with_all_three_prototypes_distinguished": sum(
            len(set(values.values())) == 3 for values in formula_to_prototype_smiles.values()
        ),
        "disconnected_fallbacks": sum(
            entry["used_disconnected_fallback"] for entry in graph_cache.values()
        ),
        "quotient_edges": {
            "min": min(edge_counts),
            "median": statistics.median(edge_counts),
            "max": max(edge_counts),
        },
        "methodological_warning": (
            "This is a finite simple-graph approximation. CrystalNN bonds are heuristic; "
            "periodic translation labels, parallel edges, geometry, bond order, oxidation "
            "states, and periodic self-loops are not encoded. It is not an invertible "
            "representation of the crystal and is not a reproduction of the manually "
            "curated Quirós et al. COD pipeline."
        ),
    }
    report_path = args.output.with_suffix(".audit.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
