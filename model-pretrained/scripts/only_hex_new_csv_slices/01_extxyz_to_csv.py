"""Export every original only_hex_new extxyz frame into a reversible CSV table.

Retain geometry, arrays, info and calculator results as JSON, plus explicit energy
units and the existing formula split. Audit IDs, energy consistency, source hashes
and round-trip geometry before permitting graph conversion. No DFT is recalculated.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread

from common import (LOGGER, ROOT, configuration, digest, json_text, provenance,
                    read_json, write_json)


def frame_row(atoms: Atoms, index: int, split_row: dict[str, str],
              settings: dict[str, Any]) -> dict[str, Any]:
    """Validate ASE Atoms and indexed split metadata; return reversible CSV row dict."""
    info = atoms.info
    identity = info["config_id"]
    if identity != split_row["config_id"] or index != int(split_row["source_index"]):
        raise ValueError(f"Frame/index mismatch at {index}: {identity}")
    if len(atoms) < 1 or not atoms.pbc.all() or atoms.get_volume() <= 0:
        raise ValueError(f"Invalid periodic cell: {identity}")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError(f"Nonfinite geometry: {identity}")
    formula = atoms.get_chemical_formula(mode="hill", empirical=True)
    if formula != split_row["formula"]:
        raise ValueError(f"Composition/index mismatch for {identity}: {formula}")
    for key in ["source_key", "polymorph", "data_origin", "frame_index"]:
        if str(info[key]) != split_row[key]:
            raise ValueError(f"Metadata mismatch for {identity}: {key}")
    if atoms.calc is None or "energy" not in atoms.calc.results:
        raise ValueError(f"Missing original total energy: {identity}")
    results = atoms.calc.results
    total, per_atom = float(results["energy"]), float(info["energy_per_atom_eV"])
    if not math.isfinite(total) or not math.isfinite(per_atom):
        raise ValueError(f"Nonfinite source energy: {identity}")
    # Energy is bound to this extxyz frame. Relaxation labels are never replaced
    # with a static energy belonging to a different geometry.
    error = abs(total / len(atoms) - per_atom)
    if error > settings["energy_tolerance_ev_atom"]:
        raise ValueError(f"Total/per-atom energy inconsistency for {identity}: {error}")
    arrays_json = json_text(atoms.arrays)
    cell_json, pbc_json, info_json = map(json_text, (atoms.cell.array, atoms.pbc, info))
    arrays = json.loads(arrays_json)
    restored = Atoms(numbers=arrays["numbers"], positions=arrays["positions"],
                     cell=json.loads(cell_json), pbc=json.loads(pbc_json), info=json.loads(info_json))
    if not (np.array_equal(restored.positions, atoms.positions)
            and np.array_equal(restored.cell.array, atoms.cell.array)
            and np.array_equal(restored.numbers, atoms.numbers)
            and np.array_equal(restored.pbc, atoms.pbc)):
        raise ValueError(f"CSV JSON geometry round-trip failed: {identity}")
    return {"structure_id": identity, "config_id": identity, "source_index": index,
            "source_key": info["source_key"], "formula_original": info["formula"],
            "reduced_formula": formula, "polymorph_family": info["polymorph"],
            "magnetic_state": info.get("magnetic_state", ""),
            "calculation_type": info["calculation"], "data_origin": info["data_origin"],
            "frame_index": info["frame_index"], "ionic_step": info.get("ionic_step", ""),
            "converged": info.get("converged", ""), "split": split_row["split"],
            "num_atoms": len(atoms), "energy_total_ev": total,
            "energy_per_atom_ev": per_atom, "energy_consistency_error_ev_atom": error,
            "energy_provenance": "extxyz frame: calculator.energy and info.energy_per_atom_eV",
            "functional_source_label": info["source_key"].rsplit("--", 1)[-1],
            "source_file": str(settings["source"].relative_to(ROOT)),
            "cell_json": cell_json, "pbc_json": pbc_json, "arrays_json": arrays_json,
            "info_json": info_json, "calculator_results_json": json_text(results)}


def export(settings: dict[str, Any]) -> dict[str, Any]:
    """Export configured extxyz and verify source/split provenance; return passed audit dict."""
    source_hash = digest(settings["source"])
    original = read_json(settings["source_manifest"])
    splits = read_json(settings["split_manifest"])
    if source_hash != original["output_sha256"][settings["source"].name] or source_hash != splits["source_sha256"]:
        raise ValueError("Source extxyz differs from original/pre-existing split manifests")
    for key, name in [("configuration_index", "configuration_index.csv"),
                      ("formula_assignment", "formula_assignment.csv")]:
        if digest(settings[key]) != splits["output_sha256"][name]:
            raise ValueError(f"Original split metadata integrity failed: {key}")
    with settings["configuration_index"].open() as stream:
        split_rows = list(csv.DictReader(stream))
    with settings["formula_assignment"].open() as stream:
        assignment_rows = list(csv.DictReader(stream))
    assignments = {row["formula"]: row["split"] for row in assignment_rows}
    if len(assignments) != len(assignment_rows):
        raise ValueError("Duplicate formula assignments")
    settings["processed"].mkdir(parents=True, exist_ok=True)
    output = settings["processed"] / "configurations.csv"
    audit_path = settings["reports"] / "audit" / "csv_audit.json"
    if output.exists():
        if audit_path.exists():
            prior = read_json(audit_path)
            if (prior["source_sha256"] == source_hash and prior["csv_sha256"] == digest(output)
                    and prior["config_sha256"] == digest(settings["config_path"])):
                LOGGER.info("reuse_verified_csv rows=%d", prior["frames"])
                return prior
        raise FileExistsError(f"Refusing to overwrite existing CSV without matching provenance: {output}")
    seen, formula_splits = set(), defaultdict(set)
    families, origins, split_counts = Counter(), Counter(), Counter()
    max_error = 0.0
    count = 0
    temporary = output.with_suffix(".csv.part")
    with temporary.open("x", newline="") as stream:
        writer = None
        for index, atoms in enumerate(iread(settings["source"], index=":", format="extxyz")):
            if index >= len(split_rows):
                raise ValueError("More frames than original configuration index")
            row = frame_row(atoms, index, split_rows[index], settings)
            if row["config_id"] in seen:
                raise ValueError(f"Duplicated frame ID: {row['config_id']}")
            if assignments.get(row["reduced_formula"]) != row["split"]:
                raise ValueError("Frame split differs from existing formula assignment")
            seen.add(row["config_id"])
            formula_splits[row["reduced_formula"]].add(row["split"])
            families[row["polymorph_family"]] += 1
            origins[row["data_origin"]] += 1
            split_counts[row["split"]] += 1
            max_error = max(max_error, row["energy_consistency_error_ev_atom"])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            count += 1
            if count % 1000 == 0:
                LOGGER.info("csv_progress frames=%d", count)
    if count != len(split_rows) or count != settings["expected_frames"]:
        raise ValueError(f"Unexpected frame count: {count}")
    if len(formula_splits) != settings["expected_formulas"] or any(len(v) != 1 for v in formula_splits.values()):
        raise ValueError("Unexpected composition count or composition leakage across splits")
    if digest(settings["source"]) != source_hash:
        raise ValueError("Original extxyz changed during export")
    temporary.replace(output)
    audit = {**provenance(settings), "status": "passed", "frames": count,
             "formulas": len(formula_splits), "families": dict(families), "data_origins": dict(origins),
             "split_frames": dict(split_counts), "split_formulas": dict(Counter(next(iter(v)) for v in formula_splits.values())),
             "duplicate_ids": 0, "formulas_crossing_splits": 0, "geometry_roundtrip_exact": True,
             "max_energy_consistency_error_ev_atom": max_error, "csv_sha256": digest(output),
             "configuration_index_sha256": digest(settings["configuration_index"]),
             "formula_assignment_sha256": digest(settings["formula_assignment"]),
             "policy": "All frames preserved; relaxation trajectory frames and static references remain distinguished."}
    write_json(audit_path, audit)
    LOGGER.info("csv_complete frames=%d formulas=%d output=%s", count, len(formula_splits), output)
    return audit


def main() -> None:
    """Run CLI export; log descriptive failures and propagate nonzero exit status."""
    _, settings = configuration(__doc__)
    try:
        export(settings)
    except Exception:
        LOGGER.exception("csv_export_failed")
        raise


if __name__ == "__main__":
    main()
