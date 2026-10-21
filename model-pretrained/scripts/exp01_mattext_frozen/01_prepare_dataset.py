"""Validate static GGA labels against final geometries and existing official SLICES.

Read the original ABS3 JSON plus the prior conversion audit. Write dataset.csv,
grouped splits and explicit exclusions; never treat relaxation as a second sample.
No original structure, energy or preprocessing notebook is modified.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from utils import (LOGGER, cli, energy_per_atom, load_config, make_splits,
                   read_json, sha256, write_json)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build dict from JSON pairs, raising on duplicated keys rather than overwriting."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicated JSON identifier/key: {key}")
        result[key] = value
    return result


def same_geometry(first: Structure, second: Structure, tolerance: float) -> tuple[bool, float]:
    """Compare ordered structures modulo periodic wraps; return agreement and max Å displacement."""
    if len(first) != len(second) or first.species != second.species:
        return False, float("inf")
    if not np.allclose(first.lattice.matrix, second.lattice.matrix, atol=tolerance, rtol=0):
        return False, float("inf")
    delta = first.frac_coords - second.frac_coords
    delta -= np.round(delta)
    maximum = float(np.linalg.norm(delta @ first.lattice.matrix, axis=1).max())
    return maximum <= tolerance, maximum


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Audit data using config dict; persist tables/splits and return critical status."""
    import json
    paths = config["paths"]
    raw = json.loads(paths["source"].read_text(), object_pairs_hook=unique_object)
    source_hash = sha256(paths["source"])
    summary = read_json(paths["slices_summary"])
    if summary["source"]["sha256"] != source_hash or not summary.get("complete_corpus"):
        raise ValueError("SLICES source hash differs or previous corpus is partial")
    prior = pd.read_csv(paths["slices"])
    if prior.calculation_id.duplicated().any():
        raise ValueError("Duplicated calculation IDs in SLICES table")
    prior = prior.set_index("calculation_id")
    rows, excluded, links, ambiguities = [], [], [], []
    structures = {}
    tolerance = config["data"]["geometry_tolerance_angstrom"]
    for key, value in raw.items():
        setup = value.get("meta", {}).get("setup", {})
        calc = setup.get("calc")
        if calc == "relax":
            excluded.append({"structure_id": key, "reason": "relaxation_reference_only"})
            continue
        if calc != "static":
            ambiguities.append({"structure_id": key, "reason": "unknown_calculation_type"})
            continue
        try:
            incar = value["meta"]["incar"]
            actual = value["meta"]["all_input_settings"]
            result = value["results"]
            if not key.endswith("--gga-static") or setup.get("xc") != "gga":
                raise ValueError("Identifier/setup does not establish static GGA provenance")
            if incar.get("GGA", "").lower() != "pe" or incar.get("NSW") != 0:
                raise ValueError("Expected GGA=Pe and NSW=0")
            if any(actual.get(flag, False) for flag in ("LHFCALC", "LMETAGGA", "LDAU")):
                raise ValueError("Incompatible hybrid/meta-GGA/+U reference")
            if result.get("convergence") is not True:
                excluded.append({"structure_id": key, "reason": "not_converged"})
                continue
            structure = Structure.from_dict(value["structure"])
            if not structure.is_ordered or not structure.is_valid(tol=0.5):
                raise ValueError("Invalid/disordered structure")
            relax_id = key.removesuffix("static") + "relax"
            relax = raw.get(relax_id)
            if relax is None:
                raise ValueError("Missing corresponding relaxation")
            related, displacement = same_geometry(
                structure, Structure.from_dict(relax["structure"]), tolerance)
            relax_structure = Structure.from_dict(relax["structure"])
            lattice_delta = float(np.abs(structure.lattice.matrix - relax_structure.lattice.matrix).max())
            coordinates_delta = structure.frac_coords - relax_structure.frac_coords
            coordinates_delta -= np.round(coordinates_delta)
            ordered_displacement = float(np.linalg.norm(coordinates_delta @ structure.lattice.matrix, axis=1).max())
            links.append({"structure_id": key, "relaxation_id": relax_id,
                          "geometry_agrees": related,
                          "maximum_displacement_angstrom": ordered_displacement,
                          "maximum_lattice_component_delta_angstrom": lattice_delta,
                          "geometry_tolerance_angstrom": tolerance})
            if not related:
                raise ValueError("Static geometry differs from final relaxation")
            energy = result.get("E_per_at")
            if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not np.isfinite(energy):
                raise ValueError("Missing/nonfinite source E_per_at")
            total = float(energy * len(structure))
            if not np.isclose(energy_per_atom(total, len(structure)), energy, atol=1e-12, rtol=0):
                raise ValueError("Energy unit consistency failed")
            if key not in prior.index:
                excluded.append({"structure_id": key, "reason": "failed_prior_official_slices_audit"})
                continue
            previous = prior.loc[key]
            if not bool(previous["slices_3d_valid"]) or not np.isclose(previous.E_per_at_eV, energy, atol=1e-12):
                raise ValueError("Previous SLICES label/provenance inconsistent")
            if int(previous.nsites) != len(structure):
                raise ValueError("SLICES cell size differs from static structure")
            indicator = setup["struc_indicator"]
            family = next((name for name in ["dist_perovskite", "hexagonal", "needle"]
                           if indicator.endswith("_" + name)), None)
            if family is None:
                raise ValueError("Unknown polymorph family")
            formula = structure.composition.reduced_formula
            structures[key] = structure
            rows.append({"structure_id": key, "composition": setup["formula_indicator"],
                         "reduced_formula": formula, "polymorph_family": family,
                         "family_origin": "initial_prototype_from_meta.setup; final_geometry_may_transform",
                         "structure_source": f"{paths['source'].relative_to(config['root'])}#{key}.structure",
                         "slices_original": previous.SLICES,
                         "slices_prior_strategy3": previous.SLICES_mattext,
                         "energy_total_ev": total, "energy_total_ev_origin": "derived: source E_per_at * num_atoms",
                         "num_atoms": len(structure), "energy_per_atom_ev": float(energy),
                         "energy_source_field": "results.E_per_at (static)",
                         "functional": "GGA (INCAR GGA=Pe; no meta-GGA/hybrid/+U)",
                         "calculation_type": "static", "converged": True,
                         "relaxation_id": relax_id, "source_file": str(paths['source'].relative_to(config['root'])),
                         "source_sha256": source_hash, "volume_angstrom3": structure.volume})
        except (ValueError, KeyError, TypeError) as error:
            ambiguities.append({"structure_id": key, "reason": str(error)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No valid static structures")
    candidates = frame.copy()
    # Selection metrics need the full set of three candidate families. Reject a
    # composition as a unit if a failed SLICES graph removes one of its candidates.
    expected = {"dist_perovskite", "hexagonal", "needle"}
    incomplete = []
    if config["data"]["require_three_families"]:
        for formula, group in frame.groupby("reduced_formula"):
            if set(group.polymorph_family) != expected or len(group) != 3:
                incomplete.append(formula)
                excluded.extend({"structure_id": key, "reason": "composition_missing_full_three_family_set"}
                                for key in group.structure_id)
        frame = frame.loc[~frame.reduced_formula.isin(incomplete)].copy()
    duplicate_rows = []
    matcher = StructureMatcher(ltol=1e-5, stol=1e-5, angle_tol=0.001,
                               primitive_cell=False, scale=False, attempt_supercell=False)
    for formula, group in candidates.groupby("reduced_formula"):
        for first, second in itertools.combinations(group.to_dict("records"), 2):
            if matcher.fit(structures[first["structure_id"]], structures[second["structure_id"]]):
                duplicate_rows.append({"reduced_formula": formula, "first_id": first["structure_id"],
                                       "second_id": second["structure_id"],
                                       "delta_energy_ev_atom": abs(first["energy_per_atom_ev"] - second["energy_per_atom_ev"])})
    blockers = []
    if ambiguities:
        blockers.append("Ambiguous static energy/structure associations; inspect energy_ambiguities.csv")
    if duplicate_rows:
        blockers.append("Geometrically duplicate physical structures require adjudication; no silent deletion")
    compositions = int(frame.reduced_formula.nunique())
    if compositions < config["data"]["minimum_compositions"]:
        blockers.append("Insufficient compositions")
    processed, tables = paths["processed"], paths["reports"] / "tables"
    frame.to_csv(processed / "dataset.csv", index=False)
    make_splits(frame, config["seed"], config["split"]["test_fraction"], config["split"]["folds"]).to_csv(
        processed / "split_assignments.csv", index=False)
    pd.DataFrame(excluded, columns=["structure_id", "reason"]).to_csv(tables / "excluded_structures.csv", index=False)
    pd.DataFrame(ambiguities, columns=["structure_id", "reason"]).to_csv(tables / "energy_ambiguities.csv", index=False)
    pd.DataFrame(links).to_csv(tables / "static_relaxation_links.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=["reduced_formula", "first_id", "second_id", "delta_energy_ev_atom"]).to_csv(
        tables / "structural_duplicates.csv", index=False)
    frame.groupby("reduced_formula").agg(structures=("structure_id", "size"),
                                        min_cell_atoms=("num_atoms", "min"), max_cell_atoms=("num_atoms", "max")).to_csv(
        tables / "composition_cell_sizes.csv")
    frame.groupby("polymorph_family").size().rename("count").to_csv(tables / "polymorph_distribution.csv")
    frame.energy_per_atom_ev.describe().rename("energy_ev_atom").to_csv(tables / "energy_statistics.csv")
    pd.DataFrame([{"structure_id": key, "converged": item.get("results", {}).get("convergence"),
                   "source_energy_per_atom_ev": item.get("results", {}).get("E_per_at"),
                   "calculation_type": item.get("meta", {}).get("setup", {}).get("calc")}
                  for key, item in raw.items()]).to_csv(tables / "all_calculation_references.csv", index=False)
    report = {"status": "failed" if blockers else "passed", "blockers": blockers,
              "source_entries": len(raw), "calculation_types": dict(Counter(
                  item.get("meta", {}).get("setup", {}).get("calc") for item in raw.values())),
              "static_candidates_with_slices": len(candidates), "validated_samples": len(frame),
              "validated_compositions": compositions, "incomplete_compositions_excluded": incomplete,
              "duplicate_ids": int(frame.structure_id.duplicated().sum()),
              "source_compositions": len({item["meta"]["setup"]["formula_indicator"] for item in raw.values()}),
              "source_nonconverged": sum(item.get("results", {}).get("convergence") is not True for item in raw.values()),
              "structural_duplicate_pairs": len(duplicate_rows), "energy_ambiguities": len(ambiguities),
              "nonfinite_selected_energies": int((~np.isfinite(frame.energy_per_atom_ev)).sum()),
              "source_sha256": source_hash,
              "energy_total_policy": "Total unavailable as independent source field; derived from published static E_per_at * atoms. E_relax never used as a total energy.",
              "family_policy": "Initial prototype labels; not a claim of final relaxed space-group identity.",
              "pretraining_overlap": "Unknown: NOMAD pretraining structure IDs are not included with this checkpoint."}
    write_json(paths["reports"] / "audit" / "dataset_audit.json", report)
    LOGGER.info("dataset samples=%d compositions=%d blockers=%d", len(frame), compositions, len(blockers))
    return report


def main() -> None:
    """Prepare original data/splits from config; exit nonzero on ambiguous references."""
    report = prepare(load_config(cli(__doc__).parse_args().config))
    if report["status"] != "passed":
        raise SystemExit("Critical dataset audit failed; see dataset_audit.json")


if __name__ == "__main__":
    main()
