#!/usr/bin/env python3
"""Audit static ABS3 labels and reuse the historical composition split."""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.common import ALT, DATASET, E1Error, REPORTS, configure_logging, read_vocab, sha256_file, verify_backbone_files, write_json

FORMULA_PART = re.compile(r"([A-Z][a-z]?)(\d*)")
LOGGER = logging.getLogger("matgpt_e1")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a reduced empirical formula.

    Args:
        formula: Reduced formula string from the source dataset.

    Returns:
        Element symbol to integer count mapping; empty when malformed.
    """
    tokens = FORMULA_PART.findall(formula)
    if not tokens or "".join(element + number for element, number in tokens) != formula:
        return {}
    return {element: int(count) if count else 1 for element, count in tokens}


def is_abs3(formula: str) -> bool:
    """Identify ternary one-to-one sulfides with stoichiometry ABS3.

    Args:
        formula: Reduced composition.

    Returns:
        True when exactly two non-sulfur elements occur once and sulfur thrice.
    """
    counts = parse_formula(formula)
    return len(counts) == 3 and counts.get("S") == 3 and sum(v == 1 for e, v in counts.items() if e != "S") == 2


def reduced_counts(formula: str) -> dict[str, int]:
    """Normalize a formula's integer counts by their greatest common divisor.

    Args:
        formula: Formula string to normalize.

    Returns:
        Reduced element counts, or an empty mapping for malformed input.
    """
    counts = parse_formula(formula)
    if not counts:
        return {}
    divisor = math.gcd(*counts.values())
    return {element: count // divisor for element, count in counts.items()}


def bool_value(value: Any) -> bool:
    """Normalize CSV boolean values.

    Args:
        value: Scalar read from a dataframe.

    Returns:
        Boolean interpretation of common CSV forms.
    """
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def run(input_csv: Path, out_dir: Path) -> dict[str, Any]:
    """Filter, validate, and audit eligible static ABS3 rows.

    Args:
        input_csv: Source configurations CSV.
        out_dir: E1 output directory, which must not contain earlier results.

    Returns:
        Audit summary written to dataset_audit.json.
    """
    if not input_csv.is_file():
        raise E1Error(f"No existe el CSV de entrada: {input_csv}")
    if (out_dir / "dataset.csv").exists() or (out_dir / "dataset_audit.json").exists():
        raise E1Error(f"Ya existen artefactos en {out_dir}; usa --out-dir nuevo para preservar resultados.")
    hashes = verify_backbone_files()
    stoi, _, block_size = read_vocab()
    frame = pd.read_csv(input_csv, low_memory=False)
    required = {"structure_id", "formula_original", "reduced_formula", "polymorph_family", "SLICES", "energy_total_ev", "num_atoms", "energy_per_atom_ev", "calculation_type", "source_file", "split", "energy_provenance", "functional_source_label", "converged", "slices_basic_valid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise E1Error(f"Faltan columnas requeridas: {missing}")
    abs3_mask = frame["reduced_formula"].fillna("").map(is_abs3)
    static_mask = frame["calculation_type"].astype(str).str.lower().eq("static")
    candidates = frame[abs3_mask & static_mask].copy()
    rejected: Counter[str] = Counter()
    valid_indices: list[int] = []
    for index, row in candidates.iterrows():
        reason = None
        if not isinstance(row["structure_id"], str) or not row["structure_id"].strip():
            reason = "missing_structure_id"
        elif pd.isna(row["energy_per_atom_ev"]) or pd.isna(row["energy_total_ev"]):
            reason = "missing_energy"
        elif pd.isna(row["num_atoms"]) or float(row["num_atoms"]) <= 0:
            reason = "invalid_num_atoms"
        elif not bool_value(row["converged"]):
            reason = "not_converged"
        elif not bool_value(row["slices_basic_valid"]):
            reason = "invalid_slices"
        elif reduced_counts(str(row["formula_original"])) != reduced_counts(str(row["reduced_formula"])):
            reason = "composition_inconsistent"
        elif not isinstance(row["SLICES"], str) or not row["SLICES"].strip():
            reason = "missing_matgpt_slices"
        else:
            # MatGPT's original vocabulary contains each three-character direction vector
            # (for example, "ooo") as one token. slices_model_input is a different
            # MatText-oriented representation that splits such vectors into characters.
            tokens = row["SLICES"].split()
            if set(tokens) - set(stoi):
                reason = "unknown_token"
            elif len(tokens) + 2 > block_size:
                reason = "sequence_exceeds_block_size"
            elif abs(float(row["energy_total_ev"]) / float(row["num_atoms"]) - float(row["energy_per_atom_ev"])) > 1e-7:
                reason = "energy_per_atom_consistency_failure"
        if reason:
            rejected[reason] += 1
        else:
            valid_indices.append(index)

    accepted = candidates.loc[valid_indices].copy()
    if accepted.empty:
        raise E1Error(f"El filtro no aceptó ninguna fila. Causas: {dict(rejected)}. No se escribió dataset vacío.")
    if accepted["structure_id"].duplicated().any():
        raise E1Error("structure_id duplicados en filas aceptadas; asociación etiqueta-estructura ambigua.")
    if accepted["split"].isna().any() or not set(accepted["split"].astype(str)) <= {"train", "val", "test"}:
        raise E1Error("El split histórico tiene valores faltantes o no reconocidos.")
    formula_split_count = accepted.groupby("reduced_formula")["split"].nunique()
    if (formula_split_count > 1).any():
        raise E1Error(f"Fuga por composición: {formula_split_count[formula_split_count > 1].index.tolist()}")
    overlap: dict[str, list[str]] = {}
    accepted["matgpt_slices"] = accepted["SLICES"].astype(str)
    for key in ("structure_id", "matgpt_slices"):
        split_count = accepted.groupby(key)["split"].nunique()
        overlap[key] = [str(value) for value in split_count[split_count > 1].index.tolist()]
        if overlap[key]:
            raise E1Error(f"Solapamiento entre splits para {key}: {overlap[key][:10]}")

    collisions: list[dict[str, Any]] = []
    for sequence, group in accepted.groupby("matgpt_slices", sort=False):
        ids = sorted(group["structure_id"].astype(str).unique())
        if len(ids) > 1:
            collisions.append({"slices_model_input": sequence, "structure_ids": ids,
                               "formulas": sorted(group["reduced_formula"].astype(str).unique()),
                               "families": sorted(group["polymorph_family"].astype(str).unique()),
                               "energy_per_atom_ev": group["energy_per_atom_ev"].astype(float).tolist()})

    out_dir.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(out_dir / "dataset.csv", index=False)
    split_summary: dict[str, Any] = {}
    for split, group in accepted.groupby("split"):
        split_summary[str(split)] = {
            "rows": int(len(group)), "compositions": int(group["reduced_formula"].nunique()),
            "families": {str(k): int(v) for k, v in group["polymorph_family"].value_counts().items()},
            "compositions_with_at_least_two_polymorphs": int((group.groupby("reduced_formula")["structure_id"].nunique() >= 2).sum()),
        }
    audit = {
        "experiment": "matgpt_e1_frozen", "status": "DATA_READY_WITH_METHOD_PROVENANCE_LIMITATION",
        "input_csv": str(input_csv), "input_sha256": sha256_file(input_csv),
        "source_file_values": sorted(accepted["source_file"].dropna().astype(str).unique().tolist()),
        "source_rows": int(len(frame)), "static_abs3_candidates": int(len(candidates)), "accepted_rows": int(len(accepted)),
        "rejected_candidate_rows_by_reason": dict(rejected),
        "excluded_non_abs3_rows": int(len(frame) - int(abs3_mask.sum())),
        "excluded_abs3_trajectory_rows": int(int(abs3_mask.sum()) - len(candidates)),
        "target": "energy_per_atom_ev", "target_units": "eV/atom",
        "target_definition": "static DFT total energy divided by atom count; not formation energy",
        "energy_provenance": sorted(accepted["energy_provenance"].dropna().astype(str).unique().tolist()),
        "functional_source_label": sorted(accepted["functional_source_label"].dropna().astype(str).unique().tolist()),
        "functional_method_limit": "Local files identify gga-static but do not specify the exact exchange-correlation functional or corrections; verify the upstream calculation provenance before interpreting absolute energies physically.",
        "composition_split_reused": True, "split_counts": split_summary, "split_overlap": overlap,
        "slices_representation_collisions": collisions, "slices_collision_count": len(collisions),
        "ranking_test_compositions": split_summary.get("test", {}).get("compositions_with_at_least_two_polymorphs", 0),
        "checkpoint_hashes": hashes, "vocabulary_size": len(stoi), "block_size": block_size,
        "max_energy_consistency_error_ev_atom": float(accepted["energy_consistency_error_ev_atom"].abs().max()),
        "slices_column_used_for_matgpt": "SLICES (canonical whitespace tokenization; direction vectors remain whole tokens)",
        "note": "No structures were removed based on representation collisions; all collisions are reported.",
    }
    write_json(out_dir / "dataset_audit.json", audit)
    formula_splits = accepted.groupby("reduced_formula")["split"].first().astype(str).to_dict()
    write_json(out_dir / "split_manifest.json", {
        "policy": "reuse exact historical formula split from configurations_slices.csv", "seed": 42,
        "split_summary": split_summary, "formula_splits": formula_splits,
        "input_sha256": audit["input_sha256"], "dataset_sha256": sha256_file(out_dir / "dataset.csv"),
        "group_key": "reduced_formula", "overlap_checks": overlap,
    })
    LOGGER.info("dataset_prepared", extra={"accepted": len(accepted), "candidates": len(candidates), "split_counts": split_summary})
    return audit


def parse_args() -> argparse.Namespace:
    """Parse CLI options with project-relative defaults.

    Returns:
        CLI arguments for dataset preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ALT / "01_data" / "processed" / "only_hex_new_csv_slices" / "configurations_slices.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORTS)
    return parser.parse_args()


def main() -> None:
    """Run the dataset audit and emit a descriptive error on failure."""
    logger = configure_logging()
    args = parse_args()
    try:
        run(args.input, args.out_dir)
    except (E1Error, OSError, ValueError, ImportError) as exc:
        logger.error("dataset_audit_failed", extra={"error": str(exc)})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
