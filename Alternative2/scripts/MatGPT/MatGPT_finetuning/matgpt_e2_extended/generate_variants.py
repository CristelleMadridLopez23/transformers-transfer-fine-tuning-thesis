#!/usr/bin/env python3
"""Build a graph-verified, MatGPT-tokenizable SLICES variant bank for E2-Aug."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Composition, Structure

from Alternative2.scripts.MatGPT.MatGPT_finetuning.matgpt_e2_extended.periodic_graph import decoded_graph, periodic_graphs_equivalent

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
INFO = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
SOURCE = ALT / "models/MatGPT-source/mat-concat"
OUT = ALT / "reports/MatGPT_finetuning/E2-Aug1/variants_development_only_v2"
LOG = logging.getLogger("matgpt_e2_extended.variants")


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with SHA-256.

    Args:
        value: Text to hash.
    Returns:
        Hexadecimal digest.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encode_ids(sequence: str, stoi: dict[str, int]) -> list[int]:
    """Tokenize one historical MatGPT SLICES sequence with boundary IDs.

    Args:
        sequence: Whitespace-delimited SLICES content.
        stoi: Pinned MatGPT token-to-index mapping.
    Returns:
        Integer IDs including exactly one start and end token.
    Raises:
        KeyError: If any token is absent from the pinned vocabulary.
    """
    return [int(stoi["<start>"]), *[int(stoi[token]) for token in sequence.split()], int(stoi["<end>"])]


def structure_from_row(row: Any) -> Structure:
    """Reconstruct a Pymatgen structure from the audited static dataset row.

    Args:
        row: Pandas row with cell_json and arrays_json fields.
    Returns:
        Periodic Pymatgen Structure with the original atomic positions.
    """
    lattice = json.loads(row.cell_json)
    arrays = json.loads(row.arrays_json)
    if not all(json.loads(row.pbc_json)):
        raise ValueError(f"E2-Aug requiere PBC 3D: {row.structure_id}")
    return Structure(lattice, arrays["numbers"], arrays["positions"], coords_are_cartesian=True, to_unit_cell=False)


def generate(data_path: Path, output: Path, variants_requested: int) -> pd.DataFrame:
    """Generate and audit variants for every historical static structure.

    Args:
        data_path: Audited 234-structure E2-A static dataset.
        output: New directory for the variant manifest.
        variants_requested: Maximum total views per structure, including view 0.
    Returns:
        DataFrame with an accepted or rejected row for every generated view.
    """
    if variants_requested < 1 or variants_requested > 4:
        raise ValueError("variants_requested debe estar entre 1 y 4.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Se preservan resultados existentes; el directorio no está vacío: {output}")
    info = json.loads(INFO.read_text(encoding="utf-8"))
    stoi: dict[str, int] = info["stoi"]
    data = pd.read_csv(data_path, low_memory=False)
    if "split" not in data.columns:
        raise RuntimeError("El dataset debe identificar la partición para excluir el test antes de convertir.")
    data = data[data.split.isin(["train", "val"])].copy().reset_index(drop=True)
    if len(data) != 210 or data.structure_id.astype(str).duplicated().any():
        raise RuntimeError("El banco de desarrollo debe cubrir 210 estructuras únicas train+validation.")
    sys.path.insert(0, str(SOURCE))
    from slices.core import SLICES  # type: ignore[import-not-found]

    backend = SLICES(graph_method="econnn", relax_model=None)
    rows: list[dict[str, Any]] = []
    for index, record in data.iterrows():
        structure_id = str(record.structure_id)
        historical = str(record.matgpt_slices).strip()
        structure = structure_from_row(record)
        expected_formula = Composition(str(record.formula_original))
        if structure.composition.reduced_composition != expected_formula.reduced_composition:
            raise RuntimeError(f"Composición reconstruida incompatible: {structure_id}")
        historical_graph = decoded_graph(backend, historical, strategy=4)
        canonical_with_sg = backend.structure2SLICES(structure, strategy=4)
        backend.from_SLICES(canonical_with_sg, strategy=4)
        canonical = backend.get_slices_by_strategy(
            4, backend.atom_symbols, backend.edge_indices, backend.to_jimages, None
        )
        structure_graph = decoded_graph(backend, canonical, strategy=4)
        base_graph_ok, base_transform = periodic_graphs_equivalent(historical_graph, structure_graph)
        if not base_graph_ok:
            raise RuntimeError(f"La representación histórica no codifica el grafo periódico de la estructura: {structure_id}")

        original_seed = int(hashlib.sha256(structure_id.encode("utf-8")).hexdigest()[:8], 16)
        random.seed(original_seed)
        candidates: list[str] = []
        generation_error = ""
        if variants_requested > 1:
            try:
                candidates = backend.structure2randomSLICES(
                    structure,
                    strategy=4,
                    num=max(variants_requested * 3, 8),
                    shuffle_atom_order=True,
                    shuffle_bond_order=True,
                    flip_bonds=True,
                    batch_multiplier=10,
                )
            except Exception as exc:  # preserve a useful per-structure rejection record
                generation_error = f"{type(exc).__name__}: {exc}"
        candidates = sorted({str(item).strip() for item in candidates if str(item).strip()}, key=sha256_text)
        seen = {historical}
        views = [(0, historical, "historical_view", base_transform)]
        rejected: list[tuple[str, str]] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                # The API currently prefixes strategy 4 with a space-group
                # encoding. The audited E2-A strings omit it and the pinned
                # vocabulary has no IDs for that prefix. Decode its graph and
                # serialize the same labelled edges without the optional prefix.
                backend.from_SLICES(candidate, strategy=4)
                candidate = backend.get_slices_by_strategy(
                    4, backend.atom_symbols, backend.edge_indices, backend.to_jimages, None
                )
                unknown = sorted(set(candidate.split()) - set(stoi))
                if unknown:
                    rejected.append((candidate, f"unknown_tokens:{','.join(unknown)}"))
                    continue
                encoded = encode_ids(candidate, stoi)
                if len(encoded) > int(info["block_size"]):
                    rejected.append((candidate, f"length_exceeds_block_size:{len(encoded)}"))
                    continue
                if not backend.check_SLICES_basic(candidate, strategy=4):
                    rejected.append((candidate, "invalid_strategy4_syntax"))
                    continue
                candidate_graph = decoded_graph(backend, candidate, strategy=4)
                graph_ok, transform = periodic_graphs_equivalent(historical_graph, candidate_graph)
                if not graph_ok:
                    rejected.append((candidate, "periodic_labelled_graph_mismatch"))
                    continue
                views.append((len(views), candidate, "graph_equivalent_signed_axis_permutation", transform))
                if len(views) >= variants_requested:
                    break
            except Exception as exc:
                rejected.append((candidate, f"decode_error:{type(exc).__name__}:{exc}"))

        for variant_id, sequence, graph_check, basis_transform in views:
            encoded = encode_ids(sequence, stoi)
            rows.append({
                "structure_id": structure_id,
                "composition": str(record.reduced_formula),
                "polymorph": str(record.polymorph_family),
                "historical_split": str(record.split),
                "original_slices": historical,
                "variant_id": variant_id,
                "variant_slices": sequence,
                "input_ids_hash": hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest(),
                "length": len(encoded),
                "graph_check": graph_check,
                "basis_transform": json.dumps(basis_transform),
                "generation_seed": original_seed,
                "library_version": "slices 2.0.12; graph_method=econnn; strategy=4; optional SG prefix omitted",
                "accepted": True,
                "reason": "historical_view" if variant_id == 0 else "periodic_graph_isomorphism_passed",
            })
        for sequence, reason in rejected:
            rows.append({
                "structure_id": structure_id,
                "composition": str(record.reduced_formula),
                "polymorph": str(record.polymorph_family),
                "historical_split": str(record.split),
                "original_slices": historical,
                "variant_id": -1,
                "variant_slices": sequence,
                "input_ids_hash": "",
                "length": len(sequence.split()) + 2,
                "graph_check": "rejected",
                "basis_transform": "",
                "generation_seed": original_seed,
                "library_version": "slices 2.0.12; graph_method=econnn; strategy=4; optional SG prefix omitted",
                "accepted": False,
                "reason": reason,
            })
        if generation_error:
            rows.append({
                "structure_id": structure_id,
                "composition": str(record.reduced_formula),
                "polymorph": str(record.polymorph_family),
                "historical_split": str(record.split),
                "original_slices": historical,
                "variant_id": -1,
                "variant_slices": "",
                "input_ids_hash": "",
                "length": 0,
                "graph_check": "generation_error",
                "basis_transform": "",
                "generation_seed": original_seed,
                "library_version": "slices 2.0.12; graph_method=econnn; strategy=4; optional SG prefix omitted",
                "accepted": False,
                "reason": generation_error,
            })
        LOG.info("structure_variants structure=%d/%d views=%d rejected=%d", index + 1, len(data), len(views), len(rejected))

    result = pd.DataFrame(rows)
    required_per_structure = result[result.accepted].groupby("structure_id").variant_id.nunique()
    if len(required_per_structure) != 210 or required_per_structure.min() < 1:
        raise RuntimeError("El banco no conserva la vista histórica para cada una de las 210 estructuras de desarrollo.")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "variant_manifest.csv", index=False)
    manifest = {
        "status": "PASS" if (required_per_structure >= 2).all() else "PARTIAL",
        "data_rows": len(data),
        "structures_with_historical_view": int(required_per_structure.size),
        "structures_with_at_least_two_views": int((required_per_structure >= 2).sum()),
        "structures_with_at_least_four_views": int((required_per_structure >= 4).sum()),
        "total_accepted_views": int(result.accepted.sum()),
        "total_rejected_candidates": int((~result.accepted).sum()),
        "strategy": 4,
        "reason_for_optional_prefix_omission": "The executed E2-A data uses strategy-4 compact edge labels without the optional space-group prefix; that preserves the pinned input vocabulary and historical token format.",
        "test_partition_used_for_generation": False,
        "historical_test_rows_excluded_before_structure_reconstruction": 24,
        "seed_rule": "first 32 bits of SHA-256(structure_id)",
    }
    (output / "variant_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("variant_bank_complete status=%s structures=%d views=%d", manifest["status"], len(required_per_structure), manifest["total_accepted_views"])
    return result


def main() -> None:
    """Parse CLI options and build the static SLICES variant bank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--views", type=int, default=4, help="Total accepted views per structure including historical view 0.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        generate(args.input, args.out, args.views)
    except Exception as exc:
        LOG.exception("variant_generation_failed error=%s", exc)
        raise


if __name__ == "__main__":
    main()
