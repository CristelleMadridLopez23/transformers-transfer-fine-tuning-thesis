#!/usr/bin/env python3
"""Audit ABS3 relaxation trajectories, MatGPT tokens, DFT consistency, and collisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
RAW = ALT / "01_data/raw_data/results(1).json"
FRAME_CACHE = ALT / "reports/MatGPT_finetuning/matgpt_e2_t0/frame_tokenization.csv"
STATIC_DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
INFO = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
SOURCE = ALT / "models/MatGPT-source/mat-concat"
OUT = ALT / "reports/MatGPT_finetuning/E2-T1"
LOG = logging.getLogger("matgpt_e2_t1")
MEV = 1000.0


def sha_text(value: str) -> str:
    """Hash a string as UTF-8 SHA-256."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encode(sequence: str, stoi: dict[str, int]) -> list[int]:
    """Encode one SLICES sequence with the pinned unconditional boundaries."""
    return [int(stoi["<start>"]), *[int(stoi[token]) for token in sequence.split()], int(stoi["<end>"])]


def stable_json_hash(value: list[int]) -> str:
    """Hash an integer input-ID sequence without ambiguous separators."""
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("ascii")).hexdigest()


def structure_from_cache(row: Any) -> Structure:
    """Reconstruct a periodic structure from cached Cartesian cell/site arrays."""
    arrays = json.loads(row.arrays_json)
    lattice = json.loads(row.cell_json)
    pbc = json.loads(row.pbc_json)
    if pbc != [True, True, True]:
        raise ValueError("Solo se admiten celdas periódicas 3D.")
    return Structure(lattice, arrays["numbers"], arrays["positions"], coords_are_cartesian=True, to_unit_cell=False)


def raw_trajectory_index(records: dict[str, Any]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Index raw DFT relaxation frames by source task and ionic step.

    Args:
        records: Parsed ``results(1).json`` mapping.
    Returns:
        Tuple of frame rows and task-level DFT metadata.
    """
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    tasks: dict[str, Any] = {}
    for source_key, record in records.items():
        if not str(source_key).endswith("gga-relax"):
            continue
        tasks[str(source_key)] = record
        for frame in record.get("trajectory", []):
            if not isinstance(frame, list) or len(frame) < 5:
                raise ValueError(f"Frame de trayectoria mal formado: {source_key}")
            step = int(frame[0])
            frames[(str(source_key), step)] = {
                "ionic_step": step,
                "energy_per_atom_ev": float(frame[1]),
                "structure": frame[2],
                "forces": frame[3],
                "stress": frame[4],
            }
    return frames, tasks


def metadata_audit(tasks: dict[str, Any]) -> dict[str, Any]:
    """Summarize GGA/PBE, corrections, and relevant VASP settings.

    Args:
        tasks: Raw relaxation-task records.
    Returns:
        Compatibility audit with distinct setting values and pass flag.
    """
    observed: dict[str, set[str]] = defaultdict(set)
    potcar_names: set[str] = set()
    missing = 0
    for record in tasks.values():
        meta = record.get("meta", {})
        incar = meta.get("incar", {}) or {}
        potcars = meta.get("potcar", {}) or {}
        if not incar or not potcars:
            missing += 1
        for key in ("GGA", "LDAU", "ENCUT", "EDIFF", "EDIFFG", "ISIF", "ISPIN", "ISMEAR", "SIGMA", "KSPACING", "NSW"):
            observed[key].add(str(incar.get(key, "MISSING")))
        for element_data in potcars.values():
            if isinstance(element_data, dict):
                potcar_names.add(str(element_data.get("name", "MISSING")) + ":" + str(element_data.get("pp", "MISSING")))
            else:
                potcar_names.add(str(element_data))
    values = {key: sorted(vals) for key, vals in observed.items()}
    pbe = bool(values.get("GGA") == ["Pe"] and values.get("LDAU") == ["False"])
    return {
        "n_relax_tasks": len(tasks),
        "missing_incar_or_potcar": missing,
        "incar_unique_values": values,
        "potcar_identifiers": sorted(potcar_names),
        "all_pbe_without_u": pbe,
        "reference_check": "PBE (GGA=Pe), LDAU=False, PAW_PBE potentials; static/relax total-energy offsets are not mixed in trajectory differences.",
    }


def audit(input_raw: Path, cache_path: Path, out: Path, graph_sample: int) -> dict[str, Any]:
    """Cross-check raw trajectories and cached SLICES, then create T1 artifacts.

    Args:
        input_raw: Raw DFT JSON.
        cache_path: Previously serialized frame/SLICES cache.
        out: New T1 report directory.
        graph_sample: Number of cached frame structures to re-encode; zero means all.
    Returns:
        Machine-readable decision summary.
    """
    for path in (input_raw, cache_path, STATIC_DATA, FOLDS, INFO):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"No se sobrescriben resultados T1 existentes: {out}")
    raw_records = json.loads(input_raw.read_text(encoding="utf-8"))
    frame_lookup_all, tasks = raw_trajectory_index(raw_records)
    cache = pd.read_csv(cache_path, low_memory=False)
    relax = cache[cache.calculation_type.astype(str).eq("relax")].copy()
    if not {"source_key", "frame_index", "sequence", "energy_per_atom_ev"}.issubset(relax.columns):
        raise RuntimeError("El cache de frames no contiene fuente, paso, SLICES y energía.")
    cache_keys = set(zip(relax.source_key.astype(str), relax.frame_index.astype(int)))
    if cache_keys - set(frame_lookup_all):
        raise RuntimeError("El cache contiene frames ausentes de results(1).json.")
    cache_lookup = {(str(row.source_key), int(row.frame_index)): row for row in relax.itertuples(index=False)}

    info = json.loads(INFO.read_text(encoding="utf-8"))
    stoi: dict[str, int] = info["stoi"]
    static = pd.read_csv(STATIC_DATA, low_memory=False)
    folds = pd.read_csv(FOLDS)
    group_by_id = folds.drop_duplicates("structure_id").set_index("structure_id").composition_group
    static["composition_group"] = static.structure_id.astype(str).map(group_by_id)
    if static.composition_group.isna().any():
        raise RuntimeError("Falta mapa de composición en la asignación de folds.")
    role_by_group = static.drop_duplicates("composition_group").set_index("composition_group").split.to_dict()
    static_by_source_key = static.drop_duplicates("source_key").set_index("source_key")
    # Exclude historical test before reconstructing or tokenizing any frame.
    # Link relaxation to static by exact source-task key, without formula guesses.
    frame_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for key, frame in frame_lookup_all.items():
        static_key = key[0].replace("gga-relax", "gga-static")
        if static_key in static_by_source_key.index:
            source_row = static_by_source_key.loc[static_key]
            if str(source_row.split) in {"train", "val"}:
                frame_lookup[key] = frame
    if not frame_lookup:
        raise RuntimeError("No hay frames enlazables a las composiciones de desarrollo.")
    fold_roles: dict[str, pd.DataFrame] = {str(key): value for key, value in folds.groupby("composition_group")}

    sys.path.insert(0, str(ALT / "scripts/MatGPT_finetuning/matgpt_e2_extended"))
    sys.path.insert(0, str(SOURCE))
    from slices.core import SLICES  # type: ignore[import-not-found]
    from periodic_graph import decoded_graph, graph_from_arrays, periodic_graphs_equivalent

    backend = SLICES(graph_method="econnn", relax_model=None)
    ordered_keys = sorted(frame_lookup)
    if graph_sample <= 0 or graph_sample >= len(ordered_keys):
        graph_checked_keys = set(ordered_keys)
    else:
        graph_checked_keys = set(sorted(ordered_keys, key=lambda key: sha_text(f"{key[0]}::{key[1]}"))[:graph_sample])

    # The earlier cache is a thinned subset of raw trajectories. Reuse cached
    # serializations only where present; create the missing frames directly
    # from the raw Pymatgen structures and compare periodic labelled graphs.
    token_rows: list[dict[str, Any]] = []
    graph_failed: list[str] = []
    cache_energy_diffs: list[float] = []
    for n, key in enumerate(ordered_keys, 1):
        source_key, frame_index = key
        raw = frame_lookup[key]
        structure = Structure.from_dict(raw["structure"])
        raw_energy = float(raw["energy_per_atom_ev"])
        cache_row = cache_lookup.get(key)
        energy_difference = np.nan
        if cache_row is not None:
            energy_difference = abs(raw_energy - float(cache_row.energy_per_atom_ev))
            cache_energy_diffs.append(energy_difference)
            if energy_difference > 1e-8:
                raise RuntimeError(f"Energía raw/cache difiere en {key}: {energy_difference:.3g} eV/átomo")
        atom_types, edge_indices, to_jimages, _ = backend.structure2crystal_graph_rep(structure)
        raw_graph = graph_from_arrays(atom_types, edge_indices, to_jimages)
        if cache_row is not None:
            sequence = str(cache_row.sequence).strip()
        else:
            encoded_with_sg = backend.structure2SLICES(structure, strategy=4)
            backend.from_SLICES(encoded_with_sg, strategy=4)
            sequence = backend.get_slices_by_strategy(4, backend.atom_symbols, backend.edge_indices, backend.to_jimages, None).strip()
        unknown = sorted(set(sequence.split()) - set(stoi))
        ids = encode(sequence, stoi) if not unknown else []
        token_hash = stable_json_hash(ids) if not unknown else ""
        forces = np.asarray(raw["forces"], dtype=float)
        stress = np.asarray(raw["stress"], dtype=float)
        force_norms = np.linalg.norm(forces, axis=1) if forces.ndim == 2 and forces.shape[1] == 3 else np.asarray([])
        result_formula = str(tasks[source_key].get("results", {}).get("formula", ""))
        composition = Composition(result_formula).reduced_formula
        static_key = source_key.replace("gga-relax", "gga-static")
        if static_key in static_by_source_key.index:
            static_row = static_by_source_key.loc[static_key]
            group = str(static_row.composition_group)
            polymorph = str(static_row.polymorph_family)
            magnetic_state = str(static_row.magnetic_state)
        else:
            group = composition
            parts = source_key.split("--")
            polymorph = parts[1].rsplit("_", 1)[-1] if len(parts) > 1 else "unknown"
            magnetic_state = parts[2] if len(parts) > 2 else "unknown"
        checked = key in graph_checked_keys
        equivalent = None
        if checked and not unknown:
            token_graph = decoded_graph(backend, sequence, strategy=4)
            equivalent, _ = periodic_graphs_equivalent(raw_graph, token_graph)
            if not equivalent:
                graph_failed.append(f"{source_key}::{frame_index}")
        token_rows.append({
            "trajectory_id": source_key,
            "source_task_id": source_key,
            "composition_group": group,
            "historical_split": role_by_group.get(group, "unassigned_source"),
            "polymorph_family": polymorph,
            "magnetic_state": magnetic_state,
            "frame_index": frame_index,
            "ionic_step": int(raw["ionic_step"]),
            "energy_per_atom_ev": raw_energy,
            "n_atoms": len(structure),
            "sequence": sequence,
            "input_ids_hash": token_hash,
            "token_length_with_boundaries": len(ids),
            "unknown_tokens": ";".join(unknown),
            "force_rms_ev_angstrom": float(np.sqrt(np.mean(forces ** 2))) if forces.size else np.nan,
            "force_max_atom_norm_ev_angstrom": float(np.max(force_norms)) if force_norms.size else np.nan,
            "stress_component_rms_source_units": float(np.sqrt(np.mean(stress ** 2))) if stress.size else np.nan,
            "raw_cache_energy_abs_diff_ev_atom": energy_difference,
            "graph_reencode_checked": checked,
            "graph_reencode_equivalent": equivalent,
        })
        if n % 250 == 0 or n == len(ordered_keys):
            LOG.info("frame_audit progress=%d/%d graph_failures=%d cache_hits=%d", n, len(ordered_keys), len(graph_failed), len(cache_energy_diffs))
    frames = pd.DataFrame(token_rows)
    if frames.input_ids_hash.eq("").any():
        raise RuntimeError("Hay secuencias de trayectoria con tokens desconocidos en vocabulario MatGPT.")
    if frames.token_length_with_boundaries.gt(int(info["block_size"])).any():
        raise RuntimeError("Al menos un frame de trayectoria supera block_size=442; no truncar.")

    frames["delta_to_last_frame_mev_atom"] = frames.groupby("trajectory_id").energy_per_atom_ev.transform(lambda x: (x - x.iloc[-1]) * MEV)
    trajectory_rows: list[dict[str, Any]] = []
    for trajectory_id, group in frames.groupby("trajectory_id", sort=True):
        seq_count = group.input_ids_hash.nunique()
        source_record = tasks[trajectory_id]
        trajectory_rows.append({
            "trajectory_id": trajectory_id,
            "composition_group": group.composition_group.iloc[0],
            "historical_split": group.historical_split.iloc[0],
            "polymorph_family": group.polymorph_family.iloc[0],
            "magnetic_state": group.magnetic_state.iloc[0],
            "n_frames": len(group),
            "n_unique_token_sequences": seq_count,
            "unique_token_ratio": seq_count / len(group),
            "energy_start_ev_atom": float(group.energy_per_atom_ev.iloc[0]),
            "energy_end_ev_atom": float(group.energy_per_atom_ev.iloc[-1]),
            "energy_range_mev_atom": float((group.energy_per_atom_ev.max() - group.energy_per_atom_ev.min()) * MEV),
            "max_abs_delta_to_last_mev_atom": float(group.delta_to_last_frame_mev_atom.abs().max()),
            "n_frames_with_gt_1mev_atom_force_rms": int((group.force_rms_ev_angstrom > 0.001).sum()),
            "raw_converged": bool(source_record.get("results", {}).get("convergence", False)),
        })
    trajectories = pd.DataFrame(trajectory_rows)

    collision_rows: list[dict[str, Any]] = []
    for token_hash, group in frames.groupby("input_ids_hash", sort=False):
        if len(group) < 2:
            continue
        energies = group.energy_per_atom_ev.to_numpy(float)
        median_energy = float(np.median(energies))
        delta = (float(energies.max()) - float(energies.min())) * MEV
        collision_rows.append({
            "input_ids_hash": token_hash,
            "n_frames": len(group),
            "n_trajectories": int(group.trajectory_id.nunique()),
            "n_compositions": int(group.composition_group.nunique()),
            "energy_min_ev_atom": float(energies.min()),
            "energy_max_ev_atom": float(energies.max()),
            "collision_range_mev_atom": delta,
            "collision_min_mae_mev_atom": float(np.mean(np.abs(energies - median_energy)) * MEV),
            "same_sequence_energy_near_equal_lt_1mev": bool(delta < 1.0),
            "composition_groups": ";".join(sorted(group.composition_group.unique())),
        })
    collisions = pd.DataFrame(collision_rows)

    # Select at most one representative frame per sequence in each development
    # trajectory, preferring the frame with the smallest force RMS. This rule
    # uses geometry diagnostics, not outer-validation energies or model scores.
    dev_frames = frames[frames.historical_split.isin(["train", "val"])].copy()
    dev_frames = dev_frames.sort_values(["trajectory_id", "force_rms_ev_angstrom", "frame_index"])
    selected = dev_frames.drop_duplicates(["trajectory_id", "input_ids_hash"], keep="first").copy()
    selected["n_unique_sequences_in_trajectory"] = selected.groupby("trajectory_id").input_ids_hash.transform("nunique")
    selected["selection_reason"] = "unique token sequence per trajectory; lowest force RMS among repeated IDs"
    selected["eligible_for_t2"] = selected.n_unique_sequences_in_trajectory.ge(2)
    selected["frame_selection_policy"] = "one frame per input_ids_hash per trajectory"
    fold_rows: list[dict[str, Any]] = []
    for candidate in selected.itertuples(index=False):
        fold_data = fold_roles.get(str(candidate.composition_group))
        if fold_data is None:
            continue
        for fold_id, fold_part in fold_data.groupby("outer_fold"):
            inner_role_values = fold_part.inner_role.unique()
            outer_role_values = fold_part.outer_role.unique()
            if len(inner_role_values) != 1 or len(outer_role_values) != 1:
                raise RuntimeError(f"Composición cruza rol de fold: {candidate.composition_group}, fold {fold_id}")
            fold_rows.append({
                "trajectory_id": candidate.trajectory_id,
                "composition_group": candidate.composition_group,
                "polymorph_family": candidate.polymorph_family,
                "frame_index": candidate.frame_index,
                "input_ids_hash": candidate.input_ids_hash,
                "energy_per_atom_ev": candidate.energy_per_atom_ev,
                "force_rms_ev_angstrom": candidate.force_rms_ev_angstrom,
                "outer_fold": int(fold_id),
                "outer_role": outer_role_values[0],
                "inner_role": inner_role_values[0],
                "eligible_for_t2_training": inner_role_values[0] == "inner_train" and bool(candidate.eligible_for_t2),
                "selection_reason": candidate.selection_reason,
            })
    selections = pd.DataFrame(fold_rows)
    per_fold_eligible_comps = {str(int(fold_id)): int(group.composition_group.nunique())
                               for fold_id, group in selections[selections.eligible_for_t2_training].groupby("outer_fold")}

    development_task_keys = {
        str(row.source_key).replace("gga-static", "gga-relax")
        for row in static[static.split.isin(["train", "val"])].drop_duplicates("source_key").itertuples(index=False)
    }
    development_tasks = {key: task for key, task in tasks.items() if key in development_task_keys}
    if len(development_tasks) != 210:
        raise RuntimeError(f"Metadatos de relajación enlazados a train+validation: {len(development_tasks)}, esperado 210.")
    dft_audit = metadata_audit(development_tasks)
    graph_equiv_count = int(frames.graph_reencode_equivalent.fillna(False).sum())
    graph_checked_count = int(frames.graph_reencode_checked.sum())
    graph_complete = graph_checked_count == len(frames) and graph_equiv_count == len(frames)
    collision_groups = len(collisions)
    distinct_collision_groups = int((collisions.collision_range_mev_atom >= 1.0).sum()) if len(collisions) else 0
    status = "GO" if dft_audit["all_pbe_without_u"] and graph_complete and min(per_fold_eligible_comps.values(), default=0) >= 10 else "LIMITED"
    if not dft_audit["all_pbe_without_u"] or graph_failed:
        status = "BLOCKED"
    decision = {
        "status": status,
        "source_raw": str(input_raw.resolve()),
        "source_cache": str(cache_path.resolve()),
        "n_raw_records": len(raw_records),
        "n_relax_tasks": len(development_tasks),
        "n_raw_relax_frames_all_partitions": len(frame_lookup_all),
        "n_development_relax_frames_audited": len(frame_lookup),
        "n_historical_test_relax_frames_excluded_before_structure_conversion": len(frame_lookup_all) - len(frame_lookup),
        "n_cached_relax_frames": len(frames),
        "n_cached_frames_matched_to_raw": len(cache_energy_diffs),
        "n_development_frames_missing_from_previous_thinned_cache": len(frame_lookup) - len(cache_energy_diffs),
        "raw_cache_max_abs_energy_diff_ev_atom": float(max(cache_energy_diffs, default=0.0)),
        "n_unique_token_sequences": int(frames.input_ids_hash.nunique()),
        "unique_token_ratio": float(frames.input_ids_hash.nunique() / len(frames)),
        "n_collision_groups": collision_groups,
        "n_collision_groups_range_ge_1mev_atom": distinct_collision_groups,
        "max_collision_range_mev_atom": float(collisions.collision_range_mev_atom.max()) if len(collisions) else 0.0,
        "mean_collision_min_mae_mev_atom": float(collisions.collision_min_mae_mev_atom.mean()) if len(collisions) else 0.0,
        "graph_reencode_checked": graph_checked_count,
        "graph_reencode_equivalent": graph_equiv_count,
        "graph_reencode_failed_ids": graph_failed,
        "all_frames_reencoded": graph_complete,
        "pbe_metadata": dft_audit,
        "development_trajectory_candidates": int(len(selected)),
        "development_compositions_with_candidates_by_fold": per_fold_eligible_comps,
        "selected_frames_rows_by_fold": len(selections),
        "test_historical_used_for_training_or_model_selection": False,
        "t2_training_gate": status,
        "limitations": [
            "T1 is a data/provenance audit, not a supervised performance comparison.",
            "The DFT source stores trajectory frames as total energy per atom; frame differences stay within one calculation.",
            "Frames sharing identical input IDs are represented once per trajectory in the candidate list; contradictory labels are not paired in T2.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out / "frame_tokenization.csv", index=False)
    trajectories.to_csv(out / "trajectory_manifest.csv", index=False)
    collisions.to_csv(out / "collision_summary.csv", index=False)
    trajectories.to_csv(out / "trajectory_statistics.csv", index=False)
    selections.to_csv(out / "selection_candidates.csv", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("t1_audit_complete status=%s frames=%d unique_ratio=%.4f collisions=%d graph=%d/%d", status, len(frames), decision["unique_token_ratio"], collision_groups, graph_equiv_count, graph_checked_count)
    return decision


def main() -> None:
    """Parse command line and run the T1 audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-raw", type=Path, default=RAW)
    parser.add_argument("--frame-cache", type=Path, default=FRAME_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--graph-sample", type=int, default=0, help="0 verifies every frame; positive values verify a deterministic sample.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        audit(args.input_raw, args.frame_cache, args.out, args.graph_sample)
    except Exception as exc:
        LOG.exception("t1_audit_failed error=%s", exc)
        raise


if __name__ == "__main__":
    main()
