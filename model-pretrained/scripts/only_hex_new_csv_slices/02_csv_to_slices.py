"""Convert audited CSV geometries with official SLICES 2.0.12 on Slurm CPU nodes.

Use econnn, relax_model=None and strategy 4, matching the prior notebook. Preserve
raw library output, canonical legacy SLICES and equivalent strategy 3 syntax.
No reconstruction, relaxation, tokenization, energy recomputation or training runs.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import signal
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from common import (LOGGER, ROOT, configuration, digest, json_text, provenance,
                    read_json, write_json)

CONVERTER: Any = None
TIMEOUT = 120
EXTRA_FIELDS = ["slices_raw", "slices_original", "slices_model_input", "SLICES", "SLICES_mattext",
                "slices_basic_valid", "slices_3d_valid", "strategy3_graph_equivalent",
                "topology_sha256", "mattext_tokenizer_verified", "status", "error_stage", "error"]


def timed_out(signum: int, frame: Any) -> None:
    """Raise TimeoutError for OS alarm signal/frame arguments; never return normally."""
    raise TimeoutError("Per-frame official SLICES conversion exceeded configured timeout")


def initialize(timeout: int) -> None:
    """Initialize one official no-relaxation SLICES converter per CPU worker; return None."""
    global CONVERTER, TIMEOUT
    from slices.core import SLICES
    CONVERTER = SLICES(graph_method="econnn", check_results=False, relax_model=None)
    TIMEOUT = timeout
    signal.signal(signal.SIGALRM, timed_out)


def convert(row: dict[str, str]) -> dict[str, Any]:
    """Convert one audited geometry row without changing labels; return result/status dict."""
    import numpy as np
    from pymatgen.core import Structure
    stage = "structure"
    output = {**row, **dict.fromkeys(EXTRA_FIELDS, ""), "status": "rejected",
              "mattext_tokenizer_verified": False}
    signal.alarm(TIMEOUT)
    try:
        arrays = json.loads(row["arrays_json"])
        structure = Structure(lattice=json.loads(row["cell_json"]), species=arrays["numbers"],
                              coords=arrays["positions"], coords_are_cartesian=True,
                              to_unit_cell=False)
        if not structure.is_valid(tol=0.5):
            raise ValueError("Interatomic separation below prior notebook's 0.5 angstrom threshold")
        # Compute graph using the exact original criterion. Do not change neighbor
        # policy to rescue a graph that fails official validation.
        stage = "official_strategy4_encoding"
        raw = CONVERTER.structure2SLICES(structure, strategy=4)
        output["slices_raw"] = raw
        CONVERTER.from_SLICES(raw, strategy=4, fix_duplicate_edge=False)
        # SLICES 2.0.12 may prefix a space-group code (SLICES-PLUS). Preserve raw;
        # export the same graph in legacy syntax, as in data/toSLICES.ipynb.
        CONVERTER.space_group_num = None
        legacy = CONVERTER.to_SLICES(strategy=4)
        s4 = " ".join(CONVERTER.get_canonical_SLICES(legacy, strategy=4).split())
        output.update(slices_original=s4, SLICES=s4)
        stage = "official_strategy4_validation"
        basic = bool(CONVERTER.check_SLICES_basic(s4, strategy=4))
        strict = bool(CONVERTER.check_SLICES(s4, strategy=4, dupli_check=True, graph_rank_check=True))
        output.update(slices_basic_valid=basic, slices_3d_valid=strict)
        if not basic or not strict:
            raise ValueError("Official strategy-4 graph validation rejected this frame")
        CONVERTER.from_SLICES(s4, strategy=4, fix_duplicate_edge=False)
        species, edges, images = [np.array(v, copy=True) for v in
                                 (CONVERTER.atom_types, CONVERTER.edge_indices, CONVERTER.to_jimages)]
        # Use the official strategy-3 serializer directly: the 2.0.12 dispatcher's
        # extra argument is incompatible with get_slices3's published signature.
        s3 = " ".join(CONVERTER.get_slices3(CONVERTER.atom_symbols, CONVERTER.edge_indices,
                                           CONVERTER.to_jimages).split())
        stage = "official_strategy3_equivalence"
        if not CONVERTER.check_SLICES(s3, strategy=3, dupli_check=True, graph_rank_check=True):
            raise ValueError("Official strategy-3 graph validation failed")
        CONVERTER.from_SLICES(s3, strategy=3, fix_duplicate_edge=False)
        equivalent = all(np.array_equal(a, b) for a, b in zip((species, edges, images),
                         (CONVERTER.atom_types, CONVERTER.edge_indices, CONVERTER.to_jimages)))
        if not equivalent or sorted(species.tolist()) != sorted(arrays["numbers"]):
            raise ValueError("Serialization changed species, node indices, edges or periodic images")
        output.update(slices_model_input=s3, SLICES_mattext=s3, strategy3_graph_equivalent=True,
                      topology_sha256=hashlib.sha256(s4.encode()).hexdigest(),
                      status="accepted", error_stage="", error="")
    except Exception as error:
        output.update(error_stage=stage, error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
    return output


def verify_inputs(settings: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate export audit and hashes; return provenance dict and ordered CSV rows."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full graph conversion requires a Slurm CPU allocation; do not run on login")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) < settings["workers"]:
        raise ValueError("Configured workers exceed allocated Slurm CPUs")
    if importlib.metadata.version("slices") != settings["slices_version"]:
        raise ValueError("SLICES version changed; audit the graph criterion before proceeding")
    audit = read_json(settings["reports"] / "audit" / "csv_audit.json")
    csv_path = settings["processed"] / "configurations.csv"
    current = provenance(settings)
    if audit["status"] != "passed" or audit["csv_sha256"] != digest(csv_path):
        raise ValueError("CSV missing, corrupted or not approved")
    for name in ["source_sha256", "config_sha256", "code_sha256"]:
        if audit[name] != current[name]:
            raise ValueError(f"Export audit is stale: {name}")
    if (digest(settings["configuration_index"]) != audit["configuration_index_sha256"]
            or digest(settings["formula_assignment"]) != audit["formula_assignment_sha256"]):
        raise ValueError("Original split metadata changed since export")
    csv.field_size_limit(16 * 1024 * 1024)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != audit["frames"] or len({r["config_id"] for r in rows}) != len(rows):
        raise ValueError("CSV count or ID uniqueness changed")
    return current, rows


def run(settings: dict[str, Any]) -> None:
    """Convert all configured rows in CPU workers; save accepted/rejected tables and real audits."""
    current, rows = verify_inputs(settings)
    output = settings["processed"]
    final = output / "configurations_slices.csv"
    report = settings["reports"]
    (report / "audit").mkdir(parents=True, exist_ok=True)
    (report / "tables").mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError("SLICES results already exist; inspect their manifest instead of overwriting")
    started = time.monotonic()
    fields = list(rows[0]) + EXTRA_FIELDS
    # Smoke is on real source data inside this allocation, with both origins and
    # all prototype families represented. It never selects a tokenizer or model.
    smoke_rows, seen = [], set()
    for row in rows:
        key = row["polymorph_family"], row["data_origin"]
        if key not in seen:
            smoke_rows.append(row)
            seen.add(key)
    counts, rejections, split_counts, origin_counts = Counter(), Counter(), Counter(), Counter()
    topologies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    temporary = final.with_suffix(".csv.part")
    valid_tmp, rejected_tmp = output / "slices_valid.csv.part", output / "slices_rejected.csv.part"
    with ProcessPoolExecutor(max_workers=settings["workers"], initializer=initialize,
                             initargs=(settings["frame_timeout_seconds"],),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        smoke = list(pool.map(convert, smoke_rows, chunksize=1))
        write_json(report / "audit" / "slices_smoke.json", {
            "passed": all(r["status"] == "accepted" for r in smoke),
            "samples": [{k: r[k] for k in ["config_id", "data_origin", "polymorph_family",
                        "status", "strategy3_graph_equivalent", "error"]} for r in smoke]})
        if any(r["status"] != "accepted" for r in smoke):
            raise RuntimeError("Real-data graph smoke failed; inspect slices_smoke.json")
        with temporary.open("x", newline="") as all_stream, valid_tmp.open("x", newline="") as good_stream, rejected_tmp.open("x", newline="") as bad_stream:
            writers = [csv.DictWriter(s, fieldnames=fields) for s in (all_stream, good_stream, bad_stream)]
            for writer in writers:
                writer.writeheader()
            for index, result in enumerate(pool.map(convert, rows, chunksize=8), start=1):
                writers[0].writerow(result)
                accepted = result["status"] == "accepted"
                writers[1 if accepted else 2].writerow(result)
                counts[result["status"]] += 1
                if accepted:
                    split_counts[result["split"]] += 1
                    origin_counts[result["data_origin"]] += 1
                    topologies[result["topology_sha256"]].append({k: result[k] for k in
                        ["config_id", "reduced_formula", "polymorph_family", "data_origin", "energy_per_atom_ev"]})
                else:
                    rejections[result["error_stage"]] += 1
                if index % 100 == 0 or index == len(rows):
                    for stream in (all_stream, good_stream, bad_stream):
                        stream.flush()
                    LOGGER.info("slices_progress frames=%d total=%d accepted=%d rejected=%d", index,
                                len(rows), counts["accepted"], counts["rejected"])
                    write_json(report / "audit" / "progress.json", {"processed": index, "total": len(rows),
                               "counts": dict(counts), "seconds": time.monotonic() - started})
    if sum(counts.values()) != len(rows) or digest(settings["source"]) != current["source_sha256"]:
        raise ValueError("Incomplete output or original source changed")
    temporary.replace(final)
    valid_tmp.replace(output / "slices_valid.csv")
    rejected_tmp.replace(output / "slices_rejected.csv")
    collision_file = report / "tables" / "topology_collisions.csv"
    with collision_file.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["topology_sha256", "frames", "families_json",
                  "origins_json", "energy_range_ev_atom", "config_ids_json"])
        writer.writeheader()
        for key, group in topologies.items():
            if len(group) > 1:
                energies = [float(r["energy_per_atom_ev"]) for r in group]
                writer.writerow({"topology_sha256": key, "frames": len(group),
                    "families_json": json_text(sorted({r["polymorph_family"] for r in group})),
                    "origins_json": json_text(sorted({r["data_origin"] for r in group})),
                    "energy_range_ev_atom": max(energies) - min(energies),
                    "config_ids_json": json_text([r["config_id"] for r in group])})
    manifest = {**current, "status": "completed_with_rejections" if counts["rejected"] else "completed",
                "input_frames": len(rows), "counts": dict(counts), "accepted_split_frames": dict(split_counts),
                "accepted_data_origins": dict(origin_counts), "rejection_stages": dict(rejections),
                "unique_topologies": len(topologies),
                "colliding_topology_groups": sum(len(g) > 1 for g in topologies.values()),
                "graph_method": "econnn", "relax_model": None, "strategies": [4, 3],
                "canonicalization": "official legacy strategy4; raw SLICES-PLUS retained separately",
                "tokenizer_compatibility": "not established; strategy3 is syntax only",
                "filter_policy": "all CSV rows accounted for; official strict failures retained in slices_rejected.csv",
                "trajectory_policy": "No deduplication; different geometries may share identical topology",
                "csv_sha256": digest(output / "configurations.csv"),
                "output_sha256": {p.name: digest(p) for p in [final, output / "slices_valid.csv", output / "slices_rejected.csv"]},
                "job_id": os.environ["SLURM_JOB_ID"], "seconds": time.monotonic() - started}
    write_json(report / "audit" / "slices_manifest.json", manifest)
    LOGGER.info("slices_complete counts=%s seconds=%.1f", dict(counts), manifest["seconds"])


def main() -> None:
    """Run CLI conversion in allocated CPUs; record descriptive failure and re-raise errors."""
    _, settings = configuration(__doc__)
    try:
        run(settings)
    except Exception as error:
        LOGGER.exception("slices_conversion_failed")
        write_json(settings["reports"] / "audit" / "conversion_failure.json",
                   {"error": f"{type(error).__name__}: {error}", "job_id": os.environ.get("SLURM_JOB_ID")})
        raise


if __name__ == "__main__":
    main()
