"""Inspect immutable checkpoint identity, source provenance and existing audit.

Uses safetensors headers and read-only NumPy memory maps; never instantiates a
model. Recounts stored audit artifacts without recomputing SLICES or retokenizing
the dataset. Source inventories retain every searched match and negative search
scope; apparent architectural compatibility is not historical certification.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gather import DEFAULT, ROOT, SNAPSHOTS, digest, write_json

LOGGER = logging.getLogger(__name__)
PATTERN = re.compile(r"SliceTokenizer|vocab_file|\bvocab\b|special_tokens|special_num_token|cls_token_id|sep_token_id|pad_token_id|mask_token_id|unk_token_id|get_vocab|DataCollatorForLanguageModeling|BertConfig|BertForMaskedLM|save_pretrained|from_pretrained|resize_token_embeddings|AutoModelForMaskedLM|max_position_embeddings|santiago_ckpt_normal|checkpoint-393000")
ARTIFACT_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "vocab.txt", "training_args.bin"}


def read(path: Path) -> Any:
    """Read a UTF-8 JSON artifact without modifying it."""
    return json.loads(path.read_text())


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write a UTF-8 CSV table with explicit columns, including empty searches."""
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_header(path: Path) -> tuple[dict[str, Any], int]:
    """Validate a safetensors JSON header and return entries and data start offset."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete safetensors length")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(path.stat().st_size - 8, 10_000_000):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(handle.read(length))
    available = path.stat().st_size - 8 - length
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if not 0 <= start <= end <= available:
            raise ValueError(f"Invalid tensor offsets: {name}")
    return header, 8 + length


def expected_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    """Return every expected public BertModel tensor shape from checkpoint config."""
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    shapes = {"embeddings.word_embeddings.weight": [config["vocab_size"], hidden],
              "embeddings.position_embeddings.weight": [config["max_position_embeddings"], hidden],
              "embeddings.token_type_embeddings.weight": [config["type_vocab_size"], hidden],
              "embeddings.LayerNorm.weight": [hidden], "embeddings.LayerNorm.bias": [hidden],
              "pooler.dense.weight": [hidden, hidden], "pooler.dense.bias": [hidden]}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"encoder.layer.{layer}."
        for component in ["query", "key", "value"]:
            shapes[prefix + f"attention.self.{component}.weight"] = [hidden, hidden]
            shapes[prefix + f"attention.self.{component}.bias"] = [hidden]
        for sub in ["attention.output", "output"]:
            shapes[prefix + f"{sub}.dense.weight"] = [hidden, hidden if sub.startswith("attention") else intermediate]
            for suffix in ["dense.bias", "LayerNorm.weight", "LayerNorm.bias"]:
                shapes[prefix + f"{sub}.{suffix}"] = [hidden]
        shapes[prefix + "intermediate.dense.weight"] = [intermediate, hidden]
        shapes[prefix + "intermediate.dense.bias"] = [intermediate]
    return shapes


def identity(output: Path) -> dict[str, Any]:
    """Verify local bytes against pinned public config/LFS metadata and all shapes."""
    import numpy as np
    model = ROOT / "model-pretrained/mattext-slices-2m"
    config = read(model / "config.json")
    manifest = read(ROOT / "model-pretrained/mattext-manifest.json")
    tree = read(output / "evidence/hf_pinned_tree.json")
    upstream = next(item for item in tree if item["path"] == "model.safetensors")
    weights = model / "model.safetensors"
    weight_hash, config_hash = digest(weights), digest(model / "config.json")
    header, offset = tensor_header(weights)
    shapes = expected_shapes(config)
    actual = {name: entry["shape"] for name, entry in header.items() if name != "__metadata__"}
    matches = actual == shapes
    if not matches:
        raise ValueError("Checkpoint tensor shapes disagree with public BertModel config")
    embedding = header["embeddings.word_embeddings.weight"]
    if embedding["dtype"] != "F32":
        raise ValueError("Read-only embedding diagnosis requires F32 checkpoint")
    matrix = np.memmap(weights, mode="r", dtype="<f4", offset=offset + embedding["data_offsets"][0], shape=tuple(embedding["shape"]))
    stats = [{"row": row, "l2_norm": float(np.linalg.norm(matrix[row].astype(np.float64))),
              "zero_components": int(np.count_nonzero(matrix[row] == 0)),
              "row_sha256": hashlib.sha256(matrix[row].tobytes()).hexdigest()}
             for row in [0, 149, 150, 151, 152, 153, 154, 155, 1000, 30521]]
    upstream_hash = upstream.get("lfs", {}).get("oid", upstream.get("lfs", {}).get("sha256"))
    if upstream_hash and upstream_hash.startswith("sha256:"):
        upstream_hash = upstream_hash.split(":", 1)[1]
    good = weight_hash == upstream_hash and (model / "config.json").read_bytes() == (output / "evidence/hf_pinned_config.json").read_bytes()
    if not good:
        raise ValueError(f"Public/local checkpoint identity mismatch: {upstream_hash}")
    result = {"status": "verified", "repository": manifest["repository"], "revision": manifest["revision"],
              "first_weight_upload_revision": "a7ae83dbd06845f9d03e120a155174f3f81454da",
              "first_weight_upload_time": "2024-06-05T03:40:59Z", "weight_sha256": weight_hash,
              "public_lfs_sha256": upstream_hash, "weight_bytes": weights.stat().st_size,
              "config_sha256": config_hash, "public_config_byte_identical": True, "config": config,
              "tensor_shapes_exact": matches, "tensors": header, "tensor_count": len(actual),
              "parameters": sum(__import__("math").prod(value) for value in actual.values()),
              "embedding_row_diagnostics": stats,
              "diagnostic_limit": "Row norms do not identify original token semantics or training data",
              "original_hf_files": [item["path"] for item in tree],
              "local_files": [str(path.relative_to(model)) for path in sorted(model.rglob("*")) if path.is_file()],
              "checkpoint_specific_tokenizer_released": False,
              "live_hf_revision": read(output / "evidence/hf_live_api.json")["sha"]}
    write_json(output / "checkpoint_identity.json", result)
    return result


def prior_audit(output: Path) -> dict[str, Any]:
    """Recount persisted row/token/collision artifacts and verify original data hashes."""
    old = output.parent
    audit = read(old / "tokenizer_audit.json")
    checks: dict[str, Any] = {}
    paths = {"slices": ROOT / "data/processed/only_hex_new_csv_slices/configurations_slices.csv",
             "source": ROOT / "data/processed/only_hex_new_csv_slices/configurations.csv",
             "conversion_manifest": ROOT / "reports/only_hex_new_csv_slices/audit/slices_manifest.json",
             "reference_tokenizer": old / "reference_tokenizer.json",
             "candidate_tokens": ROOT / "data/processed/exp02_only_hex_mattext_finetune/tokenized_candidate.jsonl"}
    for key, path in paths.items():
        actual_hash = digest(path)
        checks[key] = {"path": str(path.relative_to(ROOT)), "sha256": actual_hash,
                       "matches_prior_fingerprint": actual_hash == audit["fingerprint"][key]}
        if not checks[key]["matches_prior_fingerprint"]:
            raise ValueError(f"Prior audit artifact changed: {path}")
    with (old / "tokenizer_per_structure.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if row["status"] == "compatible_with_official_code"]
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    raw_groups: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for line in paths["candidate_tokens"].open():
        row = json.loads(line)
        ids = tuple(row["input_ids"])
        groups[ids].append(row)
        raw_groups[ids].add(row["slices_model_input"])
    stats = {"source_rows": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
             "valid_rows": len(valid), "max_length": max(int(row["length"]) for row in valid),
             "omissions": sum(int(row["nonwhitespace_omissions"]) for row in valid),
             "unknown_tokens": sum(int(row["unknown_tokens"]) for row in valid),
             "truncations": sum(row["truncated"] != "False" for row in valid),
             "unique_inputs": len(groups), "collision_groups": sum(len(group) > 1 for group in groups.values()),
             "tokenizer_induced_collision_groups": sum(len(strings) > 1 for strings in raw_groups.values()),
             "exact_official_ids_recorded_all": all(row["exact_official_ids"] == "True" for row in valid),
             "graph_equivalence_recorded_all": all(row["graph_equivalent"] == "True" for row in valid)}
    expected = {"source_rows": 8898, "valid_rows": 8712, "max_length": 401, "omissions": 0,
                "unknown_tokens": 0, "truncations": 0, "unique_inputs": 2853, "collision_groups": 1119,
                "tokenizer_induced_collision_groups": 0}
    if any(stats[key] != value for key, value in expected.items()):
        raise ValueError(f"Prior result recount mismatch: {stats}")
    log = old / "unit_tests.log"
    result = {"locally_verified": stats, "integrity": checks,
              "prior_test_log": {"sha256": digest(log), "five_tests_success_log_present": "Ran 5 tests" in log.read_text() and "\nOK" in log.read_text(),
                                 "rerun_this_audit": False},
              "limits": ["Per-row historical token/graph checks are persisted prior results, not recomputed here",
                         "Counts, ID/string collision groups and fingerprints rechecked locally; five-test log inspected",
                         "No energy metric used for certification or selecting token IDs"]}
    write_json(output / "existing_audit_verification.json", result)
    return result


def source_inventory(output: Path) -> None:
    """Inventory all pinned source keyword matches, original tests and named artifacts."""
    roots = {label: output / "evidence" / label for label in SNAPSHOTS}
    roots["zenodo_1.1.0"] = output.parent / "reference/paper_code/MatText-main"
    matches: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for label, root in roots.items():
        repository, revision = SNAPSHOTS.get(label, ("MatText", "2a594a4c69237da2daccab04e0515cf8cdcd715d"))
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if path.suffix in {".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".md", ".lock"}:
                text = path.read_text(errors="replace")
                file_hash = digest(path)
                interesting = path.name in ARTIFACT_NAMES or "vocab" in path.name or "pretrain" in str(relative) or "tests/" in str(relative)
                if interesting:
                    inventory.append({"snapshot": label, "path": str(relative), "sha256": file_hash,
                                      "named_original_artifact": path.name in ARTIFACT_NAMES,
                                      "revision": revision, "url": f"https://github.com/lamalab-org/{repository}/blob/{revision}/{relative}"})
                for line_number, line in enumerate(text.splitlines(), 1):
                    if PATTERN.search(line):
                        matches.append({"snapshot": label, "file": str(relative), "line": line_number,
                                        "observed": line.strip(), "revision": revision, "sha256": file_hash,
                                        "url": f"https://github.com/lamalab-org/{repository}/blob/{revision}/{relative}#L{line_number}"})
    csv_write(output / "source_keyword_evidence.csv", matches)
    csv_write(output / "source_artifact_inventory.csv", inventory)
    LOGGER.info("source_inventory matches=%d selected_files=%d", len(matches), len(inventory))


def decide(identity_ok: bool, reconstruction_ok: bool, coverage_ok: bool,
           exact_run_link: bool, special_protocol_verified: bool,
           material_contradictions: list[str]) -> str:
    """Return formal verdict; missing linkage alone is partial, contradictions block."""
    if not identity_ok or not reconstruction_ok or not coverage_ok or material_contradictions:
        return "BLOCKED"
    if exact_run_link and special_protocol_verified:
        return "CERTIFIED"
    return "PARTIALLY_VERIFIED"


def main() -> None:
    """Validate output location and produce read-only forensic diagnostic artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports"):
        parser.error("--output must be inside reports/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s level=%(levelname)s message=%(message)s')
    try:
        identity(output)
        prior_audit(output)
        source_inventory(output)
    except Exception:
        LOGGER.exception("forensic_diagnostics_failed")
        raise


if __name__ == "__main__":
    main()
