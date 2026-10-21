"""Validate certification deliverables and record final read-only input integrity.

This final phase checks expected real outputs and source archive checksums, not
predictive metrics. The generated inventory distinguishes newly created scripts
and reports from immutable original checkpoint/data files. No gate is changed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import logging
from pathlib import Path

from audit import csv_write, read
from gather import DEFAULT, ROOT, digest, write_json

LOGGER = logging.getLogger(__name__)
REQUIRED = ["checkpoint_identity.json", "historical_tokenizer_trace.md", "pretraining_provenance_matrix.csv",
            "vocabulary_mapping_comparison.csv", "special_tokens_audit.md", "tokenizer_golden_tests.json",
            "slices_input_consistency.md", "certification_verdict.json", "certification_report.md", "author_request.md"]


def finalize(output: Path) -> None:
    """Check all scientific outputs, original byte integrity and test logs; inventory files."""
    for name in REQUIRED:
        if not (output / name).is_file() or not (output / name).stat().st_size:
            raise FileNotFoundError(f"Required certification deliverable missing: {name}")
    identity = read(output / "checkpoint_identity.json")
    integrity = []
    for relative, expected in {
        "model-pretrained/mattext-slices-2m/model.safetensors": identity["weight_sha256"],
        "model-pretrained/mattext-slices-2m/config.json": identity["config_sha256"],
        "data/only_hex_new/configurations_only_hex_new.extxyz": "a31f436fd6f44be4f85c5110f8d5ebe37718a5b28eec5453425d6a9724271de4",
    }.items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Original input integrity failed: {relative}")
        integrity.append({"path": relative, "sha256": actual, "matches_preexisting_manifest": True})
    old = output.parent
    prior = read(old / "tokenizer_audit.json")
    for entry in read(output / "existing_audit_verification.json")["integrity"].values():
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Prior artifact changed during audit: {entry['path']}")
    scripts = ROOT / "model-pretrained/scripts/exp02_only_hex_mattext_finetune"
    code_hash = hashlib.sha256("".join(f"{path.name}:{digest(path)}\n" for path in sorted(scripts.glob("*.py"))).encode()).hexdigest()
    if code_hash != prior["fingerprint"]["code"]:
        raise ValueError("Pre-existing experiment scripts changed since prior audit")
    log = (output / "unit_tests.log").read_text()
    if "Ran 7 tests" not in log or "\nOK" not in log:
        raise ValueError("Seven required forensic regression checks have not passed")
    golden = read(output / "tokenizer_golden_tests.json")
    implementations = golden["implementations"]
    pre = implementations["xtal_preupload"]["variants"]
    if not pre["documentation_default"]["tests"][0]["official_ids_match"]:
        raise ValueError("Original documentation reference mismatch")
    if pre["archived_pretraining_specials"]["mlm_probe"]["status"] != "passed":
        raise ValueError("Historical configured MLM probe did not pass")
    if not all(case["repeat_exact"] for item in implementations.values() for variant in item["variants"].values() for case in variant.get("tests", [])):
        raise ValueError("Non-deterministic tokenizer probe")
    archive = old / "reference/MatText-main.zip"
    published = read(old / "reference/zenodo_record.json")["files"]
    record = next(item for item in published if item["key"] == archive.name)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if record["checksum"] != "md5:" + md5:
        raise ValueError("Zenodo archive checksum mismatch")
    # Syntax-check our new Python scripts without producing __pycache__.
    new_scripts = ROOT / "model-pretrained/scripts/tokenizer_certification"
    for path in new_scripts.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    write_json(output / "final_integrity.json", {"status": "passed", "original_inputs": integrity,
        "prior_audit_artifacts_unchanged": True, "prior_code_inventory_aggregate": code_hash,
        "prior_experiment_code_unchanged": True,
        "prior_code_fingerprint_from_audit": prior["fingerprint"]["code"],
        "zenodo_published_md5_verified": md5, "unit_tests_passed": 7,
        "new_python_syntax": "passed", "training": False, "slurm_submissions": 0})
    rows = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path), "change": "created by certification audit"}
            for base in [new_scripts, output] for path in sorted(base.rglob("*"))
            if path.is_file() and path.name != "created_files.csv"]
    csv_write(output / "created_files.csv", rows)
    LOGGER.info("certification_finalized files=%d original_inputs_unchanged=true", len(rows))


def main() -> None:
    """Validate report CLI location and log any failed final check explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports"):
        parser.error("--output must be inside reports/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s level=%(levelname)s message=%(message)s')
    try:
        finalize(output)
    except Exception:
        LOGGER.exception("certification_final_validation_failed")
        raise


if __name__ == "__main__":
    main()
