"""Orchestrate auditable phases and stop submission/extraction on critical failures.

The preflight continues independent audits to collect all blockers, runs small
unit tests, and runs real CPU smoke phases only after data/tokenizer acceptance.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils import (LOGGER, audit_fingerprint, cli, load_config, read_json,
                   require_gate, write_json)


def phase(name: str) -> Any:
    """Import numeric phase module by str; return module without executing its CLI."""
    return importlib.import_module(name)


def preflight(config: dict[str, Any], fetch: bool) -> dict[str, Any]:
    """Run independent audits and bounded smoke checks; return persisted acceptance status."""
    audit_dir = config["paths"]["reports"] / "audit"
    blockers, checks = [], {}
    for name, call in [
        ("inventory", lambda: phase("00_inventory").inventory(config)),
        ("dataset", lambda: phase("01_prepare_dataset").prepare(config)),
        ("tokenizer", lambda: phase("02_audit_slices").audit(config, fetch)),
    ]:
        try:
            report = call()
            checks[name] = report.get("status", "passed")
            blockers.extend(report.get("blockers", []))
            if name == "inventory" and not report["checkpoint"]["integrity_passed"]:
                blockers.append("Checkpoint SHA-256 mismatch")
            if name == "inventory":
                missing = [package for package, version in report["packages"].items()
                           if version is None and package != "xtal2txt"]
                blockers.extend(f"Missing dependency: {package}" for package in missing)
                checks["slurm_permissions"] = "verified" if (
                    config["slurm"]["account"] + "||" in report["slurm"]["association"].get("stdout", "")
                    and "State=UP" in report["slurm"]["gpu_partition"].get("stdout", "")
                    and "State=UP" in report["slurm"]["cpu_partition"].get("stdout", "")) else "failed"
                if checks["slurm_permissions"] == "failed":
                    blockers.append("Configured Slurm account or partition authorization could not be verified")
        except Exception as error:
            LOGGER.exception("preflight_phase_failed phase=%s", name)
            checks[name] = "failed"
            blockers.append(f"{name}: {type(error).__name__}: {error}")
    tests = subprocess.run([sys.executable, str(Path(__file__).with_name("test_exp01.py"))],
                           capture_output=True, text=True, timeout=90, check=False)
    (audit_dir / "unit_tests.log").write_text(tests.stdout + tests.stderr)
    checks["unit_tests"] = "passed" if tests.returncode == 0 else "failed"
    if tests.returncode:
        blockers.append("Unit tests failed; inspect unit_tests.log")
    if blockers:
        checks["real_cpu_embedding_smoke"] = "skipped: critical data/tokenizer audit failed"
        checks["development_head_smoke"] = "skipped: no approved real embeddings"
    else:
        try:
            phase("03_extract_embeddings").extract(config, smoke=True)
            checks["real_cpu_embedding_smoke"] = "passed"
            phase("04_train_regressors").train(config, smoke=True)
            checks["development_head_smoke"] = "passed"
        except Exception as error:
            LOGGER.exception("preflight_smoke_failed")
            blockers.append(f"CPU smoke: {error}")
    report = {"status": "failed" if blockers else "passed", "blockers": blockers,
              "checks": checks, "fingerprint": audit_fingerprint(config),
              "slurm_submitted": False,
              "synthetic_fixture_policy": "Unit tests contain labeled synthetic fixtures; no synthetic scientific metrics/embeddings are produced."}
    write_json(audit_dir / "preflight.json", report)
    phase("06_generate_reports").generate(config)
    LOGGER.info("preflight_complete status=%s blockers=%d", report["status"], len(blockers))
    return report


def main() -> None:
    """Dispatch configured phase; fail clearly instead of continuing through failed gates."""
    parser = cli(__doc__)
    parser.add_argument("--phase", choices=["preflight", "extract", "train-evaluate", "report"], default="preflight")
    parser.add_argument("--fetch-reference", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "preflight":
        result = preflight(config, args.fetch_reference)
        if result["status"] != "passed":
            raise SystemExit("PREFLIGHT BLOCKED: scientific jobs were not submitted; inspect audit/preflight.json")
    elif args.phase == "report":
        phase("06_generate_reports").generate(config)
    elif args.phase == "extract":
        require_gate(config)
        phase("03_extract_embeddings").extract(config)
    elif args.phase == "train-evaluate":
        require_gate(config)
        phase("04_train_regressors").train(config)
        phase("05_evaluate_models").evaluate(config)
        phase("06_generate_reports").generate(config)


if __name__ == "__main__":
    main()
