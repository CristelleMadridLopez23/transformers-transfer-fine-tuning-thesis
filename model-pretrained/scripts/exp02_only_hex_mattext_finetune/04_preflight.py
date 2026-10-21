"""Run bounded technical tests and scientific gates before any GPU fine-tuning submission.

Write preflight.json and unit test logs. CPU head smoke uses two approved training
frames only when all critical audits pass. --submit validates Slurm resources and
records a real Job ID; a failed gate never submits or trains anything.
"""
from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from workflow import (LOGGER, ROOT, cli, fingerprint, load_config, read_json,
                      require_audit, write_json)


def cpu_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Fit only the head for two steps on two approved train frames; return technical finite/gradient checks."""
    from functools import partial
    import torch
    from finetuning import collate, load_rows, model, pad_id
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    rows = []
    for row in load_rows(config):
        if row["split"] == "train" and row["reduced_formula"] not in {r["reduced_formula"] for r in rows}:
            rows.append(row)
        if len(rows) == 2:
            break
    if len(rows) != 2:
        raise ValueError("Need two approved training compositions for bounded CPU smoke")
    network = model(config)
    network.bert.requires_grad_(False)
    network.eval()
    batch = collate(rows, pad_id(config))
    before = network.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(network.classifier.parameters(), lr=config["training"]["learning_rate"])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = network(**batch)
        if not torch.isfinite(result.loss):
            raise RuntimeError("CPU head smoke generated nonfinite loss")
        result.loss.backward()
        optimizer.step()
    if torch.equal(before, network.classifier.weight):
        raise RuntimeError("CPU smoke did not update regression head")
    if any(p.grad is not None for p in network.bert.parameters()):
        raise RuntimeError("CPU smoke unexpectedly differentiated the encoder")
    return {"status": "passed", "training_frame_ids": [r["structure_id"] for r in rows],
            "optimizer_steps": 2, "encoder_gradients": False, "head_updated": True,
            "scientific_metrics": False, "saved_weights": False}


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Run independent tests, then gated real CPU smoke; return persisted critical acceptance report."""
    directory = config["paths"]["reports"] / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("test_workflow.py"))],
                            capture_output=True, text=True, timeout=120, check=False)
    (directory / "unit_tests.log").write_text(result.stdout + result.stderr)
    checks = {"unit_tests": "passed" if result.returncode == 0 else "failed"}
    blockers = [] if result.returncode == 0 else ["Technical unit tests failed"]
    try:
        require_audit(config)
        checks["scientific_audit"] = "passed"
    except Exception as error:
        checks["scientific_audit"] = "failed"
        blockers.append(str(error))
    if blockers:
        checks["real_cpu_head_smoke"] = "skipped: critical scientific/technical gate failed"
    else:
        try:
            smoke = cpu_smoke(config)
            write_json(directory / "cpu_head_smoke.json", smoke)
            checks["real_cpu_head_smoke"] = "passed"
        except Exception as error:
            LOGGER.exception("cpu_smoke_failed")
            blockers.append(f"CPU smoke: {error}")
            checks["real_cpu_head_smoke"] = "failed"
    report = {"status": "failed" if blockers else "passed", "checks": checks,
              "blockers": blockers, "fingerprint": fingerprint(config), "training_submitted": False}
    write_json(directory / "preflight.json", report)
    return report


def submit(config: dict[str, Any]) -> dict[str, Any]:
    """Validate account/resources and submit one nonduplicate final-fit GPU job; return real submission record."""
    require_audit(config)
    paths = config["paths"]
    gate = read_json(paths["reports"] / "audit/preflight.json")
    if gate["status"] != "passed" or gate["fingerprint"] != fingerprint(config):
        raise RuntimeError("Current successful preflight is required before sbatch")
    target = paths["experiment"] / "submission.json"
    with (paths["experiment"] / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            raise FileExistsError("Submission already recorded; inspect real scheduler status before any retry")
        active = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),
                                 "--name=mattext_hex_ft", "-o", "%i %T %R"], capture_output=True, text=True, check=True)
        if active.stdout.strip():
            raise RuntimeError(f"A fine-tuning job with this name is already active: {active.stdout}")
        account, partition = config["slurm"]["account"], config["slurm"]["partition"]
        association = subprocess.run(["sacctmgr", "-nP", "show", "assoc", "where",
                                     f"user={os.environ.get('USER', '')}", "format=Account,Partition,QOS"],
                                    capture_output=True, text=True, check=True)
        if account + "||" not in association.stdout:
            raise RuntimeError("Configured account is not in the user's current Slurm associations")
        conda = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True).stdout.strip()
        logs = paths["reports"] / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MATHEX_CONFIG=str(config["config_path"]),
                   MATHEX_CONDA_BASE=conda, MATHEX_ENV_PREFIX=sys.prefix)
        command = ["sbatch", "--parsable", f"--account={account}", f"--partition={partition}",
                   f"--chdir={ROOT}", f"--output={logs}/finetune_%j.out", f"--error={logs}/finetune_%j.err",
                   str(ROOT / "sh_files/exp02_only_hex_mattext_finetune/finetune.sbatch")]
        validation = subprocess.run([command[0], "--test-only", *command[1:]], env=env,
                                     cwd=ROOT, capture_output=True, text=True, timeout=45)
        write_json(paths["reports"] / "audit/slurm_validation.json", {"returncode": validation.returncode,
                   "stdout": validation.stdout, "stderr": validation.stderr, "command": command})
        if validation.returncode:
            raise RuntimeError(f"GPU resource request failed: {validation.stderr}")
        response = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=45)
        identity = response.stdout.strip().split(";")[0]
        record = {"command": command, "returncode": response.returncode,
                  "stdout": response.stdout, "stderr": response.stderr,
                  "job_id": identity if re.fullmatch(r"\d+", identity) else None,
                  "fingerprint": fingerprint(config)}
        write_json(target, record)
        if response.returncode or record["job_id"] is None:
            raise RuntimeError("Submission failed or returned unparseable ID; response preserved before any retry")
        return record


def main() -> None:
    """Run preflight and optionally submit after every gate passes; fail descriptively otherwise."""
    parser = cli(__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = preflight(config)
    LOGGER.info("preflight_complete status=%s blockers=%s", result["status"], result["blockers"])
    if result["status"] != "passed":
        raise SystemExit("PREFLIGHT BLOCKED: no fine-tuning submitted; inspect audit/preflight.json")
    if args.submit:
        LOGGER.info("submitted %s", submit(config))


if __name__ == "__main__":
    main()
