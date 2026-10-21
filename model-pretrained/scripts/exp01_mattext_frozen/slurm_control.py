"""Submit gated Slurm stages once and record real scheduler IDs/status evidence.

Only submit when all critical preflight hashes still match. Validate both resource
requests with sbatch --test-only, then use a real afterok dependency. No job is
created by the status command or on failed scientific audits.
"""
from __future__ import annotations

import fcntl
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from utils import (LOGGER, audit_fingerprint, cli, load_config, read_json,
                   require_gate, run_command, write_json)


def request(config: dict[str, Any], stage: str) -> list[str]:
    """Build sbatch argv from verified config and extraction/training stage; return argv."""
    directory = config["root"] / "sh_files/exp01_mattext_frozen"
    filename = "01_extract_embeddings.sbatch" if stage == "extract" else "02_train_evaluate.sbatch"
    return ["sbatch", "--parsable", f"--account={config['slurm']['account']}",
            f"--partition={config['slurm']['extraction_partition' if stage == 'extract' else 'training_partition']}",
            f"--chdir={config['root']}",
            f"--output={config['paths']['reports'] / 'logs' / (stage + '_%j.out')}",
            f"--error={config['paths']['reports'] / 'logs' / (stage + '_%j.err')}",
            str(directory / filename)]


def submit(config: dict[str, Any]) -> dict[str, Any]:
    """Validate preflight/resources and submit two nonduplicate jobs; return saved real IDs."""
    require_gate(config)
    paths = config["paths"]
    manifest_path = paths["experiment"] / "slurm_jobs.json"
    with (paths["experiment"] / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            previous = read_json(manifest_path)
            raise RuntimeError(f"A submission record already exists; inspect actual jobs first: {previous}")
        conda = run_command(["conda", "info", "--base"])
        if conda["returncode"] != 0:
            raise RuntimeError("Cannot locate existing conda activation script")
        conda_base = conda["stdout"].strip()
        if not (Path(conda_base) / "etc/profile.d/conda.sh").is_file():
            raise RuntimeError("Conda activation script missing")
        paths["reports"].joinpath("logs").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, EXP01_CONFIG=str(config["config_path"]),
                   EXP01_CONDA_BASE=conda_base, EXP01_ENV_PREFIX=str(Path(config["slurm"]["python"]).parent.parent))
        validation = {}
        for stage in ("extract", "train"):
            argv = request(config, stage)
            check = subprocess.run([*argv[:1], "--test-only", *argv[1:]], cwd=config["root"], env=env,
                                   capture_output=True, text=True, timeout=45, check=False)
            validation[stage] = {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
            if check.returncode:
                write_json(paths["reports"] / "audit" / "slurm_request_validation.json", validation)
                raise RuntimeError(f"Slurm resource validation rejected {stage}: {check.stderr}")
        write_json(paths["reports"] / "audit" / "slurm_request_validation.json", validation)
        manifest = {"fingerprint": audit_fingerprint(config), "jobs": [], "dependency": None}
        for stage in ("extract", "train"):
            argv = request(config, stage)
            if stage == "train":
                dependency = f"afterok:{manifest['jobs'][0]['job_id']}"
                argv.insert(1, f"--dependency={dependency}")
                manifest["dependency"] = dependency
            result = subprocess.run(argv, cwd=config["root"], env=env, capture_output=True,
                                    text=True, timeout=45, check=False)
            if result.returncode:
                manifest["submission_error"] = {"stage": stage, "stderr": result.stderr}
                write_json(manifest_path, manifest)
                raise RuntimeError(f"Slurm {stage} submission failed; earlier job IDs preserved: {result.stderr}")
            job_id = result.stdout.strip().split(";")[0]
            if not re.fullmatch(r"\d+", job_id):
                manifest["submission_error"] = {"stage": stage, "stdout": result.stdout, "stderr": result.stderr,
                                                  "reason": "Unexpected scheduler response; verify squeue before retry"}
                write_json(manifest_path, manifest)
                raise RuntimeError("Cannot parse actual sbatch job ID; submission response preserved")
            manifest["jobs"].append({"stage": stage, "job_id": job_id,
                                      "partition": "msigpu" if stage == "extract" else "msismall"})
            write_json(manifest_path, manifest)
            LOGGER.info("slurm_submitted stage=%s job_id=%s", stage, job_id)
        status(config)
        return manifest


def status(config: dict[str, Any]) -> dict[str, Any]:
    """Read recorded actual Job IDs and query squeue/sacct; return persisted status evidence."""
    manifest_path = config["paths"]["experiment"] / "slurm_jobs.json"
    if not manifest_path.exists():
        result = {"submitted": False, "jobs": [], "reason": "No Slurm submission record"}
    else:
        manifest = read_json(manifest_path)
        ids = ",".join(job["job_id"] for job in manifest["jobs"])
        result = {"submitted": bool(ids), "jobs": manifest["jobs"]}
        if ids:
            result["squeue"] = run_command(["squeue", "-j", ids, "-o", "%.18i %.12P %.30j %.12T %.40R"])
            result["sacct"] = run_command(["sacct", "-j", ids, "--parsable2", "--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES"])
    write_json(config["paths"]["reports"] / "audit" / "slurm_status.json", result)
    LOGGER.info("slurm_status %s", result)
    return result


def main() -> None:
    """Dispatch explicit submit/status CLI action; return None."""
    parser = cli(__doc__)
    parser.add_argument("action", choices=["submit", "status"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "submit":
        submit(config)
    else:
        status(config)


if __name__ == "__main__":
    main()
