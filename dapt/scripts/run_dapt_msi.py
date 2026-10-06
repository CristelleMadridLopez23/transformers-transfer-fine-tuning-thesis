#!/usr/bin/env python3
"""Reproducible full-corpus SLICES DAPT runner for an allocated MSI CUDA node."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))
from dapt_utils import (  # noqa: E402
    create_timestamped_run,
    detect_device,
    find_latest_checkpoint,
    run_dapt_pipeline,
)


def latest_resumable_run(outputs_root: Path) -> tuple[Path, Path] | None:
    candidates = []
    for run_dir in outputs_root.glob("msi_dapt_*"):
        # A completed run is an analysis artifact, not a candidate for automatic resumption.
        if (run_dir / "dapt_run_summary.json").is_file():
            continue
        checkpoint = find_latest_checkpoint(run_dir)
        if checkpoint is not None:
            candidates.append((checkpoint.stat().st_mtime, run_dir, checkpoint))
    if not candidates:
        return None
    _, run_dir, checkpoint = max(candidates, key=lambda item: item[0])
    return run_dir, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dapt/configs/dapt_msi.yaml",
                        help="YAML config path, relative to project root unless absolute.")
    parser.add_argument("--resume", nargs="?", const="auto", default="none",
                        help="Resume from latest checkpoint in latest MSI run, or provide a checkpoint directory.")
    parser.add_argument("--run-dir", default=None,
                        help="Optional existing run directory, mainly for controlled resumption.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("max_structures") is not None:
        raise ValueError("MSI full DAPT requires max_structures: null; refusing to train on a capped subset.")

    # Fail fast before loading the dataset or model if this is not an allocated CUDA node.
    device, device_info = detect_device("msi", require_cuda=True)
    print("Full DAPT is intended to run on an MSI GPU allocation.")
    print(json.dumps(device_info, indent=2))

    outputs_root = PROJECT_ROOT / "dapt" / "outputs" / "msi"
    outputs_root.mkdir(parents=True, exist_ok=True)
    resume_value = args.resume
    resume_checkpoint = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        if resume_value == "auto":
            resume_checkpoint = find_latest_checkpoint(run_dir)
    elif resume_value == "auto":
        found = latest_resumable_run(outputs_root)
        if found:
            run_dir, resume_checkpoint = found
            print(f"Auto-resume selected run: {run_dir}")
            print(f"Latest complete checkpoint: {resume_checkpoint}")
        else:
            run_dir = create_timestamped_run(outputs_root, "msi_dapt")
            print("No resumable checkpoint found; starting a new timestamped run.")
    elif resume_value not in ("none", None):
        resume_checkpoint = Path(resume_value).expanduser().resolve()
        if not resume_checkpoint.is_dir():
            parser.error(f"Checkpoint directory not found: {resume_checkpoint}")
        if args.run_dir:
            run_dir = Path(args.run_dir).resolve()
        else:
            run_dir = resume_checkpoint.parent.parent if resume_checkpoint.parent.name == "checkpoints" else resume_checkpoint.parent
    else:
        run_dir = create_timestamped_run(outputs_root, "msi_dapt")

    config["tokenizer_path"] = str((PROJECT_ROOT / config.get("tokenizer_path", "audit/tokenizers/chemberta_zinc_slices_adapted")).resolve())
    config["split_path"] = str((PROJECT_ROOT / config.get("split_path", "dapt/configs/dapt_split_assignments.csv")).resolve())
    if config.get("dataset_path"):
        data_path = Path(config["dataset_path"])
        config["dataset_path"] = str((PROJECT_ROOT / data_path).resolve() if not data_path.is_absolute() else data_path.resolve())
    config["base_model_name"] = config.get("base_model_name", "seyonec/ChemBERTa-zinc-base-v1")
    config["run_name"] = config.get("run_name", "msi_full_dapt")
    config["cli_invocation"] = {
        "argv": sys.argv,
        "config_path": str(config_path.resolve()),
        "resume": str(resume_value),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "run_dir": str(run_dir.resolve()),
        "started_at_local": datetime.now().astimezone().isoformat(),
        "device": str(device),
    }
    if resume_checkpoint is not None:
        prior_config_path = run_dir / "resolved_config.yaml"
        if prior_config_path.is_file():
            prior_config = yaml.safe_load(prior_config_path.read_text(encoding="utf-8")) or {}
            resume_keys = [
                "base_model_name", "tokenizer_path", "dataset_path", "split_path", "max_structures",
                "epochs", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "learning_rate", "weight_decay", "warmup_ratio", "mlm_probability", "max_length",
                "seed", "diagnostic_seed", "max_grad_norm", "early_stopping_patience",
                "minimum_grammar_fidelity",
            ]
            changed = {key: (prior_config.get(key), config.get(key))
                       for key in resume_keys if prior_config.get(key) != config.get(key)}
            if changed:
                raise RuntimeError(
                    "Resume refused because training/data settings differ from the checkpoint run. "
                    f"Start a new run or restore the original config. Differences: {changed}"
                )
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "cli_arguments.json").write_text(json.dumps(config["cli_invocation"], indent=2), encoding="utf-8")

    result = run_dapt_pipeline(
        config=config,
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        device_policy="msi",
        require_cuda=True,
        publish_final_model=True,
        smoke_mode=False,
        resume=str(resume_checkpoint) if resume_checkpoint else None,
    )
    summary = result["summary"]
    print("=" * 61)
    print("SLICES DAPT — MSI FULL RUN")
    print("=" * 61)
    print(f"GPU: {device_info['gpu_name']} | precision: {summary['precision']}")
    print(f"Structures: {summary['n_structures']:,} | chunks: {summary['n_chunks']:,}")
    print(f"Decision: {summary['decision']}")
    print(f"Run outputs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
