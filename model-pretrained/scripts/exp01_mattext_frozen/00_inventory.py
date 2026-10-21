"""Inventory real files, package versions, checkpoint hashes and Slurm permissions.

Reads existing project files without changing originals; writes inventory.json
and environment/slurm/git evidence under reports/exp01_mattext_frozen/audit/.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open

from utils import LOGGER, cli, load_config, read_json, run_command, sha256, write_json


def inventory(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect configured files/packages/Slurm; return and save an evidence dictionary."""
    root, paths = config["root"], config["paths"]
    packages = {}
    for name in ["torch", "transformers", "scikit-learn", "numpy", "pandas", "matplotlib",
                 "safetensors", "PyYAML", "pymatgen", "slices", "xtal2txt"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
             for path in sorted(root.rglob("*")) if path.is_file() and ".git" not in path.parts
             and "__pycache__" not in path.parts]
    manifest = read_json(paths["checkpoint_manifest"])
    weight = paths["checkpoint"] / "model.safetensors"
    expected = next(item["sha256"] for item in manifest["files"]
                    if item["path"].endswith("model.safetensors"))
    actual = sha256(weight)
    with safe_open(weight, framework="pt", device="cpu") as handle:
        shape = list(handle.get_slice("embeddings.word_embeddings.weight").get_shape())
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "host": socket.gethostname(), "packages": packages, "files": files,
              "checkpoint": {"sha256": actual, "expected_sha256": expected,
                             "integrity_passed": actual == expected,
                             "embedding_shape": shape,
                             "config": read_json(paths["checkpoint"] / "config.json")},
              "git_status": run_command(["git", "-C", str(root), "status", "--short"]),
              "git_revision": run_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
              "slurm": {"association": run_command(["sacctmgr", "-nP", "show", "assoc", "where",
                                                       f"user={os.environ.get('USER', '')}",
                                                       "format=Account,Partition,QOS"]),
                        "gpu_partition": run_command(["scontrol", "show", "partition", "msigpu"]),
                        "cpu_partition": run_command(["scontrol", "show", "partition", "msismall"])}}
    write_json(paths["reports"] / "audit" / "inventory.json", report)
    LOGGER.info("inventory files=%d checkpoint_integrity=%s", len(files), actual == expected)
    return report


def main() -> None:
    """Run inventory from --config and fail if checkpoint integrity is invalid."""
    config = load_config(cli(__doc__).parse_args().config)
    if not inventory(config)["checkpoint"]["integrity_passed"]:
        raise SystemExit("Checkpoint integrity validation failed")


if __name__ == "__main__":
    main()
