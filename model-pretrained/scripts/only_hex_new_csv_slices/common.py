"""Shared provenance and CLI utilities for lossless extxyz/CSV/SLICES preparation.

Inputs are project-relative JSON configuration and original data. Outputs are
new processed tables and audit JSON; original data and exp01 remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("only_hex_new_csv_slices")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "experiments/only_hex_new_csv_slices/config.json"


def digest(path: Path) -> str:
    """Hash input file Path in bounded chunks; return SHA-256 hexadecimal str."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_text(value: Any) -> str:
    """Encode JSON-compatible or numpy values without NaN/Inf; return compact str."""
    return json.dumps(value, default=lambda item: item.tolist(), allow_nan=False,
                      ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    """Write audit value to Path atomically; return None, rejecting nonfinite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON file Path; return decoded value or descriptive I/O error."""
    return json.loads(path.read_text())


def configuration(description: str) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse shared CLI description; return arguments and validated resolved settings."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    path = ROOT / args.config
    settings = read_json(path)
    if settings["experiment_id"] != "only_hex_new_csv_slices":
        raise ValueError("Unexpected experiment identifier")
    if settings["graph_method"] != "econnn" or settings["workers"] < 1:
        raise ValueError("Require original econnn graph policy and positive worker count")
    if settings["frame_timeout_seconds"] < 1 or settings["energy_tolerance_ev_atom"] <= 0:
        raise ValueError("Timeout and energy validation tolerance must be positive")
    for name in ["source", "source_manifest", "split_manifest", "configuration_index",
                 "formula_assignment", "processed", "reports"]:
        settings[name] = ROOT / settings[name]
    settings["config_path"] = path
    logging.basicConfig(level=settings["logging_level"],
                        format='%(asctime)s level=%(levelname)s logger=%(name)s message="%(message)s"')
    return args, settings


def provenance(settings: dict[str, Any]) -> dict[str, Any]:
    """Collect settings paths, package versions and code hashes; return JSON audit dict."""
    names = ["ase", "numpy", "pymatgen", "slices"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    return {"experiment_id": settings["experiment_id"], "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in names},
            "git_revision": revision, "config_sha256": digest(settings["config_path"]),
            "source": str(settings["source"].relative_to(ROOT)),
            "source_sha256": digest(settings["source"]),
            "code_sha256": {str(path.relative_to(ROOT)): digest(path)
                            for path in sorted(Path(__file__).parent.glob("*.py"))},
            "randomness": "none; no augmentation or dataset repartitioning"}


def main() -> None:
    """Validate CLI configuration and log experiment ID; return None."""
    _, settings = configuration(__doc__)
    LOGGER.info("configuration_valid experiment=%s", settings["experiment_id"])


if __name__ == "__main__":
    main()
