"""Shared configuration, integrity, split and SLICES helpers for Experiment 01.

Inputs are project-relative YAML, original SLICES and audited metadata; outputs
are deterministic artifacts under the experiment directories. No graph is rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("exp01")
DEFAULT_CONFIG = "experiments/exp01_mattext_frozen/config.yaml"


def project_root() -> Path:
    """Return Path: repository root inferred from this module's location."""
    return Path(__file__).resolve().parents[3]


def cli(description: str) -> argparse.ArgumentParser:
    """Build parser from description (str); return parser with shared config option."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML path (str/Path); return validated settings with resolved paths."""
    root = project_root()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    config = yaml.safe_load(target.read_text())
    if config.get("experiment_id") != "exp01_mattext_frozen":
        raise ValueError("Expected experiment_id=exp01_mattext_frozen")
    if config.get("error_policy") != "stop":
        raise ValueError("Critical audit errors must stop this experiment")
    if not 0 < config["split"]["test_fraction"] < 1 or config["split"]["folds"] < 2:
        raise ValueError("Need test_fraction in (0,1) and at least two grouped folds")
    if config["extraction"]["batch_size"] < 1 or config["extraction"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("Invalid extraction batch size or device")
    poolings = config["extraction"]["poolings"]
    if not poolings or not set(poolings) <= {"mean_pooling", "cls_pooling"}:
        raise ValueError("Unknown or empty pooling configuration")
    if config["tokenizer"]["truncation_policy"] != "reject":
        raise ValueError("No truncation policy has been scientifically validated; require reject")
    if config["evaluation"]["degeneracy_tolerance_ev_atom"] < 0:
        raise ValueError("Degeneracy tolerance must be nonnegative")
    config["root"] = root
    config["config_path"] = target.resolve()
    for name, value in config["paths"].items():
        config["paths"][name] = root / value
    logging.basicConfig(level=config.get("logging_level", "INFO"),
                        format='%(asctime)s level=%(levelname)s logger=%(name)s message="%(message)s"')
    return config


def sha256(path: Path) -> str:
    """Hash file path (Path) in chunks; return hex SHA-256 string."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically serialize JSON-compatible value to path; reject NaN/Inf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read path (Path); return parsed JSON data."""
    return json.loads(path.read_text())


def run_command(args: list[str], timeout: int = 20) -> dict[str, Any]:
    """Run read-only argv with timeout seconds; return status/stdout/stderr."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "returncode": -1, "error": str(error)}


def energy_per_atom(total_ev: float, atoms: int) -> float:
    """Divide finite total energy (eV) by positive atom count; return eV/atom."""
    if isinstance(atoms, bool) or int(atoms) != atoms or atoms <= 0 or not np.isfinite(total_ev):
        raise ValueError("Energy must be finite and atom count a positive integer")
    return float(total_ev / atoms)


def parse_slices(sequence: str, strategy: int = 4) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Parse str in SLICES strategy 3/4; return ordered species and periodic edges.

    Reject malformed indices, image vectors and unsupported symbols. This parses
    syntax only; topology/3D validity comes from the original official SLICES audit.
    """
    if strategy not in (3, 4):
        raise ValueError("Only SLICES strategies 3 and 4 are supported")
    tokens = sequence.split()
    species: list[str] = []
    position = 0
    from pymatgen.core import Element
    while position < len(tokens) and re.fullmatch(r"[A-Z][a-z]?", tokens[position]):
        if not Element.is_valid_symbol(tokens[position]):
            raise ValueError(f"Unknown element: {tokens[position]}")
        species.append(tokens[position])
        position += 1
    width = 3 if strategy == 4 else 5
    if not species or position == len(tokens) or (len(tokens) - position) % width:
        raise ValueError("Missing species/edges or incomplete edge unit")
    edges = []
    for start in range(position, len(tokens), width):
        unit = tokens[start:start + width]
        if not all(re.fullmatch(r"\d+", value) for value in unit[:2]):
            raise ValueError(f"Invalid node indices: {unit}")
        a, b = map(int, unit[:2])
        image = unit[2] if strategy == 4 else "".join(unit[2:])
        if max(a, b) >= len(species) or not re.fullmatch(r"[o+\-]{3}", image):
            raise ValueError(f"Invalid periodic edge: {unit}")
        edges.append((a, b, image))
    return species, edges


def adapt_slices(sequence: str) -> str:
    """Convert validated strategy-4 str to strategy-3 str, preserving every edge."""
    species, edges = parse_slices(sequence, 4)
    units = species + [f"{a} {b} {' '.join(image)}" for a, b, image in edges]
    result = " ".join(units)
    if parse_slices(result, 3) != (species, edges):
        raise ValueError("SLICES adapter changed structural units")
    return result


def tokenize_lossless(sequence: str, vocab: dict[str, int], max_length: int) -> tuple[list[str], list[int]]:
    """Match official longest-first vocabulary regex; return tokens/IDs or fail.

    Unlike the upstream findall implementation, unmatched non-whitespace spans
    raise an error. Multi-digit node indices follow the original digit tokenizer;
    exact span coverage proves that no digit or periodic component is omitted.
    """
    keys = sorted(vocab, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(token) for token in keys))
    tokens = []
    previous = 0
    for match in pattern.finditer(sequence):
        if sequence[previous:match.start()].strip():
            raise ValueError(f"Silent token omission: {sequence[previous:match.start()]!r}")
        tokens.append(match.group())
        previous = match.end()
    if sequence[previous:].strip():
        raise ValueError(f"Unrepresented suffix: {sequence[previous:]!r}")
    tokens = ["[CLS]", *tokens, "[SEP]"]
    if len(tokens) > max_length:
        raise ValueError(f"Truncation forbidden: {len(tokens)} > {max_length}")
    ids = [vocab[token] for token in tokens]
    if any(token in {"[UNK]", "[PAD]"} for token in tokens):
        raise ValueError("Unknown/padding token inside structural input")
    return tokens, ids


def make_splits(frame: pd.DataFrame, seed: int, test_fraction: float, folds: int) -> pd.DataFrame:
    """Assign DataFrame IDs by reduced composition; return deterministic split table."""
    if not 0 < test_fraction < 1 or folds < 2 or frame.structure_id.duplicated().any():
        raise ValueError("Invalid split settings or duplicate structure IDs")
    groups = np.array(sorted(frame.reduced_formula.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = max(1, int(np.ceil(len(groups) * test_fraction)))
    development = groups[count:]
    if len(development) < folds:
        raise ValueError("Insufficient development compositions for grouped folds")
    assignments = {group: ("test", -1) for group in groups[:count]}
    for index, group in enumerate(development):
        assignments[group] = ("development", index % folds)
    result = frame[["structure_id", "reduced_formula"]].copy()
    result["split"] = result.reduced_formula.map(lambda group: assignments[group][0])
    result["fold"] = result.reduced_formula.map(lambda group: assignments[group][1])
    return result


def audit_fingerprint(config: dict[str, Any]) -> dict[str, str]:
    """Return hashes of config, original input, weights, processed input and splits."""
    paths = config["paths"]
    files = {"config": config["config_path"], "source": paths["source"],
             "checkpoint": paths["checkpoint"] / "model.safetensors",
             "checkpoint_config": paths["checkpoint"] / "config.json",
             "dataset": paths["processed"] / "dataset.csv",
             "splits": paths["processed"] / "split_assignments.csv",
             "slices": paths["slices"], "slices_summary": paths["slices_summary"],
             "tokenized": paths["processed"] / "tokenized.jsonl",
             "vocab": config["root"] / config["tokenizer"]["vocab"],
             "evidence": config["root"] / config["tokenizer"]["evidence"]}
    result = {name: sha256(path) for name, path in files.items() if path.is_file()}
    code_paths = sorted((config["root"] / "model-pretrained/scripts/exp01_mattext_frozen").glob("*.py"))
    code_paths += sorted((config["root"] / "sh_files/exp01_mattext_frozen").glob("*"))
    result["experiment_code"] = hashlib.sha256("".join(
        f"{path.relative_to(config['root'])}:{sha256(path)}\n" for path in code_paths if path.is_file()).encode()).hexdigest()
    return result


def require_gate(config: dict[str, Any], stage: str = "preflight") -> None:
    """Require passed audit stage and unchanged hashes; raise on stale/failed gates."""
    path = config["paths"]["reports"] / "audit" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"Missing critical audit: {path}")
    report = read_json(path)
    if report.get("status") != "passed":
        raise RuntimeError(f"Critical audit {stage} failed: {report.get('blockers', [])}")
    if report.get("fingerprint") != audit_fingerprint(config):
        raise RuntimeError(f"Audit {stage} is stale; rerun preflight")


def load_embeddings(config: dict[str, Any], smoke: bool = False) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read approved embeddings and aligned IDs/splits; return metadata and pooling arrays."""
    require_gate(config, "tokenizer_audit")
    paths = config["paths"]
    directory = paths["reports"] / "audit" / "embedding_smoke" if smoke else paths["embeddings"]
    manifest = read_json(directory / "embedding_manifest.json")
    if manifest["fingerprint"] != audit_fingerprint(config):
        raise RuntimeError("Embedding provenance differs from audited inputs")
    for key, filename in [("npz_sha256", "embeddings.npz"), ("metadata_sha256", "metadata.csv"),
                          ("reference_sha256", "reference_energies.csv")]:
        if manifest[key] != sha256(directory / filename):
            raise ValueError(f"Embedding integrity mismatch: {filename}")
    metadata = pd.read_csv(directory / "metadata.csv")
    metadata = metadata.merge(pd.read_csv(directory / "reference_energies.csv"), on="structure_id", validate="one_to_one")
    metadata = metadata.merge(pd.read_csv(paths["processed"] / "split_assignments.csv"),
                              on=["structure_id", "reduced_formula"], validate="one_to_one")
    with np.load(directory / "embeddings.npz", allow_pickle=False) as store:
        arrays = {name: store[name] for name in config["extraction"]["poolings"]}
    if any(value.ndim != 2 or len(value) != len(metadata) or not np.isfinite(value).all()
           for value in arrays.values()):
        raise ValueError("Invalid embedding shape/order or nonfinite values")
    return metadata, arrays


def main() -> None:
    """Validate CLI configuration; return None, logging its experiment identifier."""
    args = cli(__doc__).parse_args()
    config = load_config(args.config)
    LOGGER.info("configuration_valid experiment=%s", config["experiment_id"])


if __name__ == "__main__":
    main()
