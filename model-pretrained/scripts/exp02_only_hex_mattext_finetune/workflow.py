"""Configuration, provenance and scientific acceptance gates for the paper-based workflow.

Read original checkpoint and converted source data; write only experiment-specific
audit/results. A tokenizer reconstruction never silently becomes checkpoint proof.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("exp02")
DEFAULT_CONFIG = "experiments/exp02_only_hex_mattext_finetune/config.json"


def sha256(path: Path) -> str:
    """Hash file Path in bounded chunks; return hexadecimal SHA-256 str."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    """Read JSON file Path; return decoded value."""
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON-compatible value atomically to Path; return None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Resolve JSON config str/Path against project root; return validated settings dict."""
    target = ROOT / path
    config = read_json(target)
    if config["experiment_id"] != "exp02_only_hex_mattext_finetune":
        raise ValueError("Unexpected experiment identifier")
    if config["tokenizer"]["truncation_policy"] != "reject":
        raise ValueError("No scientific truncation policy has been approved")
    if config["training"]["regime"] != "full_encoder_finetuning":
        raise ValueError("This protocol implements the paper's full BERT fine-tuning")
    if config["training"]["learning_rate"] <= 0 or config["training"]["batch_size"] < 1:
        raise ValueError("Invalid optimization settings")
    for key, value in config["paths"].items():
        config["paths"][key] = ROOT / value
    config["config_path"] = target
    logging.basicConfig(level=config["logging_level"],
                        format='%(asctime)s level=%(levelname)s logger=%(name)s message="%(message)s"')
    return config


def cli(description: str) -> argparse.ArgumentParser:
    """Create CLI parser from module description str; return parser with configurable paths."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def fingerprint(config: dict[str, Any]) -> dict[str, str]:
    """Return content hashes of configuration, inputs, checkpoint and workflow code."""
    paths = config["paths"]
    files = {"config": config["config_path"], "slices": paths["slices_csv"],
             "source": paths["source_csv"], "conversion_manifest": paths["conversion_manifest"],
             "weights": paths["checkpoint"] / "model.safetensors",
             "model_config": paths["checkpoint"] / "config.json",
             "reference_tokenizer": paths["reports"] / "audit" / "reference_tokenizer.json",
             "candidate_tokens": paths["processed"] / "tokenized_candidate.jsonl",
             "approved_tokens": paths["processed"] / "tokenized.jsonl",
             "checkpoint_tokenizer_evidence": ROOT / config["tokenizer"]["checkpoint_evidence"],
             "graph_helpers": ROOT / "model-pretrained/scripts/exp01_mattext_frozen/utils.py",
             "physical_metrics": ROOT / "model-pretrained/scripts/exp01_mattext_frozen/05_evaluate_models.py"}
    if files["checkpoint_tokenizer_evidence"].is_file():
        evidence = read_json(files["checkpoint_tokenizer_evidence"])
        files["original_tokenizer_artifact"] = ROOT / evidence["original_artifact"]
    result = {key: sha256(path) for key, path in files.items() if path.exists()}
    result["code"] = hashlib.sha256("".join(f"{p.name}:{sha256(p)}\n" for p in
        sorted(Path(__file__).parent.glob("*.py"))).encode()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in ["transformers", "tokenizers",
                "huggingface_hub", "torch", "safetensors", "numpy", "pandas", "pymatgen", "slices"]}
    result["runtime"] = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    return result


def require_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Require current successful scientific acceptance audit; return report or raise RuntimeError."""
    audit = read_json(config["paths"]["reports"] / "audit" / "tokenizer_audit.json")
    if audit["status"] != "passed":
        raise RuntimeError(f"Fine-tuning blocked by tokenizer/data audit: {audit['blockers']}")
    if audit["fingerprint"] != fingerprint(config):
        raise RuntimeError("Scientific audit is stale; repeat it before submitting training")
    return audit


def main() -> None:
    """Validate CLI configuration and log experiment identifier; return None."""
    config = load_config(cli(__doc__).parse_args().config)
    LOGGER.info("configuration_valid experiment=%s", config["experiment_id"])


if __name__ == "__main__":
    main()
