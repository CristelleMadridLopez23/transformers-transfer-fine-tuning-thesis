"""Shared paths, validation, vocabulary, and logging helpers for MatGPT E1."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ALT = ROOT / "Alternative2"
REPORTS = ALT / "reports" / "matgpt_e1_v2"
DATASET = REPORTS / "dataset.csv"
CONFIG = ALT / "scripts" / "matgpt_e1" / "config.yaml"
CHECKPOINT = ALT / "models" / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
SOURCE = ALT / "models" / "MatGPT-source" / "mat-concat"
SOURCE_MANIFEST = ALT / "models" / "MatGPT-source.download-manifest.json"
EXPECTED_CHECKPOINT_SHA256 = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"


class E1Error(RuntimeError):
    """Describe a validation or execution error that should stop an E1 stage."""


class JsonFormatter(logging.Formatter):
    """Format structured logging records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the level, message, and attached fields.

        Args:
            record: Python logging record.

        Returns:
            A JSON encoded log record.
        """
        base = logging.LogRecord("", 0, "", 0, "", (), None).__dict__
        fields = {k: v for k, v in record.__dict__.items() if k not in base}
        return json.dumps({"level": record.levelname, "message": record.getMessage(), **fields}, default=str)


def configure_logging() -> logging.Logger:
    """Configure and return the E1 JSON logger.

    Returns:
        Logger configured to write structured records to stderr.
    """
    logger = logging.getLogger("matgpt_e1")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 digest incrementally.

    Args:
        path: Existing file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report, creating its parent directory.

    Args:
        path: Destination path.
        payload: JSON serializable report data.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_vocab(path: Path = MODEL_INFO) -> tuple[dict[str, int], dict[int, str], int]:
    """Load MatGPT token maps and validate their inverse relation.

    Args:
        path: Checkpoint companion model_info JSON.

    Returns:
        Token IDs, inverse IDs, and maximum sequence length.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in data["stoi"].items()}
        itos = {int(k): str(v) for k, v in data["itos"].items()}
        size = int(data["block_size"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise E1Error(f"No se pudo cargar vocabulario MatGPT {path}: {exc}") from exc
    if len(stoi) != len(itos) or any(itos.get(idx) != token for token, idx in stoi.items()):
        raise E1Error("stoi e itos no son inversos; se detiene E1.")
    if not {"<start>", "<end>", "<pad>"}.issubset(stoi):
        raise E1Error("El vocabulario no incluye <start>, <end> y <pad>.")
    if size != 442:
        raise E1Error(f"block_size inesperado: {size}; E0 validó 442.")
    return stoi, itos, size


def verify_backbone_files() -> dict[str, str]:
    """Verify the original checkpoint hash and required E0 files.

    Returns:
        Hashes for the checkpoint, tokenizer metadata, and model config.
    """
    config = SOURCE / "config" / "uncond.yaml"
    required = (CHECKPOINT, MODEL_INFO, config, SOURCE / "utils" / "models.py", SOURCE_MANIFEST)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise E1Error(f"Faltan archivos validados en E0: {missing}")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise E1Error(f"Hash checkpoint no coincide con E0: {checkpoint_hash}")
    try:
        source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E1Error(f"No se pudo leer el manifiesto de código fuente E0: {exc}") from exc
    expected_commit = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
    if source_manifest.get("resolved_commit") != expected_commit:
        raise E1Error(f"Commit de MatGPT distinto al validado en E0: {source_manifest.get('resolved_commit')}")
    return {
        "checkpoint_sha256": checkpoint_hash,
        "model_info_sha256": sha256_file(MODEL_INFO),
        "config_sha256": sha256_file(config),
        "source_commit": str(source_manifest["resolved_commit"]),
    }
