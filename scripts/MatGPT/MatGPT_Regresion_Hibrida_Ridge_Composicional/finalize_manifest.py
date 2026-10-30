"""Crea un único manifiesto compacto con hashes de entradas y artefactos."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import common as cm

LOG = logging.getLogger("matgpt_hybrid_manifest")


def file_hash(path: Path) -> str:
    """Calcula SHA-256 de un archivo.

    Args:
        path: Archivo a resumir.
    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize(output: Path = cm.REPORT) -> dict[str, Any]:
    """Escribe el manifiesto final; no sobrescribe uno anterior.

    Args:
        output: Carpeta completa del experimento.
    Returns:
        Manifiesto escrito.
    """
    target = output / "manifest.json"
    if target.exists():
        raise FileExistsError(f"No se sobrescribe el manifiesto: {target}")
    resource_path = output / "resources.json"
    if not resource_path.is_file():
        raise FileNotFoundError("Falta resources.json: ejecute el wrapper medido")
    resources = json.loads(resource_path.read_text(encoding="utf-8"))
    if resources.get("exit_code") != 0:
        raise ValueError("La corrida medida terminó con error")
    files = sorted(path for path in output.rglob("*") if path.is_file() and path != target)
    code_files = sorted(path for path in Path(__file__).resolve().parent.rglob("*.py")
                        if "__pycache__" not in path.parts)
    result = {"status": "PASS", "experiment": cm.EXPERIMENT,
              "created_outputs": {path.relative_to(output).as_posix(): file_hash(path) for path in files},
              "code_sha256": {path.relative_to(cm.ROOT).as_posix(): file_hash(path) for path in code_files},
              "config_sha256": file_hash(Path(__file__).resolve().parent / "config.yaml"),
              "input_sha256": {"development_dataset": cm.sha256(cm.DATASET), "fold_assignments": cm.sha256(cm.FOLDS),
                               "frozen_lora_predictions": cm.sha256(cm.C_PREDICTIONS),
                               "h3c_replay": cm.sha256(cm.H3C_REPLAY)},
              "historical_test_accessed": False, "transformer_inference": False,
              "mlp8_or_lora_training": False, "newly_fitted_components": ["five compositional Ridge pipelines"],
              "resource_record": {k: resources[k] for k in ("started_at_utc", "ended_at_utc", "wall_seconds",
                                   "user_cpu_seconds", "system_cpu_seconds", "peak_rss_bytes", "exit_code")}}
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Manifiesto final: %s", target)
    return result


def main() -> None:
    """CLI del generador de manifiesto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cm.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    finalize(args.output.resolve())


if __name__ == "__main__":
    main()
