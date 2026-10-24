#!/usr/bin/env python3
"""Carga Ridge R4 reconstruidos y checkpoints MLP8 H2 para H3-TEST."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import H2_RUNS, REPORT_DIR, H3TestError, sha256_file
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.two_head_predictor import TwoHeadPredictor


def load_ridge(fold: int, lock: dict[str, Any]) -> Any:
    """Carga el estimador R4 congelado del fold y valida su hash.

    Args:
        fold: Fold outer histórico en el rango 0–4.
        lock: Manifiesto pre-etiquetas inmutable.
    Returns:
        Pipeline StandardScaler + Ridge ya ajustada.
    """
    if fold not in range(5):
        raise H3TestError(f"Fold Ridge inválido: {fold}.")
    record = lock["R4_historical_models"][fold]
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path.cwd() / path
    if sha256_file(path) != record["sha256"]:
        raise H3TestError(f"Ridge fold {fold} cambió después del lock.")
    pipeline = joblib.load(path)
    if float(pipeline.named_steps["ridge"].alpha) != float(record["alpha"]):
        raise H3TestError(f"Alpha Ridge cambió en fold {fold}.")
    return pipeline


def load_h2(fold: int, seed: int, lock: dict[str, Any], device: str = "cpu") -> tuple[TwoHeadPredictor, dict[str, Any]]:
    """Carga estrictamente un checkpoint H2 y lo congela para inferencia.

    Args:
        fold: Fold histórico 0–4.
        seed: Semilla histórica (42, 123, 2026).
        lock: Manifiesto pre-etiquetas inmutable.
        device: Dispositivo de inferencia.
    Returns:
        MLP8 H2 en eval y payload con escalas fijadas al entrenar.
    """
    record = next((row for row in lock["H2_historical_models"]
                   if int(row["fold"]) == fold and int(row["seed"]) == seed), None)
    if record is None:
        raise H3TestError(f"No existe H2 fold={fold}, seed={seed} en el lock.")
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path.cwd() / path
    if sha256_file(path) != record["sha256"]:
        raise H3TestError(f"Checkpoint H2 fold={fold}, seed={seed} cambió después del lock.")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model") != "H2" or payload.get("fold") != fold or payload.get("seed") != seed:
        raise H3TestError("El checkpoint cargado no corresponde al fold/semilla H2 bloqueados.")
    model = TwoHeadPredictor(width=8, dropout=0.1).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise H3TestError("H2 contiene pesos entrenables durante inferencia.")
    return model, payload


def audit_loads(lock: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """Carga los cinco Ridge y quince H2 sin ejecutar ni seleccionar modelos.

    Args:
        lock: Estrategia bloqueada antes de etiquetas test.
        device: Dispositivo compatible para las cabezas pequeñas.
    Returns:
        Conteos y pruebas estrictas de carga de checkpoints.
    """
    ridge = [load_ridge(fold, lock) for fold in range(5)]
    checked = []
    for fold in range(5):
        for seed in (42, 123, 2026):
            model, payload = load_h2(fold, seed, lock, device)
            checked.append({"fold": fold, "seed": seed, "strict_load": True,
                            "best_epoch": int(payload["best_epoch"]),
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    result = {"status": "PASS", "ridge_estimators_strictly_loaded": len(ridge),
              "h2_heads_strictly_loaded": len(checked), "h2_loads": checked,
              "backbone_loaded_here": False, "test_labels_loaded": False}
    out = REPORT_DIR / "provenance/model_load_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Valida carga estricta de todos los estimadores bloqueados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    lock_path = REPORT_DIR / "provenance/test_evaluation_lock.json"
    if not lock_path.is_file():
        raise SystemExit("H3-TEST load FAILED: falta test_evaluation_lock.json")
    try:
        result = audit_loads(json.loads(lock_path.read_text(encoding="utf-8")), args.device)
        print(json.dumps({key: result[key] for key in ("status", "ridge_estimators_strictly_loaded", "h2_heads_strictly_loaded", "test_labels_loaded")}))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3-TEST load FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
