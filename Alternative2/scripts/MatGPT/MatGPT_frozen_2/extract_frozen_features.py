#!/usr/bin/env python3
"""Extrae h_end y h_mean solo para las 210 estructuras development."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence, hidden_and_logits, load_model
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import sha256_file as existing_sha256
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import ALT, CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA, ExperimentError, MODEL_INFO, REPORT_DIR, SCRIPT_DIR, SOURCE, load_development, log, sha256_file, verify_provenance, write_json


def extract(out_dir: Path, device_name: str = "cpu", batch_size: int = 4) -> dict[str, Any]:
    """Extrae ambos pools congelados, comprueba padding y conserva ids/hashes.

    Args:
        out_dir: Carpeta nueva para NPZ, metadatos y manifiesto.
        device_name: Dispositivo local de inferencia.
        batch_size: Número máximo de secuencias por lote.

    Returns:
        Manifiesto de extracción con integridad y auditorías.
    """
    if batch_size < 1 or device_name not in {"cpu", "mps", "cuda"}:
        raise ExperimentError("batch_size debe ser positivo y dispositivo cpu/mps/cuda.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ExperimentError(f"Directorio de extracción no vacío; se preserva: {out_dir}")
    audit = verify_provenance()
    data = load_development()
    model, stoi, block, torch_module = load_model(SOURCE, MODEL_INFO, CHECKPOINT, SOURCE / "config" / "uncond.yaml")
    if torch_module.is_inference_mode_enabled():
        raise ExperimentError("No iniciar extracción bajo inference_mode externo.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA no está disponible.")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ExperimentError("MPS no está disponible.")
    device = torch.device(device_name)
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ExperimentError("El backbone debe estar completamente congelado.")
    sequences = [encode_sequence(str(value), stoi, block) for value in data.matgpt_slices]
    if any(stoi["<energy>"] == token_id for ids in sequences for token_id in ids if "<energy>" in stoi):
        raise ExperimentError("Se detectó token de energía en una secuencia SLICES.")
    pad_id = stoi["<pad>"]
    ends: list[np.ndarray] = []
    means: list[np.ndarray] = []
    padding_errors: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            width = max(map(len, chunk))
            padded = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
            for j, ids in enumerate(chunk):
                padded[j, :len(ids)] = torch.tensor(ids, device=device)
            hidden, _ = hidden_and_logits(model, padded, torch)
            for j, ids in enumerate(chunk):
                ends.append(hidden[j, len(ids) - 1].detach().cpu().float().numpy().copy())
                means.append(hidden[j, 1:len(ids) - 1].mean(0).detach().cpu().float().numpy().copy())
        # Invariancia explícita ante padding a derecha para una cadena real.
        probe = sequences[0]
        probe_tensor = torch.tensor([probe], dtype=torch.long, device=device)
        probe_hidden, _ = hidden_and_logits(model, probe_tensor, torch)
        extra_pad = torch.tensor([probe + [pad_id] * 13], dtype=torch.long, device=device)
        padded_hidden, _ = hidden_and_logits(model, extra_pad, torch)
        padding_errors = [float((probe_hidden[0, len(probe) - 1] - padded_hidden[0, len(probe) - 1]).abs().max().cpu()),
                          float((probe_hidden[0, 1:len(probe)-1].mean(0) - padded_hidden[0, 1:len(probe)-1].mean(0)).abs().max().cpu())]
    h_end = np.asarray(ends, dtype=np.float32)
    h_mean = np.asarray(means, dtype=np.float32)
    if h_end.shape != (210, 768) or h_mean.shape != (210, 768) or not np.isfinite(h_end).all() or not np.isfinite(h_mean).all():
        raise ExperimentError("Embeddings no finitos o de dimensión incorrecta.")
    if max(padding_errors) > 1e-5 or float(h_end.std(0).mean()) < 1e-8:
        raise ExperimentError(f"Falla padding/invariabilidad h_end: {padding_errors}.")
    # Comprueba que a una estructura sin padding y en batch sus pools coinciden.
    if not np.allclose(h_end[0], probe_hidden[0, len(probe)-1].cpu().numpy(), atol=1e-5, rtol=0):
        raise ExperimentError("No se reproduce h_end del primer elemento.")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "frozen_features_development.npz"
    ids = np.asarray(data.structure_id.astype(str).tolist(), dtype="U256")
    inputs = np.asarray([np.asarray(seq, dtype=np.int32) for seq in sequences], dtype=object)
    # Ragged IDs se serializan como hashes JSON para conservar allow_pickle=False.
    input_hashes = [sha256_file_bytes(np.asarray(seq, dtype=np.int32).tobytes()) for seq in sequences]
    np.savez_compressed(cache, structure_id=ids, h_end=h_end, h_mean=h_mean)
    meta = data[["structure_id", "composition_id", "polymorph_family", "split"]].copy()
    meta["input_ids_sha256"] = input_hashes
    meta_path = out_dir / "feature_metadata.csv"
    meta.to_csv(meta_path, index=False)
    manifest = {"status": "PASS", "rows": 210, "compositions": 70, "dimension": 768, "development_only": True,
        "historical_test_embeddings_extracted": False, "checkpoint_sha256": audit["checkpoint_sha256"],
        "dataset_sha256": sha256_file(DATASET), "model_info_sha256": sha256_file(MODEL_INFO),
        "model_source_sha256": sha256_file(SOURCE / "utils" / "models.py"),
        "extractor_sha256": sha256_file(Path(__file__)), "pooling": {"h_end": "actual <end> hidden after ln_f before lm_head", "h_mean": "mean over SLICES token states after ln_f; excludes start/end/pad"},
        "padding_right_invariance_max_abs_error": {"h_end": padding_errors[0], "h_mean": padding_errors[1]},
        "h_end_coordinate_std_mean": float(h_end.std(axis=0).mean()), "device": str(device), "pytorch_version": torch.__version__,
        "input_ids_sha256": dict(zip(meta.structure_id.astype(str), input_hashes)), "cache_sha256": sha256_file(cache), "metadata_sha256": sha256_file(meta_path)}
    write_json(out_dir / "embeddings_manifest.json", manifest)
    return manifest


def sha256_file_bytes(value: bytes) -> str:
    """Calcula SHA-256 de bytes de input_ids."""
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    """Valida opciones y ejecuta extracción exclusivamente local."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR / "features")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        result = extract(args.out_dir, args.device, args.batch_size)
        log(logging.INFO, "frozen_features_extracted", rows=result["rows"], dim=result["dimension"], test=False)
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(logging.ERROR, "feature_extraction_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
