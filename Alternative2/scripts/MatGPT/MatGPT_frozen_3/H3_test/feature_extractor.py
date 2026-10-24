#!/usr/bin/env python3
"""Extrae L03_MEAN, FINAL_END y FINAL_MEAN de las entradas test ya bloqueadas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import (CHECKPOINT, EXPECTED_CHECKPOINT_SHA, MODEL_INFO, REPORT_DIR,
                    SOURCE_ROOT, H3TestError, sha256_file, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import (
    encode_sequence, hidden_and_logits, load_model)


def locate_end_positions(input_ids: torch.Tensor, end_id: int) -> torch.Tensor:
    """Encuentra el único `<end>` real de cada secuencia.

    Args:
        input_ids: Tensor de IDs con padding a la derecha.
        end_id: ID original de `<end>`.
    Returns:
        Índices enteros de posición del token final.
    """
    mask = input_ids.eq(end_id)
    if not bool(torch.all(mask.sum(dim=1).eq(1))):
        raise H3TestError("Cada entrada debe contener un único token `<end>`.")
    return mask.to(torch.int64).argmax(dim=1)


def extract(batch_size: int = 4, device_name: str = "cpu") -> dict[str, Any]:
    """Extrae las tres vistas históricas sin abrir etiquetas DFT.

    Args:
        batch_size: Número de SLICES inferidas juntas.
        device_name: Dispositivo de inferencia (`cpu`, `mps` o `cuda`).
    Returns:
        Manifiesto de embeddings y controles de congelación/padding.
    """
    if batch_size < 1 or device_name not in {"cpu", "mps", "cuda"}:
        raise H3TestError("batch_size/dispositivo inválido.")
    lock_path = REPORT_DIR / "provenance/test_evaluation_lock.json"
    input_path = REPORT_DIR / "predictions/test_inputs.csv"
    output_dir = REPORT_DIR / "predictions"
    archive_path = output_dir / "test_features.npz"
    manifest_path = output_dir / "test_features_manifest.json"
    if not lock_path.is_file() or not input_path.is_file():
        raise H3TestError("Falta el lock pre-etiquetas o su archivo de entradas.")
    if archive_path.exists() or manifest_path.exists():
        raise H3TestError("Cache de test existente; se preserva y no se sobrescribe.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST_LABELS" or sha256_file(input_path) != lock["test_inputs"]["sha256"]:
        raise H3TestError("El lock o las entradas cambiaron.")
    inputs = pd.read_csv(input_path, dtype={"structure_id": str, "composition_id": str})
    forbidden = {"energy_total_ev", "energy_per_atom_ev", "E_DFT_eV_atom", "num_atoms"}
    if forbidden & set(inputs.columns) or len(inputs) != 24:
        raise H3TestError("Entradas con etiquetas o conteo distinto a 24.")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise H3TestError("Checkpoint MatGPT distinto al bloqueo histórico.")
    model, stoi, block_size, torch_module = load_model(
        SOURCE_ROOT, MODEL_INFO, CHECKPOINT, SOURCE_ROOT / "config/uncond.yaml")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise H3TestError("El backbone tiene parámetros entrenables.")
    device = torch.device(device_name)
    model.to(device).eval()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise H3TestError("CUDA solicitada pero no disponible.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise H3TestError("MPS solicitada pero no disponible.")
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    pad_id, start_id, end_id = stoi["<pad>"], stoi["<start>"], stoi["<end>"]
    sequences = [encode_sequence(value, stoi, block_size) for value in inputs.matgpt_slices.astype(str)]
    if any(row[0] != start_id or row[-1] != end_id or row.count(end_id) != 1 for row in sequences):
        raise H3TestError("Las secuencias no respetan `<start> SLICES <end>`.")
    captured: dict[str, torch.Tensor] = {}
    hook = model.transformer_decoder.layers[2].register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("layer3", output.detach()))
    l03: list[np.ndarray] = []
    final_end: list[np.ndarray] = []
    final_mean: list[np.ndarray] = []
    positions: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for begin in range(0, len(sequences), batch_size):
                batch = sequences[begin:begin + batch_size]
                width = max(map(len, batch))
                tokens = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
                for row_index, row in enumerate(batch):
                    tokens[row_index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                hidden, _ = hidden_and_logits(model, tokens, torch_module)
                end_positions = locate_end_positions(tokens, end_id)
                raw_layer = captured["layer3"]
                for row_index, row in enumerate(batch):
                    end_pos = int(end_positions[row_index].item())
                    if end_pos != len(row) - 1:
                        raise H3TestError("`<end>` se encontró en una posición inesperada.")
                    l03.append(raw_layer[1:end_pos, row_index, :].mean(dim=0).cpu().float().numpy().copy())
                    final_end.append(hidden[row_index, end_pos].cpu().float().numpy().copy())
                    final_mean.append(hidden[row_index, 1:end_pos].mean(dim=0).cpu().float().numpy().copy())
                    positions.append({"structure_id": inputs.iloc[begin + row_index].structure_id,
                                      "length": len(row), "start_position": 0, "end_position": end_pos,
                                      "slices_token_count": end_pos - 1})
    finally:
        hook.remove()
    arrays = {"L03_MEAN": np.stack(l03).astype(np.float32),
              "FINAL_END": np.stack(final_end).astype(np.float32),
              "FINAL_MEAN": np.stack(final_mean).astype(np.float32)}
    if any(value.shape != (24, 768) or not np.isfinite(value).all() for value in arrays.values()):
        raise H3TestError("Embeddings test inválidos, no finitos o con forma incorrecta.")
    probe = sequences[0]
    probe_plain = torch.tensor([probe], dtype=torch.long, device=device)
    probe_padded = torch.tensor([probe + [pad_id] * 11], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_plain, _ = hidden_and_logits(model, probe_plain, torch_module)
        layer_plain = captured["layer3"].clone()
        hidden_pad, _ = hidden_and_logits(model, probe_padded, torch_module)
        layer_pad = captured["layer3"].clone()
    end_error = float(torch.max(torch.abs(hidden_plain[0, len(probe) - 1] - hidden_pad[0, len(probe) - 1])).item())
    layer_mean_plain = layer_plain[1:len(probe)-1, 0, :].mean(dim=0)
    layer_mean_pad = layer_pad[1:len(probe)-1, 0, :].mean(dim=0)
    layer_error = float(torch.max(torch.abs(layer_mean_plain - layer_mean_pad)).item())
    if max(end_error, layer_error) > 1e-5:
        raise H3TestError(f"Invariancia al padding falló: FINAL_END={end_error}, L03_MEAN={layer_error}.")
    if any(parameter.requires_grad for parameter in model.parameters()) or any(
            not torch.equal(frozen_state[key], value.detach().cpu()) for key, value in model.state_dict().items()):
        raise H3TestError("El backbone tuvo gradientes o cambió durante inferencia.")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, structure_id=inputs.structure_id.to_numpy(dtype=str), **arrays)
    pd.DataFrame(positions).to_csv(output_dir / "test_token_positions.csv", index=False)
    manifest = {"status": "PASS", "test_labels_loaded": False, "structures": len(inputs),
                "compositions": int(inputs.composition_id.nunique()), "dimension": 768,
                "checkpoint_sha256": sha256_file(CHECKPOINT), "model_info_sha256": sha256_file(MODEL_INFO),
                "input_csv_sha256": sha256_file(input_path), "feature_archive_sha256": sha256_file(archive_path),
                "feature_ids": inputs.structure_id.tolist(), "features": {
                    "L03_MEAN": "raw block 3 output, mean over SLICES positions only, before ln_f",
                    "FINAL_END": "post-ln_f hidden state at actual end token, before lm_head",
                    "FINAL_MEAN": "post-ln_f mean over SLICES positions only"},
                "special_token_ids": {"start": start_id, "end": end_id, "pad": pad_id},
                "strict_checkpoint_load": True, "eval_mode": True, "backbone_trainable_parameters": 0,
                "all_values_finite": True, "right_padding_max_abs_error_final_end": end_error,
                "right_padding_max_abs_error_l03_mean": layer_error, "device": str(device),
                "torch_version": torch.__version__, "numpy_version": np.__version__}
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    """Extrae y verifica features del test sin acceder a las etiquetas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        result = extract(args.batch_size, args.device)
        print(json.dumps({"status": result["status"], "rows": result["structures"],
                          "test_labels_loaded": False, "padding_error": result["right_padding_max_abs_error_final_end"]}, ensure_ascii=False))
    except (H3TestError, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"H3-TEST feature extraction FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
