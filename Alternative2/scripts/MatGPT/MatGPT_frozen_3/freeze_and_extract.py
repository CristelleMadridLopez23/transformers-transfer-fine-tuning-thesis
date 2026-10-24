#!/usr/bin/env python3
"""Extrae estados congelados de capas MatGPT sin leer etiquetas de test."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence, hidden_and_logits, load_model
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA,
    MODEL_INFO, REPORT_DIR, SOURCE_DIR, ExperimentError, load_dev, log, sha256_file, write_json)


def locate_end_positions(input_ids: torch.Tensor, end_id: int) -> torch.Tensor:
    """Finds the true end-token position and rejects zero/multiple end tokens."""
    mask = input_ids.eq(int(end_id))
    counts = mask.sum(dim=1)
    if not bool(torch.all(counts.eq(1))):
        raise ExperimentError("Cada fila debe tener exactamente un token <end>.")
    return mask.to(torch.int64).argmax(dim=1)


def run(device_name: str = "cpu", batch_size: int = 4) -> dict[str, Any]:
    """Capture block 3/6/9/12 END/MEAN and final post-ln_f control.

    Args:
        device_name: cpu, cuda, mps or auto.
        batch_size: Batch size for deterministic frozen extraction.
    Returns:
        Feature provenance manifest, stored under the allowed report root.
    """
    if batch_size < 1:
        raise ExperimentError("batch_size debe ser positivo.")
    output = REPORT_DIR / "features_cache"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "layer_features.npz"
    manifest_path = output / "feature_manifest.json"
    if archive.exists() or manifest_path.exists():
        raise ExperimentError(f"Caché existente; se preserva y no se sobrescribe: {output}")
    data = load_dev()
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ExperimentError("Checkpoint E0 SHA inesperado.")
    model, stoi, block_size, torch_mod = load_model(SOURCE_DIR, MODEL_INFO, CHECKPOINT, SOURCE_DIR / "config/uncond.yaml")
    device = device_name
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA solicitada pero no disponible.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ExperimentError("MPS solicitada pero no disponible.")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    pad_id = stoi["<pad>"]
    sequences = [encode_sequence(text, stoi, block_size) for text in data.matgpt_slices.astype(str)]
    end_id = stoi["<end>"]
    if any(ids.count(end_id) != 1 or ids[-1] != end_id for ids in sequences):
        raise ExperimentError("Cada secuencia debe contener un solo <end> en la posición final.")
    block_values: dict[int, list[torch.Tensor]] = {layer: [] for layer in (3, 6, 9, 12)}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_number in block_values:
        module = model.transformer_decoder.layers[layer_number - 1]
        handles.append(module.register_forward_hook(lambda _m, _i, out, n=layer_number: captured.__setitem__(n, out.detach())))
    arrays: dict[str, list[np.ndarray]] = {f"L{layer:02d}_{pool}": [] for layer in block_values for pool in ("END", "MEAN")}
    arrays.update({"FINAL_END": [], "FINAL_MEAN": []})
    position_hashes = []
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                current = sequences[start:start + batch_size]
                length = max(map(len, current))
                ids = torch.full((len(current), length), pad_id, dtype=torch.long, device=device)
                for i, row in enumerate(current):
                    ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                hidden_final, _ = hidden_and_logits(model, ids, torch_mod)
                for row_index, row_ids in enumerate(current):
                    end_pos = int(locate_end_positions(ids[row_index:row_index+1], end_id).item())
                    if end_pos != len(row_ids) - 1:
                        raise ExperimentError("Posición <end> real inconsistente.")
                    slices_mask = torch.zeros(len(row_ids), dtype=torch.bool, device=device)
                    slices_mask[1:end_pos] = True
                    arrays["FINAL_END"].append(hidden_final[row_index, end_pos].cpu().float().numpy().copy())
                    arrays["FINAL_MEAN"].append(hidden_final[row_index, :len(row_ids)][slices_mask].mean(dim=0).cpu().float().numpy().copy())
                    position_hashes.append({"structure_id": data.iloc[start + row_index].structure_id,
                        "length": len(row_ids), "end_position": end_pos, "slices_positions": int(slices_mask.sum())})
                    for layer_number in block_values:
                        # Transformer decoder blocks emit [sequence, batch, hidden].
                        state = captured[layer_number][:, row_index, :]
                        arrays[f"L{layer_number:02d}_END"].append(state[end_pos].cpu().float().numpy().copy())
                        arrays[f"L{layer_number:02d}_MEAN"].append(state[1:end_pos].mean(dim=0).cpu().float().numpy().copy())
    finally:
        for handle in handles:
            handle.remove()
    if any(p.requires_grad for p in model.parameters()) or any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
        raise ExperimentError("El backbone fue modificado o tiene parámetros entrenables.")
    features = {key: np.stack(value).astype(np.float32) for key, value in arrays.items()}
    if any(value.shape != (210, 768) or not np.isfinite(value).all() for value in features.values()):
        raise ExperimentError("Caché de capa con forma incorrecta o NaN/Inf.")
    # Compare final END/MEAN to the established H2 feature archive and verify right-padding invariance.
    prior_cache = np.load(REPORT_DIR.parent.parent / "MatGPT_frozen_2" / "v1" / "features" / "frozen_features_development.npz", allow_pickle=False)
    prior_ids = prior_cache["structure_id"].astype(str).tolist()
    prior_end = np.asarray(prior_cache["h_end"], dtype=np.float32)
    prior_mean = np.asarray(prior_cache["h_mean"], dtype=np.float32)
    prior_cache.close()
    if prior_ids != data.structure_id.tolist():
        raise ExperimentError("IDs del cache histórico de cotejo no coinciden.")
    equivalence_end = float(np.max(np.abs(features["FINAL_END"] - prior_end)))
    equivalence_mean = float(np.max(np.abs(features["FINAL_MEAN"] - prior_mean)))
    if max(equivalence_end, equivalence_mean) > 1e-5:
        raise ExperimentError(f"Final post-ln_f no coincide con extracción E0/H2: END={equivalence_end}, MEAN={equivalence_mean}")
    sample = sequences[0]
    sample_ids = torch.tensor([sample], dtype=torch.long, device=device)
    sample_pad = torch.tensor([sample + [pad_id] * 9], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_a, _ = hidden_and_logits(model, sample_ids, torch_mod)
        hidden_b, _ = hidden_and_logits(model, sample_pad, torch_mod)
        padding_error = float((hidden_a[0, len(sample) - 1] - hidden_b[0, len(sample) - 1]).abs().max().item())
    if padding_error > 1e-5 or np.std(features["FINAL_END"], axis=0).mean() <= 1e-8:
        raise ExperimentError("Prueba padding/end-embedding no pasó.")
    np.savez_compressed(archive, structure_id=data.structure_id.to_numpy(dtype=str), **features)
    metadata = data[["structure_id", "composition_id", "polymorph_family"]].copy()
    metadata.to_csv(output / "feature_ids.csv", index=False)
    pd.DataFrame(position_hashes).to_csv(output / "token_positions.csv", index=False)
    layer_outputs = {str(i): "raw decoder block output before ln_f" for i in (3, 6, 9, 12)}
    manifest = {"status": "PASS", "rows": 210, "compositions": 70, "dimension": 768,
        "historical_test_used_or_extracted": False, "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_info_sha256": sha256_file(MODEL_INFO), "dataset_sha256": sha256_file(DATASET),
        "fold_manifest_sha256": sha256_file(REPORT_DIR / "fold_assignments_checked.csv"),
        "extractor_sha256": sha256_file(Path(__file__)), "model_source_sha256": sha256_file(SOURCE_DIR / "utils/models.py"),
        "features": {key: ("raw decoder block output at actual <end> token before ln_f" if key.endswith("_END")
            else "raw decoder block mean over valid SLICES tokens before ln_f") for key in features if key.startswith("L")},
        "final_controls": {"FINAL_END": "post-ln_f actual <end>, before lm_head", "FINAL_MEAN": "post-ln_f mean of SLICES positions"},
        "layer_output_space": layer_outputs, "end_id": stoi["<end>"], "start_id": stoi["<start>"], "pad_id": stoi["<pad>"],
        "strict_load": True, "model_eval": True, "backbone_requires_grad": False,
        "final_end_max_difference_vs_h2_cache": equivalence_end, "final_mean_max_difference_vs_h2_cache": equivalence_mean,
        "right_padding_max_abs_error": padding_error, "all_features_finite": True, "deterministic": True,
        "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "cache_sha256": sha256_file(archive), "ids_sha256": sha256_file(output / "feature_ids.csv"),
        "positions_sha256": sha256_file(output / "token_positions.csv"), "input_has_energy_tokens": False}
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    """Extrae estados de capas permitidas sin ajustar el Transformer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        result = run(args.device, args.batch_size)
        log("frozen_features_extracted", status=result["status"], variants=len(result["features"]), test_used=False)
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log("feature_extraction_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
