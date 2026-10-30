"""Reconstruye cache H3-C congelado y exige su SHA-256 histórico exacto.

El cache original no está en el árbol activo ni en Git. Esta reconstrucción
usa el checkpoint/preprocesamiento originales, únicamente train/val y eval().
Si el hash no coincide, el preflight permanece bloqueado.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG = logging.getLogger("geometry_rebuild_cache")
EXPECTED_SHA = "2f790a30ba0fa3cd961c12de82b65febf0ae651b1f217efd5df49c8c5a9cfb62"


def sha256(path: Path) -> str:
    """Hash de archivo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_rows(path: Path) -> tuple[list[str], list[str]]:
    """Extrae ID y SLICES de 210 filas train/val; ignora etiquetas y test."""
    ids: list[str] = []
    slices: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in ("train", "val"):
                continue
            ids.append(row["structure_id"])
            slices.append(row["matgpt_slices"])
    if len(ids) != 210 or len(set(ids)) != 210:
        raise ValueError("IDs de desarrollo inconsistentes")
    return ids, slices


def rebuild(root: Path, output: Path) -> dict[str, Any]:
    """Repite la extracción histórica L03–L12 y FINAL, y valida hash exacto."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"No sobrescribir cache: {output}")
    sys.path.insert(0, str(root))
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import (
        encode_sequence, hidden_and_logits, load_model)
    dataset = root / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
    ids, texts = input_rows(dataset)
    source = root / "models/MatGPT-source/mat-concat"
    checkpoint = source / "mat-uncond/weights/transformer_property_0.pth"
    info = checkpoint.with_name("model_info_property_0.json")
    if sha256(checkpoint) != "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414":
        raise ValueError("Checkpoint MatGPT distinto")
    torch.set_num_threads(2)
    model, stoi, context, torch_mod = load_model(source, info, checkpoint, source / "config/uncond.yaml")
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Backbone no congelado")
    sequences = [encode_sequence(text, stoi, context) for text in texts]
    end, pad = stoi["<end>"], stoi["<pad>"]
    if any(sequence.count(end) != 1 or sequence[-1] != end for sequence in sequences):
        raise ValueError("Token <end> inválido")
    layers = (3, 6, 9, 12)
    captured: dict[int, torch.Tensor] = {}
    handles = [model.transformer_decoder.layers[layer-1].register_forward_hook(
        lambda _module, _inputs, state, number=layer: captured.__setitem__(number, state.detach()))
        for layer in layers]
    arrays: dict[str, list[np.ndarray]] = {f"L{layer:02d}_{pool}": []
                                               for layer in layers for pool in ("END", "MEAN")}
    arrays.update({"FINAL_END": [], "FINAL_MEAN": []})
    try:
        with torch.inference_mode():
            for start in range(0, len(sequences), 4):
                batch = sequences[start:start+4]
                input_ids = torch.full((len(batch), max(map(len, batch))), pad, dtype=torch.long)
                for i, sequence in enumerate(batch):
                    input_ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                hidden, _ = hidden_and_logits(model, input_ids, torch_mod)
                for i, sequence in enumerate(batch):
                    stop = len(sequence) - 1
                    if int(input_ids[i, stop]) != end:
                        raise ValueError("Posición final de <end> incorrecta")
                    arrays["FINAL_END"].append(hidden[i, stop].cpu().float().numpy().copy())
                    arrays["FINAL_MEAN"].append(hidden[i, 1:stop].mean(dim=0).cpu().float().numpy().copy())
                    for layer in layers:
                        state = captured[layer][:, i, :]
                        arrays[f"L{layer:02d}_END"].append(state[stop].cpu().float().numpy().copy())
                        arrays[f"L{layer:02d}_MEAN"].append(state[1:stop].mean(dim=0).cpu().float().numpy().copy())
                LOG.info(json.dumps({"event": "embedding_progress", "processed": min(start+4, 210)}))
    finally:
        for handle in handles:
            handle.remove()
    values = {key: np.stack(rows).astype(np.float32) for key, rows in arrays.items()}
    if any(array.shape != (210, 768) or not np.isfinite(array).all() for array in values.values()):
        raise ValueError("Embeddings faltantes/no finitos")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, structure_id=np.asarray(ids, dtype=str), **values)
    actual = sha256(output)
    status = "PASS" if actual == EXPECTED_SHA else "FAIL"
    result = {"status": status, "rows": 210, "features": list(values),
              "cache_sha256": actual, "expected_sha256": EXPECTED_SHA,
              "checkpoint_sha256": sha256(checkpoint), "dataset_sha256": sha256(dataset),
              "historical_test_accessed": False, "backbone_trainable_parameters": 0}
    output.with_suffix(".rebuild_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    LOG.info(json.dumps({"event": "cache_rebuilt", "status": status, "sha256": actual}))
    if status != "PASS":
        raise ValueError("Cache reconstruido no coincide bit a bit con H3-C histórico")
    return result


def main() -> None:
    """Valida CLI y extrae sobre CPU sin entrenamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rebuild(args.repo_root, args.output)


if __name__ == "__main__":
    main()
