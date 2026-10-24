#!/usr/bin/env python3
"""Extract verified last-<end> hidden states from the frozen MatGPT backbone."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.common import ALT, CHECKPOINT, DATASET, E1Error, MODEL_INFO, REPORTS, SOURCE, configure_logging, read_vocab, sha256_file, verify_backbone_files, write_json


LOGGER = logging.getLogger("matgpt_e1")


def load_model(source_dir: Path, model_info: Path, checkpoint: Path, config_path: Path) -> tuple[Any, dict[str, int], int, Any]:
    """Load the official unconditional model with strict state matching.

    Args:
        source_dir: Official MatGPT mat-concat directory.
        model_info: Companion vocabulary metadata JSON.
        checkpoint: Original pretrained state dictionary.
        config_path: Official uncond.yaml path.

    Returns:
        Model in eval mode, token map, context length, and PyTorch module.
    """
    import torch

    stoi, _, block_size = read_vocab(model_info)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    train_cfg = config["training"]
    if train_cfg.get("conditioning_mode") != "unconditional" or int(train_cfg.get("num_properties", -1)) != 0:
        raise E1Error("El YAML no describe MatGPT incondicional.")
    expected_architecture = {"n_embd": 768, "n_head": 12, "n_layer": 12, "dropout": 0.1}
    if any(model_cfg.get(key) != value for key, value in expected_architecture.items()):
        raise E1Error(f"La arquitectura YAML no coincide con E0: {model_cfg}")
    source_string = str(source_dir.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    try:
        module = importlib.import_module("utils.models")
        model_class = module.ModifiedTransformerDecoder
    except (ImportError, AttributeError) as exc:
        raise E1Error(f"No se pudo importar arquitectura oficial MatGPT: {exc}") from exc
    model = model_class(vocab_size=len(stoi), n_embd=int(model_cfg["n_embd"]), n_head=int(model_cfg["n_head"]),
                        n_layer=int(model_cfg["n_layer"]), block_size=block_size, dropout=float(model_cfg["dropout"]),
                        stoi=stoi, num_properties=0, conditioning_mode="unconditional")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise E1Error("El backbone contiene parámetros entrenables.")
    return model, stoi, block_size, torch


def encode_sequence(sequence: str, stoi: dict[str, int], block_size: int) -> list[int]:
    """Encode one whitespace-tokenized SLICES string with MatGPT boundaries.

    Args:
        sequence: SLICES model input.
        stoi: Original MatGPT token mapping.
        block_size: Maximum original model sequence length.

    Returns:
        Encoded integer IDs, including start and end tokens.
    """
    tokens = sequence.split()
    unknown = sorted(set(tokens) - set(stoi))
    if unknown:
        raise E1Error(f"Tokens desconocidos: {unknown[:12]}")
    ids = [stoi["<start>"], *(stoi[token] for token in tokens), stoi["<end>"]]
    if len(ids) > block_size:
        raise E1Error(f"Secuencia de longitud {len(ids)} excede block_size={block_size}.")
    return ids


def hidden_and_logits(model: Any, token_ids: Any, torch: Any) -> tuple[Any, Any]:
    """Run the official causal decoder path through ln_f and lm_head.

    Args:
        model: Loaded official MatGPT module in eval mode.
        token_ids: Padded batch of token IDs.
        torch: Imported PyTorch module.

    Returns:
        Hidden states after ln_f and corresponding vocabulary logits.
    """
    batch, length = token_ids.shape
    device = token_ids.device
    x = model.token_embedding_table(token_ids)
    x = x + model.position_embedding_table(torch.arange(length, device=device))
    x = x.permute(1, 0, 2)
    causal_mask = model.generate_square_subsequent_mask(length).to(device)
    memory = torch.zeros((length, batch, model.n_embd), device=device)
    decoded = model.transformer_decoder(x, memory, tgt_mask=causal_mask).permute(1, 0, 2)
    hidden = model.ln_f(decoded)
    return hidden, model.lm_head(hidden)


def run(dataset_path: Path, output_dir: Path, batch_size: int, device_name: str) -> dict[str, Any]:
    """Extract embeddings and save a dependency-aware cache.

    Args:
        dataset_path: Prepared E1 dataset CSV.
        output_dir: Destination for NPZ cache, metadata, and audit JSON.
        batch_size: Number of sequences processed per batch.
        device_name: cpu, cuda, mps, or auto.

    Returns:
        Embedding audit information.
    """
    if not dataset_path.is_file():
        raise E1Error(f"No existe el dataset preparado: {dataset_path}; ejecuta prepare_dataset.py primero.")
    if any((output_dir / name).exists() for name in ("matgpt_uncond_last_token.npz", "embedding_metadata.csv", "embedding_audit.json")):
        raise E1Error(f"Ya existe caché en {output_dir}; preserva el resultado y usa otra ruta.")
    hashes = verify_backbone_files()
    frame = pd.read_csv(dataset_path, low_memory=False)
    if frame.empty:
        raise E1Error(f"El dataset preparado está vacío: {dataset_path}. Ejecuta de nuevo prepare_dataset.py actualizado.")
    if "structure_id" not in frame or frame["structure_id"].isna().any():
        raise E1Error("Falta structure_id o hay IDs estructurales vacíos en el dataset.")
    if frame["structure_id"].astype(str).duplicated().any():
        raise E1Error("Hay structure_id duplicados en el dataset preparado.")
    model, stoi, block_size, torch = load_model(SOURCE, MODEL_INFO, CHECKPOINT, SOURCE / "config" / "uncond.yaml")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise E1Error("Se solicitó CUDA pero no está disponible.")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise E1Error("Se solicitó MPS pero no está disponible.")
    device = torch.device(device_name)
    model.to(device)
    pad_id = stoi["<pad>"]
    if "matgpt_slices" not in frame:
        raise E1Error("El dataset preparado no contiene matgpt_slices; vuelve a generarlo con prepare_dataset.py actualizado.")
    encoded = [encode_sequence(str(value), stoi, block_size) for value in frame["matgpt_slices"]]
    end_positions = [len(ids) - 1 for ids in encoded]
    vectors: list[np.ndarray] = []
    max_logit_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch_ids = encoded[start:start + batch_size]
            max_len = max(map(len, batch_ids))
            tensor = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
            for row_index, ids in enumerate(batch_ids):
                tensor[row_index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            hidden, logits_manual = hidden_and_logits(model, tensor, torch)
            properties = torch.empty((len(batch_ids), 0), dtype=torch.float32, device=device)
            logits_forward, _ = model(tensor, properties)
            max_logit_error = max(max_logit_error, float((logits_manual - logits_forward).abs().max().item()))
            for row_index, ids in enumerate(batch_ids):
                vectors.append(hidden[row_index, len(ids) - 1].detach().cpu().float().numpy().copy())
    if max_logit_error > 1e-5:
        raise E1Error(f"Equivalencia logits-forward falló: max_abs_error={max_logit_error:.3g}")
    embeddings = np.stack(vectors).astype(np.float32)
    if embeddings.shape != (len(frame), model.n_embd):
        raise E1Error(f"Dimensión inesperada de embeddings: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise E1Error("Embeddings contienen NaN o Inf.")
    with torch.inference_mode():
        sample = encoded[0]
        probe = torch.tensor([sample], dtype=torch.long, device=device)
        probe_hidden, _ = hidden_and_logits(model, probe, torch)
        extra_pad = torch.tensor([sample + [pad_id] * 7], dtype=torch.long, device=device)
        padded_hidden, _ = hidden_and_logits(model, extra_pad, torch)
        padding_error = float((probe_hidden[0, len(sample) - 1] - padded_hidden[0, len(sample) - 1]).abs().max().item())
        repeat_error = float((probe_hidden[0, len(sample) - 1] - embeddings[0]).abs().max().item())
    if padding_error > 1e-5 or repeat_error > 1e-5:
        raise E1Error(f"Prueba padding/determinismo falló: padding={padding_error}, repeat={repeat_error}")
    if len(embeddings) > 1 and float(np.std(embeddings, axis=0).mean()) <= 1e-8:
        raise E1Error("Los embeddings carecen de variabilidad entre las estructuras.")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "matgpt_uncond_last_token.npz"
    np.savez_compressed(npz_path, embeddings=embeddings)
    metadata_path = output_dir / "embedding_metadata.csv"
    frame.to_csv(metadata_path, index=False)
    audit = {
        "status": "PASS", "checkpoint_sha256": hashes["checkpoint_sha256"], "model_info_sha256": hashes["model_info_sha256"],
        "dataset_sha256": sha256_file(dataset_path), "pooling": "hidden state at explicit <end> token after ln_f, before lm_head",
        "rows": len(frame), "dimension": int(embeddings.shape[1]), "shape": list(embeddings.shape), "device": str(device),
        "all_backbone_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
        "logits_forward_max_abs_error": max_logit_error, "padding_invariance_max_abs_error": padding_error,
        "repeat_determinism_max_abs_error": repeat_error, "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "embedding_norm_std": float(np.linalg.norm(embeddings, axis=1).std()), "coordinate_std_mean": float(embeddings.std(axis=0).mean()),
        "pytorch_version": torch.__version__, "numpy_version": np.__version__, "batch_size": batch_size,
        "checkpoint_path": str(CHECKPOINT), "model_info_path": str(MODEL_INFO), "dataset_path": str(dataset_path),
        "cache_sha256": sha256_file(npz_path), "metadata_sha256": sha256_file(metadata_path),
        "extractor_script_sha256": sha256_file(Path(__file__)),
        "common_helpers_sha256": sha256_file(Path(__file__).with_name("common.py")),
        "official_model_source_sha256": sha256_file(SOURCE / "utils" / "models.py"),
        "seed": 42, "cache_reuse_requires_matching_hashes": True,
    }
    write_json(output_dir / "embedding_audit.json", audit)
    LOGGER.info("embeddings_extracted", extra={"rows": len(frame), "dimension": embeddings.shape[1], "device": str(device)})
    return audit


def parse_args() -> argparse.Namespace:
    """Parse extractor CLI options.

    Returns:
        Validated command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size debe ser >= 1")
    return args


def main() -> None:
    """Run extraction and log recoverable errors."""
    logger = configure_logging()
    args = parse_args()
    try:
        run(args.dataset, args.out_dir, args.batch_size, args.device)
    except (E1Error, OSError, ValueError, ImportError, RuntimeError) as exc:
        logger.error("embedding_extraction_failed", extra={"error": str(exc)})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
