#!/usr/bin/env python3
"""Extrae masked mean pooling sobre estados MatGPT posteriores a ln_f."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, E1_REPORT, DiagnosticError, json_write, log, logger_for, sha256_file

LOGGER = logger_for("matgpt_e1_diagnostics")
CHECKPOINT = ALT / "models" / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
SOURCE = ALT / "models" / "MatGPT-source" / "mat-concat"
EXPECTED_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"


def build_model() -> tuple[Any, dict[str, int], int]:
    """Carga strictamente la arquitectura oficial y congela todo el backbone.

    Returns:
        Modelo en modo evaluación, mapa token→ID original y longitud de contexto.
    """
    if not CHECKPOINT.is_file() or not MODEL_INFO.is_file():
        raise DiagnosticError("Faltan checkpoint o vocabulario MatGPT E0.")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_SHA:
        raise DiagnosticError(f"Hash inesperado de checkpoint MatGPT: {checkpoint_hash}")
    source_manifest = ALT / "models" / "MatGPT-source.download-manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("resolved_commit") != EXPECTED_COMMIT:
        raise DiagnosticError("El código MatGPT no coincide con el commit auditado en E0.")
    info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in info["stoi"].items()}
    if any(info["itos"].get(str(v)) != k for k, v in stoi.items()) or int(info["block_size"]) != 442:
        raise DiagnosticError("Vocabulario o block_size no coincide con E0.")
    cfg = yaml.safe_load((SOURCE / "config" / "uncond.yaml").read_text(encoding="utf-8"))
    mc, tc = cfg["model"], cfg["training"]
    if (mc.get("n_embd"), mc.get("n_head"), mc.get("n_layer"), tc.get("conditioning_mode"), tc.get("num_properties")) != (768, 12, 12, "unconditional", 0):
        raise DiagnosticError(f"Configuración no corresponde a MatGPT incondicional: {cfg}")
    source_string = str(SOURCE.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    module = importlib.import_module("utils.models")
    model = module.ModifiedTransformerDecoder(vocab_size=len(stoi), n_embd=768, n_head=12, n_layer=12,
        block_size=442, dropout=float(mc["dropout"]), stoi=stoi, num_properties=0, conditioning_mode="unconditional")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, stoi, 442


def extract(dataset_path: Path, out_dir: Path, device_name: str) -> dict[str, Any]:
    """Extrae un vector por estructura, sin promediar start/end/padding.

    Args:
        dataset_path: Dataset E1 que conserva IDs, SLICES y split históricos.
        out_dir: Directorio nuevo del caché mean pooling.
        device_name: cpu, cuda, mps o auto.

    Returns:
        Auditoría escrita junto al caché.
    """
    if not dataset_path.is_file():
        raise DiagnosticError(f"No existe dataset E1: {dataset_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "matgpt_uncond_masked_mean.npz"
    metadata_path = out_dir / "embedding_metadata.csv"
    audit_path = out_dir / "embedding_audit.json"
    pooling_audit_path = out_dir / "pooling_audit.json"
    if any(path.exists() for path in (cache_path, metadata_path, audit_path, pooling_audit_path)):
        raise DiagnosticError(f"El destino ya contiene caché; usa un directorio versionado nuevo: {out_dir}")
    frame = pd.read_csv(dataset_path, low_memory=False)
    if frame.empty or frame.structure_id.astype(str).duplicated().any() or "matgpt_slices" not in frame:
        raise DiagnosticError("Dataset vacío, IDs repetidos o falta matgpt_slices.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("CUDA no está disponible.")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise DiagnosticError("MPS no está disponible.")
    device = torch.device(device_name)
    model, stoi, block_size = build_model()
    model.to(device)
    vectors: list[np.ndarray] = []
    lengths: list[int] = []
    max_logit_error = 0.0
    pad_id, start_id, end_id = stoi["<pad>"], stoi["<start>"], stoi["<end>"]
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            tokens = str(row.matgpt_slices).split()
            unknown = sorted(set(tokens) - set(stoi))
            if unknown:
                raise DiagnosticError(f"Tokens no reconocidos en {row.structure_id}: {unknown[:8]}")
            ids = [start_id, *(stoi[token] for token in tokens), end_id]
            if len(ids) > block_size or len(tokens) == 0:
                raise DiagnosticError(f"Longitud/token vacío inválido en {row.structure_id}: {len(ids)}")
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            hidden, logits_manual = hidden_and_logits(model, tensor, torch)
            properties = torch.empty((1, 0), dtype=torch.float32, device=device)
            logits_forward, _ = model(tensor, properties)
            max_logit_error = max(max_logit_error, float((logits_manual - logits_forward).abs().max().cpu()))
            # Convención exacta: todos los tokens SLICES, excluidos <start>, <end> y <pad>.
            vectors.append(hidden[0, 1:len(ids)-1].mean(dim=0).cpu().float().numpy().copy())
            lengths.append(len(tokens))
        probe_tokens = str(frame.iloc[0].matgpt_slices).split()
        probe_ids = [start_id, *(stoi[token] for token in probe_tokens), end_id]
        probe = torch.tensor([probe_ids], dtype=torch.long, device=device)
        probe_hidden, _ = hidden_and_logits(model, probe, torch)
        padded_probe = torch.tensor([probe_ids + [pad_id] * 7], dtype=torch.long, device=device)
        padded_hidden, _ = hidden_and_logits(model, padded_probe, torch)
        padding_error = float((probe_hidden[0, 1:len(probe_ids)-1].mean(0) - padded_hidden[0, 1:len(probe_ids)-1].mean(0)).abs().max().cpu())
        repeat_error = float((probe_hidden[0, 1:len(probe_ids)-1].mean(0) - torch.from_numpy(vectors[0]).to(device)).abs().max().cpu())
    if max_logit_error > 1e-5:
        raise DiagnosticError(f"lm_head(hidden) no reproduce logits: error={max_logit_error:.3g}")
    if padding_error > 1e-5 or repeat_error > 1e-5:
        raise DiagnosticError(f"La invariancia/reproducibilidad del masked mean falló: padding={padding_error:.3g}, repeat={repeat_error:.3g}")
    embeddings = np.stack(vectors).astype(np.float32)
    if embeddings.shape != (len(frame), 768) or not np.isfinite(embeddings).all():
        raise DiagnosticError(f"Embedding mean inválido: {embeddings.shape}")
    if float(embeddings.std(axis=0).mean()) <= 1e-8:
        raise DiagnosticError("El caché mean pooling carece de variabilidad.")
    np.savez_compressed(cache_path, embeddings=embeddings)
    frame = frame.copy()
    frame["slices_token_length"] = lengths
    frame.to_csv(metadata_path, index=False)
    audit = {"status": "PASS", "pooling": "mean de hidden states tras ln_f y antes de lm_head sobre todos los tokens SLICES; excluye start/end/pad",
             "checkpoint_sha256": sha256_file(CHECKPOINT), "model_info_sha256": sha256_file(MODEL_INFO),
             "source_commit": EXPECTED_COMMIT, "dataset_sha256": sha256_file(dataset_path),
             "cache_sha256": sha256_file(cache_path), "metadata_sha256": sha256_file(metadata_path),
             "rows": len(frame), "dimension": 768, "shape": list(embeddings.shape), "device": str(device),
             "all_backbone_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
             "logits_forward_max_abs_error": max_logit_error, "padding_invariance_max_abs_error": padding_error,
             "repeat_determinism_max_abs_error": repeat_error, "padding_policy": "inferencia individual sin padding artificial; probe adicional con padding final",
             "sequence_length_min": int(min(lengths)), "sequence_length_median": float(np.median(lengths)),
             "sequence_length_max": int(max(lengths)), "sequence_lengths": lengths,
             "dataset_path": str(dataset_path), "checkpoint_path": str(CHECKPOINT)}
    json_write(audit_path, audit)
    json_write(pooling_audit_path, audit)
    log(LOGGER, logging.INFO, "masked_mean_extracted", rows=len(frame), dimension=768, device=str(device), sequence_length_max=max(lengths))
    return audit


def hidden_and_logits(model: Any, token_ids: torch.Tensor, torch_module: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Ejecuta decoder causal y proyecta hidden states mediante lm_head.

    Args:
        model: Decoder MatGPT cargado en modo evaluación.
        token_ids: IDs tokenizados de forma `[batch, longitud]`.
        torch_module: Módulo PyTorch usado para crear máscara y memoria dummy.

    Returns:
        Estados posteriores a ln_f y logits correspondientes.
    """
    batch, length = token_ids.shape
    device = token_ids.device
    x = model.token_embedding_table(token_ids) + model.position_embedding_table(torch.arange(length, device=device))
    x = x.permute(1, 0, 2)
    mask = model.generate_square_subsequent_mask(length).to(device)
    memory = torch_module.zeros((length, batch, model.n_embd), device=device)
    decoded = model.transformer_decoder(x, memory, tgt_mask=mask).permute(1, 0, 2)
    hidden = model.ln_f(decoded)
    return hidden, model.lm_head(hidden)


def main() -> None:
    """Valida la CLI y extrae el caché masked mean E1.2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=E1_REPORT / "dataset.csv")
    parser.add_argument("--out-dir", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    try:
        extract(args.dataset, args.out_dir, args.device)
    except (DiagnosticError, OSError, ValueError, RuntimeError, KeyError, ImportError, AttributeError) as exc:
        log(LOGGER, logging.ERROR, "masked_mean_extraction_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
