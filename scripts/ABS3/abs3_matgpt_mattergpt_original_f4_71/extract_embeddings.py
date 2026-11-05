"""Extrae FINAL_END de los extractores originales MatGPT y MatterGPT para ABS3.

La extracción usa las cadenas SLICES históricas por `structure_id`, mantiene
MatterGPT condicionado con [1.0, 2.0] y guarda cachés auditables 210×768.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71"
DATASET = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
MATGPT_SOURCE = ROOT / "models/MatGPT-source/mat-concat"
MATGPT_CHECKPOINT = MATGPT_SOURCE / "mat-uncond/weights/transformer_property_0.pth"
MATGPT_INFO = MATGPT_CHECKPOINT.with_name("model_info_property_0.json")
MATGPT_CONFIG = MATGPT_SOURCE / "config/uncond.yaml"
MATTER_DIR = ROOT / "models/MatterGPT"
EXPECTED_MATGPT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_MATTER_SHA = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
PROPERTY_VECTOR = (1.0, 2.0)
LOGGER = logging.getLogger("abs3_original_extractors")


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo por bloques.

    Args:
        path: Archivo a resumir.
    Returns:
        Digest hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset() -> pd.DataFrame:
    """Carga y valida la cohorte de desarrollo, sin leer el test.

    Args:
        None.
    Returns:
        Dataset ordenado con 210 estructuras y siete descriptores.
    """
    frame = pd.read_csv(DATASET, low_memory=False)
    required = {"structure_id", "composition_id", "family", "energy_dft_ev_atom", "SLICES",
                "a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
                "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3", "outer_fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas ABS3: {sorted(missing)}")
    if len(frame) != 210 or frame.structure_id.nunique() != 210 or frame.composition_id.nunique() != 70:
        raise ValueError("El conjunto debe contener 210 estructuras y 70 composiciones")
    if frame.groupby("composition_id").size().ne(3).any():
        raise ValueError("Se requiere un triplete completo por composición")
    if frame.SLICES.isna().any() or not frame.SLICES.eq(frame.slices_original).all():
        raise ValueError("Las SLICES no coinciden byte a byte con la entrada histórica")
    geometry = frame[["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
                      "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]].to_numpy(float)
    if not np.isfinite(geometry).all() or not np.isfinite(frame.energy_dft_ev_atom.to_numpy(float)).all():
        raise ValueError("Targets o geometría contienen NaN/Inf")
    return frame.sort_values(["composition_id", "family"]).reset_index(drop=True)


def load_matgpt() -> tuple[torch.nn.Module, dict[str, int], int, dict[str, Any]]:
    """Carga MatGPT original con compatibilidad estricta de arquitectura.

    Args:
        None.
    Returns:
        Modelo, vocabulario, longitud máxima y auditoría de carga.
    """
    from utils.models import ModifiedTransformerDecoder

    observed = sha256(MATGPT_CHECKPOINT)
    if observed != EXPECTED_MATGPT_SHA:
        raise ValueError(f"SHA MatGPT inesperado: {observed}")
    info = json.loads(MATGPT_INFO.read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in info["stoi"].items()}
    block_size = int(info["block_size"])
    config = yaml.safe_load(MATGPT_CONFIG.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    training_cfg = config["training"]
    if (int(model_cfg["n_embd"],) != 768 or int(model_cfg["n_head"]) != 12
            or int(model_cfg["n_layer"]) != 12 or training_cfg.get("conditioning_mode") != "unconditional"):
        raise ValueError("La configuración local no coincide con MatGPT incondicional de 12 bloques")
    model = ModifiedTransformerDecoder(vocab_size=len(stoi), n_embd=int(model_cfg["n_embd"]),
        n_head=int(model_cfg["n_head"]), n_layer=int(model_cfg["n_layer"]), block_size=block_size,
        dropout=float(model_cfg["dropout"]), stoi=stoi, num_properties=0,
        conditioning_mode="unconditional")
    state = torch.load(MATGPT_CHECKPOINT, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Carga estricta MatGPT falló: {result}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    audit = {"checkpoint": str(MATGPT_CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": observed,
             "tokenizer_sha256": sha256(MATGPT_INFO), "config_sha256": sha256(MATGPT_CONFIG),
             "n_layer": 12, "n_embd": 768, "block_size": block_size,
             "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}
    return model, stoi, block_size, audit


def load_mattergpt() -> tuple[torch.nn.Module, dict[str, int], int, dict[str, Any]]:
    """Carga MatterGPT original estrictamente y crea el tokenizador publicado.

    Args:
        None.
    Returns:
        Modelo, tokenizador, contexto y auditoría de carga.
    """
    if sha256(MATTER_DIR / "pytorch_model.pt") != EXPECTED_MATTER_SHA:
        raise ValueError("SHA del checkpoint original MatterGPT no coincide")
    config = json.loads((MATTER_DIR / "config.json").read_text(encoding="utf-8"))
    from model import GPT, GPTConfig

    model = GPT(GPTConfig(config["vocab_size"], config["block_size"], **{
        key: value for key, value in config.items()
        if key not in {"model_type", "architectures", "vocab_size", "block_size"}}))
    state = torch.load(MATTER_DIR / "pytorch_model.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Carga estricta MatterGPT falló: {result}")
    vocab = sorted(set((MATTER_DIR / "Voc_prior").read_text(encoding="utf-8").splitlines()) | {"<", ">"})
    stoi = {token: index for index, token in enumerate(vocab)}
    if len(stoi) != int(config["vocab_size"]) or int(config["num_props"]) != 2:
        raise ValueError("Voc_prior/config.json no coinciden con MatterGPT auditado")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    audit = {"checkpoint": str((MATTER_DIR / "pytorch_model.pt").relative_to(ROOT)),
             "checkpoint_sha256": EXPECTED_MATTER_SHA, "tokenizer_sha256": sha256(MATTER_DIR / "Voc_prior"),
             "config_sha256": sha256(MATTER_DIR / "config.json"), "n_layer": 12,
             "n_embd": 768, "block_size": int(config["block_size"]),
             "property_vector": list(PROPERTY_VECTOR), "missing_keys": result.missing_keys,
             "unexpected_keys": result.unexpected_keys}
    return model, stoi, int(config["block_size"]), audit


def encode(frame: pd.DataFrame, stoi: dict[str, int], block_size: int, matter: bool) -> list[list[int]]:
    """Codifica las SLICES con el tokenizador propio, sin OOV ni truncamiento.

    Args:
        frame: Dataset de desarrollo.
        stoi: Mapa token→ID original.
        block_size: Contexto máximo.
        matter: Añade el prefijo y cierre propios de MatterGPT.
    Returns:
        Secuencias de token IDs en orden de las filas.
    """
    output: list[list[int]] = []
    for row in frame.itertuples():
        tokens = str(row.SLICES).split()
        unknown = sorted(set(tokens) - set(stoi))
        if unknown:
            raise ValueError(f"OOV {row.structure_id}: {unknown[:8]}")
        if matter:
            tokens = [">", *tokens, "<"]
        else:
            tokens = ["<start>", *tokens, "<end>"]
        ids = [stoi[token] for token in tokens]
        if len(ids) > block_size:
            raise ValueError(f"Truncamiento requerido {row.structure_id}: {len(ids)}>{block_size}")
        output.append(ids)
    return output


def padded_batch(sequences: list[list[int]], rows: list[int], pad_id: int) -> tuple[Tensor, Tensor]:
    """Crea lote con padding derecho y posición de token terminal.

    Args:
        sequences: Input IDs del corpus.
        rows: Índices de filas que forman el lote.
        pad_id: Token de padding original.
    Returns:
        IDs tensor y posiciones terminales.
    """
    width = max(len(sequences[index]) for index in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for batch_index, row_index in enumerate(rows):
        ids[batch_index, :len(sequences[row_index])] = torch.tensor(sequences[row_index], dtype=torch.long)
    positions = torch.tensor([len(sequences[index]) - 1 for index in rows], dtype=torch.long)
    return ids, positions


def extract_matgpt(model: torch.nn.Module, seqs: list[list[int]], pad_id: int, batch_size: int) -> np.ndarray:
    """Extrae MatGPT FINAL_END posterior a transformer_decoder y ln_f.

    Args:
        model: MatGPT original.
        seqs: IDs de secuencia con start/end.
        pad_id: ID `<pad>`.
        batch_size: Límite de memoria por lote.
    Returns:
        Matriz float32 [210,768].
    """
    vectors = np.empty((len(seqs), 768), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(seqs), batch_size):
            rows = list(range(start, min(start + batch_size, len(seqs))))
            ids, positions = padded_batch(seqs, rows, pad_id)
            length, batch = ids.shape[1], ids.shape[0]
            hidden = model.token_embedding_table(ids) + model.position_embedding_table(torch.arange(length))
            hidden = hidden.permute(1, 0, 2)
            mask = model.generate_square_subsequent_mask(length)
            memory = torch.zeros((length, batch, model.n_embd))
            hidden = model.transformer_decoder(hidden, memory, tgt_mask=mask).permute(1, 0, 2)
            hidden = model.ln_f(hidden)
            batch_ids = torch.arange(batch)
            vectors[rows] = hidden[batch_ids, positions].cpu().numpy()
            LOGGER.info("MatGPT extraction %d/%d", min(start + batch_size, len(seqs)), len(seqs))
    return vectors


def extract_mattergpt(model: torch.nn.Module, seqs: list[list[int]], batch_size: int) -> np.ndarray:
    """Extrae MatterGPT FINAL_END en ln_f, preservando prop=[1.0,2.0].

    Args:
        model: MatterGPT original.
        seqs: IDs de secuencia con `>` inicial y `<` terminal.
        batch_size: Límite de memoria por lote.
    Returns:
        Matriz float32 [210,768].
    """
    vectors = np.empty((len(seqs), 768), dtype=np.float32)
    model.eval()
    for start in range(0, len(seqs), batch_size):
        rows = list(range(start, min(start + batch_size, len(seqs))))
        width = max(len(seqs[index]) for index in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        for batch_index, row_index in enumerate(rows):
            ids[batch_index, :len(seqs[row_index])] = torch.tensor(seqs[row_index], dtype=torch.long)
        prop = torch.tensor(PROPERTY_VECTOR, dtype=torch.float32).view(1, 1, 2).expand(len(rows), 1, 2)
        capture: list[Tensor] = []
        hook = model.ln_f.register_forward_hook(lambda _module, _inputs, output: capture.append(output))
        try:
            with torch.inference_mode():
                model(ids, prop=prop)
        finally:
            hook.remove()
        if len(capture) != 1:
            raise RuntimeError("MatterGPT no produjo exactamente una salida ln_f")
        terminal = torch.tensor([len(seqs[index]) - 1 for index in rows], dtype=torch.long) + 1
        vectors[rows] = capture[0][torch.arange(len(rows)), terminal].cpu().numpy()
        LOGGER.info("MatterGPT extraction %d/%d", min(start + batch_size, len(seqs)), len(seqs))
    return vectors


def main() -> None:
    """Audita tokens/checkpoints y genera cachés originales no entrenadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size debe ser positivo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    frame = load_dataset()
    data_hash = sha256(DATASET)
    started = time.monotonic()
    sys.path.insert(0, str(MATGPT_SOURCE))
    mat_model, mat_stoi, mat_ctx, mat_audit = load_matgpt()
    mat_seqs = encode(frame, mat_stoi, mat_ctx, matter=False)
    mat_hidden = extract_matgpt(mat_model, mat_seqs, mat_stoi["<pad>"], args.batch_size)
    del mat_model
    sys.path.insert(0, str(MATTER_DIR))
    matter_model, matter_stoi, matter_ctx, matter_audit = load_mattergpt()
    matter_seqs = encode(frame, matter_stoi, matter_ctx, matter=True)
    matter_hidden = extract_mattergpt(matter_model, matter_seqs, args.batch_size)
    del matter_model
    if mat_hidden.shape != (210, 768) or matter_hidden.shape != (210, 768):
        raise ValueError(f"Dimensión embedding inesperada: {mat_hidden.shape}, {matter_hidden.shape}")
    if not np.isfinite(mat_hidden).all() or not np.isfinite(matter_hidden).all():
        raise FloatingPointError("Embedding contiene NaN/Inf")
    embedding_dir = EXPERIMENT / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, matrix, audit in (("matgpt_original", mat_hidden, mat_audit),
                                ("mattergpt_original", matter_hidden, matter_audit)):
        path = embedding_dir / f"{name}_final_end.npz"
        np.savez_compressed(path, structure_id=frame.structure_id.to_numpy(str), embeddings=matrix)
        outputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                         "shape": list(matrix.shape), "checkpoint_sha256": audit["checkpoint_sha256"],
                         "tokenizer_sha256": audit["tokenizer_sha256"], "condition": audit.get("property_vector")}
    manifest_rows = []
    for model_name, sequences, matrix in (("MatGPT_original", mat_seqs, mat_hidden),
                                          ("MatterGPT_original", matter_seqs, matter_hidden)):
        for index, row in frame.iterrows():
            manifest_rows.append({"model": model_name, "structure_id": row.structure_id,
                                  "sequence_tokens": len(sequences[index]), "embedding_dim": matrix.shape[1],
                                  "embedding_norm": float(np.linalg.norm(matrix[index])),
                                  "dataset_sha256": data_hash})
    pd.DataFrame(manifest_rows).to_csv(embedding_dir / "embedding_manifest.csv", index=False)
    metadata = {"status": "PASS", "dataset_path": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": data_hash, "structures": 210, "compositions": 70,
        "oov_tokens": 0, "truncated_sequences": 0,
        "matgpt": {**mat_audit, "sequence_token_min": min(map(len, mat_seqs)),
                   "sequence_token_max": max(map(len, mat_seqs)), **outputs["matgpt_original"]},
        "mattergpt": {**matter_audit, "sequence_token_min": min(map(len, matter_seqs)),
                       "sequence_token_max": max(map(len, matter_seqs)), **outputs["mattergpt_original"]},
        "extraction_seconds": time.monotonic() - started, "device": "cpu",
        "historical_test_accessed": False}
    (embedding_dir / "embedding_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("Embedding extraction complete: %s", json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
