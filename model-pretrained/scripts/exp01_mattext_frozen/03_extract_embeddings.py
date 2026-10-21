"""Extract frozen MatText mean/CLS embeddings only after successful critical audits.

Reads approved tokenized JSONL/checkpoint, writes one NPZ plus aligned metadata
and hashes. CPU smoke uses at most two compositions per development fold.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import BertConfig, BertModel

from utils import (LOGGER, audit_fingerprint, cli, load_config, read_json,
                   require_gate, write_json)


def load_frozen(checkpoint: Any) -> BertModel:
    """Load checkpoint Path strictly into BertModel; return eval encoder with zero gradients."""
    model = BertModel(BertConfig.from_json_file(str(checkpoint / "config.json")))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    model.requires_grad_(False)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 0:
        raise RuntimeError("Encoder has trainable parameters")
    return model


def pool_hidden(hidden: torch.Tensor, attention: torch.Tensor,
                structural: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool B×L×D hidden states using B×L masks; return structural mean and CLS tensors."""
    if hidden.ndim != 3 or hidden.shape[:2] != attention.shape or structural.shape != attention.shape:
        raise ValueError("Invalid hidden/mask shapes")
    mask = (attention.bool() & structural.bool()).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("No structural tokens remain for mean pooling")
    mean = (hidden * mask).sum(dim=1) / counts
    return mean, hidden[:, 0, :]


def encode_batch(model: BertModel, sequences: list[list[int]], pad_id: int,
                 device: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode ID sequences with explicit length masks; return float32 mean/CLS arrays.

    Padding uses the tokenizer's pad ID. It is not inferred from config.pad_token_id
    because the latter may retain the original BERT default despite a custom vocab.
    """
    maximum = max(map(len, sequences))
    if maximum > model.config.max_position_embeddings:
        raise ValueError("Truncation is forbidden")
    ids = torch.full((len(sequences), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    structural = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        if len(sequence) < 3:
            raise ValueError("Expected CLS + structural tokens + SEP")
        ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[index, :len(sequence)] = 1
        structural[index, 1:len(sequence) - 1] = 1
    with torch.inference_mode():
        hidden = model(input_ids=ids, attention_mask=attention).last_hidden_state
        mean, cls = pool_hidden(hidden, attention, structural)
    return mean.cpu().numpy().astype("float32"), cls.cpu().numpy().astype("float32")


def extract(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Extract config's approved inputs; return saved manifest, rejecting stale caches."""
    require_gate(config, "tokenizer_audit")
    if read_json(config["paths"]["reports"] / "audit" / "dataset_audit.json")["status"] != "passed":
        raise RuntimeError("Energy/structure audit failed")
    if not smoke:
        require_gate(config)
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Full extraction must run in a Slurm allocation, never on login")
    paths = config["paths"]
    frame = pd.read_csv(paths["processed"] / "dataset.csv")
    splits = pd.read_csv(paths["processed"] / "split_assignments.csv")
    records = [json.loads(line) for line in (paths["processed"] / "tokenized.jsonl").read_text().splitlines()]
    ids = [row["structure_id"] for row in records]
    if ids != frame.structure_id.tolist() or any(row.get("candidate_only", True) for row in records):
        raise ValueError("Unapproved inputs or input ID order mismatch")
    if smoke:
        development = set(splits.loc[splits.split == "development", "structure_id"])
        selected = frame.loc[frame.structure_id.isin(development)].merge(
            splits[["structure_id", "fold"]], on="structure_id", validate="one_to_one")
        selected = selected.drop_duplicates("reduced_formula").groupby("fold", group_keys=False).head(2)
        selected_ids = set(selected.structure_id)
        records = [row for row in records if row["structure_id"] in selected_ids]
        frame = frame.set_index("structure_id").loc[[row["structure_id"] for row in records]].reset_index()
    fingerprint = audit_fingerprint(config)
    output = paths["reports"] / "audit" / "embedding_smoke" if smoke else paths["embeddings"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "embedding_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError("Embeddings exist for other inputs/config; use a new experiment ID")
        from utils import sha256
        if manifest.get("npz_sha256") != sha256(output / "embeddings.npz"):
            raise RuntimeError("Cached embedding integrity failure")
        LOGGER.info("reusing_verified_embeddings directory=%s", output)
        return manifest
    torch.set_num_threads(config["extraction"]["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    device = "cpu" if smoke else config["extraction"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    model = load_frozen(paths["checkpoint"]).to(device)
    vocab = read_json(config["root"] / config["tokenizer"]["vocab"])
    started = time.monotonic()
    batch_size = config["extraction"]["batch_size"]
    means, clses = [], []
    for offset in range(0, len(records), batch_size):
        sequences = [row["input_ids"] for row in records[offset:offset + batch_size]]
        mean, cls = encode_batch(model, sequences, vocab["[PAD]"], device)
        if offset == 0:
            repeated = encode_batch(model, sequences, vocab["[PAD]"], device)
            if not all(np.allclose(first, second, rtol=0, atol=config["extraction"]["determinism_atol"])
                       for first, second in zip((mean, cls), repeated)):
                raise RuntimeError("Frozen extraction is not deterministic")
        means.append(mean)
        clses.append(cls)
    arrays = {"mean_pooling": np.concatenate(means), "cls_pooling": np.concatenate(clses)}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Nonfinite embeddings")
    np.savez_compressed(output / "embeddings.npz", **arrays)
    frame[["structure_id", "reduced_formula", "polymorph_family"]].to_csv(output / "metadata.csv", index=False)
    frame[["structure_id", "energy_per_atom_ev"]].to_csv(output / "reference_energies.csv", index=False)
    from utils import sha256
    manifest = {"fingerprint": fingerprint, "npz_sha256": sha256(output / "embeddings.npz"),
                "metadata_sha256": sha256(output / "metadata.csv"),
                "reference_sha256": sha256(output / "reference_energies.csv"),
                "shape": list(arrays["mean_pooling"].shape), "poolings": list(arrays),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "encoder_trainable_parameters": 0, "dropout_disabled": not model.training,
                "determinism_passed": True, "device": device,
                "seconds": time.monotonic() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                "mean_policy": "exclude padding plus first CLS and final SEP; no structural tokens excluded by numeric ID"}
    write_json(manifest_path, manifest)
    LOGGER.info("embeddings_saved shape=%s encoder_trainable=0", manifest["shape"])
    return manifest


def main() -> None:
    """Run approved extraction; --smoke uses up to two compositions per development fold."""
    parser = cli(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    extract(load_config(args.config), args.smoke)


if __name__ == "__main__":
    main()
