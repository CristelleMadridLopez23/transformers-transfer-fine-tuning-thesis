"""Independent full-BERT regression utilities matching the paper's supervised task flow.

Only token IDs and length-based masks enter BERT. Original DFT-GGA frame energies
are targets in eV/atom; they never become features. Checkpoint loading is strict.
"""
from __future__ import annotations

import json
import importlib.metadata
import math
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification

from workflow import read_json, require_audit


class FrameDataset(Dataset):
    """Dataset of approved frame dictionaries; no information is inferred from test labels."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store approved row dicts; reject empty datasets or candidate-only records; return None."""
        if not rows or any(row.get("candidate_only", True) for row in rows):
            raise ValueError("Dataset must contain approved, non-candidate records")
        if any(not math.isfinite(row["energy_per_atom_ev"]) for row in rows):
            raise ValueError("Nonfinite original energy target")
        self.rows = rows

    def __len__(self) -> int:
        """Return integer frame count."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return approved row at integer index; inputs remain separate from energy labels."""
        return self.rows[index]


def collate(rows: list[dict[str, Any]], pad_id: int, maximum: int = 512) -> dict[str, torch.Tensor]:
    """Pad row ID lists by length using explicit pad_id; return IDs, mask and eV/atom labels."""
    if not rows or any(not r["input_ids"] for r in rows):
        raise ValueError("Empty input batch/sequence")
    length = max(len(r["input_ids"]) for r in rows)
    if length > maximum:
        raise ValueError("Truncation is not permitted")
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[index, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        mask[index, :n] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "labels": torch.tensor([r["energy_per_atom_ev"] for r in rows], dtype=torch.float32)}


def load_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load approved JSONL after current critical audit; return row dicts with exact IDs."""
    require_audit(config)
    path = config["paths"]["processed"] / "tokenized.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(r.get("candidate_only", True) for r in rows):
        raise ValueError("Missing approved tokenized data")
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["reduced_formula"], set()).add(row["split"])
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError("Composition leakage across original partitions")
    return rows


def partition(rows: list[dict[str, Any]], fold: int, seed: int,
              folds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return training/validation rows: original fixed split or train-only grouped CV fold."""
    training = [r for r in rows if r["split"] == "train"]
    if fold == -1:
        validation = [r for r in rows if r["split"] == "val"]
    else:
        if not 0 <= fold < folds or folds < 2:
            raise ValueError("Invalid grouped CV fold")
        groups = np.array(sorted({r["reduced_formula"] for r in training}))
        if len(groups) < folds:
            raise ValueError("Not enough training compositions for grouped CV")
        np.random.default_rng(seed).shuffle(groups)
        valid_groups = set(groups[fold::folds])
        validation = [r for r in training if r["reduced_formula"] in valid_groups]
        training = [r for r in training if r["reduced_formula"] not in valid_groups]
    if not training or not validation:
        raise ValueError("Empty training/validation partition")
    if {r["reduced_formula"] for r in training} & {r["reduced_formula"] for r in validation}:
        raise ValueError("Composition leakage in fitting split")
    if any(r["split"] == "test" for r in training + validation):
        raise ValueError("Test rows cannot participate in training/model selection")
    return training, validation


def model(config: dict[str, Any]) -> BertForSequenceClassification:
    """Strictly load original encoder into CLS-pooled BERT regressor; return all-trainable model."""
    checkpoint = config["paths"]["checkpoint"]
    reference = read_json(config["paths"]["reports"] / "audit/reference_tokenizer.json")
    for name, version in reference["versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Audited runtime changed: {name}")
    settings = BertConfig.from_json_file(str(checkpoint / "config.json"))
    settings.num_labels = 1
    settings.problem_type = "regression"
    network = BertForSequenceClassification(settings)
    network.bert.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    network.requires_grad_(True)
    if not all(p.requires_grad for p in network.parameters()):
        raise RuntimeError("Paper protocol requires full encoder fine-tuning")
    if network.bert.embeddings.word_embeddings.weight.shape[0] != settings.vocab_size:
        raise RuntimeError("Original embedding matrix was resized")
    return network


def pad_id(config: dict[str, Any]) -> int:
    """Return historical tokenizer padding ID from the verified runtime probe."""
    return read_json(config["paths"]["reports"] / "audit/reference_tokenizer.json")["special_ids"]["pad_token"]


def main() -> None:
    """Validate configuration and require scientific gate; return None without starting training."""
    from workflow import cli, load_config
    require_audit(load_config(cli(__doc__).parse_args().config))


if __name__ == "__main__":
    main()
