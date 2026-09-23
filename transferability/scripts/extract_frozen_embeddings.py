#!/usr/bin/env python3
"""Extract frozen, mean-pooled embeddings from a selected SMILES column."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_INPUT = TRANSFERABILITY_ROOT / "data" / "SMILES" / "composition_smiles.csv"
DEFAULT_CONFIG = TRANSFERABILITY_ROOT / "config" / "models.json"
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--smiles-column", default="composition_smiles")
    parser.add_argument(
        "--representation-name",
        default="composition_smiles",
        help="Filesystem-safe label used in output filenames and metadata.",
    )
    parser.add_argument("--models", nargs="+", default=["chemberta", "molformer"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing files to be fetched instead of requiring the local cache.",
    )
    return parser.parse_args()


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolved_revision(alias: str, spec: dict, cache_dir: Path) -> str:
    manifest_path = cache_dir.parent / "download_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        downloaded = manifest.get("models", {}).get(alias, {})
        if downloaded.get("resolved_revision"):
            return downloaded["resolved_revision"]
    return spec.get("revision", "main")


def mean_pool(last_hidden_state, attention_mask, special_tokens_mask, torch):
    keep = attention_mask.bool() & ~special_tokens_mask.bool()
    # Defensive fallback for an empty sequence after removing special tokens.
    empty = keep.sum(dim=1) == 0
    if empty.any():
        keep[empty] = attention_mask[empty].bool()
    weights = keep.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def main() -> None:
    args = parse_args()
    # Remote-code modules are copied to a separate HF cache. Keep that cache
    # project-local too, both for portability and for restricted environments.
    hf_root = args.cache_dir.resolve().parent
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_root / "modules"))

    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Install numpy, torch, and transformers "
            "from requirements_tesis.txt."
        ) from exc

    registry = load_registry(args.config)
    unknown = sorted(set(args.models) - set(registry))
    if unknown:
        raise SystemExit(f"Unknown model aliases: {', '.join(unknown)}")

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    if args.smiles_column not in rows[0]:
        raise SystemExit(
            f"Column {args.smiles_column!r} is absent from {args.input}; "
            f"available columns: {', '.join(rows[0])}"
        )
    if not args.representation_name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--representation-name may contain only letters, numbers, '_' and '-'")

    # Embed each unique string once, then map it back to every crystal row.
    unique_smiles = list(dict.fromkeys(row[args.smiles_column] for row in rows))
    smile_to_index = {smiles: index for index, smiles in enumerate(unique_smiles)}
    row_indices = np.asarray([smile_to_index[row[args.smiles_column]] for row in rows])
    record_ids = np.asarray([row["record_id"] for row in rows])
    device = choose_device(torch, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; records: {len(rows)}; unique inputs: {len(unique_smiles)}")

    for alias in args.models:
        spec = registry[alias]
        revision = resolved_revision(alias, spec, args.cache_dir)
        common = {
            "pretrained_model_name_or_path": spec["repo_id"],
            "revision": revision,
            "cache_dir": str(args.cache_dir),
            "local_files_only": not args.allow_download,
            "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        }
        print(f"Loading {alias}: {spec['repo_id']}@{revision}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(**common)
        model_kwargs = dict(common)
        if alias == "molformer":
            model_kwargs["deterministic_eval"] = True
        model = AutoModel.from_pretrained(**model_kwargs).to(device)
        model.eval()

        configured_limits = [
            value
            for value in (
                getattr(tokenizer, "model_max_length", None),
                getattr(model.config, "max_position_embeddings", None),
            )
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        max_length = min(configured_limits) if configured_limits else 512

        pooled_batches = []
        token_counts = []
        untruncated_token_counts = []
        unknown_counts = []
        for start in range(0, len(unique_smiles), args.batch_size):
            batch_smiles = unique_smiles[start : start + args.batch_size]
            raw_encoded = tokenizer(batch_smiles, padding=False, truncation=False)
            untruncated_token_counts.extend(len(ids) for ids in raw_encoded["input_ids"])
            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            token_counts.extend(encoded["attention_mask"].sum(dim=1).tolist())
            if tokenizer.unk_token_id is None:
                unknown_counts.extend([0] * len(batch_smiles))
            else:
                unknown_counts.extend(
                    ((encoded["input_ids"] == tokenizer.unk_token_id) & encoded["attention_mask"].bool())
                    .sum(dim=1)
                    .tolist()
                )
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**model_inputs)
                pooled = mean_pool(
                    outputs.last_hidden_state,
                    model_inputs["attention_mask"],
                    special_tokens_mask.to(device),
                    torch,
                )
            pooled_batches.append(pooled.float().cpu().numpy())

        unique_embeddings = np.concatenate(pooled_batches, axis=0)
        embeddings = unique_embeddings[row_indices]
        output_path = args.output_dir / f"{alias}_{args.representation_name}.npz"
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            record_ids=record_ids,
            input_strings=np.asarray([row[args.smiles_column] for row in rows]),
            smiles_column=np.asarray(args.smiles_column),
            representation_name=np.asarray(args.representation_name),
            model_alias=np.asarray(alias),
            repo_id=np.asarray(spec["repo_id"]),
            revision=np.asarray(revision),
            pooling=np.asarray("last_hidden_state_mean_without_special_tokens"),
        )
        audit = {
            "model_alias": alias,
            "repo_id": spec["repo_id"],
            "revision": revision,
            "input_table": str(args.input.resolve()),
            "smiles_column": args.smiles_column,
            "representation_name": args.representation_name,
            "pooling": "last_hidden_state_mean_without_special_tokens",
            "embedding_shape": list(embeddings.shape),
            "records": len(rows),
            "unique_inputs": len(unique_smiles),
            "token_count_min": min(token_counts),
            "token_count_max": max(token_counts),
            "untruncated_token_count_min": min(untruncated_token_counts),
            "untruncated_token_count_max": max(untruncated_token_counts),
            "model_max_length_used": max_length,
            "unique_inputs_truncated": int(
                sum(count > max_length for count in untruncated_token_counts)
            ),
            "unknown_token_total": int(sum(unknown_counts)),
            "unique_inputs_with_unknown_tokens": int(sum(count > 0 for count in unknown_counts)),
            "device": device,
        }
        with output_path.with_suffix(".audit.json").open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(audit, indent=2), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
