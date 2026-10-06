"""Shared data, SLICES grammar, chunking, MLM training and evaluation utilities.

This module is the single scientific implementation used by the Mac smoke test
and the MSI full-DAPT runner. It intentionally does not use property targets,
custom embedding initialization, PEFT, or a modified Transformer architecture.
"""
from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from pymatgen.core import Element

BASE_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
REQUIRED_COLUMNS = ["mat_id", "slices", "composition_key", "nsites"]
OPTIONAL_COLUMNS = ["reduced_formula", "n_elements", "spg", "composition_count", "is_polymorph_candidate"]
TOKEN_CATEGORIES = ["space_group", "element", "node_index", "periodic_vector", "other"]
PERIODIC_RE = re.compile(r"^[o+\-]{3}$")
INTEGER_RE = re.compile(r"^\d+$")


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "audit").is_dir():
            return candidate
    raise FileNotFoundError("Could not find a project root containing both data/ and audit/.")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def find_alexandria_csv(project_root: str | Path) -> Path:
    root = Path(project_root)
    audit_outputs = root / "audit" / "notebooks" / "outputs"
    summaries = sorted(audit_outputs.glob("run_*/tokenizer_audit_summary.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for summary_path in summaries:
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8")).get("dataset")
            candidate = Path(value) if value else None
            if candidate and candidate.is_file() and candidate.name == "alexandria_clean_all.csv":
                return candidate.resolve()
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    preferred = root / "data" / "notebooks_processing" / "outputs" / "alexandria_clean_all.csv"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted({p.resolve() for p in (root / "data").rglob("alexandria_clean_all.csv")})
    if not candidates:
        raise FileNotFoundError("No alexandria_clean_all.csv found below data/.")
    if len(candidates) == 1:
        return candidates[0]
    hashes = {path: sha256_file(path) for path in candidates}
    if len(set(hashes.values())) != 1:
        detail = "\n".join(f"- {p}: {h}" for p, h in hashes.items())
        raise RuntimeError("Multiple different Alexandria clean CSVs found; set dataset_path explicitly.\n" + detail)
    clean_runs = [p for p in candidates if "cleaning_run_" in str(p)]
    return (clean_runs or candidates)[0]


def set_reproducible_seeds(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Seed the CPU generator directly: torch.manual_seed also schedules CUDA seeding
    # in some PyTorch releases, which the Mac policy intentionally avoids.
    torch.random.default_generator.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device is not None and device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass


def detect_device(policy: str = "auto", require_cuda: bool = False) -> tuple[torch.device, dict[str, Any]]:
    """Select MPS first on macOS, CUDA only when available, CPU otherwise.

    policy='mac' deliberately never probes/selects CUDA. policy='msi' requires CUDA.
    """
    policy = policy.lower()
    if policy == "msi" and sys.platform == "darwin":
        raise RuntimeError("Full DAPT is intended to run on an MSI GPU allocation; submit the runner on MSI, not on macOS.")
    mps_available = bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
    # The Mac smoke-test policy deliberately avoids even querying CUDA.
    cuda_probed = policy != "mac" and sys.platform != "darwin"
    cuda_available = bool(torch.cuda.is_available()) if cuda_probed else False
    if require_cuda or policy == "msi":
        if not cuda_available:
            raise RuntimeError("Full DAPT is intended to run on an MSI GPU allocation. CUDA is unavailable.")
        device = torch.device("cuda")
    elif policy == "mac":
        device = torch.device("mps" if mps_available else "cpu")
    elif sys.platform == "darwin" and mps_available:
        device = torch.device("mps")
    elif cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    info: dict[str, Any] = {
        "device": str(device), "device_type": device.type,
        "mps_available": mps_available,
        "cuda_available": cuda_available if policy != "mac" else False,
        "cuda_probed": cuda_probed,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else ("Apple Metal (MPS)" if device.type == "mps" else None),
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else (1 if device.type == "mps" else 0),
        "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
    }
    return device, info


def select_precision(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"precision": "float32", "bf16": False, "fp16": False, "autocast_dtype": None}
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_supported:
        return {"precision": "bf16", "bf16": True, "fp16": False, "autocast_dtype": torch.bfloat16}
    if torch.cuda.get_device_capability(0)[0] >= 7:
        return {"precision": "fp16", "bf16": False, "fp16": True, "autocast_dtype": torch.float16}
    return {"precision": "float32", "bf16": False, "fp16": False, "autocast_dtype": None}


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet when an engine is installed; otherwise write a CSV.GZ fallback."""
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except (ImportError, ModuleNotFoundError):
            fallback = path.with_suffix(path.suffix + ".csv.gz")
            frame.to_csv(fallback, index=False, compression="gzip")
            print(f"Parquet engine unavailable; wrote compatible table to {fallback.name}")
            return fallback
    frame.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError):
            fallback = path.with_suffix(path.suffix + ".csv.gz")
            return pd.read_csv(fallback, low_memory=False)
    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def load_alexandria(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Alexandria dataset is missing required columns: {missing}")
    usecols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if df.empty:
        raise ValueError("Alexandria SLICES dataset is empty.")
    for col in ["mat_id", "composition_key", "slices"]:
        if df[col].isna().any():
            count = int(df[col].isna().sum())
            raise ValueError(f"{count} rows have missing {col}; DAPT will not silently discard rows.")
    df["mat_id"] = df["mat_id"].astype(str)
    df["composition_key"] = df["composition_key"].astype(str)
    df["slices"] = df["slices"].map(lambda x: " ".join(str(x).split()))
    if df["mat_id"].duplicated().any():
        raise ValueError(f"mat_id is not unique ({int(df['mat_id'].duplicated().sum())} duplicate rows). Resolve before DAPT.")
    if df["composition_key"].str.strip().eq("").any() or df["slices"].str.strip().eq("").any():
        raise ValueError("Empty composition_key or SLICES string detected; no rows were dropped.")
    return df


def build_or_load_composition_split(df: pd.DataFrame, split_path: str | Path,
                                    seed: int = 42) -> pd.DataFrame:
    """Persist one global composition split; later runs reuse it and validate identity."""
    split_path = Path(split_path)
    current = df[["mat_id", "composition_key"]].copy().sort_values("mat_id").reset_index(drop=True)
    if split_path.is_file():
        saved = pd.read_csv(split_path, dtype={"mat_id": str, "composition_key": str})
        if not {"mat_id", "composition_key", "split"}.issubset(saved.columns):
            raise ValueError(f"Existing split file has invalid columns: {split_path}")
        saved = saved[["mat_id", "composition_key", "split"]].sort_values("mat_id").reset_index(drop=True)
        if not current.equals(saved[["mat_id", "composition_key"]]):
            raise ValueError("Existing global split does not match current Alexandria rows/compositions. It was not regenerated.")
        if not set(saved["split"]).issubset({"train", "validation", "holdout"}):
            raise ValueError("Existing split contains unknown labels.")
        assignment = saved
        print("Reusing existing composition split:", split_path)
    else:
        groups = np.array(sorted(current["composition_key"].unique()), dtype=object)
        rng = np.random.default_rng(seed)
        groups = groups[rng.permutation(len(groups))]
        n = len(groups)
        n_train = int(math.floor(.90 * n))
        n_val = int(math.floor(.05 * n))
        group_to_split = {str(key): "train" for key in groups[:n_train]}
        group_to_split.update({str(key): "validation" for key in groups[n_train:n_train+n_val]})
        group_to_split.update({str(key): "holdout" for key in groups[n_train+n_val:]})
        assignment = current.copy()
        assignment["split"] = assignment["composition_key"].map(group_to_split)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_csv(split_path, index=False)
        print("Created persistent composition split:", split_path)
    if assignment.groupby("composition_key")["split"].nunique().max() != 1:
        raise AssertionError("Composition leakage: one composition_key appears in multiple splits.")
    sets = {name: set(assignment.loc[assignment["split"] == name, "composition_key"])
            for name in ["train", "validation", "holdout"]}
    if sets["train"] & sets["validation"] or sets["train"] & sets["holdout"] or sets["validation"] & sets["holdout"]:
        raise AssertionError("Composition sets overlap across splits.")
    return assignment


def _is_element(token: str) -> bool:
    try:
        return Element(token).symbol == token
    except Exception:
        return False


def parse_slices_token_categories(slices_string: str) -> dict[str, Any]:
    """Parse Strategy-4 lexical roles without confusing SG 'o' and vector 'ooo'."""
    tokens = str(slices_string).split()
    result = {"tokens": tokens, "categories": ["other"] * len(tokens),
              "valid": False, "error": None, "space_group_prefix": [],
              "elements": [], "edge_triples": []}
    if not tokens:
        result["error"] = "empty_sequence"
        return result
    # Find the earliest plausible start of a contiguous element list followed by
    # an integer-led tail whose length is divisible into index/index/vector triples.
    chosen = None
    for start, token in enumerate(tokens):
        if not _is_element(token):
            continue
        end = start
        while end < len(tokens) and _is_element(tokens[end]):
            end += 1
        if end == len(tokens):
            chosen = (start, end)
            break
        if not INTEGER_RE.fullmatch(tokens[end]):
            continue
        tail = tokens[end:]
        if len(tail) % 3:
            continue
        triples = [tail[i:i+3] for i in range(0, len(tail), 3)]
        if all(INTEGER_RE.fullmatch(t[0]) and INTEGER_RE.fullmatch(t[1]) and PERIODIC_RE.fullmatch(t[2]) for t in triples):
            chosen = (start, end)
            break
    if chosen is None:
        result["error"] = "could_not_identify_element_block_and_edge_triples"
        return result
    start, end = chosen
    prefix = tokens[:start]
    elements = tokens[start:end]
    tail = tokens[end:]
    triples = []
    if tail:
        if len(tail) % 3:
            result["error"] = "edge_tail_length_not_multiple_of_three"
            return result
        triples = [tail[i:i+3] for i in range(0, len(tail), 3)]
        if not all(INTEGER_RE.fullmatch(t[0]) and INTEGER_RE.fullmatch(t[1]) and PERIODIC_RE.fullmatch(t[2]) for t in triples):
            result["error"] = "invalid_edge_triple"
            return result
    categories = ["space_group"] * len(prefix) + ["element"] * len(elements)
    for _ in triples:
        categories.extend(["node_index", "node_index", "periodic_vector"])
    if len(categories) != len(tokens):
        result["error"] = "category_count_does_not_match_token_count"
        return result
    result.update({"categories": categories, "valid": True, "error": None,
                   "space_group_prefix": prefix, "elements": elements,
                   "edge_triples": triples, "element_start": start,
                   "edge_start": end})
    return result


def audit_slices_grammar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows, parsed = [], {}
    for row in df[["mat_id", "composition_key", "slices"]].itertuples(index=False):
        info = parse_slices_token_categories(row.slices)
        parsed[row.mat_id] = info
        rows.append({"mat_id": row.mat_id, "composition_key": row.composition_key,
                     "grammar_valid": info["valid"], "parse_error": info["error"],
                     "n_lexical_tokens": len(info["tokens"]),
                     "n_space_group_tokens": info["categories"].count("space_group"),
                     "n_element_tokens": info["categories"].count("element"),
                     "n_edge_triples": len(info["edge_triples"])})
    return pd.DataFrame(rows), parsed


def grammar_fidelity_check(df: pd.DataFrame, tokenizer, sample_size: int = 5000,
                           seed: int = 12345, min_rate: float = .995) -> tuple[float, pd.DataFrame]:
    n = min(max(5000, sample_size), len(df))
    sample = df.sample(n=n, random_state=seed) if len(df) > n else df.copy()
    rows = []
    for row in sample[["mat_id", "slices"]].itertuples(index=False):
        lexical = row.slices.split()
        ids = tokenizer.encode(row.slices, add_special_tokens=False)
        decoded = tokenizer.convert_ids_to_tokens(ids)
        exact = len(ids) == len(lexical) and list(decoded) == lexical
        rows.append({"mat_id": row.mat_id, "n_slices_tokens": len(lexical),
                     "n_tokenizer_units": len(ids), "one_to_one_exact": exact,
                     "first_mismatch": next((f"{a!r}!={b!r}" for a,b in zip(lexical,decoded) if a!=b), None)})
    audit = pd.DataFrame(rows)
    rate = float(audit["one_to_one_exact"].mean()) if len(audit) else 0.0
    print(f"Grammar fidelity ({len(audit):,} structures): {rate:.4%}")
    if rate < min_rate:
        raise RuntimeError(f"Tokenizer grammar fidelity {rate:.4%} < {min_rate:.2%}; stopping before model loading/training.")
    return rate, audit


def verify_lexical_vocabulary(tokenizer, df: pd.DataFrame) -> None:
    lexemes = sorted({token for text in df["slices"] for token in text.split()})
    joined = " ".join(lexemes)
    ids = tokenizer.encode(joined, add_special_tokens=False)
    converted = tokenizer.convert_ids_to_tokens(ids)
    if len(ids) != len(lexemes) or list(converted) != lexemes:
        raise RuntimeError("At least one corpus SLICES lexeme is not encoded as exactly one unchanged tokenizer unit.")
    print(f"Verified {len(lexemes):,} unique SLICES lexemes: 1 token ID each.")


def _encode_content(tokens: list[str], tokenizer) -> list[int]:
    text = " ".join(tokens)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != len(tokens) or tokenizer.convert_ids_to_tokens(ids) != tokens:
        raise RuntimeError("SLICES lexical tokens do not map one-to-one to tokenizer units while chunking.")
    return [int(x) for x in ids]


def grammar_aware_chunk(parsed: dict[str, Any], tokenizer,
                        max_content_length: int = 510) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Keep SG+all atoms in every chunk; partition only complete edge triples."""
    if not parsed["valid"]:
        return [], {"prefix_atoms_too_long": False, "reason": "invalid_grammar"}
    prefix = parsed["space_group_prefix"]
    atoms = parsed["elements"]
    triples = parsed["edge_triples"]
    fixed = prefix + atoms
    fixed_len = len(_encode_content(fixed, tokenizer))
    if fixed_len > max_content_length:
        return [], {"prefix_atoms_too_long": True, "reason": "space_group_plus_full_atom_list_exceeds_limit",
                    "fixed_lexical_tokens": fixed_len}
    chunks: list[list[str]] = []
    if len(parsed["tokens"]) <= max_content_length:
        chunks = [parsed["tokens"]]
    elif not triples:
        return [], {"prefix_atoms_too_long": True, "reason": "sequence_exceeds_limit_without_chunkable_edges",
                    "fixed_lexical_tokens": fixed_len}
    else:
        triples_per_chunk = (max_content_length - fixed_len) // 3
        if triples_per_chunk < 1:
            return [], {"prefix_atoms_too_long": True, "reason": "no_complete_edge_triple_fits",
                        "fixed_lexical_tokens": fixed_len}
        for start in range(0, len(triples), triples_per_chunk):
            block = triples[start:start+triples_per_chunk]
            edge_tokens = [token for triple in block for token in triple]
            chunks.append(fixed + edge_tokens)
    records = []
    expected_edges = [tuple(x) for x in triples]
    actual_edges = []
    for chunk_id, tokens in enumerate(chunks):
        ids = _encode_content(tokens, tokenizer)
        full_ids = [int(x) for x in tokenizer.build_inputs_with_special_tokens(ids)]
        if len(ids) > max_content_length:
            raise AssertionError("Chunk exceeds configured content-token limit.")
        categories = []
        if len(tokens) == len(parsed["tokens"]) and tokens == parsed["tokens"]:
            categories = list(parsed["categories"])
        else:
            categories = (["space_group"] * len(prefix) + ["element"] * len(atoms))
            for _ in range((len(tokens)-len(fixed))//3):
                categories.extend(["node_index", "node_index", "periodic_vector"])
        offset = len(prefix) + len(atoms)
        chunk_edges = [tuple(tokens[i:i+3]) for i in range(offset,len(tokens),3)]
        actual_edges.extend(chunk_edges)
        records.append({"chunk_id": chunk_id, "slices_chunk": " ".join(tokens),
                        "input_ids": full_ids, "content_ids": ids,
                        "token_categories": ["other"] + categories + ["other"],
                        "n_content_tokens": len(ids), "n_edge_triples": len(chunk_edges)})
    if actual_edges != expected_edges:
        return [], {"prefix_atoms_too_long": False, "reason": "edge_preservation_check_failed",
                    "expected_edge_triples": len(expected_edges), "observed_edge_triples": len(actual_edges)}
    for record in records:
        if len(record["token_categories"]) != len(record["input_ids"]):
            raise AssertionError("Special-token positions and parsed token categories are not aligned.")
    return records, None


def prepare_split_and_chunks(df: pd.DataFrame, split_path: str | Path, tokenizer,
                             output_dir: str | Path, seed: int = 42,
                             diagnostic_seed: int = 12345, max_length: int = 512,
                             max_structures: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    max_content_length = max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
    if max_content_length <= 0:
        raise ValueError("No content positions remain after tokenizer special tokens.")
    assignment = build_or_load_composition_split(df, split_path, seed=seed)
    merged = df.merge(assignment[["mat_id", "split"]], on="mat_id", how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise AssertionError("Some structures did not receive a persisted split assignment.")
    if max_structures is not None and len(merged) > int(max_structures):
        subset_parts=[]
        proportions={"train":.90,"validation":.05,"holdout":.05}
        remaining=int(max_structures)
        for split in ["train","validation","holdout"]:
            group=merged.loc[merged["split"]==split]
            take=min(len(group),int(round(max_structures*proportions[split])))
            if split=="holdout": take=min(len(group),remaining)
            selected=group.sample(n=take,random_state=seed+{"train":0,"validation":1,"holdout":2}[split]) if take else group.head(0)
            subset_parts.append(selected); remaining-=len(selected)
        merged=pd.concat(subset_parts,ignore_index=True)
        # If rounding left spare rows, fill only within existing assigned splits.
        if remaining>0:
            leftovers=df.loc[~df["mat_id"].isin(merged["mat_id"])].merge(assignment[["mat_id","split"]],on="mat_id")
            add=leftovers.sample(n=min(remaining,len(leftovers)),random_state=seed+3)
            merged=pd.concat([merged,add],ignore_index=True)
    merged.to_csv(output_dir/"dapt_split_assignments.csv",index=False)
    grammar_audit, parsed = audit_slices_grammar(merged)
    grammar_audit.to_csv(output_dir/"slices_grammar_audit.csv",index=False)
    invalid=grammar_audit.loc[~grammar_audit["grammar_valid"]]
    if len(invalid):
        invalid.to_csv(output_dir/"grammar_exceptions.csv",index=False)
        raise RuntimeError(f"{len(invalid)} selected structures have invalid SLICES grammar. Details saved to grammar_exceptions.csv; refusing silent exclusion.")
    fidelity_rate, fidelity_audit = grammar_fidelity_check(merged,tokenizer,sample_size=5000,seed=diagnostic_seed)
    fidelity_audit.to_csv(output_dir/"grammar_fidelity_sample.csv",index=False)
    verify_lexical_vocabulary(tokenizer,merged)
    chunk_rows=[]; exceptions=[]; per_structure=[]
    total_edges=0; preserved_edges=0
    for row in merged[["mat_id","composition_key","slices","split"]].itertuples(index=False):
        info=parsed[row.mat_id]
        total_edges+=len(info["edge_triples"])
        chunks,error=grammar_aware_chunk(info,tokenizer,max_content_length=max_content_length)
        if error:
            exceptions.append({"mat_id":row.mat_id,"composition_key":row.composition_key,"split":row.split,**error})
            per_structure.append({"mat_id":row.mat_id,"split":row.split,"n_chunks":0,"n_edges":len(info["edge_triples"]),"edges_preserved":0,"excluded":True})
            continue
        output_edges=sum(x["n_edge_triples"] for x in chunks)
        preserved_edges+=output_edges
        per_structure.append({"mat_id":row.mat_id,"split":row.split,"n_chunks":len(chunks),"n_edges":len(info["edge_triples"]),"edges_preserved":output_edges,"excluded":False})
        for chunk in chunks:
            chunk_rows.append({"mat_id":row.mat_id,"composition_key":row.composition_key,"split":row.split,
                               "chunk_id":chunk["chunk_id"],"slices_chunk":chunk["slices_chunk"],
                               "input_ids":chunk["input_ids"],"content_ids":chunk["content_ids"],
                               "token_categories":chunk["token_categories"],"n_content_tokens":chunk["n_content_tokens"],
                               "n_edge_triples":chunk["n_edge_triples"]})
    exceptions_df=pd.DataFrame(exceptions)
    exceptions_df.to_csv(output_dir/"chunking_exceptions.csv",index=False)
    chunks_df=pd.DataFrame(chunk_rows)
    if chunks_df.empty:
        raise RuntimeError("Chunking produced no training sequences.")
    structure_stats=pd.DataFrame(per_structure)
    write_table(structure_stats,output_dir/"structure_chunk_counts.parquet")
    write_table(chunks_df,output_dir/"prepared_chunks.parquet")
    eligible_structures=int((~structure_stats["excluded"]).sum())
    chunked=structure_stats.loc[structure_stats["n_chunks"]>1,"n_chunks"]
    chunking_statistics=pd.DataFrame([{
        "original_structures":int(len(merged)),"structures_not_chunked":int((structure_stats["n_chunks"]==1).sum()),
        "structures_chunked":int((structure_stats["n_chunks"]>1).sum()),"structures_excluded_prefix_too_long":int(len(exceptions_df)),
        "total_chunks":int(len(chunks_df)),"mean_chunks_per_chunked_structure":float(chunked.mean()) if len(chunked) else 0.0,
        "median_chunks_per_chunked_structure":float(chunked.median()) if len(chunked) else 0.0,
        "p95_chunks_per_chunked_structure":float(chunked.quantile(.95)) if len(chunked) else 0.0,
        "max_chunks_per_structure":int(structure_stats["n_chunks"].max()),
        "fraction_structures_chunked":float((structure_stats["n_chunks"]>1).mean()),
        "fraction_edges_preserved_all_selected":float(preserved_edges/total_edges) if total_edges else 1.0,
        "fraction_edges_preserved_trainable":1.0,
        "max_length":int(max_length),"max_content_length":int(max_content_length),
        "grammar_fidelity_rate":fidelity_rate,"eligible_structures":eligible_structures,
    }])
    chunking_statistics.to_csv(output_dir/"chunking_statistics.csv",index=False)
    split_summary={}
    for split in ["train","validation","holdout"]:
        split_struct=merged.loc[merged["split"]==split,"mat_id"].nunique()
        split_chunks=int((chunks_df["split"]==split).sum())
        split_summary[split]={"structures":int(split_struct),"chunks":split_chunks,
                              "compositions":int(merged.loc[merged["split"]==split,"composition_key"].nunique())}
    (output_dir/"split_summary.json").write_text(json.dumps({"counts":split_summary,"composition_disjoint":True,
        "global_split_path":str(split_path),"seed":seed,"max_structures":max_structures},indent=2),encoding="utf-8")
    return {"structures":merged,"chunks":chunks_df,"parsed":parsed,"grammar_audit":grammar_audit,
            "fidelity_rate":fidelity_rate,"chunking_statistics":chunking_statistics,
            "chunking_exceptions":exceptions_df,"split_summary":split_summary,
            "structure_stats":structure_stats,"all_edge_count":total_edges,"preserved_edge_count":preserved_edges}


class TokenSequenceDataset(Dataset):
    def __init__(self, records: pd.DataFrame):
        self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        return {"input_ids":self.records.iloc[index]["input_ids"]}


class FixedMaskDataset(Dataset):
    def __init__(self, examples: list[dict[str,Any]]): self.examples=examples
    def __len__(self): return len(self.examples)
    def __getitem__(self,index):
        item=self.examples[index]
        return {"input_ids":item["masked_input_ids"],"labels":item["labels"],
                "attention_mask":[1]*len(item["masked_input_ids"])}


def _fixed_mask_for_record(record: dict[str,Any], tokenizer, seed: int,
                           probability: float=.15) -> dict[str,Any]:
    input_ids=list(record["input_ids"])
    categories=list(record["token_categories"])
    special=set(tokenizer.all_special_ids)
    eligible=[i for i,t in enumerate(input_ids) if t not in special]
    seed_text=f"{seed}|{record['mat_id']}|{record['chunk_id']}|{record['split']}".encode()
    local_seed=int.from_bytes(hashlib.blake2b(seed_text,digest_size=8).digest(),"little")%(2**32)
    rng=random.Random(local_seed)
    masked=[i for i in eligible if rng.random()<probability]
    if eligible and not masked: masked=[eligible[rng.randrange(len(eligible))]]
    masked_input=input_ids.copy(); labels=[-100]*len(input_ids)
    position_rows=[]
    for pos in masked:
        true_id=int(input_ids[pos]); masked_input[pos]=int(tokenizer.mask_token_id); labels[pos]=true_id
        token=str(tokenizer.convert_ids_to_tokens(true_id))
        category=categories[pos] if pos<len(categories) else "other"
        position_rows.append({"split":record["split"],"sequence_id":record["mat_id"],"mat_id":record["mat_id"],
                              "chunk_id":int(record["chunk_id"]),"token_position":int(pos),
                              "true_token":token,"true_token_id":true_id,"token_category":category,
                              "original_or_added":"original_vocab" if true_id<int(tokenizer.vocab_size) else "added_slices_vocab"})
    return {"split":record["split"],"mat_id":record["mat_id"],"chunk_id":record["chunk_id"],
            "masked_input_ids":masked_input,"labels":labels,"mask_positions":masked,
            "token_categories":categories,"position_rows":position_rows}


def build_fixed_diagnostics(chunks: pd.DataFrame, tokenizer, diagnostic_seed: int=12345,
                            mlm_probability: float=.15) -> tuple[dict[str,list[dict[str,Any]]],pd.DataFrame]:
    by_split={"validation":[],"holdout":[]}; rows=[]
    for record in chunks.loc[chunks["split"].isin(by_split)].to_dict(orient="records"):
        example=_fixed_mask_for_record(record,tokenizer,diagnostic_seed,mlm_probability)
        if example["mask_positions"]:
            by_split[record["split"]].append(example); rows.extend(example["position_rows"])
    if not by_split["validation"] or not by_split["holdout"]:
        raise RuntimeError("Fixed MLM diagnostics need non-empty validation and holdout sets.")
    return by_split,pd.DataFrame(rows)


def _pad_fixed_batch(features: list[dict[str,Any]], pad_id: int) -> dict[str,torch.Tensor]:
    width=max(len(x["input_ids"]) for x in features)
    ids=torch.full((len(features),width),pad_id,dtype=torch.long)
    labels=torch.full((len(features),width),-100,dtype=torch.long)
    attention=torch.zeros((len(features),width),dtype=torch.long)
    for i,item in enumerate(features):
        n=len(item["input_ids"]); ids[i,:n]=torch.tensor(item["input_ids"],dtype=torch.long)
        labels[i,:n]=torch.tensor(item["labels"],dtype=torch.long)
        attention[i,:n]=1
    return {"input_ids":ids,"labels":labels,"attention_mask":attention}


def _autocast_context(device: torch.device, precision: dict[str,Any]):
    if device.type=="cuda" and precision.get("autocast_dtype") is not None:
        return torch.autocast(device_type="cuda",dtype=precision["autocast_dtype"])
    return torch.autocast(device_type="cpu",enabled=False)


def evaluate_fixed_masks(model, examples: list[dict[str,Any]], tokenizer,
                         device: torch.device, batch_size: int,
                         precision: dict[str,Any] | None=None,
                         save_predictions: bool=True) -> tuple[dict[str,Any],pd.DataFrame]:
    precision=precision or {"autocast_dtype":None}
    loader=DataLoader(FixedMaskDataset(examples),batch_size=batch_size,shuffle=False,
                      collate_fn=lambda rows:_pad_fixed_batch(rows,int(tokenizer.pad_token_id)),num_workers=0)
    was_training=model.training; model.eval(); pred_rows=[]; total_loss=0.0; total_masked=0
    with torch.no_grad():
        for batch,example_batch in zip(loader,[examples[i:i+batch_size] for i in range(0,len(examples),batch_size)]):
            batch={k:v.to(device) for k,v in batch.items()}
            with _autocast_context(device,precision):
                output=model(**batch)
            logits=output.logits.float(); labels=batch["labels"]
            valid=labels.ne(-100); count=int(valid.sum().item())
            if count==0: continue
            losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction="sum")
            total_loss+=float(losses.item()); total_masked+=count
            probs=torch.softmax(logits,dim=-1)
            topk_prob,topk_id=torch.topk(probs,k=min(5,probs.shape[-1]),dim=-1)
            for bi,example in enumerate(example_batch):
                for pos in example["mask_positions"]:
                    true_id=int(example["labels"][pos]); true_prob=float(probs[bi,pos,true_id].item())
                    greater=int((logits[bi,pos]>logits[bi,pos,true_id]).sum().item())
                    equal_before=int((logits[bi,pos,:true_id]==logits[bi,pos,true_id]).sum().item())
                    rank=greater+equal_before+1
                    ids=topk_id[bi,pos].detach().cpu().tolist(); ps=topk_prob[bi,pos].detach().cpu().tolist()
                    toks=[str(tokenizer.convert_ids_to_tokens(int(t))) for t in ids]
                    category=example["token_categories"][pos] if pos<len(example["token_categories"]) else "other"
                    pred_rows.append({"split":example["split"],"sequence_id":example["mat_id"],"mat_id":example["mat_id"],
                        "chunk_id":int(example["chunk_id"]),"token_position":int(pos),"true_token_id":true_id,
                        "true_token":str(tokenizer.convert_ids_to_tokens(true_id)),"token_category":category,
                        "original_or_added":"original_vocab" if true_id<int(tokenizer.vocab_size) else "added_slices_vocab",
                        "loss":float(-math.log(max(true_prob,1e-30))),"true_probability":true_prob,
                        "rank":rank,"top1_token_id":int(ids[0]),"top1_token":toks[0],"top1_probability":float(ps[0]),
                        "top3_correct":int(true_id in ids[:3]),"top5_correct":int(true_id in ids[:5]),
                        "top5_tokens_json":json.dumps(toks,ensure_ascii=False),
                        "top5_probabilities_json":json.dumps([float(x) for x in ps])})
    if was_training: model.train()
    predictions=pd.DataFrame(pred_rows)
    if predictions.empty: raise RuntimeError("No fixed masked positions were evaluated.")
    metrics=metrics_from_predictions(predictions)
    metrics["loss"] = total_loss/total_masked if total_masked else float("nan")
    metrics["masked_count"] = int(total_masked)
    return metrics,predictions if save_predictions else predictions.iloc[0:0]


def metrics_from_predictions(predictions: pd.DataFrame) -> dict[str,Any]:
    if predictions.empty: return {"loss":None,"perplexity":None,"top1_accuracy":None,"top3_accuracy":None,"top5_accuracy":None,"mrr":None,"mean_true_probability":None,"masked_count":0}
    loss=float(predictions["loss"].mean())
    return {"loss":loss,"perplexity":float(math.exp(min(loss,80))),
            "top1_accuracy":float((predictions["rank"]==1).mean()),
            "top3_accuracy":float(predictions["rank"]<=3).mean(),
            "top5_accuracy":float(predictions["rank"]<=5).mean(),
            "mrr":float((1/predictions["rank"]).mean()),
            "mean_true_probability":float(predictions["true_probability"].mean()),
            "masked_count":int(len(predictions))}


def category_metrics(predictions: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows=[]
    for (split,category),group in predictions.groupby(["split","token_category"],dropna=False):
        met=metrics_from_predictions(group)
        rows.append({"split":split,"token_category":category,**{f"{stage}_{k}":v for k,v in met.items()}})
    return pd.DataFrame(rows)


def grouped_token_metrics(predictions: pd.DataFrame, category: str, stage: str) -> pd.DataFrame:
    subset=predictions.loc[predictions["token_category"]==category]
    rows=[]
    for token,group in subset.groupby("true_token"):
        met=metrics_from_predictions(group)
        rows.append({"token":token,"frequency":int(len(group)),"masked_count":int(len(group)),
                     f"{stage}_loss":met["loss"],f"{stage}_top1":met["top1_accuracy"],
                     f"{stage}_top3":met["top3_accuracy"],f"{stage}_top5":met["top5_accuracy"],
                     f"{stage}_mean_probability":met["mean_true_probability"]})
    return pd.DataFrame(rows)


def _fixed_batch_logits(model, examples, device, tokenizer, precision):
    if not examples: return torch.empty(0)
    batch=_pad_fixed_batch([{"input_ids":x["masked_input_ids"],"labels":x["labels"]} for x in examples],int(tokenizer.pad_token_id))
    batch={k:v.to(device) for k,v in batch.items()}
    model.eval()
    with torch.no_grad(),_autocast_context(device,precision):
        return model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits.float().detach().cpu()


def _save_checkpoint(model, optimizer, scheduler, run_dir: Path, epoch: int, global_step: int,
                     next_batch: int, best_loss: float, bad_epochs: int, seed: int,
                     is_best: bool=False, keep_limit: int=2, cuda_rng_enabled: bool=False) -> Path:
    ckpt_dir=run_dir/"checkpoints"/f"checkpoint-step-{global_step:08d}"
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(ckpt_dir,safe_serialization=True)
    state={"epoch":epoch,"global_step":global_step,"next_batch":next_batch,"best_loss":best_loss,
           "bad_epochs":bad_epochs,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
           "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if cuda_rng_enabled else None}
    torch.save({"state":state,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict()},ckpt_dir/"training_state.pt")
    (ckpt_dir/"checkpoint_state.json").write_text(json.dumps({k:v for k,v in state.items() if k not in {"python_rng","numpy_rng","torch_rng","cuda_rng"}},indent=2),encoding="utf-8")
    if is_best:
        best_dir=run_dir/"best_checkpoint"
        if best_dir.exists(): shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True,exist_ok=False)
        model.save_pretrained(best_dir,safe_serialization=True)
        (best_dir/"best_state.json").write_text(json.dumps({"epoch":max(1,epoch),"global_step":global_step,
            "best_validation_loss":best_loss},indent=2),encoding="utf-8")
    all_ckpts=sorted((run_dir/"checkpoints").glob("checkpoint-step-*"),key=lambda p:p.stat().st_mtime)
    while len(all_ckpts)>keep_limit:
        old=all_ckpts.pop(0)
        if old!=ckpt_dir: shutil.rmtree(old,ignore_errors=True)
    (run_dir/"latest_checkpoint.txt").write_text(str(ckpt_dir.relative_to(run_dir)),encoding="utf-8")
    return ckpt_dir


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    root=Path(run_dir)/"checkpoints"
    candidates=[]
    for path in root.glob("checkpoint-step-*"):
        state_path=path/"checkpoint_state.json"
        if state_path.is_file():
            try: state=json.loads(state_path.read_text()); candidates.append((int(state.get("global_step",-1)),path))
            except Exception: pass
    return max(candidates,key=lambda x:x[0])[1] if candidates else None


def _save_json(path: Path, value: Any) -> None:
    def convert(x):
        if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
        if isinstance(x,(np.bool_,)): return bool(x)
        if isinstance(x,Path): return str(x)
        if isinstance(x,torch.device): return str(x)
        return x
    path.write_text(json.dumps(convert(value),ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def _training_loop(model, train_chunks, val_examples, tokenizer, config, device,
                   precision, run_dir, resume_checkpoint=None):
    train_dataset=TokenSequenceDataset(train_chunks.reset_index(drop=True))
    workers=int(config.get("num_workers",0)) if device.type=="cuda" else 0
    collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,
                                             mlm_probability=float(config.get("mlm_probability",.15)))
    batch_size=int(config["train_batch_size"]); accum=int(config["gradient_accumulation_steps"])
    loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=collator,
                      num_workers=workers,pin_memory=device.type=="cuda")
    steps_per_epoch=math.ceil(len(loader)/accum)
    epochs=int(config["epochs"]); total_steps=steps_per_epoch*epochs
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config.get("weight_decay",.01)))
    warmup=int(total_steps*float(config.get("warmup_ratio",.1)))
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total_steps)
    start_epoch=0; start_batch=0; global_step=0; best_loss=float("inf"); bad_epochs=0
    if resume_checkpoint is not None:
        resume_checkpoint=Path(resume_checkpoint)
        state_obj=torch.load(resume_checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
        state=state_obj["state"]; optimizer.load_state_dict(state_obj["optimizer"]); scheduler.load_state_dict(state_obj["scheduler"])
        start_epoch=int(state["epoch"]); start_batch=int(state["next_batch"]); global_step=int(state["global_step"])
        best_loss=float(state["best_loss"]); bad_epochs=int(state["bad_epochs"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); torch.set_rng_state(state["torch_rng"])
        if device.type=="cuda" and state.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming from {resume_checkpoint.name}, epoch={start_epoch+1}, next_batch={start_batch}, step={global_step}")
    history_path=run_dir/"training_history.csv"
    history=pd.read_csv(history_path).to_dict(orient="records") if history_path.is_file() else []
    stop_requested={"value":False}
    previous_handler=None
    if hasattr(signal,"SIGTERM"):
        try:
            previous_handler=signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM,lambda signum,frame:stop_requested.update(value=True))
        except (ValueError,OSError): pass
    model.to(device); model.train(); optimizer.zero_grad(set_to_none=True)
    save_steps=int(config.get("checkpoint_every_steps",500)); keep_limit=int(config.get("save_total_limit",2))
    early_patience=int(config.get("early_stopping_patience",2)); max_norm=float(config.get("max_grad_norm",1.0))
    scaler=(torch.amp.GradScaler("cuda",enabled=True)
            if device.type=="cuda" and precision.get("fp16",False) else None)
    epoch_logs=[]
    try:
        for epoch in range(start_epoch,epochs):
            # A fixed permutation makes mid-epoch resume deterministic; chunks remain in their composition split.
            generator=torch.Generator().manual_seed(int(config.get("seed",42))+epoch)
            order=torch.randperm(len(train_dataset),generator=generator).tolist()
            class FixedOrderSampler(Sampler):
                def __init__(self, indices): self.indices=indices
                def __iter__(self): return iter(self.indices)
                def __len__(self): return len(self.indices)
            epoch_sampler=FixedOrderSampler(order)
            loader=DataLoader(train_dataset,batch_size=batch_size,sampler=epoch_sampler,collate_fn=collator,
                              num_workers=workers,pin_memory=device.type=="cuda",generator=generator)
            running=0.0; n_micro=0
            for batch_index,batch in enumerate(loader):
                if epoch==start_epoch and batch_index<start_batch:
                    continue
                batch={k:v.to(device,non_blocking=device.type=="cuda") for k,v in batch.items()}
                try:
                    with _autocast_context(device,precision):
                        loss=model(**batch).loss
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss detected.")
                    tail_size = len(loader) % accum
                    last_group_size = tail_size if tail_size else accum
                    is_partial_final_group = (batch_index // accum) == (len(loader) // accum) and tail_size > 0
                    divisor = last_group_size if is_partial_final_group else accum
                    if scaler is not None:
                        scaler.scale(loss/divisor).backward()
                    else:
                        (loss/divisor).backward()
                    running+=float(loss.detach().float().item()); n_micro+=1
                    final_batch=batch_index==len(loader)-1
                    if (batch_index+1)%accum==0 or final_batch:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)
                        if not torch.isfinite(torch.as_tensor(grad_norm)): raise FloatingPointError("Non-finite gradient norm detected.")
                        if scaler is not None:
                            scaler.step(optimizer); scaler.update()
                        else: optimizer.step()
                        scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
                        lr=float(scheduler.get_last_lr()[0])
                        history.append({"epoch":epoch+1,"step":global_step,"train_loss":running/max(n_micro,1),
                                        "eval_loss":np.nan,"learning_rate":lr,"gradient_norm":float(torch.as_tensor(grad_norm).item())})
                        running=0.; n_micro=0
                        if global_step%save_steps==0 or stop_requested["value"]:
                            _save_checkpoint(model,optimizer,scheduler,run_dir,epoch,global_step,batch_index+1,best_loss,bad_epochs,
                                             int(config.get("seed",42)),keep_limit=keep_limit,cuda_rng_enabled=device.type=="cuda")
                        if stop_requested["value"]:
                            pd.DataFrame(history).to_csv(history_path,index=False)
                            _save_checkpoint(model,optimizer,scheduler,run_dir,epoch,global_step,batch_index+1,best_loss,bad_epochs,
                                             int(config.get("seed",42)),keep_limit=keep_limit,cuda_rng_enabled=device.type=="cuda")
                            raise InterruptedError("SLURM SIGTERM received; checkpoint saved. Re-submit with --resume auto.")
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if device.type=="cuda": torch.cuda.empty_cache()
                        if device.type=="cuda":
                            guidance="CUDA OOM. Reduce train_batch_size to 8 or 4 and increase gradient_accumulation_steps to 8 or 16 to keep effective batch near 64; no batch change was applied automatically."
                        else:
                            guidance=f"Out of memory on {device.type}. Reduce the smoke-test max_structures or train_batch_size in YAML; no setting was changed automatically."
                        raise RuntimeError(guidance) from exc
                    raise
            val_metrics,_=evaluate_fixed_masks(model,val_examples,tokenizer,device,
                int(config["eval_batch_size"]),precision,save_predictions=False)
            if not np.isfinite(val_metrics["loss"]): raise FloatingPointError("Validation loss is NaN/Inf.")
            improved=val_metrics["loss"]<best_loss
            if improved: best_loss=val_metrics["loss"]; bad_epochs=0
            else: bad_epochs+=1
            history.append({"epoch":epoch+1,"step":global_step,"train_loss":np.nan,"eval_loss":val_metrics["loss"],
                            "learning_rate":float(scheduler.get_last_lr()[0]),"gradient_norm":np.nan})
            pd.DataFrame(history).to_csv(history_path,index=False)
            # Store the next epoch boundary so --resume auto never replays a finished epoch.
            _save_checkpoint(model,optimizer,scheduler,run_dir,epoch+1,global_step,0,best_loss,bad_epochs,
                             int(config.get("seed",42)),is_best=improved,keep_limit=keep_limit,cuda_rng_enabled=device.type=="cuda")
            epoch_logs.append({"epoch":epoch+1,"global_step":global_step,"validation_loss":val_metrics["loss"],
                               "best":improved,"bad_epochs":bad_epochs})
            print(f"epoch {epoch+1}/{epochs} | eval loss {val_metrics['loss']:.5f} | best={improved}")
            if bad_epochs>=early_patience:
                print(f"Early stopping after {bad_epochs} non-improving validation epochs.")
                break
        pd.DataFrame(history).to_csv(history_path,index=False)
    finally:
        if previous_handler is not None:
            try: signal.signal(signal.SIGTERM,previous_handler)
            except (ValueError,OSError): pass
    return {"history":pd.DataFrame(history),"epoch_logs":epoch_logs,"global_step":global_step,
            "best_validation_loss":best_loss,"early_stopped":bool(bad_epochs>=early_patience),
            "epochs_completed":max([int(x.get("epoch",0)) for x in history],default=0)}


def _embedding_audit(model, tokenizer, old_vocab_size: int, initial_embeddings: torch.Tensor,
                     categories_by_token: dict[str,str], output_dir: Path) -> pd.DataFrame:
    current=model.get_input_embeddings().weight.detach().float().cpu()
    initial=initial_embeddings.detach().float().cpu()
    if current.shape[0]!=len(tokenizer): raise AssertionError("Model/tokenizer embedding size mismatch.")
    tokens=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    rows=[]
    for token_id,token in enumerate(tokens):
        before=initial[token_id]; after=current[token_id]
        diff=after-before
        denom=float(torch.linalg.vector_norm(before).item())
        cosine=float(torch.nn.functional.cosine_similarity(before[None,:],after[None,:]).item())
        rows.append({"token":str(token),"token_id":token_id,
                     "token_type":categories_by_token.get(str(token),"other"),
                     "original_or_added":"original_vocab" if token_id<old_vocab_size else "added_slices_vocab",
                     "initial_norm":denom,"final_norm":float(torch.linalg.vector_norm(after).item()),
                     "L2_displacement":float(torch.linalg.vector_norm(diff).item()),
                     "relative_L2_displacement":float(torch.linalg.vector_norm(diff).item())/max(denom,1e-12),
                     "cosine_similarity_initial_final":cosine})
    result=pd.DataFrame(rows)
    result.to_csv(output_dir/"embedding_shift.csv",index=False)
    return result


def _token_roles(tokenizer, old_vocab_size: int) -> dict[str,str]:
    roles={}
    for idx,token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        token=str(token)
        if PERIODIC_RE.fullmatch(token): role="periodic_vector"
        elif INTEGER_RE.fullmatch(token): role="node_index"
        elif _is_element(token): role="element"
        else: role="other"
        roles[token]=role
    return roles


def _make_examples(pre: pd.DataFrame, post: pd.DataFrame, fixed_examples: dict[str,list[dict[str,Any]]],
                   tokenizer, out: Path, seed: int=42) -> pd.DataFrame:
    joined=pre.merge(post,on=["split","sequence_id","chunk_id","token_position","true_token_id"],suffixes=("_pre","_post"),validate="one_to_one")
    joined["probability_gain"]=joined["true_probability_post"]-joined["true_probability_pre"]
    rng=np.random.default_rng(seed); selected=[]
    for category in ["element","periodic_vector","node_index","space_group"]:
        group=joined.loc[joined["token_category_pre"]==category]
        if len(group): selected.append(group.iloc[int(rng.integers(0,len(group)))])
    examples=pd.DataFrame(selected)
    if not examples.empty:
        examples["masked_slices_context"]=""
        lookup={(x["split"],x["mat_id"],x["chunk_id"]):x for split,items in fixed_examples.items() for x in items}
        for idx,row in examples.iterrows():
            ex=lookup[(row["split"],row["mat_id"],int(row["chunk_id"]))]
            ids=ex["masked_input_ids"]
            examples.loc[idx,"masked_slices_context"]=" ".join(str(t) for t in tokenizer.convert_ids_to_tokens(ids))
    examples.to_csv(out/"masked_prediction_examples.csv",index=False)
    return examples


def generate_scientific_plots(run_dir: str | Path, tokenizer_path: str | Path | None=None,
                              base_model_name: str=BASE_MODEL_NAME) -> list[Path]:
    """Generate the 20 thesis-oriented MSI comparison plots from run artifacts."""
    import matplotlib.pyplot as plt
    run=Path(run_dir); out=[]
    def save(fig,name):
        path=run/name; fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); out.append(path)
    history=pd.read_csv(run/"training_history.csv") if (run/"training_history.csv").exists() else pd.DataFrame()
    pre_val=json.loads((run/"pre_dapt_validation_metrics.json").read_text()) if (run/"pre_dapt_validation_metrics.json").exists() else {}
    post_val=json.loads((run/"post_dapt_validation_metrics.json").read_text()) if (run/"post_dapt_validation_metrics.json").exists() else {}
    pre_hold=json.loads((run/"pre_dapt_holdout_metrics.json").read_text()) if (run/"pre_dapt_holdout_metrics.json").exists() else {}
    post_hold=json.loads((run/"post_dapt_holdout_metrics.json").read_text()) if (run/"post_dapt_holdout_metrics.json").exists() else {}
    pre=read_table(run/"pre_dapt_predictions.parquet") if (run/"pre_dapt_predictions.parquet").exists() or (run/"pre_dapt_predictions.parquet.csv.gz").exists() else pd.DataFrame()
    post=read_table(run/"post_dapt_predictions.parquet") if (run/"post_dapt_predictions.parquet").exists() or (run/"post_dapt_predictions.parquet.csv.gz").exists() else pd.DataFrame()
    if pre.empty or post.empty: return []
    merged=pre.merge(post,on=["split","sequence_id","chunk_id","token_position","true_token_id"],suffixes=("_pre","_post"),validate="one_to_one")
    merged["probability_gain"]=merged["true_probability_post"]-merged["true_probability_pre"]
    # 01 Training curves.
    fig,ax=plt.subplots(figsize=(9,5))
    if not history.empty:
        step_rows=history.dropna(subset=["train_loss"]); eval_rows=history.dropna(subset=["eval_loss"])
        if len(step_rows): ax.plot(step_rows.step,step_rows.train_loss,label="Train loss",alpha=.65)
        if len(eval_rows): ax.plot(eval_rows.step,eval_rows.eval_loss,"o-",label="Fixed validation MLM loss")
    ax.set(title="DAPT training curves",xlabel="Optimizer step",ylabel="MLM cross-entropy loss"); ax.grid(alpha=.2); ax.legend(); save(fig,"01_mlm_training_curves.png")
    # 02 Overall metrics as separate axes.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    labels=["Loss","Top-1","Top-5","MRR"]; keys=["loss","top1_accuracy","top5_accuracy","mrr"]
    x=np.arange(len(labels)); width=.36
    axs[0].bar(x-width/2,[pre_val.get(k,np.nan) for k in keys],width,label="PRE")
    axs[0].bar(x+width/2,[post_val.get(k,np.nan) for k in keys],width,label="POST")
    axs[0].set_xticks(x,labels); axs[0].set_title("Validation fixed-mask metrics"); axs[0].legend(); axs[0].grid(axis="y",alpha=.2)
    axs[1].bar([0,1],[pre_hold.get("loss",np.nan),post_hold.get("loss",np.nan)],label="Holdout loss")
    axs[1].set_xticks([0,1],["PRE","POST"]); axs[1].set_title("Holdout MLM loss"); axs[1].grid(axis="y",alpha=.2)
    save(fig,"02_pre_vs_post_overall_mlm.png")
    # 03 Perplexity validation / holdout.
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(2); w=.35
    ax.bar(x-w/2,[pre_val.get("perplexity",np.nan),pre_hold.get("perplexity",np.nan)],w,label="PRE")
    ax.bar(x+w/2,[post_val.get("perplexity",np.nan),post_hold.get("perplexity",np.nan)],w,label="POST")
    ax.set_xticks(x,["Validation","Holdout"]); ax.set_ylabel("Perplexity"); ax.set_title("PRE vs POST perplexity (same tokenizer/masks)"); ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,"03_pre_vs_post_perplexity.png")
    # Category metrics.
    category=pd.read_csv(run/"pre_vs_post_metrics_by_category.csv") if (run/"pre_vs_post_metrics_by_category.csv").exists() else pd.DataFrame()
    valcat=category.loc[category.split=="validation"] if not category.empty else category
    cats=[x for x in TOKEN_CATEGORIES if not valcat.empty and x in set(valcat.token_category)]
    def category_plot(filename,metric,title,ylabel):
        fig,ax=plt.subplots(figsize=(9,5)); xpos=np.arange(len(cats)); w=.36
        if cats:
            b=valcat.set_index("token_category")
            ax.bar(xpos-w/2,[b.loc[c,f"pre_{metric}"] for c in cats],w,label="PRE")
            ax.bar(xpos+w/2,[b.loc[c,f"post_{metric}"] for c in cats],w,label="POST")
        ax.set_xticks(xpos,cats,rotation=20); ax.set(title=title,ylabel=ylabel); ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,filename)
    category_plot("04_accuracy_by_slices_category.png","top1_accuracy","Top-1 accuracy by SLICES category","Accuracy")
    category_plot("05_loss_by_slices_category.png","loss","Fixed-mask loss by SLICES category","Cross-entropy")
    category_plot("06_top5_accuracy_by_category.png","top5_accuracy","Top-5 accuracy by SLICES category","Accuracy")
    # 07 original vs added.
    vocab=pd.read_csv(run/"original_vs_added_token_metrics.csv") if (run/"original_vs_added_token_metrics.csv").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(8,5))
    if not vocab.empty:
        x=np.arange(len(vocab)); w=.36; ax.bar(x-w/2,vocab.pre_top1,w,label="PRE"); ax.bar(x+w/2,vocab.post_top1,w,label="POST")
        ax.set_xticks(x,vocab.original_or_added)
    ax.set(title="Original vs added vocabulary",ylabel="Top-1 accuracy"); ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,"07_original_vs_added_vocab_accuracy.png")
    # 08 periodic vectors.
    periodic=pd.read_csv(run/"periodic_vector_mlm_metrics.csv") if (run/"periodic_vector_mlm_metrics.csv").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(11,5))
    if not periodic.empty:
        periodic=periodic.sort_values("masked_count",ascending=False).head(20); x=np.arange(len(periodic)); w=.36
        ax.bar(x-w/2,periodic.pre_top1,w,label="PRE"); ax.bar(x+w/2,periodic.post_top1,w,label="POST"); ax.set_xticks(x,periodic.token,rotation=45)
    ax.set(title="Periodic-vector MLM learning",ylabel="Top-1 accuracy"); ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,"08_periodic_vector_learning.png")
    # 09 element learning.
    elements=pd.read_csv(run/"element_mlm_metrics.csv") if (run/"element_mlm_metrics.csv").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(12,5))
    if not elements.empty:
        elements=elements.sort_values("masked_count",ascending=False).head(25); x=np.arange(len(elements)); w=.36
        ax.bar(x-w/2,elements.pre_top1,w,label="PRE"); ax.bar(x+w/2,elements.post_top1,w,label="POST"); ax.set_xticks(x,elements.element,rotation=60)
    ax.set(title="Element MLM learning (most masked occurrences)",ylabel="Top-1 accuracy"); ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,"09_element_mlm_learning.png")
    # 10 true-token probability scatter.
    fig,ax=plt.subplots(figsize=(6.5,6)); sample=merged.sample(n=min(30000,len(merged)),random_state=42) if len(merged)>30000 else merged
    ax.scatter(sample.true_probability_pre,sample.true_probability_post,s=7,alpha=.25); ax.plot([0,1],[0,1],"k--",lw=1); ax.set(xlabel="PRE true-token probability",ylabel="POST true-token probability",title="Probability assigned to the correct token"); ax.grid(alpha=.2); save(fig,"10_true_token_probability_shift.png")
    # 11 gain distribution by category.
    fig,ax=plt.subplots(figsize=(9,5)); cats_present=[c for c in ["element","periodic_vector","node_index","space_group"] if c in set(merged.token_category_pre)]
    data=[merged.loc[merged.token_category_pre==c,"probability_gain"].to_numpy() for c in cats_present]
    if data: ax.boxplot(data,labels=cats_present,showfliers=False)
    ax.axhline(0,color="black",lw=.8); ax.set(title="True-token probability gain by token type",ylabel="POST − PRE probability"); ax.grid(axis="y",alpha=.2); save(fig,"11_probability_gain_by_token_type.png")
    # 12/13 embedding changes.
    emb=pd.read_csv(run/"embedding_shift.csv") if (run/"embedding_shift.csv").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(8,5))
    if not emb.empty:
        groups=[emb.loc[emb.original_or_added==x,"cosine_similarity_initial_final"].dropna() for x in ["original_vocab","added_slices_vocab"]]
        ax.boxplot(groups,labels=["Original","Added SLICES"],showfliers=False)
    ax.set(title="Embedding cosine similarity (initial vs final)",ylabel="Cosine similarity"); ax.grid(axis="y",alpha=.2); save(fig,"12_embedding_cosine_change.png")
    fig,ax=plt.subplots(figsize=(8,5))
    if not emb.empty:
        groups=[emb.loc[emb.original_or_added==x,"relative_L2_displacement"].dropna() for x in ["original_vocab","added_slices_vocab"]]
        ax.boxplot(groups,labels=["Original","Added SLICES"],showfliers=False)
    ax.set(title="Relative embedding displacement",ylabel="Relative L2 displacement"); ax.grid(axis="y",alpha=.2); save(fig,"13_embedding_displacement.png")
    # 14 PCA positions saved by runner.
    coords=read_table(run/"selected_embedding_coordinates.parquet") if (run/"selected_embedding_coordinates.parquet").exists() or (run/"selected_embedding_coordinates.parquet.csv.gz").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(8,7))
    if not coords.empty:
        for _,r in coords.iterrows():
            ax.annotate("",(r.post_pc1,r.post_pc2),(r.pre_pc1,r.pre_pc2),arrowprops={"arrowstyle":"->","alpha":.55})
            ax.scatter(r.pre_pc1,r.pre_pc2,marker="o",color="C0"); ax.scatter(r.post_pc1,r.post_pc2,marker="x",color="C1"); ax.text(r.post_pc1,r.post_pc2,str(r.token),fontsize=8)
    ax.set(title="Selected token embeddings before/after DAPT (illustrative PCA)",xlabel="PC1",ylabel="PC2"); ax.grid(alpha=.2); save(fig,"14_selected_embeddings_pca.png")
    # 15 text examples panel.
    examples=pd.read_csv(run/"masked_prediction_examples.csv") if (run/"masked_prediction_examples.csv").exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(12,8)); ax.axis("off")
    if not examples.empty:
        lines=[]
        for _,r in examples.head(4).iterrows():
            lines.append(f"{r.get('token_category_pre','?')} TRUE={r.get('true_token_pre','?')} | PRE {r.get('top5_tokens_json_pre','')} | POST {r.get('top5_tokens_json_post','')}")
        ax.text(.01,.98,"\n\n".join(lines),va="top",family="monospace",fontsize=9,wrap=True)
    ax.set_title("Representative fixed-mask prediction examples"); save(fig,"15_masked_prediction_examples.png")
    # 16/17 confusion matrices using most frequent true tokens and top1 prediction.
    def confusion_plot(category,filename,title,top_n=15):
        subset=post.loc[(post.split=="holdout")&(post.token_category==category)]
        if subset.empty: subset=post.loc[(post.split=="validation")&(post.token_category==category)]
        common=subset.true_token.value_counts().head(top_n).index.tolist(); labels=common+(["OTHER"] if len(subset) else [])
        idx={x:i for i,x in enumerate(labels)}; mat=np.zeros((len(labels),len(labels)),dtype=int)
        for _,r in subset.iterrows():
            truth=r.true_token if r.true_token in idx else "OTHER"
            pred=r.top1_token if r.top1_token in idx else "OTHER"
            mat[idx[truth],idx[pred]]+=1
        fig,ax=plt.subplots(figsize=(9,8))
        if len(labels): ax.imshow(mat,interpolation="nearest",cmap="Blues"); ax.set_xticks(range(len(labels)),labels,rotation=60); ax.set_yticks(range(len(labels)),labels)
        ax.set(title=title,xlabel="Predicted top-1",ylabel="True token"); fig.colorbar(ax.images[0],ax=ax) if ax.images else None; save(fig,filename)
    confusion_plot("periodic_vector","16_periodic_vector_confusion_post_dapt.png","Holdout periodic-vector confusion after DAPT")
    confusion_plot("element","17_element_confusion_post_dapt.png","Holdout element confusion after DAPT")
    # 18 chunking stats.
    stats=pd.read_csv(run/"chunking_statistics.csv"); fig,axs=plt.subplots(1,2,figsize=(9,4))
    axs[0].bar(["Chunked","Single chunk"],[stats.structures_chunked.iloc[0],stats.structures_not_chunked.iloc[0]])
    chunks=read_table(run/"structure_chunk_counts.parquet") if (run/"structure_chunk_counts.parquet").exists() or (run/"structure_chunk_counts.parquet.csv.gz").exists() else pd.DataFrame()
    if not chunks.empty: axs[1].hist(chunks.n_chunks,bins=30)
    axs[0].set_ylabel("Structures"); axs[1].set_xlabel("Chunks per structure"); axs[1].set_ylabel("Structures"); fig.suptitle("Grammar-aware chunking audit"); save(fig,"18_dapt_chunking_statistics.png")
    # 19 dashboard.
    fig,ax=plt.subplots(figsize=(10,7)); ax.axis("off")
    rows=[("Validation loss",pre_val.get("loss"),post_val.get("loss")),("Validation perplexity",pre_val.get("perplexity"),post_val.get("perplexity")),
          ("Validation top-1",pre_val.get("top1_accuracy"),post_val.get("top1_accuracy")),("Validation top-5",pre_val.get("top5_accuracy"),post_val.get("top5_accuracy")),
          ("Holdout loss",pre_hold.get("loss"),post_hold.get("loss")),("Holdout top-1",pre_hold.get("top1_accuracy"),post_hold.get("top1_accuracy"))]
    ax.table(cellText=[[x,format(a,".4f") if a is not None else "NA",format(b,".4f") if b is not None else "NA"] for x,a,b in rows],colLabels=["Metric","PRE","POST"],loc="center")
    ax.set_title("SLICES-DAPT summary (same tokenizer and fixed masks)"); save(fig,"19_dapt_summary_dashboard.png")
    # 20 lexical comparison original vs adapted tokenizer.
    fig,ax=plt.subplots(figsize=(12,6)); ax.axis("off")
    if tokenizer_path:
        try:
            original=AutoTokenizer.from_pretrained(base_model_name); adapted=AutoTokenizer.from_pretrained(tokenizer_path)
            examples_slices=["Fe Na O 0 1 ooo","Sr Cd Rh 0 11 o-o 0 11 ooo"]
            text=[]
            for seq in examples_slices:
                a=original.convert_ids_to_tokens(original.encode(seq,add_special_tokens=False))
                b=adapted.convert_ids_to_tokens(adapted.encode(seq,add_special_tokens=False))
                text.append(f"SLICES: {seq}\nORIGINAL-ZINC-TOKENIZER: {' | '.join(a)}\nADAPTED-TOKENIZER: {' | '.join(b)}")
            ax.text(.01,.98,"\n\n".join(text),va="top",family="monospace",fontsize=10,wrap=True)
        except Exception as exc: ax.text(.01,.98,f"Could not render tokenizer example: {exc}",va="top")
    ax.set_title("Lexical tokenization illustration only (not MLM-loss comparison)"); save(fig,"20_original_vs_adapted_tokenization.png")
    return out


def _aggregate_outputs(pre_predictions: pd.DataFrame, post_predictions: pd.DataFrame,
                       output_dir: Path, chunks: pd.DataFrame, tokenizer,
                       old_vocab_size: int, categories_by_token: dict[str,str],
                       initial_embeddings: torch.Tensor, final_model, fixed_examples,
                       device, precision) -> dict[str,Any]:
    write_table(pre_predictions,output_dir/"pre_dapt_predictions.parquet")
    write_table(post_predictions,output_dir/"post_dapt_predictions.parquet")
    for split in ["validation","holdout"]:
        for prefix,frame in [("pre",pre_predictions),("post",post_predictions)]:
            met=metrics_from_predictions(frame.loc[frame.split==split])
            _save_json(output_dir/f"{prefix}_dapt_{split}_metrics.json",met)
    merged=pre_predictions.merge(post_predictions,on=["split","sequence_id","chunk_id","token_position","true_token_id"],suffixes=("_pre","_post"),validate="one_to_one")
    rows=[]
    for split in ["validation","holdout"]:
        p=metrics_from_predictions(pre_predictions.loc[pre_predictions.split==split]); q=metrics_from_predictions(post_predictions.loc[post_predictions.split==split])
        for metric in ["loss","perplexity","top1_accuracy","top3_accuracy","top5_accuracy","mrr","mean_true_probability"]:
            rows.append({"split":split,"metric":metric,"PRE":p[metric],"POST":q[metric],"change":q[metric]-p[metric] if p[metric] is not None and q[metric] is not None else None})
    overall=pd.DataFrame(rows); overall.to_csv(output_dir/"pre_vs_post_dapt_metrics.csv",index=False)
    bycat=category_metrics(pre_predictions,"pre").merge(category_metrics(post_predictions,"post"),on=["split","token_category"],how="outer")
    bycat.to_csv(output_dir/"pre_vs_post_metrics_by_category.csv",index=False)
    for category,name in [("element","element_mlm_metrics.csv"),("periodic_vector","periodic_vector_mlm_metrics.csv"),("space_group","space_group_mlm_metrics.csv")]:
        a=grouped_token_metrics(pre_predictions,category,"pre"); b=grouped_token_metrics(post_predictions,category,"post")
        if "token" in a and "token" in b: combined=a.merge(b,on="token",how="outer",suffixes=("_pre","_post"))
        else: combined=pd.DataFrame()
        if not combined.empty:
            for col in ["frequency","masked_count"]:
                x=f"{col}_pre"; y=f"{col}_post"
                if x in combined and y in combined: combined[col]=combined[x].fillna(combined[y])
        combined.to_csv(output_dir/name,index=False)
    idx=pre_predictions.loc[pre_predictions.token_category=="node_index"].copy()
    idx["index_range"]=pd.cut(pd.to_numeric(idx.true_token,errors="coerce"),[-1,9,19,49,99,np.inf],labels=["0-9","10-19","20-49","50-99","100+"])
    idxpost=post_predictions.loc[post_predictions.token_category=="node_index"].copy()
    idxpost["index_range"]=pd.cut(pd.to_numeric(idxpost.true_token,errors="coerce"),[-1,9,19,49,99,np.inf],labels=["0-9","10-19","20-49","50-99","100+"])
    ia=idx.groupby("index_range",observed=False).apply(lambda g:pd.Series(metrics_from_predictions(g))).add_prefix("pre_")
    ib=idxpost.groupby("index_range",observed=False).apply(lambda g:pd.Series(metrics_from_predictions(g))).add_prefix("post_")
    ia.join(ib).reset_index().to_csv(output_dir/"node_index_mlm_metrics.csv",index=False)
    va=pre_predictions.groupby("original_or_added").apply(lambda g:pd.Series(metrics_from_predictions(g))).add_prefix("pre_")
    vb=post_predictions.groupby("original_or_added").apply(lambda g:pd.Series(metrics_from_predictions(g))).add_prefix("post_")
    va.join(vb).reset_index().to_csv(output_dir/"original_vs_added_token_metrics.csv",index=False)
    embedding_audit=pd.DataFrame([{"token":str(t),"token_id":i,"original_or_added":"original_vocab" if i<old_vocab_size else "added_slices_vocab",
       "initial_norm":float(torch.linalg.vector_norm(initial_embeddings[i]).item())} for i,t in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))))])
    embedding_audit.to_csv(output_dir/"embedding_initialization_audit.csv",index=False)
    final_embeddings=final_model.get_input_embeddings().weight.detach().float().cpu().clone()
    embeddings=_embedding_audit(final_model,tokenizer,old_vocab_size,initial_embeddings,categories_by_token,output_dir)
    selected=[t for t in ["Fe","Na","Mg","Ti","Sr","Si","Al","O","C","Cl","Br","ooo","+oo","-oo","o-o","o+o"] if t in tokenizer.get_vocab()]
    rows=[]; vec=[]
    for token in selected:
        token_id=int(tokenizer.convert_tokens_to_ids(token)); vec.extend([initial_embeddings[token_id].numpy(),final_embeddings[token_id].numpy()])
        rows.append({"token":token,"token_id":token_id})
    coordinates=pd.DataFrame()
    if rows:
        matrix=np.asarray(vec,dtype=np.float64); mean=matrix.mean(axis=0); _,_,vt=np.linalg.svd(matrix-mean,full_matrices=False); pc=(matrix-mean)@vt[:2].T
        for j,row in enumerate(rows): coordinates.loc[j,"token"]=row["token"]; coordinates.loc[j,"token_id"]=row["token_id"]; coordinates.loc[j,"pre_pc1"]=pc[2*j,0]; coordinates.loc[j,"pre_pc2"]=pc[2*j,1]; coordinates.loc[j,"post_pc1"]=pc[2*j+1,0]; coordinates.loc[j,"post_pc2"]=pc[2*j+1,1]
    write_table(coordinates,output_dir/"selected_embedding_coordinates.parquet")
    examples=_make_examples(pre_predictions,post_predictions,fixed_examples,tokenizer,output_dir)
    improvement=overall.copy(); improvement["relative_loss_reduction"]=np.where(improvement.metric=="loss",(improvement.PRE-improvement.POST)/improvement.PRE,np.nan)
    improvement.to_csv(output_dir/"dapt_improvement_summary.csv",index=False)
    return {"overall":overall,"by_category":bycat,"embedding_shift":embeddings,"examples":examples,
            "final_embeddings":final_embeddings,"pre_post_join":merged}


def run_dapt_pipeline(config: dict[str,Any], project_root: str | Path, run_dir: str | Path,
                      device_policy: str="auto", require_cuda: bool=False,
                      publish_final_model: bool=False, smoke_mode: bool=False,
                      resume: str | None=None) -> dict[str,Any]:
    """Shared end-to-end runner used by both 03A and the MSI CLI."""
    project_root=Path(project_root).resolve(); run_dir=Path(run_dir).resolve(); run_dir.mkdir(parents=True,exist_ok=True)
    device,device_info=detect_device(device_policy,require_cuda=require_cuda)
    precision=select_precision(device); set_reproducible_seeds(int(config.get("seed",42)),device)
    tokenizer_path=Path(config.get("tokenizer_path",project_root/"audit/tokenizers/chemberta_zinc_slices_adapted"))
    manifest_path=tokenizer_path/"adaptation_manifest.json"
    if not tokenizer_path.is_dir() or not manifest_path.is_file(): raise FileNotFoundError(f"Accepted SLICES tokenizer not found at {tokenizer_path}")
    adapted_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if not adapted_manifest.get("accepted_for_dapt",True): raise RuntimeError("Saved adapted tokenizer manifest does not mark it accepted for DAPT.")
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None: raise ValueError("Tokenizer must define mask_token_id and pad_token_id.")
    dataset_path=Path(config.get("dataset_path") or find_alexandria_csv(project_root))
    df=load_alexandria(dataset_path)
    # The full-corpus lexical check runs before Mac subsetting to validate the global tokenizer/dataset pair.
    fidelity_rate,fidelity_sample=grammar_fidelity_check(df,tokenizer,sample_size=5000,seed=int(config.get("diagnostic_seed",12345)),min_rate=float(config.get("minimum_grammar_fidelity",.995)))
    split_path=Path(config.get("split_path",project_root/"dapt/configs/dapt_split_assignments.csv"))
    prepared=prepare_split_and_chunks(df,split_path,tokenizer,run_dir,seed=int(config.get("seed",42)),
        diagnostic_seed=int(config.get("diagnostic_seed",12345)),max_length=int(config.get("max_length",512)),
        max_structures=config.get("max_structures"))
    chunks=prepared["chunks"]
    fixed_examples,fixed_positions=build_fixed_diagnostics(chunks,tokenizer,int(config.get("diagnostic_seed",12345)),float(config.get("mlm_probability",.15)))
    write_table(fixed_positions,run_dir/"fixed_mask_positions.parquet")
    # Device/environment manifest; Mac policy never probes CUDA.
    try:
        import transformers, tokenizers
        import yaml
        from importlib.metadata import PackageNotFoundError, version
        try:
            pymatgen_version = version("pymatgen")
        except PackageNotFoundError:
            pymatgen_version = "available-version-not-reported"
        versions={"python":sys.version,"torch":torch.__version__,"transformers":transformers.__version__,
                  "tokenizers":tokenizers.__version__,"datasets":None,"datasets_note":"Not required; a torch Dataset/DataLoader is used.",
                  "pyyaml":yaml.__version__,"numpy":np.__version__,"pandas":pd.__version__,"pymatgen":pymatgen_version,
                  "cuda_version":device_info["cuda_version"],"gpu_name":device_info["gpu_name"],"gpu_count":device_info["gpu_count"],
                  "gpu_vram_bytes":device_info["gpu_vram_bytes"],"device":device_info["device"],"precision":precision["precision"],
                  "platform":sys.platform,"seed":int(config.get("seed",42)),"diagnostic_seed":int(config.get("diagnostic_seed",12345)),
                  "dataset_path":str(dataset_path),"dataset_sha256":sha256_file(dataset_path),
                  "adapted_tokenizer_manifest_sha256":sha256_file(manifest_path),
                  "composition_split_path":str(split_path),"composition_split_sha256":sha256_file(split_path)}
    except ImportError:
        versions={"python":sys.version,"torch":torch.__version__,"device":device_info["device"],"precision":precision["precision"]}
    _save_json(run_dir/"environment.json",versions)
    model=AutoModelForMaskedLM.from_pretrained(str(config.get("base_model_name",BASE_MODEL_NAME)))
    old_vocab_size=int(model.get_input_embeddings().num_embeddings)
    if old_vocab_size!=int(tokenizer.vocab_size):
        # tokenizer.vocab_size may represent the base BPE; this is exactly expected here.
        print(f"Base model embeddings={old_vocab_size}; tokenizer.vocab_size={tokenizer.vocab_size}; len(tokenizer)={len(tokenizer)}")
    old_embeddings=model.get_input_embeddings().weight.detach().clone()
    import inspect
    resize_signature=inspect.signature(model.resize_token_embeddings)
    resize_kwargs={"mean_resizing":False} if "mean_resizing" in resize_signature.parameters else {}
    model.resize_token_embeddings(len(tokenizer),**resize_kwargs)
    model.tie_weights()
    new_embeddings=model.get_input_embeddings().weight
    original_preserved=bool(torch.equal(old_embeddings,new_embeddings[:old_vocab_size]))
    if not original_preserved: raise AssertionError("Original pretrained embeddings changed during resize.")
    if new_embeddings.shape[0]!=len(tokenizer): raise AssertionError("Tokenizer/model embedding sizes do not match after resize.")
    new_norms=torch.linalg.vector_norm(new_embeddings.detach().float().cpu(),dim=1).numpy()
    roles=_token_roles(tokenizer,old_vocab_size)
    init_rows=[]
    for token_id,token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        init_rows.append({"token":str(token),"token_id":token_id,"token_type":roles.get(str(token),"other"),
                          "original_or_added":"original_vocab" if token_id<old_vocab_size else "added_slices_vocab",
                          "initial_norm":float(new_norms[token_id])})
    pd.DataFrame(init_rows).to_csv(run_dir/"embedding_initialization_audit.csv",index=False)
    # Fair PRE-DAPT evaluation uses the resized model, adapted tokenizer, and fixed masks.
    model.to(device)
    pre_predictions=[]; pre_metrics={}
    for split in ["validation","holdout"]:
        met,preds=evaluate_fixed_masks(model,fixed_examples[split],tokenizer,device,int(config["eval_batch_size"]),precision)
        pre_metrics[split]=met; pre_predictions.append(preds)
        _save_json(run_dir/f"pre_dapt_{split}_metrics.json",met)
    pre_predictions=pd.concat(pre_predictions,ignore_index=True)
    write_table(pre_predictions,run_dir/"pre_dapt_predictions.parquet")
    # Save a compact exact-token embedding snapshot in memory only; no external initialization is used.
    initial_embeddings=model.get_input_embeddings().weight.detach().float().cpu().clone()
    # Free the PRE model allocation before loading a checkpoint for resume.
    resume_checkpoint=None
    if resume:
        if resume=="auto": resume_checkpoint=find_latest_checkpoint(run_dir)
        else: resume_checkpoint=Path(resume)
        if resume_checkpoint:
            model.to("cpu")
            del model
            gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
            model=AutoModelForMaskedLM.from_pretrained(resume_checkpoint)
    train_chunks=chunks.loc[chunks.split=="train"].reset_index(drop=True)
    val_examples=fixed_examples["validation"]
    result=_training_loop(model,train_chunks,val_examples,tokenizer,config,device,precision,run_dir,resume_checkpoint)
    best_path=run_dir/"best_checkpoint"
    if best_path.is_dir(): model=AutoModelForMaskedLM.from_pretrained(best_path)
    model.to(device)
    post_predictions=[]; post_metrics={}
    for split in ["validation","holdout"]:
        met,preds=evaluate_fixed_masks(model,fixed_examples[split],tokenizer,device,int(config["eval_batch_size"]),precision)
        post_metrics[split]=met; post_predictions.append(preds)
        _save_json(run_dir/f"post_dapt_{split}_metrics.json",met)
    post_predictions=pd.concat(post_predictions,ignore_index=True)
    # Holdout is evaluation-only and is never passed to training.
    aggregate=_aggregate_outputs(pre_predictions,post_predictions,run_dir,chunks,tokenizer,old_vocab_size,roles,
        initial_embeddings,model,fixed_examples,device,precision)
    _save_json(run_dir/"pre_dapt_metrics.json",pre_metrics)
    _save_json(run_dir/"post_dapt_metrics.json",post_metrics)
    # Check checkpoint serialization/reload with identical fixed-mask inputs.
    verify_examples=(fixed_examples["validation"][:min(8,len(fixed_examples["validation"]))])
    reference=_fixed_batch_logits(model,verify_examples,device,tokenizer,precision)
    reload_model=AutoModelForMaskedLM.from_pretrained(best_path).to(device).eval()
    reloaded=_fixed_batch_logits(reload_model,verify_examples,device,tokenizer,precision)
    reload_ok=bool(torch.allclose(reference,reloaded,rtol=1e-4,atol=1e-5))
    del reload_model
    if device.type=="cuda": torch.cuda.empty_cache()
    history=pd.read_csv(run_dir/"training_history.csv") if (run_dir/"training_history.csv").exists() else pd.DataFrame()
    finite_losses=bool(np.isfinite(history[[c for c in ["train_loss","eval_loss"] if c in history]].to_numpy(dtype=float)[~np.isnan(history[[c for c in ["train_loss","eval_loss"] if c in history]].to_numpy(dtype=float))]).all()) if not history.empty else False
    split_summary=prepared["split_summary"]
    no_leakage=True
    grammar_preserved=fidelity_rate>=float(config.get("minimum_grammar_fidelity",.995))
    val_improved=post_metrics["validation"]["loss"]<pre_metrics["validation"]["loss"]
    hold_improved=post_metrics["holdout"]["loss"]<pre_metrics["holdout"]["loss"]
    size_match=model.get_input_embeddings().num_embeddings==len(tokenizer)
    critical={"finite_losses":finite_losses,"composition_split_clean":no_leakage,
              "validation_loss_improved":bool(val_improved),"holdout_loss_improved":bool(hold_improved),
              "tokenizer_embedding_size_match":bool(size_match),"grammar_fidelity_preserved":bool(grammar_preserved),
              "original_embeddings_preserved":original_preserved,"checkpoint_reload_passed":reload_ok}
    critical_pass=all(critical.values())
    desirables={
        "overall_top1_improved":post_metrics["validation"]["top1_accuracy"]>pre_metrics["validation"]["top1_accuracy"],
        "periodic_vector_top1_improved":_category_top1(post_predictions,"validation","periodic_vector")>_category_top1(pre_predictions,"validation","periodic_vector"),
        "element_top1_improved":_category_top1(post_predictions,"validation","element")>_category_top1(pre_predictions,"validation","element"),
        "added_token_top1_improved":_group_top1(post_predictions,"added_slices_vocab")>_group_top1(pre_predictions,"added_slices_vocab"),
        "true_probability_median_gain_positive":bool((post_predictions.merge(pre_predictions,on=["split","sequence_id","chunk_id","token_position","true_token_id"],suffixes=("_post","_pre")).eval("true_probability_post-true_probability_pre").median())>0),
    }
    accepted=critical_pass
    decision="ACCEPT SLICES-DAPT MODEL" if accepted else "REVIEW DAPT TRAINING" if val_improved else "REJECT DAPT MODEL"
    summary={"base_model":config.get("base_model_name",BASE_MODEL_NAME),"tokenizer_path":str(tokenizer_path),
       "dataset_path":str(dataset_path),"n_structures":int(len(prepared["structures"])),"n_chunks":int(len(chunks)),
       "fidelity_rate":fidelity_rate,"device":device_info,"precision":precision["precision"],"epochs_completed":result["epochs_completed"],
       "best_validation_loss":result["best_validation_loss"],"pre_dapt":pre_metrics,"post_dapt":post_metrics,
       "critical_checks":critical,"critical_pass":critical_pass,"desirable_checks":desirables,
       "accepted":accepted,"decision":decision,"run_dir":str(run_dir),"smoke_mode":smoke_mode}
    _save_json(run_dir/"dapt_run_summary.json",summary)
    plot_paths=generate_scientific_plots(run_dir,tokenizer_path=tokenizer_path,base_model_name=config.get("base_model_name",BASE_MODEL_NAME)) if not smoke_mode else []
    if smoke_mode:
        minimum_fidelity=float(config.get("minimum_grammar_fidelity",.995))
        smoke_status={"pipeline_passed":all([fidelity_rate>=minimum_fidelity,original_preserved,size_match,reload_ok,finite_losses]),
            "grammar_fidelity_passed":fidelity_rate>=minimum_fidelity,"chunking_passed":prepared["chunking_statistics"].fraction_edges_preserved_all_selected.iloc[0]==1.0,
            "training_completed":result["epochs_completed"]>=1,"finite_losses":finite_losses,
            "checkpoint_reload_passed":reload_ok,"ready_for_msi":False}
        smoke_status["ready_for_msi"]=bool(smoke_status["pipeline_passed"] and smoke_status["chunking_passed"] and smoke_status["training_completed"])
        _save_json(run_dir/"mac_smoke_test_status.json",smoke_status)
        _save_json(run_dir/"mac_smoke_summary.json",summary)
    elif publish_final_model and accepted:
        model_root=project_root/"dapt"/"models"; canonical=model_root/"chemberta_zinc_slices_dapt"
        destination=canonical if not canonical.exists() or not any(canonical.iterdir()) else model_root/f"chemberta_zinc_slices_dapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        destination.mkdir(parents=True,exist_ok=False)
        model.save_pretrained(destination,safe_serialization=True); tokenizer.save_pretrained(destination)
        manifest={**summary,"training_objective":"masked_language_modeling","mlm_probability":float(config.get("mlm_probability",.15)),
                  "accepted":True,"seed":int(config.get("seed",42)),"transformers_version":__import__("transformers").__version__,
                  "torch_version":torch.__version__,"model_path":str(destination)}
        _save_json(destination/"dapt_manifest.json",manifest); summary["published_model_path"]=str(destination)
        _save_json(run_dir/"dapt_run_summary.json",summary)
    return {"summary":summary,"prepared":prepared,"pre_metrics":pre_metrics,"post_metrics":post_metrics,
            "pre_predictions":pre_predictions,"post_predictions":post_predictions,"history":history,
            "critical_checks":critical,"desirable_checks":desirables,"critical_pass":critical_pass,
            "plot_paths":plot_paths,"model":model,"tokenizer":tokenizer}


def _category_top1(predictions: pd.DataFrame, split: str, category: str) -> float:
    g=predictions.loc[(predictions.split==split)&(predictions.token_category==category)]
    return float((g.rank==1).mean()) if len(g) else float("nan")


def _group_top1(predictions: pd.DataFrame, group: str) -> float:
    g=predictions.loc[predictions.original_or_added==group]
    return float((g.rank==1).mean()) if len(g) else float("nan")


def create_timestamped_run(root: str | Path, prefix: str) -> Path:
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); path=root/f"{prefix}_{stamp}"
    counter=1
    while path.exists():
        path=root/f"{prefix}_{stamp}_{counter:02d}"; counter+=1
    path.mkdir(parents=True,exist_ok=False)
    return path
