"""Execute the historical official tokenizer classes with Transformers 4.38.2.

Read pinned 2024 source and the paper's archived pretraining special-token order.
Load class bodies unchanged via AST, excluding module-level network downloads.
Write vocabulary/ID and wrapper probes; never declare original-checkpoint proof.
"""
from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, List

from workflow import LOGGER, ROOT, cli, load_config, read_json, sha256, write_json


def official_class(source: Path, vocabulary: Path) -> Any:
    """Compile unchanged historical class nodes from source Path; return official SliceTokenizer class."""
    from transformers import PreTrainedTokenizer
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"NumTokenizer", "Xtal2txtTokenizer", "SliceTokenizer"}]
    if len(classes) != 3:
        raise ValueError("Expected three official class definitions")
    scope = {"json": json, "os": os, "re": re, "List": List,
             "PreTrainedTokenizer": PreTrainedTokenizer,
             "SLICE_VOCAB": str(vocabulary), "SLICE_RT_VOCAB": str(vocabulary)}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), scope)
    return scope["SliceTokenizer"]


def probe(config: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the official source's IDs and HF call using config; return audit evidence dict."""
    import yaml
    if importlib.metadata.version("transformers") != "4.38.2":
        raise RuntimeError("Run in isolated historical runtime; do not alter dmc or infer IDs from current Transformers")
    paths = config["paths"]
    reference = paths["reports"] / "audit" / "reference"
    source = reference / config["tokenizer"]["source_file"]
    archive = reference / "paper_code" / "MatText-main"
    template = archive / "conf/model/archived_base/pretrain_template.yaml"
    special = yaml.safe_load(template.read_text())["special_tokens"]
    original_vocab = reference / "slice_vocab.txt"
    if __import__("hashlib").md5(original_vocab.read_bytes()).hexdigest() != "ea6ea28b3e7bf8a5a7e2f4f514968639":
        raise ValueError("Original published vocabulary checksum failed")
    # The official constructor writes numbered vocabulary JSON beside vocab_file.
    # Give it an explicit working copy so published references remain immutable.
    working = paths["reports"] / "audit" / "tokenizer_work"
    working.mkdir(parents=True, exist_ok=True)
    vocabulary = working / "slice_vocab.txt"
    shutil.copyfile(original_vocab, vocabulary)
    tokenizer_type = official_class(source, vocabulary)
    tokenizer = tokenizer_type(vocab_file=str(vocabulary), special_num_token=False,
                               special_tokens=special, model_max_length=512,
                               truncation=False, padding=False)
    example = "Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -"
    tokens = tokenizer.tokenize(example)
    single = tokenizer(example, truncation=False, padding=False)
    batch = tokenizer([example, "Ba S 0 1 o o o"], truncation=False, padding=True)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if single["input_ids"] != token_ids:
        raise ValueError("Historical __call__ adds a different special-token envelope than tokenize")
    if tokens.count("[CLS]") != 1 or tokens.count("[SEP]") != 1:
        raise ValueError("Historical reference must contain exactly one CLS/SEP")
    if tokenizer.get_vocab()["o o o"] != 0 or tokenizer.get_vocab()["0"] != 139:
        raise ValueError("Original structural token IDs changed")
    # Deliberately test raw compact syntax: the upstream regex may silently omit
    # periodic images. The audited adapter must catch these unmatched spans.
    compact = "Ba S 0 1 ooo"
    result = {"status": "passed", "source_commit": config["tokenizer"]["source_commit"],
              "source_sha256": sha256(source), "template_sha256": sha256(template),
              "published_vocab_sha256": sha256(original_vocab), "vocab": tokenizer.get_vocab(),
              "special_tokens": special, "special_ids": {key: getattr(tokenizer, key + "_id") for key in special},
              "special_wrapping": "single_CLS_SEP", "special_num_token": False,
              "versions": {name: importlib.metadata.version(name) for name in ["transformers", "tokenizers", "huggingface_hub", "torch", "safetensors", "numpy", "pandas", "pymatgen", "slices"]},
              "python": sys.version, "example": example, "tokens": tokens, "input_ids": single["input_ids"],
              "attention_mask": single["attention_mask"], "batch": dict(batch),
              "raw_compact_example": compact, "raw_compact_tokens": tokenizer.tokenize(compact),
              "default_special_ids_are_not_pretrain_config_ids": True,
              "checkpoint_specific_serialized_tokenizer": False}
    write_json(paths["reports"] / "audit" / "reference_tokenizer.json", result)
    LOGGER.info("historical_tokenizer_verified vocab=%d specials=%s", len(result["vocab"]), result["special_ids"])
    return result


def main() -> None:
    """Run historical source probe; log descriptive errors and fail instead of assuming compatibility."""
    config = load_config(cli(__doc__).parse_args().config)
    try:
        probe(config)
    except Exception:
        LOGGER.exception("reference_tokenizer_probe_failed")
        raise


if __name__ == "__main__":
    main()
