"""Run small deterministic historical tokenizer probes, never model training.

Executes the original three tokenizer class bodies unchanged with Transformers
4.38.2. Module-level vocabulary downloads are replaced by explicitly hashed
local historical files. Constructor output is confined to report working copies.
MLM diagnostics only collate two sequences; no model, optimizer or GPU is used.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, List

from gather import DEFAULT, ROOT, digest, write_json

LOGGER = logging.getLogger(__name__)
GA = "Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -"
GA_IDS = [149, 57, 57, 41, 41, 139, 142, 24, 139, 141, 20, 139, 140, 8, 150]
SPECIAL = {"unk_token": "[UNK]", "pad_token": "[PAD]", "cls_token": "[CLS]",
           "sep_token": "[SEP]", "mask_token": "[MASK]", "eos_token": "[EOS]", "bos_token": "[BOS]"}


def class_ast(source: Path) -> ast.Module:
    """Extract the three original tokenizer classes without changing their bodies."""
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef)
             and node.name in {"NumTokenizer", "Xtal2txtTokenizer", "SliceTokenizer"}]
    if len(nodes) != 3:
        raise ValueError(f"Missing historical class definitions: {source}")
    return ast.Module(body=nodes, type_ignores=[])


def instantiate(source: Path, vocabulary: Path, special: dict[str, str] | None) -> Any:
    """Instantiate unchanged official source with default or source-defined specials."""
    from transformers import PreTrainedTokenizer
    scope: dict[str, Any] = {"json": json, "os": os, "re": re, "List": List,
                             "PreTrainedTokenizer": PreTrainedTokenizer,
                             "SLICE_VOCAB": str(vocabulary), "SLICE_RT_VOCAB": str(vocabulary)}
    exec(compile(class_ast(source), str(source), "exec"), scope)
    kwargs: dict[str, Any] = {"vocab_file": str(vocabulary), "special_num_token": False, "model_max_length": 512}
    if special is not None:
        kwargs["special_tokens"] = dict(special)
    return scope["SliceTokenizer"](**kwargs)


def record(tokenizer: Any, text: str, name: str, reference: list[int] | None = None,
           padding: bool = False) -> dict[str, Any]:
    """Record exact encoding, masks, tokens and repeatability for a single string."""
    options: dict[str, Any] = {"truncation": False, "padding": "max_length" if padding else False}
    if padding:
        options["max_length"] = 512
    encoded = dict(tokenizer(text, **options))
    repeated = dict(tokenizer(text, **options))
    ids = encoded["input_ids"]
    return {"name": name, "original_string": text, "tokens": tokenizer.tokenize(text), **encoded,
            "effective_length": sum(encoded["attention_mask"]),
            "special_positions": [{"position": i, "token": tokenizer.convert_ids_to_tokens(value), "id": value}
                                  for i, value in enumerate(ids) if value in tokenizer.all_special_ids],
            "repeat_exact": encoded == repeated,
            "official_reference_available": reference is not None,
            "official_ids_match": ids == reference if reference is not None else None,
            "official_mask_match": None,
            "local_unpadded_mask_all_ones": all(encoded["attention_mask"]) if not padding else None,
            "reference_scope": "published documentation, not original pretraining batch" if reference is not None else "local diagnostic only"}


def mlm_probe(tokenizer: Any) -> dict[str, Any]:
    """Test len and fixed-seed MLM collation, recording errors rather than patching source."""
    import torch
    from transformers import DataCollatorForLanguageModeling
    result: dict[str, Any] = {"seed": 42, "training_performed": False}
    try:
        result["len_tokenizer"] = len(tokenizer)
    except Exception as error:
        result["len_error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        torch.manual_seed(42)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        inputs = [dict(tokenizer(GA)), dict(tokenizer("Ba S 0 1 o o o"))]
        batch = collator(inputs)
        result["collation"] = {key: value.tolist() for key, value in batch.items()}
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error_message=str(error))
    return result


def probe(output: Path) -> None:
    """Compare pinned historical implementations on official and real-project inputs."""
    if importlib.metadata.version("transformers") != "4.38.2":
        raise RuntimeError("Requires isolated Transformers 4.38.2 runtime; see README")
    import torch
    torch.set_num_threads(1)
    evidence = output / "evidence"
    candidates = ROOT / "data/processed/exp02_only_hex_mattext_finetune/tokenized_candidate.jsonl"
    selected: dict[str, dict[str, Any]] = {}
    longest: dict[str, Any] = {}
    for line in candidates.open():
        row = json.loads(line)
        if len(row["input_ids"]) > len(longest.get("input_ids", [])):
            longest = row
        if len(selected) < 4:
            selected.setdefault(row["reduced_formula"], row)
    selected["longest"] = longest
    results: dict[str, Any] = {"versions": {name: importlib.metadata.version(name) for name in
        ["transformers", "tokenizers", "huggingface_hub", "torch", "numpy"]}, "python": sys.version,
        "execution": "CPU tokenizer and two-example collation only; no model inference/training",
        "original_training_batch_found": False, "author_reference_vector_found": False, "implementations": {}}
    for label in ["xtal_preupload", "xtal_june5", "xtal_2026"]:
        source = evidence / label / "src/xtal2txt/tokenizer.py"
        if not source.exists():
            matches = list((evidence / label).rglob("tokenizer.py"))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous tokenizer source: {matches}")
            source = matches[0]
        # The pre-upload snapshot contains its own vocabulary; this prevents
        # silently substituting a later external artifact into older code.
        own_vocab = [path for path in (evidence / label).rglob("slice_vocab.txt")]
        original = own_vocab[0] if len(own_vocab) == 1 else output.parent / "reference/slice_vocab.txt"
        working = output / "working" / label
        working.mkdir(parents=True, exist_ok=True)
        vocab_file = working / "slice_vocab.txt"
        shutil.copyfile(original, vocab_file)
        entry: dict[str, Any] = {"source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
            "class_ast_sha256": __import__("hashlib").sha256(ast.dump(class_ast(source), include_attributes=False).encode()).hexdigest(),
            "vocab_source": str(original.relative_to(ROOT)), "vocab_sha256": digest(original), "variants": {}}
        for variant, specials in [("documentation_default", None), ("archived_pretraining_specials", SPECIAL)]:
            try:
                tokenizer = instantiate(source, vocab_file, specials)
                tests = [record(tokenizer, GA, "official_Ga_P", GA_IDS if specials is None else None)]
                if label == "xtal_preupload":
                    tests.append(record(tokenizer, "  " + GA.replace("Ga Ga", "Ga   Ga") + "  ",
                                        "extra_spaces_between_chemical_tokens", GA_IDS if specials is None else None))
                if label == "xtal_preupload" and specials is None:
                    original_tests = evidence / label / "tests/tokenizer/test_slice_tokenizer.py"
                    tree = ast.parse(original_tests.read_text())
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name == "test_tokenizer":
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Call) and len(decorator.args) >= 2:
                                    for number, (text, expected_tokens) in enumerate(ast.literal_eval(decorator.args[1])):
                                        item = record(tokenizer, text, f"official_original_test_{number}")
                                        item["official_expected_tokens"] = expected_tokens
                                        item["official_tokens_match"] = item["tokens"] == expected_tokens
                                        item["reference_scope"] = "original pre-upload unit test token list; no original training IDs/mask"
                                        tests.append(item)
                if label == "xtal_preupload" and specials is not None:
                    for name, row in selected.items():
                        item = record(tokenizer, row["slices_model_input"], f"real_{name}", padding=name == "longest")
                        item.update(structure_id=row["structure_id"], source_slices=row["slices_original"],
                                    matches_prior_candidate=item["input_ids"][:item["effective_length"]] == row["input_ids"])
                        tests.append(item)
                    tests.extend([record(tokenizer, "Ba S 0 1 o o o", "structural_zero_and_padding", padding=True),
                                  record(tokenizer, "[CLS] Ba [MASK] S 0 1 o o o [SEP]", "literal_special_tokens"),
                                  record(tokenizer, "Ba S 10 21 + o -", "multiple_digit_indices"),
                                  record(tokenizer, "Ba S 0 1 ooo", "raw_compact_omission_diagnostic"),
                                  record(tokenizer, "Ba @ S 0 1 o o o", "unknown_symbol_omission_diagnostic")])
                entry["variants"][variant] = {"status": "executed", "vocab": tokenizer.get_vocab(),
                    "special_ids": {name: getattr(tokenizer, name + "_id") for name in SPECIAL},
                    "tests": tests, "mlm_probe": mlm_probe(tokenizer)}
            except Exception as error:
                entry["variants"][variant] = {"status": "failed", "error_type": type(error).__name__, "error_message": str(error)}
                LOGGER.exception("historical_variant_failed source=%s variant=%s", label, variant)
        results["implementations"][label] = entry
        LOGGER.info("historical_probe_completed source=%s", label)
    write_json(output / "tokenizer_golden_tests.json", results)


def main() -> None:
    """Validate CLI report path and execute deterministic CPU-only probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports"):
        parser.error("--output must be inside reports/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s level=%(levelname)s message=%(message)s')
    try:
        probe(output)
    except Exception:
        LOGGER.exception("deterministic_probe_failed")
        raise


if __name__ == "__main__":
    main()
