"""Audit MatterGPT provenance, conditional tokenization, coverage, and frozen features."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CONDITION, DATASET, FOLDS, MODEL_DIR, REPORT_DIR, ROOT, SCRIPT_DIR, TEST_INPUTS, environment, load_development, logger, prepare_dirs, sha256, write_json

EXPECTED_CKPT = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"


def load_original() -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Instantiate the vendored GPT and strictly load its pinned checkpoint."""
    sys.path.insert(0, str(MODEL_DIR))
    from model import GPT, GPTConfig
    config = json.loads((MODEL_DIR / "config.json").read_text())
    model = GPT(GPTConfig(config["vocab_size"], config["block_size"], **{k: v for k, v in config.items() if k not in {"vocab_size", "block_size", "architectures", "model_type"}}))
    state = torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Carga strict=True no fue limpia")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("El backbone conserva parámetros entrenables")
    return model, config, state


def tokenizer_vocab() -> tuple[list[str], dict[str, int]]:
    """Reconstruct the exact vendored SimpleTokenizer vocabulary ordering."""
    tokens = sorted(set((MODEL_DIR / "Voc_prior").read_text().splitlines() + ["<", ">"] ))
    return tokens, {token: index for index, token in enumerate(tokens)}


def tokenize_slices(text: str, stoi: dict[str, int]) -> list[int]:
    """Encode the public `>` SLICES prefix and `<` termination convention."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("SLICES vacío")
    tokens = text.split()
    unknown = [token for token in tokens if token not in stoi]
    if unknown:
        raise ValueError(f"SLICES contiene OOV: {unknown[:8]}")
    return [stoi[">"]] + [stoi[token] for token in tokens] + [stoi["<"]]


def extract(model: torch.nn.Module, ids: list[int], condition: tuple[float, float], device: torch.device) -> dict[str, np.ndarray]:
    """Extract selected block means and final token states using verified hooks."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    prop = torch.tensor([condition], dtype=torch.float32, device=device)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index in range(len(model.blocks)):
        handles.append(model.blocks[index].register_forward_hook(lambda _m, _i, out, ix=index: captured.__setitem__(ix + 1, out[0])))
    try:
        with torch.inference_mode():
            logits, _loss, _attn, _embed = model(x, prop=prop)
        # `>` is at input position 0; SLICES positions are 1..len-2; `<` last.
        features: dict[str, np.ndarray] = {}
        for layer in (3, 6, 9, 12):
            raw = captured[layer][0]
            slc = raw[1:-1]
            features[f"BLOCK{layer:02d}_MEAN"] = slc.mean(dim=0).cpu().numpy()
            features[f"BLOCK{layer:02d}_LAST"] = slc[-1].cpu().numpy()
            if layer == 12:
                final = model.ln_f(raw)
                features["FINAL_MEAN"] = final[1:-1].mean(dim=0).cpu().numpy()
                features["FINAL_END"] = final[-1].cpu().numpy()
        features["_LOGITS"] = logits[0].cpu().numpy()
        return features
    finally:
        for handle in handles:
            handle.remove()


def preflight() -> dict[str, Any]:
    """Run strict checkpoint, fixed-condition, tokenizer, and data audits."""
    prepare_dirs()
    model, config, state = load_original()
    checkpoint_hash = sha256(MODEL_DIR / "pytorch_model.pt")
    if checkpoint_hash != EXPECTED_CKPT:
        raise RuntimeError(f"SHA checkpoint inesperado: {checkpoint_hash}")
    tokens, stoi = tokenizer_vocab()
    if len(tokens) != config["vocab_size"]:
        raise RuntimeError(f"Vocabulario {len(tokens)} != config {config['vocab_size']}")
    # HF config BOS/EOS metadata do not match the vendored sorted SimpleTokenizer.
    token_audit = {"vocab_size": len(tokens), "sha256": sha256(MODEL_DIR / "Voc_prior"),
                   "bos_config_id": config.get("bos_token_id"), "eos_config_id": config.get("eos_token_id"),
                   "public_prefix_text": ">", "public_prefix_id": stoi[">"],
                   "terminator_text": "<", "terminator_id": stoi["<"],
                   "config_special_ids_match_tokenizer": config.get("bos_token_id") == stoi[">"] and config.get("eos_token_id") == stoi["<"],
                   "roundtrip_examples": ["S", "S Se Te", "0 1 ooo", "Cu Nb S", "0 13 ooo", "> S <"]}
    roundtrip_candidates = [
        "S", "Se", "Te", "Cu Nb S", "0 1 ooo", "0 13 ooo", "S S S S",
        "Mg Pb S 0 12 ooo", "Ba Ge S 1 14 oo+", "Hf Sr S 2 19 o-o",
        "Nb Zn S 3 18 -o+", "Pb V S 4 16 o++", "S Se 0 0 ooo",
        "o--", "0 13 ooo 1 17 oo-",
    ]
    roundtrip_results=[]
    for sample in roundtrip_candidates:
        sample_ids=[stoi[token] for token in sample.split()]
        decoded=" ".join(tokens[index] for index in sample_ids)
        roundtrip_results.append({"input":sample,"pass":decoded==sample,"token_count":len(sample_ids)})
    if not all(item["pass"] for item in roundtrip_results):
        raise RuntimeError("Falló ida/vuelta de tokens de vocabulario")
    token_audit["roundtrip_15"] = roundtrip_results
    data, folds = load_development()
    test_inputs = pd.read_csv(TEST_INPUTS)
    source_test_meta = pd.read_csv(DATASET, usecols=["structure_id", "reduced_formula", "polymorph_family", "split", "matgpt_slices"], dtype={"structure_id": str})
    source_test_meta = source_test_meta[source_test_meta.split == "test"].copy()
    aligned_test = test_inputs.merge(source_test_meta, on="structure_id", suffixes=("_input", "_source"), validate="one_to_one")
    if len(aligned_test) != 24 or set(source_test_meta.structure_id) != set(test_inputs.structure_id):
        raise RuntimeError("Los 24 IDs de test no coinciden con el dataset fuente")
    if not (aligned_test.composition_id == aligned_test.reduced_formula).all() or not (aligned_test.polymorph_family_input == aligned_test.polymorph_family_source).all() or not (aligned_test.matgpt_slices_input == aligned_test.matgpt_slices_source).all():
        raise RuntimeError("Test inputs y dataset fuente discrepan en fórmula/familia/SLICES")
    if set(data.composition_id) & set(test_inputs.composition_id):
        raise RuntimeError("Hay composición compartida entre development y test")
    # Coverage over test inputs uses strings only, no label table or target columns.
    all_slices = data[["structure_id", "slices"]].to_dict("records") + test_inputs.rename(columns={"matgpt_slices": "slices"})[["structure_id", "slices"]].to_dict("records")
    lengths, oov = [], []
    token_counts: dict[str, int] = {}
    for row in all_slices:
        toks = str(row["slices"]).split()
        for token in toks: token_counts[token] = token_counts.get(token, 0) + 1
        lengths.append(len(toks) + 2)
        bad = sorted(set(t for t in toks if t not in stoi))
        if bad:
            oov.append({"structure_id": row["structure_id"], "tokens": bad})
    if max(lengths) > int(config["block_size"]):
        raise RuntimeError(f"La secuencia más larga ({max(lengths)}) excede block_size={config['block_size']}")
    if oov:
        raise RuntimeError(f"OOV en {len(oov)} estructuras; no se recodifican silenciosamente")
    # Small forward checks establish condition arity, hooks and finite features.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    first_ids = tokenize_slices(str(all_slices[0]["slices"]), stoi)
    features_a = extract(model, first_ids, CONDITION, device)
    features_b = extract(model, first_ids, (-2.0, 1.0), device)
    if not all(np.isfinite(v).all() for v in features_a.values()):
        raise RuntimeError("Forward instrumentado produce NaN/Inf")
    # Hook instrumentation must leave the original forward logits unchanged.
    with torch.inference_mode():
        plain_logits = model(torch.tensor([first_ids], dtype=torch.long, device=device), prop=torch.tensor([CONDITION], dtype=torch.float32, device=device))[0][0].cpu().numpy()
    logit_delta = float(np.max(np.abs(plain_logits - features_a["_LOGITS"])))
    if logit_delta > 1e-6: raise RuntimeError(f"Paridad de logits falló: {logit_delta}")
    # Right-padding/suffix invariance for actual positions, checked in batches 1, 2 and 5.
    padding_checks=[]
    sample_sequences=[tokenize_slices(str(row["slices"]),stoi) for row in all_slices[:5]]
    for batch_size in (1,2,5):
        seqs=sample_sequences[:batch_size]; max_len=max(map(len,seqs)); padded=[s+[stoi["<"]]*(max_len-len(s)) for s in seqs]
        x=torch.tensor(padded,dtype=torch.long,device=device); prop=torch.tensor([CONDITION]*batch_size,dtype=torch.float32,device=device)
        states={}
        def save_last(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
            """Capture the block output without replacing the module return."""
            states["last"] = output[0]
        handle=model.blocks[-1].register_forward_hook(save_last)
        try:
            with torch.inference_mode(): batch_logits=model(x,prop=prop)[0]
        finally: handle.remove()
        errors=[]
        for row_idx,seq in enumerate(seqs):
            with torch.inference_mode(): solo_logits=model(torch.tensor([seq],dtype=torch.long,device=device),prop=prop[row_idx:row_idx+1])[0][0]
            errors.append(float(torch.max(torch.abs(batch_logits[row_idx,:len(seq)]-solo_logits)).item()))
        max_error=max(errors,default=0.0)
        if max_error>1e-5: raise RuntimeError(f"Invariancia a sufijo derecho falló batch={batch_size}: {max_error}")
        padding_checks.append({"batch_size":batch_size,"max_logit_abs_error":max_error,"mechanism":"right suffix is ignored for prior causal positions; cache itself uses unpadded sequences"})
    report = {
        "status": "PASS", "checkpoint_sha256": checkpoint_hash, "model_revision": "ee83624d85e5e60e6a8f48bc72bf2663f87273cd",
        "teacher_forcing_serialization_verified_from_training_loader": False,
        "serialization_used": "> SLICES <; prefix is shown in the public inference example; closing token is inferred from the local SimpleTokenizer/decoder; training data loader is absent from the vendored checkpoint files",
        "model_config": config, "strict_load": True, "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "weights_l2_norm": float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in model.parameters())).item()),
        "vocabulary_audit": token_audit, "condition": list(CONDITION), "condition_source": "public model-card example; constant, independent of DFT labels",
        "condition_feature_change_norm": float(np.linalg.norm(features_a["FINAL_END"] - features_b["FINAL_END"])),
        "condition_sensitivity": {name:{"l2_change":float(np.linalg.norm(features_a[name]-features_b[name])),"cosine":float(np.dot(features_a[name],features_b[name])/(np.linalg.norm(features_a[name])*np.linalg.norm(features_b[name])))} for name in ("BLOCK03_MEAN","FINAL_MEAN","FINAL_END")},
        "logits_hook_parity_max_abs_error": logit_delta, "right_padding_checks": padding_checks,
        "data": {"development_structures": len(data), "development_compositions": int(data.composition_id.nunique()),
                 "folds": int(folds.outer_fold.nunique()), "test_structures_inputs_only": len(test_inputs),
                 "test_compositions_inputs_only": int(test_inputs.composition_id.nunique()),
                 "slice_tokens_min": min(lengths) - 2, "slice_tokens_median": float(np.median(lengths)) - 2,
                 "slice_tokens_max": max(lengths) - 2, "oov_structures": 0,
                 "element_token_counts": {symbol: token_counts.get(symbol,0) for symbol in ("S","Se","Te","Cu","Nb","Sn","Zn","Pb","V","Sr","Hf","Mg","Ba","Ge")},
                 "numeric_index_token_count": int(sum(count for token,count in token_counts.items() if token.isdigit())),
                 "periodic_edge_token_count": int(sum(count for token,count in token_counts.items() if token and token[0] in "o+-")),
                 "dataset_sha256": sha256(DATASET), "folds_sha256": sha256(FOLDS), "test_inputs_sha256": sha256(TEST_INPUTS)},
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in [MODEL_DIR / "model.py", MODEL_DIR / "mattergpt_wrapper.py", MODEL_DIR / "config.json", MODEL_DIR / "Voc_prior", MODEL_DIR / "pytorch_model.pt"]},
        "environment": environment(), "device": str(device), "caveat": "Vendored public sample supports > prefix and < termination; config BOS/EOS ids disagree with SimpleTokenizer ordering. Treat the special-token IDs from actual tokenizer, not config metadata."
    }
    write_json(REPORT_DIR / "provenance/preflight.json", report)
    write_json(REPORT_DIR / "tokenizer/audit.json", token_audit | {"coverage": report["data"]})
    return report


def extract_cache() -> dict[str, Any]:
    """Create development-only frozen feature cache and test input cache."""
    audit_path = REPORT_DIR / "provenance/preflight.json"
    if not audit_path.exists() or json.loads(audit_path.read_text()).get("status") != "PASS":
        raise RuntimeError("Ejecute primero preflight PASS")
    data, _folds = load_development()
    test = pd.read_csv(TEST_INPUTS)
    tokens, stoi = tokenizer_vocab()
    del tokens
    model, _config, _state = load_original()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs: list[dict[str, Any]] = []
    for row in data[["structure_id", "composition_id", "polymorph_family", "slices", "energy_dft_ev_atom", "outer_fold"]].to_dict("records"):
        feat = extract(model, tokenize_slices(row["slices"], stoi), CONDITION, device)
        outputs.append(row | {k: v for k, v in feat.items() if not k.startswith("_")})
    # Cache test inputs before label access; the lock later binds this file hash.
    test_outputs: list[dict[str, Any]] = []
    for row in test.to_dict("records"):
        feat = extract(model, tokenize_slices(row["matgpt_slices"], stoi), CONDITION, device)
        test_outputs.append({**row, **{k: v.tolist() for k, v in feat.items() if not k.startswith("_")}})
    dev_meta = pd.DataFrame([{k: r[k] for k in ("structure_id", "composition_id", "polymorph_family", "energy_dft_ev_atom", "outer_fold")} for r in outputs])
    feature_names = tuple(k for k in outputs[0] if k.startswith("BLOCK") or k.startswith("FINAL"))
    dev_features = {name: np.stack([r[name] for r in outputs]).astype(np.float32) for name in feature_names}
    np.savez_compressed(REPORT_DIR / "features/development_features.npz", **dev_features, structure_id=dev_meta.structure_id.to_numpy(str))
    dev_meta.to_csv(REPORT_DIR / "features/development_metadata.csv", index=False)
    test_meta = pd.DataFrame([{k: r[k] for k in ("structure_id", "composition_id", "polymorph_family", "split")} for r in test_outputs])
    test_features = {name: np.stack([r[name] for r in test_outputs]).astype(np.float32) for name in feature_names}
    np.savez_compressed(REPORT_DIR / "test/test_features_prelabel.npz", **test_features, structure_id=test_meta.structure_id.to_numpy(str))
    test_meta.to_csv(REPORT_DIR / "test/test_metadata_prelabel.csv", index=False)
    cache_manifest = {"status": "PASS", "features": list(dev_features), "dimension": int(dev_features["FINAL_END"].shape[1]),
                     "development_rows": len(dev_meta), "test_rows_without_labels": len(test_meta), "condition": list(CONDITION),
                     "checkpoint_sha256": sha256(MODEL_DIR / "pytorch_model.pt"),
                     "development_cache_sha256": sha256(REPORT_DIR / "features/development_features.npz"),
                     "test_cache_sha256": sha256(REPORT_DIR / "test/test_features_prelabel.npz"),
                     "feature_extractor_script_sha256": sha256(Path(__file__).resolve()),
                     "caveat": "FINAL_END denotes the actual closing '<' token following the SLICES string; it is not a MatGPT </end> token."}
    write_json(REPORT_DIR / "features/embeddings_manifest.json", cache_manifest)
    return cache_manifest


def main() -> None:
    """Parse audit phase and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "extract"), required=True)
    args = parser.parse_args()
    log = logger("mattergpt_audit")
    try:
        result = preflight() if args.phase == "preflight" else extract_cache()
        log.info("mattergpt_audit_%s_passed", args.phase, extra={"result": result})
    except Exception as exc:
        log.exception("mattergpt_audit_failed phase=%s error=%s", args.phase, exc)
        raise


if __name__ == "__main__":
    main()
