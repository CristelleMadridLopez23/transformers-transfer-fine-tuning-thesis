"""Run the critical local MatText tokenizer-to-checkpoint compatibility preflight."""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any
from math import prod

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "Alternative2/models/MatText-2M-SLICES"
OUT = ROOT / "Alternative2/reports/MatText_frozen_1/provenance"
TOKEN_OUT = ROOT / "Alternative2/reports/MatText_frozen_1/tokenizer_audit"
log = logging.getLogger("mattext_preflight")


def _sha256(path: Path) -> str:
    """Calculate SHA-256 in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a project artifact."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _tensor_header(path: Path) -> dict[str, Any]:
    """Read safetensors metadata without loading tensor values."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Checkpoint does not have a valid safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        header = json.loads(stream.read(size))
    return header


def _tokenizer_probe(model_path: Path) -> dict[str, Any]:
    """Probe AutoTokenizer root and recovered SLICES folder using local files only."""
    result: dict[str, Any] = {"transformers_version": None, "root_load": {}, "recovered_slice_load": {}}
    try:
        import transformers
        from transformers import AutoTokenizer
    except Exception as exc:  # dependency/runtime absence is recorded as a failed probe
        result["import_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["transformers_version"] = transformers.__version__
    sample = "Ba Ti S 0 1 o o +"
    for name, folder in (("root_load", model_path), ("recovered_slice_load", model_path / "tokenizer/slice")):
        try:
            tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            encoded = tokenizer(sample, add_special_tokens=True, truncation=False)
            token_ids = list(encoded["input_ids"])
            result[name] = {
                "status": "loaded",
                "class": type(tokenizer).__name__,
                "len_tokenizer": len(tokenizer),
                "vocab_size": int(tokenizer.vocab_size),
                "pad_token": tokenizer.pad_token,
                "pad_token_id": tokenizer.pad_token_id,
                "cls_token_id": tokenizer.cls_token_id,
                "sep_token_id": tokenizer.sep_token_id,
                "unk_token_id": tokenizer.unk_token_id,
                "sample": sample,
                "input_ids": token_ids,
                "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                "unk_count_in_sample_body": sum(token_id == tokenizer.unk_token_id for token_id in token_ids[1:-1]),
            }
        except Exception as exc:
            result[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return result


def audit() -> dict[str, Any]:
    """Write checkpoint, tokenizer, and special-ID evidence; never load labels."""
    config = _json(MODEL / "config.json")
    snapshot = _json(ROOT / "Alternative2/models/MatText-2M-SLICES.download-manifest.json")
    recovered = _json(MODEL / "tokenizer/slice/tokenizer.json")
    tokenizer_config = _json(MODEL / "tokenizer/slice/tokenizer_config.json")
    recovered_vocab = recovered["model"]["vocab"]
    slice_tokens = (MODEL / "vocab/slice_vocab.txt").read_text(encoding="utf-8").splitlines()
    state = _tensor_header(MODEL / "model.safetensors")
    word_embeddings = state.get("embeddings.word_embeddings.weight")
    if not word_embeddings:
        raise KeyError("Checkpoint has no embeddings.word_embeddings.weight")
    auto_probe = _tokenizer_probe(MODEL)
    embedding_rows = int(word_embeddings["shape"][0])
    parameter_count = sum(prod(tensor["shape"]) for name, tensor in state.items() if name != "__metadata__")
    root_probe = auto_probe.get("root_load", {})
    recovered_probe = auto_probe.get("recovered_slice_load", {})
    root_has_unknown_body = root_probe.get("status") == "loaded" and root_probe.get("unk_count_in_sample_body", 0) > 0
    recovered_vocab_size = len(recovered_vocab)
    blockers = []
    if embedding_rows != recovered_vocab_size:
        blockers.append(f"recovered_tokenizer_size_{recovered_vocab_size}_does_not_match_embedding_rows_{embedding_rows}")
    if config.get("pad_token_id") != recovered_vocab.get("[PAD]"):
        blockers.append("recovered_pad_token_id_does_not_match_model_config")
    if root_has_unknown_body:
        blockers.append("AutoTokenizer_from_checkpoint_root_maps_SLICES_body_to_UNK")
    if snapshot.get("file_count") and not any(item in {"tokenizer.json", "vocab.txt", "tokenizer_config.json"} for item in (file["path"] for file in snapshot.get("files", []))):
        blockers.append("pinned_checkpoint_snapshot_does_not_contain_original_tokenizer_files")
    status = "BLOCKED" if blockers else "PASS"
    result = {
        "status": status,
        "stage": "A_CHECKPOINT_AND_TOKENIZER_AUDIT",
        "model_id": snapshot.get("repo_id"),
        "revision": snapshot.get("resolved_commit"),
        "checkpoint_sha256": _sha256(MODEL / "model.safetensors"),
        "config_sha256": _sha256(MODEL / "config.json"),
        "model_config": {key: config.get(key) for key in ("architectures", "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings", "pad_token_id", "transformers_version")},
        "parameter_count": parameter_count,
        "word_embedding_shape": word_embeddings["shape"],
        "word_embedding_row_count": embedding_rows,
        "recovered_tokenizer_sha256": _sha256(MODEL / "tokenizer/slice/tokenizer.json"),
        "recovered_tokenizer_size": recovered_vocab_size,
        "transformers_version_used_for_local_tokenizer_probe": auto_probe.get("transformers_version"),
        "mattext_slice_vocab_base_count": len(slice_tokens),
        "recovered_base_vocab_preserves_documented_slice_order": [token for token, _ in sorted(recovered_vocab.items(), key=lambda pair: pair[1])][:len(slice_tokens)] == slice_tokens,
        "recovered_special_ids": {key: recovered_vocab.get(key) for key in ("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]")},
        "recovered_tokenizer_config_specials": {key: tokenizer_config.get(key) for key in ("unk_token", "pad_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token")},
        "root_auto_tokenizer_probe": root_probe,
        "recovered_slice_auto_tokenizer_probe": recovered_probe,
        "download_manifest": snapshot,
        "blockers": blockers,
        "downstream_stages_executed": [],
        "test_targets_read": False,
        "interpretation": "The default AutoTokenizer load from the checkpoint directory is a 5-token fallback and maps all body tokens in the probe SLICES to UNK. The separately recovered MatText SLICES tokenizer has 156 entries while the learned checkpoint has 30,522 word-embedding rows; its pad ID also disagrees. Exact pretrained token-to-row identity remains unverified. Do not extract features or fit probes until the model's exact original tokenizer is recovered.",
        "python_version": sys.version,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TOKEN_OUT.mkdir(parents=True, exist_ok=True)
    (TOKEN_OUT / "compatibility_audit.json").write_text(json.dumps({key: result[key] for key in ("status", "revision", "word_embedding_shape", "word_embedding_row_count", "recovered_tokenizer_size", "mattext_slice_vocab_base_count", "recovered_base_vocab_preserves_documented_slice_order", "recovered_special_ids", "root_auto_tokenizer_probe", "recovered_slice_auto_tokenizer_probe", "blockers", "interpretation")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (TOKEN_OUT / "compatibility_audit.md").write_text(
        "# Auditoría de compatibilidad tokenizer–pesos\n\n"
        f"**Estado: {status}.** El checkpoint tiene {embedding_rows:,} filas en `embeddings.word_embeddings.weight`; el tokenizer SLICES recuperado tiene {recovered_vocab_size} IDs. "
        f"`AutoTokenizer` desde la raíz local: {root_probe.get('class', 'no cargó')}, vocabulario {root_probe.get('len_tokenizer', 'N/D')}, "
        f"UNKs del cuerpo SLICES diagnóstico {root_probe.get('unk_count_in_sample_body', 'N/D')}. `pad_token_id` modelo={config.get('pad_token_id')}, "
        f"tokenizer recuperado={recovered_vocab.get('[PAD]')}. Bloqueos: `{blockers}`.\n\n"
        "No se extrajeron features ni se leyeron etiquetas DFT. Recuperar el tokenizer original y demostrar la correspondencia exacta token→fila antes de continuar.\n",
        encoding="utf-8",
    )
    report_root = OUT.parent
    tracked = [
        ROOT / "Alternative2/scripts/MatText_frozen_1/config.yaml",
        ROOT / "Alternative2/scripts/MatText_frozen_1/run_experiment.py",
        Path(__file__).resolve(),
        OUT / "preflight.json",
        TOKEN_OUT / "compatibility_audit.json",
        TOKEN_OUT / "compatibility_audit.md",
        report_root / "MatText_scientific_report.md",
        report_root / "MatText_closure_log.md",
        MODEL / "config.json",
        MODEL / "model.safetensors",
        MODEL / "tokenizer/slice/tokenizer.json",
        MODEL / "vocab/slice_vocab.txt",
    ]
    manifest = {
        "experiment": "MatText_frozen_1",
        "status": status,
        "created_date": "2026-10-24",
        "model_id": snapshot.get("repo_id"),
        "revision": snapshot.get("resolved_commit"),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "parameter_count_from_tensor_shapes": parameter_count,
        "test_targets_read": False,
        "downstream_training_or_evaluation_run": False,
        "blockers": blockers,
        "artifact_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in tracked if path.is_file()},
    }
    (report_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("preflight status=%s blockers=%d report=%s", status, len(blockers), OUT / "preflight.json")
    return result


def main() -> None:
    """Run the hard-gated, labels-free Stage A audit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = audit()
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
