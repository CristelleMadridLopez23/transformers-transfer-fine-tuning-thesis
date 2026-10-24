"""Audit whether the recovered MatText-SLICES tokenizer indexes checkpoint embeddings."""
from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "Alternative2/models/MatText-2M-SLICES"
REPORT = ROOT / "Alternative2/reports/MatText/tokenizer_weights_audit"
logger = logging.getLogger("mattext_tokenizer_weights_audit")


def sha256(path: Path) -> str:
    """Return SHA-256 for a file, streamed in fixed-size blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only the safetensors JSON header, without loading model tensors."""
    with path.open("rb") as stream:
        header_length_bytes = stream.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(stream.read(header_length))
    return header


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-object roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit() -> dict[str, Any]:
    """Compare recovered tokenizer dimensions and special IDs with learned weights."""
    config = _read_json(MODEL / "config.json")
    tokenizer = _read_json(MODEL / "tokenizer/slice/tokenizer.json")
    tokenizer_config = _read_json(MODEL / "tokenizer/slice/tokenizer_config.json")
    tokenizer_vocab = tokenizer["model"]["vocab"]
    header = read_safetensors_header(MODEL / "model.safetensors")
    embedding = header.get("embeddings.word_embeddings.weight")
    if not embedding:
        raise KeyError("embeddings.word_embeddings.weight is absent from checkpoint")

    slice_vocab = (MODEL / "vocab/slice_vocab.txt").read_text(encoding="utf-8").splitlines()
    ordered_tokens = [token for token, _ in sorted(tokenizer_vocab.items(), key=lambda item: item[1])]
    recovered_prefix_matches_slice = ordered_tokens[: len(slice_vocab)] == slice_vocab
    recovered_specials = {name: tokenizer_vocab.get(name) for name in ("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]")}
    model_rows = int(embedding["shape"][0])
    model_hidden = int(embedding["shape"][1])
    tokenizer_size = len(tokenizer_vocab)
    official_slice_default_size = len(slice_vocab) + 2  # [CLS] and [SEP] per MatText tokenizer docs.
    snapshot_manifest = _read_json(ROOT / "Alternative2/models/MatText-2M-SLICES.download-manifest.json")
    downloaded_paths = {entry["path"] for entry in snapshot_manifest.get("files", [])}
    missing_tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt") if name not in downloaded_paths]

    findings = {
        "model_id": snapshot_manifest.get("repo_id"),
        "resolved_revision": snapshot_manifest.get("resolved_commit"),
        "checkpoint_sha256": sha256(MODEL / "model.safetensors"),
        "config_sha256": sha256(MODEL / "config.json"),
        "recovered_tokenizer_json_sha256": sha256(MODEL / "tokenizer/slice/tokenizer.json"),
        "recovered_slice_vocab_sha256": sha256(MODEL / "vocab/slice_vocab.txt"),
        "model_config_vocab_size": int(config["vocab_size"]),
        "actual_word_embedding_shape": embedding["shape"],
        "recovered_tokenizer_vocab_size": tokenizer_size,
        "slice_vocab_base_size": len(slice_vocab),
        "documented_mattext_slice_default_vocab_size": official_slice_default_size,
        "recovered_vocab_matches_slice_vocab_order_for_base_tokens": recovered_prefix_matches_slice,
        "recovered_special_token_ids": recovered_specials,
        "model_pad_token_id": config.get("pad_token_id"),
        "tokenizer_pad_token_id": recovered_specials["[PAD]"],
        "tokenizer_config_special_ids": {"pad": tokenizer_config.get("pad_token"), "cls": tokenizer_config.get("cls_token"), "sep": tokenizer_config.get("sep_token"), "mask": tokenizer_config.get("mask_token"), "bos": tokenizer_config.get("bos_token"), "eos": tokenizer_config.get("eos_token")},
        "downloaded_snapshot_files": sorted(downloaded_paths),
        "tokenizer_artifacts_absent_from_checkpoint_snapshot": missing_tokenizer_files,
        "checkpoint_vocab_rows_match_recovered_tokenizer": model_rows == tokenizer_size,
        "checkpoint_vocab_rows_match_model_config": model_rows == int(config["vocab_size"]),
        "conclusion": "INCOMPATIBLE_DIMENSIONS; exact learned token-to-row map cannot be established from the local checkpoint snapshot",
        "interpretation": "The recovered SLICES token strings largely reproduce MatText's documented compact vocabulary, but its 156 IDs cannot directly index the checkpoint's 30,522 learned word-embedding rows as the original tokenizer without a verified ID-to-row mapping. The checkpoint snapshot omits tokenizer artifacts; the model config and tensor shape agree with 30,522, not 156.",
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "tokenizer_weights_audit.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [
        ("Checkpoint vocabulary (config)", findings["model_config_vocab_size"]),
        ("Checkpoint word-embedding rows (tensor)", model_rows),
        ("Recovered tokenizer.json vocabulary", tokenizer_size),
        ("MatText documented SLICES vocab + CLS/SEP", official_slice_default_size),
        ("Recovered PAD ID / checkpoint PAD ID", f"{recovered_specials['[PAD]']} / {config.get('pad_token_id')}"),
    ]
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    report = f'''# Auditoría de compatibilidad entre tokenizer y pesos de MatText\n\n**Dictamen: INCOMPATIBLE por tamaño de vocabulario; correspondencia exacta de IDs no verificable.**\n\nEl checkpoint local es `{findings['model_id']}` en la revisión `{findings['resolved_revision']}`. Se leyó el header de safetensors sin cargar ni alterar los pesos. Las formas de configuración y tensor son coherentes entre sí, pero no con el tokenizer recuperado:\n\n| Comprobación | Valor |\n|---|---:|\n{table}\n\nEl tokenizer recuperado conserva en orden los {len(slice_vocab)} tokens base de `slice_vocab.txt` ({'sí' if recovered_prefix_matches_slice else 'no'}). Esto concuerda con la documentación oficial de `SliceTokenizer`, que muestra esos tokens y añade `[CLS]` y `[SEP]` como IDs 149 y 150. Sin embargo, el `tokenizer.json` recuperado agrega siete especiales: `{recovered_specials}`. Su `[PAD]` es ID 150, mientras que el config del modelo fija `pad_token_id=0`.\n\nLa revisión descargada de Hugging Face contiene {len(downloaded_paths)} archivos y no incluye `tokenizer.json`, `tokenizer_config.json` ni `vocab.txt`. El config y los pesos sí declaran/contienen 30,522 filas de embedding. Por eso el tokenizador compacto puede ser un tokenizador válido para el framework MatText, pero **no se puede emparejar con seguridad con estas filas aprendidas**. Usar sus IDs con el modelo daría tensores dentro del rango, pero no demuestra que, por ejemplo, el ID de `Ga` seleccione la fila que aprendió para `Ga`; además el padding especial difiere.\n\n## Resultado científico\n\nLa verificación previa del tokenizador recuperado **no queda validada contra estos pesos**. La forma de embedding demuestra que no es el mismo vocabulario de 156 filas. La hipótesis más probable es que este checkpoint espera un vocabulario BERT de 30,522 entradas, pero los artefactos locales no incluyen el archivo de vocabulario necesario para demostrar los IDs exactos. No debe usarse el tokenizer recuperado para extraer embeddings/finetunear este checkpoint hasta recuperar la versión exacta del tokenizer usada durante el pretraining o verificar contra un ejemplo de inferencia oficial que exponga IDs/tokens.\n\n## Procedencia y huellas\n\n- Checkpoint SHA-256: `{findings['checkpoint_sha256']}`\n- Config SHA-256: `{findings['config_sha256']}`\n- Tokenizer recuperado SHA-256: `{findings['recovered_tokenizer_json_sha256']}`\n- `slice_vocab.txt` SHA-256: `{findings['recovered_slice_vocab_sha256']}`\n- Pesos: tensor `embeddings.word_embeddings.weight`, forma `{embedding['shape']}`.\n- Archivos tokenizador ausentes del snapshot: `{missing_tokenizer_files}`.\n\n## Referencias\n\n- Documentación oficial de [MatText tokenizers](https://lamalab-org.github.io/MatText/docs/tokenizers/): vocabulario `SliceTokenizer`, `[CLS]` y `[SEP]`.\n- [Repositorio MatText](https://github.com/lamalab-org/MatText).\n- [Checkpoint MatText-slices-2m y ejemplo AutoTokenizer](https://huggingface.co/n0w0f/MatText-slices-2m).\n- Alampara et al., [MatText, arXiv:2406.17295](https://arxiv.org/abs/2406.17295).\n'''
    (REPORT / "tokenizer_weights_audit.md").write_text(report, encoding="utf-8")
    logger.info("tokenizer_weights_audit_complete conclusion=%s model_rows=%d recovered_vocab=%d report=%s", findings["conclusion"], model_rows, tokenizer_size, REPORT / "tokenizer_weights_audit.md")
    return findings


def main() -> None:
    """Run the static audit and save JSON and Markdown results."""
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","message":"%(message)s"}')
    audit()


if __name__ == "__main__":
    main()
