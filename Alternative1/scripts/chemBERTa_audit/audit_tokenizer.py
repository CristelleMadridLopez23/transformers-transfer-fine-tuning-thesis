#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_spaces(text):
    return " ".join(text.split())


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def load_samples(args):
    samples = [
        ("elementos", "C Cl Br Si Na Fe Co Ni"),
        ("mayusculas", "Co CO co"),
        ("indices", "0 1 9 10 11 99 100"),
        ("periodicidad", "o o o + - o - + o"),
        ("slices_simple", "Ba S 0 1 o o o"),
        (
            "slices_documentacion",
            "Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -",
        ),
        ("espacios_simples", "Ba S 0 1 o o o"),
        ("espacios_multiples", "Ba  S  0  1  o  o  o"),
    ]
    # Son pruebas de tokenización, no validaciones de estructuras SLICES.
    if args.csv:
        with Path(args.csv).open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if args.column not in (reader.fieldnames or []):
                raise ValueError(
                    f"No existe la columna {args.column!r}. "
                    f"Columnas: {reader.fieldnames}"
                )
            for index, row in enumerate(reader, start=2):
                text = row.get(args.column)
                if text is not None and text.strip():
                    samples.append((f"csv_fila_{index}", text))
    elif args.txt:
        with Path(args.txt).open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream, start=1):
                text = line.rstrip("\r\n")
                if text.strip():
                    samples.append((f"txt_linea_{index}", text))
    return samples


def describe_tokenizer(tokenizer):
    attributes = (
        "bos_token", "eos_token", "cls_token", "sep_token",
        "pad_token", "unk_token", "mask_token",
    )
    special = {
        name: {
            "token": getattr(tokenizer, name, None),
            "id": getattr(tokenizer, name + "_id", None),
        }
        for name in attributes
    }
    return {
        "class": type(tokenizer).__name__,
        "is_fast": tokenizer.is_fast,
        "vocab_size": tokenizer.vocab_size,
        "len_tokenizer": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "special_tokens": special,
        "special_tokens_map": tokenizer.special_tokens_map,
        "added_vocab": tokenizer.get_added_vocab(),
        "init_kwargs": tokenizer.init_kwargs,
    }


def inspect_sample(tokenizer, name, text):
    raw = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    options = {
        "add_special_tokens": True,
        "truncation": False,
        "padding": False,
        "return_special_tokens_mask": True,
    }
    if tokenizer.is_fast:
        options["return_offsets_mapping"] = True

    encoded = tokenizer(text, **options)
    decoded = tokenizer.decode(
        raw,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    unknown_count = (
        sum(value == tokenizer.unk_token_id for value in raw)
        if tokenizer.unk_token_id is not None else 0
    )

    return {
        "name": name,
        "text": text,
        "tokens_without_specials": tokenizer.convert_ids_to_tokens(raw),
        "ids_without_specials": raw,
        "tokens_with_specials": tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        ),
        "ids_with_specials": encoded["input_ids"],
        "special_tokens_mask": encoded["special_tokens_mask"],
        "offset_mapping": encoded.get("offset_mapping"),
        "attention_mask": encoded.get("attention_mask"),
        "decoded_without_specials": decoded,
        "roundtrip_exact": decoded == text,
        "roundtrip_ignoring_whitespace": (
            canonical_spaces(decoded) == canonical_spaces(text)
        ),
        "unknown_count": unknown_count,
        "raw_token_count": len(raw),
        "total_token_count": len(encoded["input_ids"]),
        "whitespace_units": len(text.split()),
        "tokens_per_whitespace_unit": len(raw) / max(1, len(text.split())),
    }


def summarize(rows, limit):
    lengths = [row["total_token_count"] for row in rows]
    raw_count = sum(row["raw_token_count"] for row in rows)
    unknown_count = sum(row["unknown_count"] for row in rows)
    return {
        "samples": len(rows),
        "unique_input_strings": len({row["text"] for row in rows}),
        "total_raw_tokens": raw_count,
        "unknown_tokens": unknown_count,
        "unknown_fraction": unknown_count / max(1, raw_count),
        "samples_with_unknown": sum(
            row["unknown_count"] > 0 for row in rows
        ),
        "exact_roundtrip_failures": sum(
            not row["roundtrip_exact"] for row in rows
        ),
        "roundtrip_failures_ignoring_whitespace": sum(
            not row["roundtrip_ignoring_whitespace"] for row in rows
        ),
        "length_min": min(lengths),
        "length_median": statistics.median(lengths),
        "length_p95": percentile(lengths, 0.95),
        "length_max": max(lengths),
        "evaluation_limit_including_special_tokens": limit,
        "samples_exceeding_limit": sum(length > limit for length in lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelo",
        default="model-pretrained/chemberta-77m-mlm",
    )
    parser.add_argument(
        "--salida",
        default="reports/chemberta_tokenizer_audit",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", help="CSV con una columna de cadenas SLICES")
    source.add_argument("--txt", help="TXT con una cadena por línea")
    parser.add_argument("--column", default="slices")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Umbral de evaluación; no activa truncamiento",
    )
    args = parser.parse_args()

    model_dir = Path(args.modelo).expanduser().resolve()
    output = Path(args.salida).expanduser().resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if output == model_dir or model_dir in output.parents:
        raise ValueError(
            "La salida debe estar fuera de la carpeta original del modelo."
        )
    output.mkdir(parents=True, exist_ok=True)

    versions = {
        package: importlib.metadata.version(package)
        for package in ("transformers", "tokenizers")
    }
    write_json(output / "versions.json", versions)

    config_path = model_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file() else {}
    )

    file_names = (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json",
        "vocab.json", "vocab.txt", "merges.txt",
    )
    files = []
    for name in file_names:
        path = model_dir / name
        record = {"name": name, "exists": path.is_file()}
        if path.is_file():
            record.update(
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
            if name == "merges.txt":
                lines = path.read_text(encoding="utf-8").splitlines()
                record["noncomment_merge_lines"] = sum(
                    bool(line.strip()) and not line.startswith("#")
                    for line in lines
                )
        files.append(record)
    write_json(output / "source_files.json", files)

    tokenizers = {}
    loading = {}

    for label, use_fast in (("fast", True), ("slow", False)):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=use_fast,
            )
            tokenizers[label] = tokenizer
            loading[label] = {
                "loaded": True,
                "requested_use_fast": use_fast,
                "actual_class": type(tokenizer).__name__,
                "actual_is_fast": tokenizer.is_fast,
            }
        except Exception as exc:
            loading[label] = {
                "loaded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    write_json(output / "loading.json", loading)
    if not tokenizers:
        raise RuntimeError(
            f"No se pudo cargar ningún tokenizador. Consulta {output / 'loading.json'}"
        )

    samples = load_samples(args)
    results = {}
    metadata = {}
    summaries = {}

    for label, tokenizer in tokenizers.items():
        metadata[label] = describe_tokenizer(tokenizer)
        vocab = tokenizer.get_vocab()

        with (output / f"vocabulary_{label}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "token", "is_special"])
            special_ids = set(tokenizer.all_special_ids)
            for token, token_id in sorted(
                vocab.items(), key=lambda item: (item[1], item[0])
            ):
                writer.writerow([token_id, token, token_id in special_ids])

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            backend_json = json.loads(backend.to_str())
            write_json(output / f"backend_{label}.json", backend_json)
            model = backend_json.get("model") or {}
            pipeline = {
                key: backend_json.get(key)
                for key in (
                    "normalizer", "pre_tokenizer",
                    "post_processor", "decoder",
                    "added_tokens", "truncation", "padding",
                )
            }
            pipeline["model"] = {
                key: value for key, value in model.items()
                if key not in {"vocab", "merges"}
            }
            pipeline["model"]["merge_count"] = len(model.get("merges") or [])
            write_json(output / f"pipeline_{label}.json", pipeline)

        rows = [
            inspect_sample(tokenizer, name, text)
            for name, text in samples
        ]
        results[label] = rows
        write_json(output / f"samples_{label}.json", rows)

        limit = args.max_length
        if limit is None:
            declared = tokenizer.model_max_length
            if isinstance(declared, int) and 0 < declared < 1_000_000:
                limit = declared
            else:
                raise ValueError(
                    "El tokenizador no declara un contexto finito útil. "
                    "Vuelve a ejecutar con --max-length N."
                )
        if limit <= 0:
            raise ValueError("--max-length debe ser positivo.")

        # No se resta automáticamente nada a max_position_embeddings:
        # su interpretación depende de la arquitectura del modelo.
        all_summary = summarize(rows, limit)
        corpus_rows = [
            row for row in rows
            if row["name"].startswith(("csv_", "txt_"))
        ]
        summaries[label] = {
            "all_samples": all_summary,
            "user_corpus": (
                summarize(corpus_rows, limit) if corpus_rows else None
            ),
        }

        id_groups = defaultdict(list)
        for row in rows:
            id_groups[tuple(row["ids_without_specials"])].append(row)

        collisions = []
        for ids, group in id_groups.items():
            texts = sorted({row["text"] for row in group})
            if len(texts) > 1:
                collisions.append(
                    {
                        "ids": list(ids),
                        "texts": texts,
                        "whitespace_only_difference": (
                            len({canonical_spaces(text) for text in texts}) == 1
                        ),
                    }
                )
        write_json(output / f"collisions_{label}.json", collisions)

        padding_probe = {"tested": False}
        if tokenizer.pad_token_id is not None:
            try:
                batch = tokenizer(
                    ["Ba S", "Ba S 0 1 o o o"],
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                )
                padding_probe = {
                    "tested": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "encoding": dict(batch),
                }
            except Exception as exc:
                padding_probe["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / f"padding_{label}.json", padding_probe)

    write_json(output / "tokenizer_metadata.json", metadata)
    write_json(output / "summary.json", summaries)

    comparison = {
        "both_loaded": "fast" in results and "slow" in results,
        "distinct_fast_and_slow_implementations": (
            "fast" in tokenizers
            and "slow" in tokenizers
            and tokenizers["fast"].is_fast
            and not tokenizers["slow"].is_fast
        ),
        "differences": [],
    }
    if comparison["both_loaded"]:
        for fast_row, slow_row in zip(results["fast"], results["slow"]):
            if fast_row["ids_with_specials"] != slow_row["ids_with_specials"]:
                comparison["differences"].append(
                    {
                        "name": fast_row["name"],
                        "text": fast_row["text"],
                        "fast_ids": fast_row["ids_with_specials"],
                        "slow_ids": slow_row["ids_with_specials"],
                    }
                )
    write_json(output / "fast_slow_comparison.json", comparison)

    model_checks = {}
    configured_vocab = config.get("vocab_size")
    for label, tokenizer in tokenizers.items():
        ids = list(tokenizer.get_vocab().values())
        model_checks[label] = {
            "config_vocab_size": configured_vocab,
            "maximum_token_id": max(ids) if ids else None,
            "ids_outside_config_vocab": (
                sorted({value for value in ids if value >= configured_vocab})
                if isinstance(configured_vocab, int) else None
            ),
            "config_pad_token_id": config.get("pad_token_id"),
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "config_max_position_embeddings": config.get(
                "max_position_embeddings"
            ),
            "weights_loaded": False,
        }
    write_json(output / "model_config_checks.json", model_checks)

    print(f"\nAuditoría guardada en: {output}")
    print("\nCarga:")
    print(json.dumps(loading, indent=2, ensure_ascii=False))

    print("\nResumen:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))

    print(
        "\nDiferencias fast/slow:",
        len(comparison["differences"]),
    )
    print(
        "\nNo se modificaron vocabularios, tokens especiales ni pesos. "
        "Esta auditoría no valida la sintaxis ni la geometría SLICES."
    )


if __name__ == "__main__":
    main()