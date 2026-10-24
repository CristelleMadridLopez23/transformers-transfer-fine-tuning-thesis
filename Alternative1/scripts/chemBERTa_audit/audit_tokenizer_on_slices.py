#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría de ChemBERTa + SLICES (solo lectura; no modifica tokenizer/checkpoint).

Instalación mínima:
    pip install "transformers>=4.45" torch tokenizers safetensors
Opcional para Alexandria JSON grandes:
    pip install ijson
Opcional para convertir estructuras a SLICES:
    pip install pymatgen  # además de la implementación SLICES del proyecto

Ejemplos:
    python auditoria_chemberta_slices.py \
      --checkpoint model-pretrained/chemberta-77m-mlm \
      --input data/alexandria_slices.csv --column slices \
      --require-valid --out reports/audit

    python auditoria_chemberta_slices.py \
      --checkpoint model-pretrained/chemberta-77m-mlm \
      --tokenizer data/tokenizer_adaptado \
      --reference-tokenizer model-pretrained/chemberta-77m-mlm \
      --input data/slices.jsonl --out reports/adapted

    # Solo si el JSON tiene estructuras pymatgen y NO tiene SLICES:
    python auditoria_chemberta_slices.py \
      --checkpoint model-pretrained/chemberta-77m-mlm \
      --input data/alexandria_00000.json --generate-slices \
      --out reports/alexandria

Salida: summary.json, records.csv, failures.csv, collisions.csv,
        lexical_units.csv, new_tokens.csv, probes.json.
Código de salida: 0=sin bloqueos críticos; 1=con bloqueos; 2=error de ejecución.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PERIODIC = set((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf "
    "Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split())
PROBES = [
    "Ba S 0 1 o o o",
    "Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -",
    "0 1 9 10 11 99 100",
    "o o o + - o - + o",
    "C Cl Br Si Na Fe Co Ni",
    "Co CO co",
]
INPUT_EXT = (".csv", ".tsv", ".jsonl", ".ndjson", ".json", ".txt",
             ".csv.gz", ".jsonl.gz", ".json.gz", ".parquet")


def normalized(s):
    return " ".join(s.split())


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz") else open(path, "r", encoding="utf-8")


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    index = (len(s) - 1) * p / 100
    a, b = int(index), min(int(index) + 1, len(s) - 1)
    return round(s[a] + (s[b] - s[a]) * (index - a), 3)


def list_inputs(paths):
    files = []
    for inp in paths:
        p = Path(inp).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"No existe input: {p}")
        if p.is_dir():
            files.extend(x for x in p.rglob("*") if x.is_file() and x.name.lower().endswith(INPUT_EXT))
        else:
            files.append(p)
    return sorted(set(files))


def iter_json_records(path):
    # Streaming cuando ijson está disponible y el contenedor es un array.
    with open_text(path) as f:
        lead = f.read(65536).lstrip()
    try:
        import ijson
    except ImportError:
        ijson = None
    prefix = None
    if lead.startswith("["):
        prefix = "item"
    elif lead.startswith("{"):
        for key in ("entries", "records", "structures", "data", "materials", "items"):
            if re.search(r'"' + key + r'"\s*:\s*\[', lead):
                prefix = key + ".item"
                break
    if ijson is not None and prefix is not None:
        with open_text(path) as f:
            yield from ijson.items(f, prefix)
        return
    if path.stat().st_size > 100_000_000:
        print(f"[AVISO] JSON grande sin streaming: {path}. Instala ijson si falta.", file=sys.stderr)
    with open_text(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ("entries", "records", "structures", "data", "materials", "items"):
            if isinstance(obj.get(key), list):
                yield from obj[key]
                break
        else:
            yield obj


def iter_records(path):
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz", ".tsv")):
        with open_text(path) as f:
            delim = "\t" if name.endswith(".tsv") else ","
            yield from csv.DictReader(f, delimiter=delim)
    elif name.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
        with open_text(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif name.endswith((".json", ".json.gz")):
        yield from iter_json_records(path)
    elif name.endswith(".txt"):
        with open_text(path) as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    elif name.endswith(".parquet"):
        import pandas as pd
        for _, item in pd.read_parquet(path).iterrows():
            yield item.to_dict()
    else:
        raise ValueError(f"Extensión no soportada: {path}")


def dotted_get(row, dotted):
    val = row
    for part in dotted.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val


def extract_slices(row, column):
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None
    if column:
        val = dotted_get(row, column)
        return val if isinstance(val, str) else None
    names = ("slices", "SLICES", "slices_string", "slices_text",
             "slices_repr", "slices_representation", "slices_str")
    for obj in (row, row.get("data"), row.get("properties")):
        if isinstance(obj, dict):
            for key in names:
                if isinstance(obj.get(key), str):
                    return obj[key]
    return None


def is_true_flag(value):
    """Reconocer validaciones booleanas de CSV, Parquet y JSON, sin tratar None como True."""
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in ("true", "1", "t", "yes", "y", "si", "sí")
    return False


def validation_flags(row, basic_col, valid_col):
    """Obtener y evaluar ambas columnas; ausencias o valores no reconocidos fallan cerrado."""
    if not isinstance(row, dict):
        return None, None
    return dotted_get(row, basic_col), dotted_get(row, valid_col)


def identifier(row, fallback):
    if isinstance(row, dict):
        for obj in (row, row.get("data")):
            if isinstance(obj, dict):
                for key in ("mat_id", "material_id", "entry_id", "id", "task_id"):
                    if obj.get(key) is not None:
                        return str(obj[key])
    return str(fallback)


def make_slices(row, backend):
    from pymatgen.core import Structure
    if not isinstance(row, dict):
        raise ValueError("El registro no contiene una estructura de pymatgen")
    struct = row.get("structure") or (row.get("data") or {}).get("structure")
    if not isinstance(struct, dict):
        raise ValueError("No se encontró `structure` (dict pymatgen). Convierte la estructura primero.")
    return backend.structure2SLICES(Structure.from_dict(struct), strategy=4)


def unit_type(word):
    if word in PERIODIC:
        return "element"
    if word.isascii() and word.isdecimal():
        return "index"
    if word in ("o", "+", "-"):
        return "periodicity"
    return "other"


def analyze(tok, s, known_vocab):
    # Nunca truncar: queremos detectar qué se perdería en el pipeline.
    enc = tok(s, add_special_tokens=False, truncation=False,
              return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    covered = bytearray(len(s))
    for start, end in offsets:
        for pos in range(max(0, start), min(end, len(s))):
            covered[pos] = 1
    lost = [s[i] for i in range(len(s)) if not s[i].isspace() and not covered[i]]
    decoded = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    unknown = sum(int(i == tok.unk_token_id) for i in ids) if tok.unk_token_id is not None else 0

    units = []
    for match in re.finditer(r"\S+", s):
        w, start, end = match.group(0), *match.span()
        overlapping = [(a, b) for a, b in offsets if a < end and b > start]
        intact = len(overlapping) == 1 and overlapping[0][0] <= start and overlapping[0][1] >= end
        unit_chars_covered = all(covered[j] for j in range(start, end))
        units.append({"unit": w, "type": unit_type(w), "is_in_vocab": w in known_vocab,
                      "single_token": intact, "covered": unit_chars_covered,
                      "fragments": len(overlapping)})
    return {"ids": ids, "length": len(ids) + tok.num_special_tokens_to_add(pair=False),
            "unknown": unknown, "lost": "".join(lost), "roundtrip": decoded,
            "exact": decoded == s, "normalized_equal": normalized(decoded) == normalized(s),
            "units": units}


def file_fingerprints(folder):
    out = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt"):
        p = folder / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def verify_checkpoint(checkpoint, tok, do_load, examples):
    from transformers import AutoConfig, AutoModel
    import torch
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    cfg_pad = getattr(config, "pad_token_id", None)
    cfg_vocab = getattr(config, "vocab_size", None)
    maxpos = getattr(config, "max_position_embeddings", None)
    report = {
        "config_model_type": getattr(config, "model_type", None),
        "config_vocab_size": cfg_vocab, "tokenizer_size": len(tok),
        "config_pad_token_id": cfg_pad, "tokenizer_pad_token_id": tok.pad_token_id,
        "config_max_position_embeddings": maxpos, "tokenizer_model_max_length": tok.model_max_length,
        "weights_loaded": False,
        "warnings": [], "errors": []
    }
    if cfg_pad != tok.pad_token_id:
        report["errors"].append("El PAD ID de tokenizer y config NO coincide; afecta máscaras/posiciones.")
    if cfg_vocab is not None and len(tok) > cfg_vocab:
        report["errors"].append("El vocabulario extendido excede config.vocab_size; se requiere resize_token_embeddings y adaptación.")
    for name in ("pad_token", "unk_token", "mask_token", "cls_token", "sep_token"):
        if getattr(tok, name + "_id", None) is None:
            report["errors"].append(f"Falta token esencial: {name}")
    if not do_load:
        report["warnings"].append("--skip-weights activado: no se verificaron pesos del checkpoint")
        return report
    weight_files = list(checkpoint.glob("*.safetensors")) + list(checkpoint.glob("pytorch_model*.bin"))
    weight_files += list(checkpoint.glob("*.index.json"))
    if not weight_files:
        report["errors"].append("No hay archivos de pesos en el directorio (safetensors/bin/index).")
        return report
    try:
        kwargs = dict(local_files_only=True, output_loading_info=True)
        try:
            loaded = AutoModel.from_pretrained(str(checkpoint), **kwargs)
        except TypeError as e:
            if "output_loading_info" not in str(e):
                raise
            loaded = AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
        if isinstance(loaded, tuple):
            model, info = loaded
        else:
            model, info = loaded, {}
        model.eval()
        report["weights_loaded"] = True
        report["model_class"] = type(model).__name__
        report["input_embedding_rows"] = model.get_input_embeddings().num_embeddings
        report["embedding_dim"] = model.get_input_embeddings().embedding_dim
        report["missing_keys"] = list(info.get("missing_keys", []))[:60]
        report["unexpected_keys"] = list(info.get("unexpected_keys", []))[:60]
        critical_missing = [k for k in info.get("missing_keys", [])
                            if "word_embeddings" in k or "encoder.layer" in k or "encoder.layers" in k]
        if critical_missing:
            report["errors"].append("Faltan pesos esenciales del encoder: " + str(critical_missing[:8]))
        if model.get_input_embeddings().num_embeddings < len(tok):
            report["errors"].append("El encoder no tiene filas de embeddings suficientes para el tokenizer.")
        if examples:
            try:
                # Forward real con una muestra existente (sin truncar si cabe).
                candidates = [s for s in examples if s.strip()] + ["Ba S 0 1 o o o"]
                limit = tok.model_max_length
                if not isinstance(limit, int) or limit > 100000:
                    limit = 512
                example = next((s for s in candidates
                                if len(tok(s, truncation=False)["input_ids"]) <= limit), None)
                if example is None:
                    raise ValueError("No hay ejemplos cortos para el forward de comprobación")
                encoded = tok(example, return_tensors="pt", truncation=False)
                with torch.no_grad():
                    outputs = model(**encoded)
                report["forward_ok"] = bool(torch.isfinite(outputs.last_hidden_state).all().item())
                report["forward_shape"] = list(outputs.last_hidden_state.shape)
                if not report["forward_ok"]:
                    report["errors"].append("El forward produjo NaN o Inf")
            except Exception as e:
                report["forward_ok"] = False
                report["errors"].append(f"Fallo de forward con pesos reales: {type(e).__name__}: {e}")
        del model
    except Exception as e:
        report["errors"].append(f"No se pudieron cargar los pesos: {type(e).__name__}: {e}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Auditoría no destructiva ChemBERTa/SLICES")
    ap.add_argument("--checkpoint", required=True, help="Carpeta local del modelo con config y pesos")
    ap.add_argument("--tokenizer", help="Carpeta del tokenizador candidato; por defecto checkpoint")
    ap.add_argument("--reference-tokenizer", help="Carpeta del tokenizador original para comparar IDs")
    ap.add_argument("--input", action="append", required=True, help="Corpus o carpeta; admite varios --input")
    ap.add_argument("--column", default=None, help="Columna de SLICES; admite data.slices")
    ap.add_argument("--require-valid", action="store_true", help="Auditar solo filas cuyo slices_basic_valid y slices_valid sean ambos True")
    ap.add_argument("--basic-valid-column", default="slices_basic_valid", help="Nombre de la columna de validez básica")
    ap.add_argument("--valid-column", default="slices_valid", help="Nombre de la columna de validez completa")
    ap.add_argument("--out", required=True, help="Carpeta de resultados")
    ap.add_argument("--generate-slices", action="store_true", help="Generar SLICES de structure dict, estrategia 4")
    ap.add_argument("--check-slices", action="store_true", help="Usar check_SLICES_basic si está disponible")
    ap.add_argument("--skip-weights", action="store_true", help="Solo verifica config/tokenizer, NO checkpoint real")
    ap.add_argument("--limit", type=int, default=None, help="Auditar únicamente N registros (depuración)")
    ap.add_argument("--max-issues", type=int, default=200, help="Máximo de ejemplos en failures.csv")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    tok_dir = Path(args.tokenizer).expanduser().resolve() if args.tokenizer else checkpoint
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_dir() or not tok_dir.is_dir():
        ap.error("--checkpoint y --tokenizer deben existir localmente")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("La auditoría de spans necesita tokenizer fast (offset_mapping)")
    base = None
    if args.reference_tokenizer:
        base = AutoTokenizer.from_pretrained(args.reference_tokenizer, local_files_only=True, use_fast=True)
    vocab = tok.get_vocab()
    orig_vocab = base.get_vocab() if base else {}
    new_vocab = {w: i for w, i in vocab.items() if w not in orig_vocab} if base else {}
    changed_ids = [(w, i, vocab[w]) for w, i in orig_vocab.items() if w in vocab and i != vocab[w]]
    removed_tokens = [w for w in orig_vocab if w not in vocab]
    with open(out / "new_tokens.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["token", "new_id", "was_in_reference"])
        for w, i in sorted(new_vocab.items(), key=lambda x: x[1]):
            wr.writerow([w, i, False])

    backend = None
    if args.generate_slices or args.check_slices:
        try:
            from slices import SLICES
        except ImportError as e:
            raise RuntimeError("La opción --generate-slices/--check-slices requiere la implementación SLICES instalada") from e
        backend = SLICES(graph_method="econnn", relax_model=None)
    if args.check_slices and not callable(getattr(backend, "check_SLICES_basic", None)):
        raise RuntimeError("La implementación de SLICES no expone check_SLICES_basic")

    files = list_inputs(args.input)
    if not files:
        raise RuntimeError("No se encontraron archivos del corpus")
    lengths = []
    units_count = Counter()
    units_stats = {}
    seen_text = set()
    seen_ids = {}
    collisions = []
    probes = {}
    failures_count = Counter()
    examples = []
    n_rows = 0
    n_valid = 0
    n_rejected_basic = 0
    n_rejected_full = 0
    n_rejected_both = 0
    n_filtered_validation = 0
    n_no_slices = 0
    n_unknown = 0
    n_reconstruction = 0
    n_loss = 0
    n_overflow = 0
    n_check_fail = 0
    n_duplicates = 0
    n_generated = 0
    n_unk = 0
    per_record = out / "records.csv"
    failures_path = out / "failures.csv"
    with open(per_record, "w", newline="", encoding="utf-8") as f_rec, \
         open(failures_path, "w", newline="", encoding="utf-8") as f_fail:
        wr = csv.DictWriter(f_rec, fieldnames=["source", "id", "length_with_specials",
            "lexical_units", "unknown_ids", "lost_chars", "roundtrip_equal",
            "semantic_chars_equal", "truncated_if_model_limit", "other_units", "slices_parser_valid"])
        wf = csv.DictWriter(f_fail, fieldnames=["source", "id", "reason", "input", "decoded_or_detail"])
        wr.writeheader(); wf.writeheader()

        def issue(source, ident, reason, s, detail):
            failures_count[reason] += 1
            if sum(failures_count.values()) <= args.max_issues:
                wf.writerow({"source": source, "id": ident, "reason": reason,
                             "input": s[:2000], "decoded_or_detail": str(detail)[:2000]})

        for file in files:
            print(f"[CORPUS] {file}", flush=True)
            for local_idx, row in enumerate(iter_records(file), 1):
                if args.limit is not None and n_rows >= args.limit:
                    break
                n_rows += 1
                ident = identifier(row, f"{file.name}:{local_idx}")
                if args.require_valid:
                    basic_raw, full_raw = validation_flags(
                        row, args.basic_valid_column, args.valid_column
                    )
                    basic_ok = is_true_flag(basic_raw)
                    full_ok = is_true_flag(full_raw)
                    if not (basic_ok and full_ok):
                        n_filtered_validation += 1
                        n_rejected_basic += int(not basic_ok)
                        n_rejected_full += int(not full_ok)
                        n_rejected_both += int(not basic_ok and not full_ok)
                        continue
                s = extract_slices(row, args.column)
                if (not s or not s.strip()) and args.generate_slices:
                    try:
                        s = make_slices(row, backend)
                        n_generated += 1
                    except Exception as e:
                        issue(file.name, ident, "slices_generation_error", "", str(e))
                if not isinstance(s, str) or not s.strip():
                    n_no_slices += 1
                    issue(file.name, ident, "missing_slices", "", "No hay columna SLICES")
                    continue
                n_valid += 1
                s = s.strip()
                if len(examples) < 10:
                    examples.append(s)
                normalized_s = normalized(s)
                if normalized_s in seen_text:
                    n_duplicates += 1
                seen_text.add(normalized_s)
                try:
                    result = analyze(tok, s, vocab)
                except Exception as e:
                    issue(file.name, ident, "encoding_error", s, str(e))
                    continue
                lengths.append(result["length"])
                n_unk += result["unknown"]
                if result["unknown"]:
                    n_unknown += 1
                    issue(file.name, ident, "unknown_id", s, str(result["unknown"]))
                if result["lost"]:
                    n_loss += 1
                    issue(file.name, ident, "character_loss", s, "Caracteres perdidos: " + result["lost"])
                if not result["normalized_equal"]:
                    n_reconstruction += 1
                    issue(file.name, ident, "roundtrip_mismatch", s, result["roundtrip"])
                limit = tok.model_max_length
                if not isinstance(limit, int) or limit > 100000:
                    limit = 512
                overflow = result["length"] > limit
                if overflow:
                    n_overflow += 1
                    issue(file.name, ident, "length_exceeds_limit", s, f"{result['length']} > {limit}")
                key = tuple(result["ids"])
                old = seen_ids.get(key)
                if old is None:
                    seen_ids[key] = (normalized_s, ident)
                elif old[0] != normalized_s:
                    collisions.append({"id_first": old[1], "id_second": ident,
                                       "text_first": old[0], "text_second": normalized_s,
                                       "token_ids": " ".join(map(str, key))})
                    issue(file.name, ident, "token_id_collision", s, "Igual que " + old[1])
                other = 0
                for item in result["units"]:
                    u = item["unit"]
                    units_count[item["type"]] += 1
                    if item["type"] == "other":
                        other += 1
                    entry = units_stats.setdefault(u, {"type": item["type"], "occurrences": 0,
                        "single_token": 0, "covered": 0, "present_in_vocab": int(item["is_in_vocab"])})
                    entry["occurrences"] += 1
                    entry["single_token"] += int(item["single_token"])
                    entry["covered"] += int(item["covered"])
                parser_valid = "not_checked"
                if args.check_slices:
                    try:
                        parser_valid = bool(backend.check_SLICES_basic(s))
                    except Exception as e:
                        parser_valid = f"error: {e}"
                    if parser_valid is not True:
                        n_check_fail += 1
                        issue(file.name, ident, "slices_parser_failed", s, parser_valid)
                wr.writerow({"source": file.name, "id": ident, "length_with_specials": result["length"],
                    "lexical_units": len(result["units"]), "unknown_ids": result["unknown"],
                    "lost_chars": result["lost"][:100], "roundtrip_equal": result["exact"],
                    "semantic_chars_equal": result["normalized_equal"],
                    "truncated_if_model_limit": overflow, "other_units": other,
                    "slices_parser_valid": parser_valid})
                if n_rows % 5000 == 0:
                    print(f"  {n_rows:,} registros / {n_valid:,} con SLICES", flush=True)
            if args.limit is not None and n_rows >= args.limit:
                break

    if n_valid == 0:
        if args.require_valid:
            raise RuntimeError(
                "0 cadenas SLICES elegibles. Verifica --column y que "
                f"{args.basic_valid_column} / {args.valid_column} contengan True. "
                f"Filtradas: {n_filtered_validation} de {n_rows} filas."
            )
        raise RuntimeError("0 cadenas SLICES. Indica --column o utiliza --generate-slices con estructuras pymatgen.")

    for label, phrase in enumerate(PROBES):
        probes[f"probe_{label}"] = {"text": phrase, **{k: v for k, v in analyze(tok, phrase, vocab).items() if k != "units"}}
    write_json(out / "probes.json", probes)
    with open(out / "lexical_units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit", "type", "occurrences", "in_vocab", "covered_rate", "single_token_rate"])
        for word, u in sorted(units_stats.items(), key=lambda x: -x[1]["occurrences"]):
            n = u["occurrences"]
            w.writerow([word, u["type"], n, u["present_in_vocab"],
                        round(u["covered"] / n, 6), round(u["single_token"] / n, 6)])
    with open(out / "collisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id_first", "id_second", "text_first", "text_second", "token_ids"])
        w.writeheader(); w.writerows(collisions[:10000])
    checkpoint_report = verify_checkpoint(checkpoint, tok, not args.skip_weights, examples)
    noninvertible = n_loss > 0 or n_reconstruction > 0 or bool(collisions)
    block = (
        noninvertible or n_unk > 0 or n_overflow > 0 or n_no_slices > 0 or n_check_fail > 0
        or bool(changed_ids or removed_tokens) or bool(checkpoint_report["errors"])
        or not checkpoint_report["weights_loaded"]
    )
    summary = {
        "status": "BLOCKED" if block else "PASS",
        "input_files": [str(f) for f in files],
        "checkpoint": str(checkpoint), "tokenizer": str(tok_dir),
        "files_sha256": file_fingerprints(tok_dir),
        "records_seen": n_rows, "records_with_slices": n_valid,
        "validation_filter": {
            "enabled": args.require_valid,
            "basic_column": args.basic_valid_column,
            "full_column": args.valid_column,
            "rows_rejected_by_filter": n_filtered_validation,
            "rows_failing_basic": n_rejected_basic,
            "rows_failing_full": n_rejected_full,
            "rows_failing_both": n_rejected_both,
            "rows_eligible_before_slices_check": n_rows - n_filtered_validation,
        },
        "structures_converted_to_slices": n_generated, "missing_slices": n_no_slices,
        "unique_normalized_slices": len(seen_text), "duplicate_slices": n_duplicates,
        "sequences_with_unknown_ids": n_unknown, "unknown_token_occurrences": n_unk,
        "sequences_with_uncovered_characters": n_loss,
        "sequences_with_roundtrip_mismatch": n_reconstruction,
        "sequence_id_collisions": len(collisions),
        "sequences_over_tokenizer_max_length": n_overflow,
        "slices_parser_failures": n_check_fail if args.check_slices else None,
        "sequence_length_with_specials": {"median": percentile(lengths, 50), "p95": percentile(lengths, 95),
                "p99": percentile(lengths, 99), "max": max(lengths) if lengths else None},
        "lexical_types": dict(units_count), "unique_lexical_units": len(units_stats),
        "units_not_in_vocab": sorted(w for w, v in units_stats.items() if not v["present_in_vocab"])[:100],
        "reference_comparison": {"enabled": base is not None, "new_token_count": len(new_vocab),
                                 "changed_original_ids": changed_ids[:100], "removed_original_tokens": removed_tokens[:100]},
        "checkpoint_verification": checkpoint_report,
        "issues_by_type": dict(failures_count),
        "note": "Una coincidencia de IDs de tokens no implica estructura cristalina físicamente válida."
    }
    write_json(out / "summary.json", summary)
    print("\n========== AUDITORÍA ==========")
    print(f"Estado: {summary['status']}")
    print(f"Cadenas: {n_valid:,}; distintas: {len(seen_text):,}; duplicadas: {n_duplicates:,}")
    if args.require_valid:
        print(f"Filtro de validación: {n_filtered_validation:,} excluidas; {n_rows-n_filtered_validation:,} elegibles")
    print(f"Pérdida de caracteres: {n_loss:,}; roundtrip fallido: {n_reconstruction:,}; colisiones: {len(collisions):,}")
    print(f"UNK: {n_unk:,}; demasiado largas: {n_overflow:,}")
    print(f"Checkpoint cargado: {checkpoint_report['weights_loaded']}; problemas: {checkpoint_report['errors']}")
    print(f"Resultados: {out}")
    return 1 if block else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
