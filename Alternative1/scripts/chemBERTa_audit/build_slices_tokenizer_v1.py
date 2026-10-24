#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye y audita un tokenizer SLICES v1 compatible por IDs con ChemBERTa.

Uso (desde la raiz del repositorio):
 python scripts/chemBERTa_audit/build_slices_tokenizer_v1.py \
   --checkpoint model-pretrained/chemberta-77m-mlm \
   --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
   --column slices --require-valid --out reports/slices_tokenizer_v1

Usar --prepare-only para inspeccionar vocabulario/gramatica sin librerias de Hugging Face.
Usar --smoke-model para probar la carga de pesos, resize y forward de MLM sin entrenar.
No se sobrescribe NUNCA el checkpoint fuente ni se truncan secuencias silenciosamente.
Requisitos para construcción: tokenizers, transformers. Smoke: tambien torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ELEMS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split())
EDGE_INDEX = re.compile(r"[0-9]+\Z")
JIMAGE = re.compile(r"[o+\-]{3}\Z")
TRUTHY = {"true", "1", "yes", "y", "t"}
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"]


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def percentiles(xs):
    if not xs:
        return None
    data = sorted(xs)
    return {"min": data[0], "median": median(data), "p95": data[math.ceil(.95 * len(data))-1],
            "p99": data[math.ceil(.99 * len(data))-1], "max": data[-1]}


def norm(text):
    # La normalizacion autorizada es SOLO de espacios, NO de mayusculas o simbolos.
    return " ".join(text.split())


def parse_slices_strategy4_heuristic(units):
    """Localiza de derecha a izquierda (i, j, jimage), luego especies y cabecera.

    NO reemplaza SLICES.check_SLICES / parser oficial. Es una comprobacion sintactica.
    """
    stop = len(units)
    edges = []
    while (stop >= 3 and EDGE_INDEX.fullmatch(units[stop - 3])
           and EDGE_INDEX.fullmatch(units[stop - 2])
           and JIMAGE.fullmatch(units[stop - 1])):
        edges.append((int(units[stop-3]), int(units[stop-2]), units[stop-1]))
        stop -= 3
    edges.reverse()
    start = stop
    while start > 0 and units[start-1] in ELEMS:
        start -= 1
    header, species = units[:start], units[start:stop]
    flags = []
    if not edges:
        flags.append("no_edge_triplets")
    if not species:
        flags.append("no_species")
    if edges and species and any(max(a,b) >= len(species) for a,b,_ in edges):
        flags.append("edge_index_out_of_species_range")
    return header, species, edges, flags


def load_checkpoint_vocab(ckpt: Path):
    """Extrae mapeo completo de IDs, incluidos especiales anadidos (591 y 592)."""
    tj = ckpt / "tokenizer.json"
    if tj.is_file():
        obj = json.loads(tj.read_text(encoding="utf-8"))
        vocab = dict(obj["model"]["vocab"])
        for item in obj.get("added_tokens", []):
            name, idx = item["content"], int(item["id"])
            if name in vocab and vocab[name] != idx:
                raise ValueError(f"ID contradictorio para token {name!r}")
            vocab[name] = idx
    else:
        vj = ckpt / "vocab.json"
        if not vj.is_file():
            raise FileNotFoundError(f"No encuentro tokenizer.json ni vocab.json en {ckpt}")
        vocab = json.loads(vj.read_text(encoding="utf-8"))
        aj = ckpt / "added_tokens.json"
        if aj.is_file():
            vocab.update(json.loads(aj.read_text(encoding="utf-8")))
    ids = list(vocab.values())
    if len(set(ids)) != len(ids):
        raise ValueError("IDs originales duplicados: el vocabulario de origen es inconsistente")
    for special in SPECIALS[:5]:
        if special not in vocab:
            raise ValueError(f"Falta token especial obligatorio {special}")
    cp = ckpt / "config.json"
    if not cp.is_file():
        raise FileNotFoundError(f"Falta config.json en {ckpt}")
    config = json.loads(cp.read_text(encoding="utf-8"))
    size = int(config["vocab_size"])
    if max(ids) >= size:
        raise ValueError(f"max ID {max(ids)} excede vocab_size config={size}")
    return {str(k): int(v) for k,v in vocab.items()}, config


def read_and_profile(path: Path, text_col, require_valid, max_rows):
    total = 0
    skipped = Counter()
    counts = Counter()
    role_counts = defaultdict(Counter)
    headers = Counter()
    prefixes = Counter()
    max_index = Counter()
    jimages = Counter()
    parse_flags = Counter()
    lengths = []
    records = []
    seen_ids = {}
    seen_strings = {}
    duplicates = Counter()
    collision_by_material = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if text_col not in cols:
            raise ValueError(f"No existe columna {text_col!r}; columnas: {cols}")
        if require_valid and not {"slices_basic_valid", "slices_valid"}.issubset(cols):
            raise ValueError("--require-valid exige columnas slices_basic_valid y slices_valid")
        idcol = next((x for x in ("mat_id", "material_id", "entry_id", "id") if x in cols), None)
        compcol = next((x for x in ("formula_pretty", "reduced_formula", "formula", "composition") if x in cols), None)
        protcol = next((x for x in ("prototype_id", "prototype") if x in cols), None)
        composition_groups, prototype_groups = Counter(), Counter()
        for line, row in enumerate(reader, 2):
            if max_rows and total >= max_rows:
                break
            total += 1
            if require_valid:
                if str(row["slices_basic_valid"]).strip().lower() not in TRUTHY:
                    skipped["invalid_basic"] += 1
                    continue
                if str(row["slices_valid"]).strip().lower() not in TRUTHY:
                    skipped["invalid_full"] += 1
                    continue
            text = norm(row.get(text_col, "") or "")
            if not text:
                skipped["empty"] += 1
                continue
            name = str(row.get(idcol) or f"line_{line}").strip() if idcol else f"line_{line}"
            units = text.split()
            counts.update(units)
            header, species, edges, flags = parse_slices_strategy4_heuristic(units)
            role_counts["header"].update(header)
            role_counts["species"].update(species)
            role_counts["edge_index"].update(str(n) for a,b,_ in edges for n in (a,b))
            role_counts["periodicity"].update(offset for _,_,offset in edges)
            for flag in flags:
                parse_flags[flag] += 1
            headers[" ".join(header)] += 1
            prefixes[len(header)] += 1
            jimages.update(offset for _,_,offset in edges)
            if edges:
                max_index[max(max(a,b) for a,b,_ in edges)] += 1
            lengths.append(len(units)+2)
            if name in seen_ids:
                duplicates["repeated_mat_id_rows"] += 1
            seen_ids[name] = 1
            if text in seen_strings:
                duplicates["repeated_normalized_slices_rows"] += 1
            seen_strings[text] = 1
            if compcol and row.get(compcol):
                composition_groups[row[compcol]] += 1
            if protcol and row.get(protcol):
                prototype_groups[row[protcol]] += 1
            records.append((name, text, units, len(units)+2))
            if len(records) % 10000 == 0:
                print(f"[PERFIL] {len(records):,} SLICES elegibles", flush=True)
    dup_report = {
        **dict(duplicates),
        "same_composition_groups": sum(x>1 for x in composition_groups.values()),
        "same_prototype_groups": sum(x>1 for x in prototype_groups.values()),
        "structural_duplicates_verified": False,
        "note": "Grupos por composicion/prototipo son candidatos; NO prueba de equivalencia geometrica."
    }
    profile = {
        "rows_total": total, "rows_valid_nonempty": len(records), "excluded": dict(skipped),
        "unique_lexical_units": len(counts), "lengths_1_unit_1_token_plus_2": percentiles(lengths),
        "over_512_lexical": sum(n > 512 for n in lengths),
        "syntactic_flags_heuristic": dict(parse_flags), "unique_headers": len(headers),
        "max_edge_index_observed": max(max_index, default=None),
        "jimage_triplets_unique": len(jimages), "duplication": dup_report,
        "grammar_validation": "HEURISTIC_ONLY_NOT_OFFICIAL"
    }
    return records, counts, role_counts, headers, max_index, jimages, profile


def complete_vocab(original: dict[str,int], model_rows: int, lexical_units):
    vocab = dict(original)
    fillers = []
    existing_ids = set(vocab.values())
    for idx in range(model_rows):
        if idx not in existing_ids:
            name = f"[ORIGINAL_UNUSED_EMBED_ROW_{idx}]"
            if name in vocab:
                raise ValueError("Colision en nombre de token reservado")
            vocab[name] = idx
            fillers.append((name, idx))
    next_id = model_rows
    added = []
    for unit in sorted(lexical_units):
        if unit not in vocab:
            vocab[unit] = next_id
            added.append((unit, next_id))
            next_id += 1
    ids = sorted(vocab.values())
    if ids != list(range(len(ids))):
        raise AssertionError("Vocabulario del candidato no es contiguo")
    for name, index in original.items():
        if vocab[name] != index:
            raise AssertionError(f"Se reasigno el ID heredado de {name}")
    return vocab, added, fillers


def build_hf_tokenizer(vocab, out_dir, limit):
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:
        raise RuntimeError("Para construir instala 'tokenizers' y 'transformers' en el entorno de tesis") from e
    if any(s not in vocab for s in SPECIALS):
        raise RuntimeError("Faltan tokens especiales en vocabulario heredado")
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    # Con WordPiece cleanup=False decodifica las unidades con espacios, sin alterar signos.
    backend.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    backend.add_special_tokens(SPECIALS)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        bos_token="<s>", eos_token="</s>",
        model_max_length=limit, clean_up_tokenization_spaces=False
    )
    new_vocab = fast.get_vocab()
    if len(fast) != len(vocab) or any(new_vocab.get(k) != v for k,v in vocab.items()):
        raise RuntimeError("Al crear PreTrainedTokenizerFast cambiaron IDs o longitud de vocabulario")
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    # Confirmar que otro proceso/AutoTokenizer pueda recargar artefactos desde disco.
    from transformers import AutoTokenizer
    reloaded = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True, local_files_only=True)
    if len(reloaded) != len(vocab) or any(reloaded.get_vocab().get(k) != v for k,v in vocab.items()):
        raise RuntimeError("Reapertura de candidate_tokenizer altera IDs o tamaño")
    return reloaded


def evaluate_candidate(tok, records, limit, out):
    collisions = {}
    collisions_examples = []
    errors = []
    oversize = []
    counters = Counter()
    for i, (name, text, units, lexical_length) in enumerate(records, 1):
        seq_ids = tok.encode(text, add_special_tokens=False, truncation=False)
        decoded_units = tok.convert_ids_to_tokens(seq_ids)
        decoded_string = tok.decode(seq_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        if decoded_units != units:
            counters["mismatch_units"] += 1
            if len(errors) < 50:
                errors.append({"mat_id": name, "type": "lexical_unit_mismatch",
                               "input_prefix": text[:350], "tokens_prefix": " ".join(decoded_units[:100])})
        if norm(decoded_string) != text:
            counters["decode_mismatch"] += 1
            if len(errors) < 50:
                errors.append({"mat_id": name, "type": "decode_mismatch",
                               "input_prefix": text[:350], "tokens_prefix": decoded_string[:350]})
        if tok.unk_token_id in seq_ids:
            counters["unknown_sequences"] += 1
        if len(seq_ids)+2 > limit:
            counters["exceed_limit"] += 1
            oversize.append({"mat_id":name, "length":len(seq_ids)+2,
                             "unit_count":len(units)})
        h = hashlib.sha256(",".join(map(str,seq_ids)).encode()).hexdigest()
        first = collisions.get(h)
        if first and first[1] != text:
            counters["collision_extra_rows"] += 1
            if len(collisions_examples) < 100:
                collisions_examples.append({"mat_id_a":first[0], "mat_id_b":name,
                                            "text_a":first[1][:1000], "text_b":text[:1000]})
        elif first is None:
            collisions[h] = (name, text)
        if i % 10000 == 0:
            print(f"[VALIDACION] {i:,}/{len(records):,}", flush=True)
    save_csv(out / "lexical_errors.csv", ["mat_id","type","input_prefix","tokens_prefix"], errors)
    save_csv(out / "collision_examples.csv", ["mat_id_a","mat_id_b","text_a","text_b"], collisions_examples)
    save_csv(out / "long_sequences.csv", ["mat_id","length","unit_count"], oversize)
    return {
        "tested": len(records), "lexical_unit_errors": counters["mismatch_units"],
        "roundtrip_normalized_errors": counters["decode_mismatch"],
        "unknown_sequences": counters["unknown_sequences"],
        "collision_extra_rows": counters["collision_extra_rows"],
        "sequences_over_limit": counters["exceed_limit"],
        "lossless_lexically": all(counters[x] == 0 for x in
                                 ["mismatch_units", "decode_mismatch", "unknown_sequences", "collision_extra_rows"])
    }


def smoke_model(checkpoint, tok, records):
    """No guarda pesos. Usa posiciones explicitas: NO resuelve discrepancia PAD para entrenamiento."""
    try:
        import torch
        from transformers import AutoModelForMaskedLM
    except ImportError as e:
        raise RuntimeError("--smoke-model requiere torch, transformers") from e
    model = AutoModelForMaskedLM.from_pretrained(str(checkpoint), local_files_only=True)
    embedding = model.get_input_embeddings()
    before = embedding.weight.detach().cpu().clone()
    original_rows = before.size(0)
    try:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(len(tok))
    after = model.get_input_embeddings().weight.detach().cpu()
    original_exact = bool(torch.equal(before, after[:original_rows]))
    shorts = [s for _,s,_,n in records if n < 80][:2]
    if not shorts:
        return {"weights_loaded": True, "original_rows": original_rows,
                "new_rows":len(tok), "old_weights_bitwise_preserved":original_exact,
                "forward": "SKIPPED_NO_SHORT_EXAMPLES"}
    model.eval()
    batch = tok(shorts, padding=True, truncation=False, return_tensors="pt")
    mask = batch["attention_mask"]
    model_pad = int(model.config.pad_token_id)
    pos = mask.long().cumsum(dim=1) * mask.long() + model_pad
    max_pos = int(model.config.max_position_embeddings)
    if int(pos.max()) >= max_pos:
        raise ValueError(f"position_ids exceden max_position_embeddings ({max_pos})")
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=mask,
                       position_ids=pos).logits
    return {"weights_loaded": True, "original_rows": original_rows,
            "new_rows":len(tok), "old_weights_bitwise_preserved":original_exact,
            "forward": "OK" if bool(torch.isfinite(logits).all()) else "NONFINITE",
            "shapes": list(logits.shape),
            "warning": "Se usaron position_ids explícitos; esto no corrige permanentemente PAD desalineado."}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--column", default="slices")
    ap.add_argument("--require-valid", action="store_true")
    ap.add_argument("--out", default="reports/slices_tokenizer_v1")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=0, help="0=todas las filas; >0=prueba")
    ap.add_argument("--prepare-only", action="store_true", help="sin dependencias HuggingFace")
    ap.add_argument("--smoke-model", action="store_true", help="carga MLM, redimensiona en RAM, prueba forward")
    args = ap.parse_args()
    source = Path(args.input)
    checkpoint = Path(args.checkpoint)
    if not source.is_file():
        ap.error(f"No existe --input {source}")
    if not checkpoint.is_dir():
        ap.error(f"No existe --checkpoint {checkpoint}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31-1)
    print("[1/4] Analizando corpus y gramática heurística", flush=True)
    records, counts, roles, headers, max_index, jimages, profile = read_and_profile(
        source, args.column, args.require_valid, args.max_rows)
    if not records:
        raise RuntimeError("No se encontraron SLICES elegibles")
    save_csv(out/"lexical_units.csv", ["unit","frequency","header","species","edge_index","periodicity"],
             [{"unit":u,"frequency":n,"header":roles["header"][u],
               "species":roles["species"][u], "edge_index":roles["edge_index"][u],
               "periodicity":roles["periodicity"][u]} for u,n in counts.most_common()])
    save_csv(out/"header_patterns.csv", ["header","frequency"],
             [{"header":h,"frequency":n} for h,n in headers.most_common()])
    save_csv(out/"periodicity_triplets.csv", ["jimage","frequency"],
             [{"jimage":k,"frequency":v} for k,v in jimages.most_common()])
    save_csv(out/"max_edge_index_distribution.csv", ["max_index","structures"],
             [{"max_index":k,"structures":v} for k,v in sorted(max_index.items())])
    print("[2/4] Preparando IDs sin alterar vocabulario original", flush=True)
    original, config = load_checkpoint_vocab(checkpoint)
    original_size = int(config["vocab_size"])
    vocab, additions, fillers = complete_vocab(original, original_size, counts)
    save_json(out/"vocab_candidate_manifest.json", {
        "old_entries_accessible":len(original), "old_max_id":max(original.values()),
        "old_model_embedding_rows_declared": original_size,
        "original_reserved_rows_filled": [{"token":t,"id":i} for t,i in fillers],
        "new_slices_token_count":len(additions), "new_token_id_start":additions[0][1] if additions else None,
        "final_vocab_size":len(vocab), "special_tokens_original_ids":{k:original.get(k) for k in SPECIALS}
    })
    save_csv(out/"added_slices_tokens.csv", ["token","id","frequency"],
             [{"token":t,"id":i,"frequency":counts[t]} for t,i in additions])
    summary = {"source":str(source), "checkpoint":str(checkpoint),
               "strategy4":profile, "candidate_vocab": {
                   "old_accessible":len(original), "original_embedding_rows":original_size,
                   "reserved_old_rows":len(fillers), "lexical_units":len(counts),
                   "old_exact_units":len(counts)-len(additions),"new_units":len(additions),
                   "final_size":len(vocab)},
               "checkpoint_guardrails": {
                   "model_pad_token_id":config.get("pad_token_id"),
                   "candidate_pad_token_id":original["[PAD]"],
                   "pad_matches":config.get("pad_token_id")==original["[PAD]"],
                   "max_position_embeddings":config.get("max_position_embeddings")},
               "structural_equivalence": "NOT_VERIFIED",
               "official_SLICES_strategy4_grammar": "NOT_VERIFIED_HEURISTIC_ONLY"}
    if args.prepare_only:
        summary["status"] = "PREFLIGHT_COMPLETE_TOKENIZER_NOT_BUILT"
        save_json(out/"summary.json",summary)
        print(json.dumps({"status":summary["status"],"corpus":len(records),
                          "new_tokens":len(additions),"output":str(out)}, indent=2))
        return 0
    print("[3/4] Construyendo WordLevel + WhitespaceSplit", flush=True)
    candidate = build_hf_tokenizer(vocab, out/"candidate_tokenizer", args.max_length)
    summary["tokenizer_evaluation"] = evaluate_candidate(candidate, records, args.max_length, out)
    print("[4/4] Ejecutando guardrails de checkpoint", flush=True)
    if args.smoke_model:
        try:
            summary["model_smoke"] = smoke_model(checkpoint, candidate, records)
        except Exception as ex:
            summary["model_smoke"] = {"status":"FAILED", "error":f"{type(ex).__name__}: {ex}"}
    ok = summary["tokenizer_evaluation"]["lossless_lexically"]
    if not ok:
        summary["status"]="BLOCKED_TOKENIZATION"
    else:
        summary["status"]="TOKENIZER_LEXICALLY_PASS_MLM_NOT_YET_CLEARED"
    summary["mlm_training_ready"] = bool(ok and not summary["tokenizer_evaluation"]["sequences_over_limit"]
        and summary["checkpoint_guardrails"]["pad_matches"]
        and summary.get("model_smoke",{}).get("forward")=="OK"
        and summary["official_SLICES_strategy4_grammar"]=="VERIFIED")
    summary["critical_next_steps"] = [
        "Contrastar el parser heuristic con la version instalada de SLICES strategy=4",
        "Evaluar equivalencia cristalografica real para splits/deduplicacion",
        "Resolver la discrepancia PAD/posiciones sin modificar el checkpoint original",
        "Excluir/documentar secuencias >limite o estudiar contexto mayor sin truncamiento ciego",
        "Adaptar embeddings nuevos mediante DAPT antes de usarlos como caracteristicas fijas"
    ]
    save_json(out/"summary.json", summary)
    print("\n=== SLICES TOKENIZER v1 ===")
    print("Estado:",summary["status"])
    print("SLICES:",len(records),"| unidades:",len(counts),"| tokens nuevos:",len(additions))
    print("Evaluación:",json.dumps(summary["tokenizer_evaluation"],ensure_ascii=False))
    print("Checkpoint:",summary["checkpoint_guardrails"])
    print("Reportes:",out.resolve())
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}",file=sys.stderr)
        raise SystemExit(2)
