#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoría léxica y de integridad de SLICES strategy=4, sin modificar datos ni checkpoint.

Requisitos base: Python >= 3.9 (stdlib). Opcionales: transformers+tokenizers
para auditoría del tokenizador; pymatgen para --structure-match.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split())
INTEGER = re.compile(r"\d+\Z")
JIMAGE = re.compile(r"[o+\-]{3}\Z")
TRUE = {"true", "1", "yes", "y", "t"}
BAD_LIMIT = 300


def valid(s):
    return str(s).strip().lower() in TRUE


def choose_col(fields, requested, candidates):
    if requested:
        if requested not in fields:
            raise ValueError(f"Columna solicitada no existe: {requested!r}; disponibles: {fields}")
        return requested
    return next((name for name in candidates if name in fields), None)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_structure_obj(obj):
    """Solo igualdad JSON, NO equivalencia geométrica de cristales."""
    if not isinstance(obj, dict):
        return obj
    # Se conserva el JSON entero: evita declarar equivalencia con una huella incompleta.
    return obj


def parse_strategy4(text):
    """Separa tripletas finales (i, j, jimage), especies y prefijo de cabecera.

    No presupone una sintaxis concreta para el prefijo; reporta lo observado.
    """
    units = text.split()
    k = len(units)
    links = []
    while k >= 3 and JIMAGE.fullmatch(units[k - 1]) and INTEGER.fullmatch(units[k - 2]) and INTEGER.fullmatch(units[k - 3]):
        links.append((int(units[k - 3]), int(units[k - 2]), units[k - 1]))
        k -= 3
    links.reverse()
    j = k
    while j > 0 and units[j - 1] in ELEMENTS:
        j -= 1
    prefix, species, residue = units[:j], units[j:k], units[k:]
    errors = []
    if not links:
        errors.append("sin_tripletas_de_enlace_detectadas")
    if not species:
        errors.append("sin_especies_detectadas")
    if links and species:
        if any(i >= len(species) or p >= len(species) for i, p, _ in links):
            errors.append("indices_fuera_de_rango_segun_lista_de_especies")
    # El parseo es heurístico: su éxito no demuestra decodificación correcta SLICES.
    return units, prefix, species, links, errors


def percentiles(values):
    if not values:
        return {}
    vals = sorted(values)
    def pct(x):
        return vals[max(0, math.ceil(x * len(vals)) - 1)]
    return {"min": vals[0], "median": median(vals), "p95": pct(.95), "p99": pct(.99), "max": vals[-1]}


def element_composition(species):
    if not species:
        return ""
    counts = Counter(species)
    return "|".join(f"{sym}:{counts[sym]}" for sym in sorted(counts))


def json_cell(s):
    s = (s or "").strip()
    if not s or s.lower() in {"null", "none", "nan"}:
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def try_load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Instala transformers y tokenizers para --tokenizer") from e
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    if not tok.is_fast:
        raise RuntimeError("Esta comparación de offsets necesita un tokenizador Fast")
    return tok


def tokenizer_diag(tok, unit):
    enc = tok(unit, add_special_tokens=False, truncation=False,
              return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    decoded = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    cover = [False] * len(unit)
    for lo, hi in offsets:
        for pos in range(max(lo, 0), min(hi, len(unit))):
            cover[pos] = True
    unk = tok.unk_token_id
    return {
        "in_vocab_exact": int(unit in tok.get_vocab()),
        "n_pieces": len(ids),
        "n_unk": (ids.count(unk) if unk is not None else 0),
        "isolated_decode": decoded,
        "isolated_lossless": int(decoded == unit),
        "offset_chars_covered": sum(cover),
        "offset_chars_total": len(unit),
        "offset_coverage": round(sum(cover) / len(unit), 6) if unit else 1.0,
        "ids": " ".join(map(str, ids)),
        "offsets": str(offsets),
    }


def audit_context(tok, text):
    enc = tok(text, add_special_tokens=False, truncation=False,
              return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    decoded = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # Dos pruebas distintas: igualdad completa y preservación de caracteres no blancos.
    nows = lambda s: re.sub(r"\s+", "", s)
    covered = [False] * len(text)
    for lo, hi in offsets:
        for i in range(max(lo, 0), min(hi, len(text))):
            covered[i] = True
    relevant = [i for i, ch in enumerate(text) if not ch.isspace()]
    return {
        "pieces": len(ids),
        "exact_roundtrip": int(decoded == text),
        "nonspace_roundtrip": int(nows(decoded) == nows(text)),
        "offset_nonspace_coverage": round(
            sum(covered[i] for i in relevant) / len(relevant), 6
        ) if relevant else 1.0,
        "decoded_prefix": decoded[:220],
        "token_ids_prefix": " ".join(map(str, ids[:80])),
        "tokens_truncated_in_report": int(len(ids) > 80),
        "ids_hash": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
    }


def geometry_matches(candidates, out, max_pairs):
    """Comparación parcial, acotada, opcional con StructureMatcher.

    Agrupa por composición y número de sitios, compara solo candidatos guardados.
    No demuestra ausencia de duplicados si no hay coincidencias.
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError as e:
        raise RuntimeError("Para --structure-match instala pymatgen") from e
    matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False)
    pairs, matched, errors = 0, 0, 0
    rows = []
    for key, bucket in candidates.items():
        if len(bucket) < 2:
            continue
        structs = []
        for item in bucket:
            try:
                structs.append((item["id"], Structure.from_dict(item["structure"])))
            except Exception:
                errors += 1
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                if pairs >= max_pairs:
                    write_csv(out / "structure_matcher_pairs.csv",
                              ["id_a", "id_b", "composition", "same_structure"], rows)
                    return {"comparisons": pairs, "matched": matched, "parse_errors": errors,
                            "exhaustive": False, "limited_by_max_pairs": True}
                pairs += 1
                equal = matcher.fit(structs[i][1], structs[j][1])
                matched += int(bool(equal))
                rows.append({"id_a": structs[i][0], "id_b": structs[j][0],
                             "composition": key[0], "same_structure": int(bool(equal))})
    write_csv(out / "structure_matcher_pairs.csv",
              ["id_a", "id_b", "composition", "same_structure"], rows)
    return {"comparisons": pairs, "matched": matched, "parse_errors": errors,
            "exhaustive": False, "note": "solo subconjunto limitado de candidatos"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, help="CSV con strings SLICES strategy=4")
    p.add_argument("--out", default="reports/slices_strategy4_deep_audit")
    p.add_argument("--column", default="slices")
    p.add_argument("--require-valid", action="store_true")
    p.add_argument("--id-col", help="columna material ID; autodetecta mat_id/material_id/id")
    p.add_argument("--prototype-col", help="columna prototype_id, si existe")
    p.add_argument("--structure-col", help="columna de JSON pymatgen Structure, si existe")
    p.add_argument("--tokenizer", help="ruta al tokenizer original o adaptado (opcional)")
    p.add_argument("--context-samples", type=int, default=1000,
                   help="número de estructuras muestreadas para probar offsets/contexto")
    p.add_argument("--full-tokenization", action="store_true",
                   help="tokenizar TODAS las estructuras y detectar colisiones de IDs")
    p.add_argument("--structure-match", action="store_true",
                   help="comparaciones parciales con StructureMatcher, requiere pymatgen y estructura JSON")
    p.add_argument("--max-matcher-pairs", type=int, default=500)
    p.add_argument("--max-per-geometry-group", type=int, default=6)
    p.add_argument("--max-rows", type=int, default=0, help="0 procesa todo; >0 solo primeras filas")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    source = Path(args.input)
    if not source.is_file():
        p.error(f"No existe el CSV: {source}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    tok = try_load_tokenizer(args.tokenizer) if args.tokenizer else None
    rng = random.Random(args.seed)

    counts = Counter()
    lexical = Counter()
    roles = defaultdict(Counter)
    headers = Counter()
    header_pos = defaultdict(Counter)
    edge_jimages = Counter()
    jimage_component = [Counter(), Counter(), Counter()]
    edge_indices = Counter()
    max_indices = Counter()
    lengths = []
    atoms = []
    edges = []
    comp_counts = Counter()
    comp_proto = Counter()
    seen_slices = {}
    seen_matids = {}
    seen_exact_structs = {}
    ids_collision = {}
    geometry_candidates = defaultdict(list)
    problematic_rows = []
    dup_slices = []
    dup_matids = []
    dup_exact = []
    token_collisions = []
    sample = []

    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if args.column not in fields:
            p.error(f"No existe la columna {args.column!r}. Columnas: {fields}")
        if args.require_valid and not {"slices_basic_valid", "slices_valid"}.issubset(fields):
            p.error("--require-valid exige slices_basic_valid y slices_valid")
        id_col = choose_col(fields, args.id_col,
                            ["mat_id", "material_id", "entry_id", "id", "material_id_alexandria"])
        proto_col = choose_col(fields, args.prototype_col,
                               ["prototype_id", "prototype", "structural_prototype"])
        struct_col = choose_col(fields, args.structure_col,
                                ["structure", "structure_json", "pymatgen_structure"])
        print("[COLUMNAS]", {"id": id_col, "prototype": proto_col,
                              "structure": struct_col, "available": fields})

        for row_number, row in enumerate(reader, 2):
            if args.max_rows and counts["rows_total"] >= args.max_rows:
                break
            counts["rows_total"] += 1
            if args.require_valid:
                if not valid(row.get("slices_basic_valid")):
                    counts["excluded_basic"] += 1
                    continue
                if not valid(row.get("slices_valid")):
                    counts["excluded_full"] += 1
                    continue
            text = (row.get(args.column) or "").strip()
            if not text:
                counts["excluded_empty"] += 1
                continue
            counts["eligible"] += 1
            record_id = str(row.get(id_col, "")).strip() if id_col else f"row_{row_number}"
            if not record_id:
                record_id = f"row_{row_number}"
            proto = str(row.get(proto_col, "")).strip() if proto_col else ""
            units, prefix, species, links, parse_issues = parse_strategy4(text)
            normalized = " ".join(units)
            n_units = len(units)
            lengths.append(n_units + 2)
            atoms.append(len(species))
            edges.append(len(links))
            counts["over_512_lexical"] += int(n_units + 2 > 512)
            if parse_issues:
                counts["parse_issues"] += 1
                if len(problematic_rows) < BAD_LIMIT:
                    problematic_rows.append({"line": row_number, "mat_id": record_id,
                                             "issues": ";".join(parse_issues),
                                             "head": " ".join(units[:30])})
            for u in units:
                lexical[u] += 1
            for j, u in enumerate(prefix):
                roles[u]["header"] += 1
                header_pos[j][u] += 1
            for u in species:
                roles[u]["element"] += 1
            for i, j, offset in links:
                si, sj = str(i), str(j)
                roles[si]["edge_index"] += 1
                roles[sj]["edge_index"] += 1
                roles[offset]["jimage"] += 1
                edge_indices[i] += 1
                edge_indices[j] += 1
                edge_jimages[offset] += 1
                for idx, ch in enumerate(offset):
                    jimage_component[idx][ch] += 1
            if links:
                max_indices[max(max(i, j) for i, j, _ in links)] += 1
            headers[" ".join(prefix)] += 1
            comp = element_composition(species)
            if comp:
                comp_counts[comp] += 1
                comp_proto[(comp, proto)] += 1
            if normalized in seen_slices:
                counts["duplicate_lexical_strings"] += 1
                if len(dup_slices) < BAD_LIMIT:
                    dup_slices.append({"first_id": seen_slices[normalized], "next_id": record_id,
                                       "slices_prefix": normalized[:280]})
            else:
                seen_slices[normalized] = record_id
            if id_col:
                if record_id in seen_matids:
                    counts["repeated_mat_id_rows"] += 1
                    old_norm = seen_matids[record_id]
                    if len(dup_matids) < BAD_LIMIT:
                        dup_matids.append({"mat_id": record_id,
                                           "same_slices": int(old_norm == normalized),
                                           "first_prefix": old_norm[:180],
                                           "other_prefix": normalized[:180]})
                else:
                    seen_matids[record_id] = normalized
            struct_obj = json_cell(row.get(struct_col)) if struct_col else None
            if struct_obj is not None:
                counts["structures_json_parseable"] += 1
                blob = json.dumps(clean_structure_obj(struct_obj), sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":"))
                digest = hashlib.sha256(blob.encode()).hexdigest()
                if digest in seen_exact_structs:
                    counts["duplicate_identical_structure_json"] += 1
                    if len(dup_exact) < BAD_LIMIT:
                        dup_exact.append({"first_id": seen_exact_structs[digest],
                                          "next_id": record_id, "json_sha256": digest})
                else:
                    seen_exact_structs[digest] = record_id
                if args.structure_match and comp and len(geometry_candidates[(comp, len(species))]) < args.max_per_geometry_group:
                    geometry_candidates[(comp, len(species))].append(
                        {"id": record_id, "structure": struct_obj})
            if args.context_samples and tok:
                item = (record_id, normalized)
                if len(sample) < args.context_samples:
                    sample.append(item)
                else:
                    chosen = rng.randint(0, counts["eligible"] - 1)
                    if chosen < args.context_samples:
                        sample[chosen] = item
            if tok and args.full_tokenization:
                # No se trunca para no crear colisiones artificiales por contexto máximo.
                seq_ids = tok(normalized, add_special_tokens=True,
                              truncation=False)["input_ids"]
                digest = hashlib.sha256(",".join(map(str, seq_ids)).encode()).hexdigest()
                old = ids_collision.get(digest)
                if old is not None and old[1] != normalized:
                    counts["tokenizer_collisions_rows"] += 1
                    if len(token_collisions) < BAD_LIMIT:
                        token_collisions.append({"first_id": old[0], "next_id": record_id,
                                                 "first_text": old[1][:300], "next_text": normalized[:300],
                                                 "ids_sha256": digest})
                else:
                    ids_collision.setdefault(digest, (record_id, normalized))
            if counts["rows_total"] % 10000 == 0:
                print(f"[PROGRESO] filas={counts['rows_total']:,} SLICES={counts['eligible']:,}", flush=True)

    write_csv(out / "lexical_units.csv",
              ["unit", "occurrences", "header_count", "element_count", "edge_index_count", "jimage_count"],
              ({"unit": u, "occurrences": c, "header_count": roles[u]["header"],
                "element_count": roles[u]["element"], "edge_index_count": roles[u]["edge_index"],
                "jimage_count": roles[u]["jimage"]}
               for u, c in lexical.most_common()))
    write_csv(out / "header_units.csv", ["position", "unit", "occurrences"],
              ({"position": pos, "unit": u, "occurrences": n}
               for pos, counter in sorted(header_pos.items()) for u, n in counter.most_common()))
    write_csv(out / "header_templates.csv", ["header", "occurrences"],
              ({"header": h, "occurrences": n} for h, n in headers.most_common()))
    write_csv(out / "jimage_vectors.csv", ["jimage", "occurrences", "x", "y", "z"],
              ({"jimage": u, "occurrences": n, "x": u[0], "y": u[1], "z": u[2]}
               for u, n in edge_jimages.most_common()))
    write_csv(out / "index_occurrences.csv", ["index", "occurrences"],
              ({"index": k, "occurrences": n} for k, n in sorted(edge_indices.items())))
    write_csv(out / "max_index_per_structure.csv", ["maximum_index", "structures"],
              ({"maximum_index": k, "structures": n} for k, n in sorted(max_indices.items())))
    write_csv(out / "composition_prototype_groups.csv", ["composition_from_slices", "prototype", "structures"],
              ({"composition_from_slices": k[0], "prototype": k[1], "structures": n}
               for k, n in comp_proto.most_common()))
    write_csv(out / "possible_duplicate_slices.csv", ["first_id", "next_id", "slices_prefix"], dup_slices)
    write_csv(out / "repeated_mat_ids.csv", ["mat_id", "same_slices", "first_prefix", "other_prefix"], dup_matids)
    write_csv(out / "identical_structure_json.csv", ["first_id", "next_id", "json_sha256"], dup_exact)
    write_csv(out / "strategy4_parse_warnings.csv", ["line", "mat_id", "issues", "head"], problematic_rows)
    write_csv(out / "tokenizer_collisions_full.csv", ["first_id", "next_id", "first_text", "next_text", "ids_sha256"], token_collisions)

    tok_summary = None
    if tok:
        print(f"[TOKENIZER] Analizando {len(lexical)} unidades aisladas")
        lexical_diags = []
        for u, freq in lexical.most_common():
            lexical_diags.append({"unit": u, "occurrences": freq, **tokenizer_diag(tok, u)})
        write_csv(out / "tokenizer_lexical_diagnostics.csv",
                  ["unit", "occurrences", "in_vocab_exact", "n_pieces", "n_unk",
                   "isolated_decode", "isolated_lossless", "offset_chars_covered",
                   "offset_chars_total", "offset_coverage", "ids", "offsets"], lexical_diags)
        context_rows = []
        sample_seen = {}
        sample_collisions = []
        for rec_id, s in sample:
            result = audit_context(tok, s)
            context_rows.append({"mat_id": rec_id, "text_prefix": s[:250], **result})
            h = result["ids_hash"]
            if h in sample_seen and sample_seen[h][1] != s:
                if len(sample_collisions) < BAD_LIMIT:
                    sample_collisions.append({"first_id": sample_seen[h][0], "next_id": rec_id,
                                              "first_text": sample_seen[h][1][:300], "next_text": s[:300],
                                              "ids_sha256": h})
            else:
                sample_seen[h] = (rec_id, s)
        write_csv(out / "tokenizer_context_sample.csv",
                  ["mat_id", "text_prefix", "pieces", "exact_roundtrip", "nonspace_roundtrip",
                   "offset_nonspace_coverage", "decoded_prefix", "token_ids_prefix",
                   "tokens_truncated_in_report", "ids_hash"], context_rows)
        write_csv(out / "tokenizer_collisions_sample.csv",
                  ["first_id", "next_id", "first_text", "next_text", "ids_sha256"], sample_collisions)
        tok_summary = {
            "path": args.tokenizer,
            "vocab_size": len(tok), "is_fast": bool(tok.is_fast),
            "pad_token_id": tok.pad_token_id,
            "isolated_units": len(lexical_diags),
            "isolated_units_lossless": sum(r["isolated_lossless"] for r in lexical_diags),
            "isolated_units_in_vocab_exact": sum(r["in_vocab_exact"] for r in lexical_diags),
            "isolated_units_single_piece": sum(r["n_pieces"] == 1 for r in lexical_diags),
            "context_samples": len(context_rows),
            "context_nonspace_roundtrip_ok": sum(r["nonspace_roundtrip"] for r in context_rows),
            "context_exact_roundtrip_ok": sum(r["exact_roundtrip"] for r in context_rows),
            "sample_collision_examples": len(sample_collisions),
            "full_tokenization": args.full_tokenization,
            "full_tokenization_collision_rows": counts["tokenizer_collisions_rows"] if args.full_tokenization else None,
            "note": "No equiparar in_vocab_exact, offsets o roundtrip con cobertura semántica; validar colisiones explícitas.",
        }

    geo_summary = None
    if args.structure_match:
        if not struct_col:
            print("[WARN] No existe columna estructura: se omite StructureMatcher")
        else:
            geo_summary = geometry_matches(geometry_candidates, out, args.max_matcher_pairs)

    summary = {
        "input": str(source), "columns": {"id": id_col, "prototype": proto_col,
                                          "structure": struct_col, "slices": args.column},
        "counts": dict(counts),
        "unique_lexical_units": len(lexical),
        "lexical_length_plus_2_special_tokens": percentiles(lengths),
        "lexical_over_512_rate": round(counts["over_512_lexical"] / counts["eligible"], 6) if counts["eligible"] else None,
        "atom_count_inferred": percentiles(atoms),
        "edge_triplets_inferred": percentiles(edges),
        "unique_compositions_derived_from_species": len(comp_counts),
        "unique_header_templates": len(headers),
        "index_max_observed": max(edge_indices) if edge_indices else None,
        "unique_jimage_vectors": len(edge_jimages),
        "jimage_component_counts": [dict(c) for c in jimage_component],
        "duplicate_evidence": {
            "slices_equal_after_normalize": counts["duplicate_lexical_strings"],
            "repeated_material_ids_rows": counts["repeated_mat_id_rows"] if id_col else None,
            "identical_full_structure_json_rows": counts["duplicate_identical_structure_json"] if struct_col else None,
            "caveat": "Misma composición y prototipo NO prueban cristales equivalentes. SLICES distintos NO prueban estructuras distintas. Igualdad de JSON solo detecta idénticos literales normalizados."},
        "tokenizer": tok_summary, "structure_matcher_subset": geo_summary,
        "method_notes": [
            "Parser de strategy=4 basado en tripletas finales (indice, indice, vector de tres caracteres o/+/−); no sustituye SLICES.check_SLICES.",
            "La segmentación por espacios es exploratoria y no asegura compatibilidad posicional con ChemBERTa.",
            "Vocabulario=conjunto de unidades completas, no cobertura real del tokenizer. La cobertura se evalúa por decodificación aislada y contexto.",
            "Las pruebas de offsets indican qué caracteres se enlazaron a tokens; NO garantizan que se conserve su valor.",
            "La prueba de colisiones completa no aplica truncamiento. La de contexto usa muestreo reproducible.",
            "La comparación geométrica (si se solicita) es parcial y acotada; ausencia de matches no implica unicidad cristalográfica.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SLICES strategy=4 · AUDITORÍA PROFUNDA ===")
    print(f"Estructuras: {counts['eligible']:,} | unidades léxicas: {len(lexical)}")
    print(f"Largo mediano/p95/p99/max: {summary['lexical_length_plus_2_special_tokens']}")
    print(f"Más de 512: {counts['over_512_lexical']:,} ({summary['lexical_over_512_rate']:.2%})")
    print(f"Parseo con incidencias: {counts['parse_issues']:,}; índices máximos: {summary['index_max_observed']}")
    print(f"Cadenas idénticas: {counts['duplicate_lexical_strings']:,}; IDs repetidos: {counts['repeated_mat_id_rows']:,}")
    if tok_summary:
        print(f"Tokenizador: unidades recuperables aisladas {tok_summary['isolated_units_lossless']}/{len(lexical)}; "
              f"ejemplos contexto sin pérdida {tok_summary['context_nonspace_roundtrip_ok']}/{tok_summary['context_samples']}")
        if args.full_tokenization:
            print(f"Colisiones de IDs en corpus completo: {counts['tokenizer_collisions_rows']:,}")
    print("Reportes:", out.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
