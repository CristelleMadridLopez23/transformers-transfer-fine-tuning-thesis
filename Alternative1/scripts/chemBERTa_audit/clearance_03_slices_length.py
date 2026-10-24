#!/usr/bin/env python3
"""Clearance 03 — length and composition-retention audit for SLICES v1.

Does not truncate, split, train, or modify model/tokenizer files.
Usage:
    python clearance_03_slices_length.py --input path.csv \
      --tokenizer reports/chemberta_pad_clearance_v2/tokenizer_pad_aligned \
      --out reports/slices_length_clearance
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def yes(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * q
    a, b = int(math.floor(k)), int(math.ceil(k))
    return round(sorted_values[a] * (b-k) + sorted_values[b] * (k-a), 3) if a != b else sorted_values[a]


def bucket(length, cutoff):
    if length <= 128:
        return "000-128"
    if length <= 256:
        return "129-256"
    if length <= 384:
        return "257-384"
    if length <= cutoff:
        return f"385-{cutoff}"
    if length <= 768:
        return f"{cutoff + 1}-768"
    if length <= 1024:
        return "769-1024"
    if length <= 2048:
        return "1025-2048"
    return ">2048"


def nonempty(*things):
    for value in things:
        if str(value or "").strip():
            return str(value).strip()
    return ""


def run(args, tokenizer=None):
    input_file = Path(args.input)
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, local_files_only=True)

    model_max_length = getattr(tokenizer, "model_max_length", None)
    pad_id = tokenizer.pad_token_id
    if pad_id != 1:
        raise ValueError(f"Se requiere tokenizador PAD alineado a id 1; recibido {pad_id}")
    sep = args.column
    sizes = []
    outliers = []
    group_counts = defaultdict(lambda: [0, 0])
    bins = Counter()
    short_count = 0
    n_eligible = 0
    dropped_filter = 0
    empty = 0
    missing_groups = 0
    missing_pad = 0
    count_gt_model = 0
    sites_short, sites_long = [], []
    csv_manifest = output / "length_manifest.csv"
    csv_oversized = output / "over_limit.csv"
    fieldnames = ["mat_id", "source_row_index", "composition_key", "formula", "prototype_id", "nsites", "is_polymorph_candidate", "length_with_specials", "source_lexical_units", "over_limit", "reason"]

    with input_file.open("r", encoding="utf-8-sig", newline="") as fi, \
         csv_manifest.open("w", encoding="utf-8", newline="") as fm, \
         csv_oversized.open("w", encoding="utf-8", newline="") as fo:
        reader = csv.DictReader(fi)
        fields = set(reader.fieldnames or [])
        needed = {sep}
        if args.require_valid:
            needed |= {"slices_valid", "slices_basic_valid"}
        if not needed.issubset(fields):
            raise ValueError(f"Faltan columnas: {sorted(needed-fields)}")
        mw = csv.DictWriter(fm, fieldnames=fieldnames); mw.writeheader()
        ow = csv.DictWriter(fo, fieldnames=fieldnames); ow.writeheader()
        for lineno, row in enumerate(reader, 1):
            if args.require_valid and not (yes(row["slices_valid"]) and yes(row["slices_basic_valid"])):
                dropped_filter += 1
                continue
            s = str(row.get(sep) or "").strip()
            if not s:
                empty += 1
                continue
            # Only report, never truncate.
            ids = tokenizer.encode(s, add_special_tokens=True, truncation=False)
            length = len(ids)
            if pad_id in ids:
                missing_pad += 1
            if model_max_length and isinstance(model_max_length, int) and model_max_length < 1_000_000 and length > model_max_length:
                count_gt_model += 1
            over = length > args.limit
            lengths_group = nonempty(row.get("composition_key"), row.get("reduced_formula"), row.get("formula"))
            if lengths_group:
                group_counts[lengths_group][0] += 1
                if not over:
                    group_counts[lengths_group][1] += 1
            else:
                missing_groups += 1
            try:
                ns = float(row.get("nsites") or "nan")
            except (ValueError, TypeError):
                ns = math.nan
            if math.isfinite(ns):
                (sites_long if over else sites_short).append(ns)
            entry = {
                "mat_id": row.get("mat_id", ""),
                "source_row_index": row.get("source_row_index", lineno),
                "composition_key": lengths_group,
                "formula": row.get("formula", ""),
                "prototype_id": row.get("prototype_id", ""),
                "nsites": row.get("nsites", ""),
                "is_polymorph_candidate": row.get("is_polymorph_candidate", ""),
                "length_with_specials": length,
                "source_lexical_units": len(s.split()),
                "over_limit": int(over),
                "reason": f"EXCEEDS_{args.limit}" if over else "FIT",
            }
            mw.writerow(entry)
            if over:
                ow.writerow(entry)
                outliers.append(entry)
            else:
                short_count += 1
            bins[bucket(length, args.limit)] += 1
            sizes.append(length)
            n_eligible += 1
            if n_eligible % 10000 == 0:
                print(f"[LENGTH] {n_eligible:,} SLICES", flush=True)

    if not sizes:
        raise ValueError("No hay secuencias elegibles")
    sizes.sort()
    n_comps = len(group_counts)
    originally_polymorphic = [c for c, counts in group_counts.items() if counts[0] >= 2]
    became_nonpolymorphic = [c for c in originally_polymorphic if group_counts[c][1] < 2]
    fully_excluded = [c for c, (n, kept) in group_counts.items() if kept == 0]
    partial = [c for c, (n, kept) in group_counts.items() if kept > 0 and kept < n]
    with (output / "affected_compositions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["composition_key", "total_structures", "within_limit", "over_limit", "possible_polymorph_group_before", "possible_polymorph_group_after", "lost_polymorph_group"])
        for name, (n, kept) in sorted(group_counts.items()):
            if kept < n:
                w.writerow([name, n, kept, n-kept, int(n >= 2), int(kept >= 2), int(n >= 2 and kept < 2)])
    report = {
        "status": "LENGTH_AUDIT_COMPLETE_POLICY_PENDING" if outliers else "LENGTH_AUDIT_PASS_NO_EXCLUSIONS",
        "tokenizer": args.tokenizer,
        "input": str(input_file),
        "limit_including_specials": args.limit,
        "model_max_length_metadata": model_max_length,
        "total_filtered_out": dropped_filter,
        "valid_empty": empty,
        "tested": n_eligible,
        "fit": short_count,
        "over_limit": len(outliers),
        "over_limit_pct": round(100*len(outliers)/n_eligible, 3),
        "nonpadding_tokens_with_pad_id": missing_pad,
        "length": {"min": sizes[0], "median": median(sizes), "p95": percentile(sizes, .95), "p99": percentile(sizes, .99), "max": sizes[-1]},
        "length_buckets": dict(sorted(bins.items())),
        "count_exceeding_tokenizer_model_max_length": count_gt_model,
        "median_nsites_fit": median(sites_short) if sites_short else None,
        "median_nsites_over_limit": median(sites_long) if sites_long else None,
        "composition_impact": {
            "groups_with_composition_key": n_comps,
            "rows_missing_composition_key_and_fallback": missing_groups,
            "groups_with_2_or_more_before": len(originally_polymorphic),
            "groups_with_2_or_more_after": sum(1 for counts in group_counts.values() if counts[1] >= 2),
            "groups_falling_below_two_after_length_filter": len(became_nonpolymorphic),
            "groups_entirely_excluded": len(fully_excluded),
            "groups_partially_excluded": len(partial),
        },
        "caveats": [
            "El corte 512 es conservador para el checkpoint; validar tambien posicionamientos al limite si se modifica.",
            "Esta prueba no hace truncamiento, splits ni deduplicacion estructural.",
            "composition_key/reduced_formula/formula son identificadores de grupo aproximados: validar consistencia y stoichiometria.",
            "Las estructuras excluidas pueden afectar la distribucion de polimorfos y la evaluacion de energia relativa.",
            "Las entradas que superan la longitud deben tratarse con una politica experimental documentada.",
        ],
        "files": {"manifest": str(csv_manifest), "oversized": str(csv_oversized), "affected_compositions": str(output / "affected_compositions.csv")},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== CLEARANCE 03: LONGITUD ===")
    print(f"Estado: {report['status']}")
    print(f"Fit: {short_count:,} / {n_eligible:,}; exceden {args.limit}: {len(outliers):,} ({report['over_limit_pct']}%)")
    print(f"Longitudes: {report['length']}")
    print(f"Polimorfos afectados por filtro: {len(became_nonpolymorphic)} grupos que caerían por debajo de 2")
    print(f"Archivos: {output.resolve()}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV Alexandria limpio")
    p.add_argument("--tokenizer", required=True, help="Directorio tokenizer_pad_aligned (Clearance 02)")
    p.add_argument("--out", default="reports/chemberta_length_clearance")
    p.add_argument("--column", default="slices")
    p.add_argument("--require-valid", action="store_true")
    p.add_argument("--limit", type=int, default=512)
    args = p.parse_args()
    if args.limit < 385:
        p.error("--limit debe ser >= 385 para los buckets del reporte")
    run(args)


if __name__ == "__main__":
    main()
