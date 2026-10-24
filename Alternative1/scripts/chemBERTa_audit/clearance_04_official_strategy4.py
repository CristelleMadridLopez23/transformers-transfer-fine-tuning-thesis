#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clearance 04: compare existing SLICES strings with installed official SLICES strategy=4.

Uses SLICES.from_SLICES(..., fix_duplicate_edge=False) and an encode roundtrip,
optionally check_SLICES_basic and check_SLICES. No structural relaxation,
no tokenization, no checkpoint modifications, and no changes to source data.

A passing sampled check does NOT amount to full-corpus validation.
"""
import argparse
import csv
import hashlib
import inspect
import json
import random
from collections import Counter
from pathlib import Path


def yes(x):
    return str(x).strip().lower() in {"true", "yes", "1", "y"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sample_from_csv(path, column, require_valid, sample_short, sample_long, seed, cutoff):
    rng = random.Random(seed)
    reservoirs = {"short": [], "long": []}
    counts = Counter()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {column} | ({"slices_valid", "slices_basic_valid"} if require_valid else set())
        if not required.issubset(fields):
            raise ValueError(f"Faltan columnas {sorted(required-fields)}")
        for row_i, row in enumerate(reader, 1):
            if require_valid and not (yes(row["slices_valid"]) and yes(row["slices_basic_valid"])):
                counts["invalid_excluded"] += 1
                continue
            s = (row.get(column) or "").strip()
            if not s:
                counts["empty_excluded"] += 1
                continue
            length = len(s.split()) + 2
            cat = "long" if length > cutoff else "short"
            counts[f"all_{cat}"] += 1
            n = counts[f"all_{cat}"]
            lim = sample_long if cat == "long" else sample_short
            item = {"mat_id": row.get("mat_id", ""), "row_i": row_i, "slices": s,
                    "length": length, "category": cat,
                    "source_basic_valid": row.get("slices_basic_valid", ""),
                    "source_full_valid": row.get("slices_valid", "")}
            if lim == 0:  # full-stratum audit
                reservoirs[cat].append(item)
            elif len(reservoirs[cat]) < lim:
                reservoirs[cat].append(item)
            else:
                ix = rng.randrange(n)
                if ix < lim:
                    reservoirs[cat][ix] = item
    return reservoirs["short"] + reservoirs["long"], counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--column", default="slices")
    p.add_argument("--require-valid", action="store_true")
    p.add_argument("--short-sample", type=int, default=1500,
                   help="0 significa auditar todas las secuencias cortas")
    p.add_argument("--long-sample", type=int, default=175,
                   help="0 significa auditar todas las secuencias largas")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-basic", action="store_true",
                   help="También ejecutar SLICES.check_SLICES_basic en las muestras")
    p.add_argument("--check-full", action="store_true",
                   help="MUY costoso: ejecutar SLICES.check_SLICES(..., graph_rank_check=False)")
    args = p.parse_args()
    source = Path(args.input)
    if not source.is_file():
        p.error(f"No existe {source}")
    if args.short_sample < 0 or args.long_sample < 0:
        p.error("Muestras deben ser >= 0; 0 = conjunto completo de ese estrato")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] Muestreo estratificado por longitud", flush=True)
    samples, population = sample_from_csv(source, args.column, args.require_valid,
                                           args.short_sample, args.long_sample, args.seed, args.limit)

    print("[2/3] Importando implementación oficial de SLICES", flush=True)
    try:
        from slices.core import SLICES
        import slices.core as slices_core
        backend = SLICES(graph_method="econnn", relax_model=None)
        installed_file = Path(inspect.getfile(slices_core)).resolve()
        installed_hash = sha(installed_file)
    except Exception as exc:
        result = {"status": "BLOCKED_SLICES_IMPORT", "error": repr(exc),
                  "sample_total": len(samples), "population": dict(population),
                  "note": "Use a Python environment with the project's official SLICES implementation installed."}
        (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(f"No se pudo importar la implementación instalada: {exc}")

    counters = Counter()
    errors = []
    fieldnames = ["mat_id", "row_i", "category", "length", "official_decoding_ok", "roundtrip_ok",
                  "basic_valid", "full_valid", "original_prefix", "regenerated_prefix", "exception"]
    with (out / "official_strategy4_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print("[3/3] Validación con decoder/encoder oficial", flush=True)
        for i, row in enumerate(samples, 1):
            s = row["slices"]
            record = {k: row[k] for k in ("mat_id", "row_i", "category", "length")}
            record.update({"official_decoding_ok": False, "roundtrip_ok": False,
                           "basic_valid": "NOT_RUN", "full_valid": "NOT_RUN",
                           "original_prefix": " ".join(s.split()[:12]), "regenerated_prefix": "", "exception": ""})
            try:
                backend.from_SLICES(s, strategy=4, fix_duplicate_edge=False)
                record["official_decoding_ok"] = True
                regenerated = backend.get_slices_by_strategy(
                    4, list(backend.atom_symbols), backend.edge_indices,
                    backend.to_jimages, backend.space_group_num)
                record["regenerated_prefix"] = " ".join(regenerated.split()[:12])
                record["roundtrip_ok"] = " ".join(s.split()) == " ".join(regenerated.split())
                if args.check_basic:
                    record["basic_valid"] = bool(backend.check_SLICES_basic(s, strategy=4))
                if args.check_full:
                    record["full_valid"] = bool(backend.check_SLICES(s, strategy=4, graph_rank_check=False))
            except Exception as exc:
                record["exception"] = str(exc)[:250]
            writer.writerow(record)
            counters["tested"] += 1
            counters[f"sample_{row['category']}"] += 1
            counters["decoder_fail"] += int(not record["official_decoding_ok"])
            counters["roundtrip_fail"] += int(not record["roundtrip_ok"])
            if args.check_basic:
                counters["basic_false"] += int(record["basic_valid"] is not True)
            if args.check_full:
                counters["full_false"] += int(record["full_valid"] is not True)
            if record["exception"] or not record["roundtrip_ok"] or record["basic_valid"] is False or record["full_valid"] is False:
                if len(errors) < 100:
                    errors.append(record)
            if i % 250 == 0:
                print(f"[OFFICIAL] {i:,}/{len(samples):,}", flush=True)

    failure_count = counters["decoder_fail"] + counters["roundtrip_fail"]
    if args.check_basic:
        failure_count += counters["basic_false"]
    if args.check_full:
        failure_count += counters["full_false"]
    full_short = args.short_sample == 0 or args.short_sample >= population["all_short"]
    full_long = args.long_sample == 0 or args.long_sample >= population["all_long"]
    mode = "FULL" if full_short and full_long else "SAMPLED"
    status = (f"OFFICIAL_STRATEGY4_{mode}_PASS" if failure_count == 0
              else f"OFFICIAL_STRATEGY4_{mode}_ISSUES")
    summary = {"status": status, "mode": mode, "source": str(source), "source_sha256": sha(source),
               "official_core_file": str(installed_file), "official_core_sha256": installed_hash,
               "sample_counts": dict(counters), "corpus_counts": dict(population),
               "check_basic_was_run": args.check_basic, "check_full_was_run": args.check_full,
               "errors_first_100": errors, "warnings": [
                   "OFFICIAL_STRATEGY4_SAMPLED_PASS is not full-corpus validation.",
                   "A byte/token roundtrip does not establish crystal structural equivalence.",
                   "full check uses graph_rank_check=False, appropriate to property-prediction validation per project source.",
                   "Interpret discrepancies in the context of the installed source hash and its SLICES version.",
               ]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Estado: {status} | muestras: {len(samples):,} | fallos decodificación: {counters['decoder_fail']} | fallos roundtrip: {counters['roundtrip_fail']}")
    print(f"Reportes: {out.resolve()}")
    if failure_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
