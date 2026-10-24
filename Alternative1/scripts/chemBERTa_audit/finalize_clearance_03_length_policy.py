#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze a lossless, nontruncating length policy after Clearance 03.

This program reads the *already tokenized* length_manifest.csv; it does NOT
recompute sequence lengths, build ML splits, label polymorphs, or train a model.

Outputs candidates only. Mandatory later steps: official SLICES validation,
structural deduplication, composition-disjoint splits, and energy-target checks.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = (
    "mat_id", "source_row_index", "composition_key", "formula", "prototype_id",
    "nsites", "is_polymorph_candidate", "length_with_specials",
    "source_lexical_units", "over_limit", "reason",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="03_reports/chemberta_length_clearance/length_manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--expect-total", type=int, default=0)
    parser.add_argument("--expect-over", type=int, default=-1)
    parser.add_argument("--expect-groups-lost", type=int, default=-1)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_file():
        parser.error(f"No se encontró: {manifest}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        absent = set(FIELDS).difference(reader.fieldnames or [])
        if absent:
            parser.error(f"Manifest incompleto; faltan: {sorted(absent)}")
        raw_rows = list(reader)
    if not raw_rows:
        parser.error("Manifest vacío")

    counts = defaultdict(lambda: [0, 0])
    fit_rows, over_rows = [], []
    problems = []
    ids, rowidx = set(), set()
    for i, r in enumerate(raw_rows, start=1):
        try:
            length = int(r["length_with_specials"])
            over = int(r["over_limit"])
        except (ValueError, TypeError):
            problems.append(f"fila {i}: longitud o indicador no entero")
            continue
        if length <= 0 or over not in (0, 1):
            problems.append(f"fila {i}: longitud/over_limit inválidos")
        if (length > args.limit) != bool(over):
            problems.append(f"fila {i}: over_limit no coincide con longitud y límite")
        if r["reason"] not in ("FIT", f"EXCEEDS_{args.limit}"):
            problems.append(f"fila {i}: reason inesperado {r['reason']!r}")
        if (r["reason"] == "FIT") != (over == 0):
            problems.append(f"fila {i}: reason/over_limit inconsistentes")
        group = r["composition_key"].strip()
        if not group:
            problems.append(f"fila {i}: composition_key vacío")
        else:
            counts[group][0] += 1
            if not over:
                counts[group][1] += 1
        mat = r["mat_id"].strip()
        if mat:
            if mat in ids:
                problems.append(f"fila {i}: mat_id repetido {mat!r}")
            ids.add(mat)
        else:
            problems.append(f"fila {i}: mat_id faltante")
        source_idx = r["source_row_index"].strip()
        if source_idx:
            if source_idx in rowidx:
                problems.append(f"fila {i}: source_row_index repetido {source_idx!r}")
            rowidx.add(source_idx)
        (over_rows if over else fit_rows).append(r)

    lost_groups = sorted(
        comp for comp, (before, after) in counts.items() if before >= 2 and after < 2
    )
    final_groups = {comp for comp, (_, after) in counts.items() if after >= 2}
    downstream_candidates = [r for r in fit_rows if r["composition_key"].strip() in final_groups]
    all_groups = []
    for comp, (before, after) in sorted(counts.items()):
        all_groups.append({
            "composition_key": comp, "before": before, "fit_512": after,
            "over_512": before - after, "eligible_preliminary_polymorph_group": int(after >= 2),
            "lost_polymorph_group": int(before >= 2 and after < 2),
            "completely_excluded": int(after == 0),
        })

    if args.expect_total > 0 and len(raw_rows) != args.expect_total:
        problems.append(f"Se esperaban {args.expect_total} filas, encontradas {len(raw_rows)}")
    if args.expect_over >= 0 and len(over_rows) != args.expect_over:
        problems.append(f"Se esperaban {args.expect_over} secuencias largas, encontradas {len(over_rows)}")
    if args.expect_groups_lost >= 0 and len(lost_groups) != args.expect_groups_lost:
        problems.append(f"Se esperaban {args.expect_groups_lost} grupos perdidos, encontrados {len(lost_groups)}")

    # No silently approve a corrupted manifest or a different corpus.
    approved = not problems
    status = "LENGTH_POLICY_FROZEN" if approved else "LENGTH_POLICY_BLOCKED_INCONSISTENT_MANIFEST"

    # Reproducible outputs are available even on failure for inspection.
    write_csv(out / "dapt_candidate_manifest_UNSPLIT.csv", FIELDS, fit_rows)
    write_csv(out / "over_512_exclusion_manifest.csv", FIELDS, over_rows)
    write_csv(out / "downstream_polymorph_candidates_UNSPLIT.csv", FIELDS, downstream_candidates)
    write_csv(out / "composition_retention.csv", list(all_groups[0]) if all_groups else ["composition_key"], all_groups)
    write_csv(out / "polymorph_groups_lost_by_limit.csv", ["composition_key", "before", "fit_512", "over_512", "eligible_preliminary_polymorph_group", "lost_polymorph_group", "completely_excluded"], [r for r in all_groups if r["lost_polymorph_group"]])

    report = {
        "status": status,
        "scope": "Length-only policy. NOT a DAPT-ready train split or final downstream cohort.",
        "source_manifest": str(manifest),
        "source_sha256": sha256(manifest),
        "limit_including_special_tokens": args.limit,
        "total": len(raw_rows),
        "fit": len(fit_rows),
        "over_limit": len(over_rows),
        "over_limit_pct": round(100 * len(over_rows) / len(raw_rows), 4),
        "composition_groups": len(counts),
        "polymorph_groups_before_length": sum(1 for a, _ in counts.values() if a >= 2),
        "polymorph_groups_after_length": len(final_groups),
        "polymorph_groups_lost_by_length": len(lost_groups),
        "preliminary_downstream_candidate_structures": len(downstream_candidates),
        "entirely_excluded_composition_groups": sum(1 for a, b in counts.values() if b == 0),
        "partially_excluded_composition_groups": sum(1 for a, b in counts.values() if 0 < b < a),
        "policy": {
            "mlm_candidate": "Complete SLICES, <= 512 tokens including special tokens; never truncate.",
            "long_sequences": "Quarantine as out-of-scope for original checkpoint; preserve intact for sensitivity analyses.",
            "downstream_candidate": "At least 2 eligible structures per composition after length filter; not yet a valid supervised set.",
            "split": "Must assign composition-disjoint train/val/test BEFORE MLM. DAPT on train only; test excluded for inductive evaluation.",
        },
        "known_remaining_gates": [
            "Validate official SLICES strategy=4 parser/checks against installed version",
            "Verify structural equivalence and canonical composition grouping",
            "Verify energy targets and 0-K polymorph definitions; freeze composition-disjoint split",
            "Audit MLM masking/random replacement & initialize new embeddings",
        ],
        "issues": problems[:100],
        "issues_total": len(problems),
        "files": ["dapt_candidate_manifest_UNSPLIT.csv", "over_512_exclusion_manifest.csv", "downstream_polymorph_candidates_UNSPLIT.csv", "composition_retention.csv", "polymorph_groups_lost_by_limit.csv"],
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Estado: {status}")
    print(f"Candidatos DAPT sin dividir: {len(fit_rows):,}; excluidos: {len(over_rows):,} ({report['over_limit_pct']} %)")
    print(f"Grupos con >=2 antes/después: {report['polymorph_groups_before_length']:,}/{report['polymorph_groups_after_length']:,}; perdidos: {len(lost_groups):,}")
    print(f"Candidatos downstream preliminares sin dividir: {len(downstream_candidates):,}")
    print(f"Resultados: {out.resolve()}")
    if not approved:
        print("Problemas:")
        for issue in problems[:15]:
            print(" -", issue)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
