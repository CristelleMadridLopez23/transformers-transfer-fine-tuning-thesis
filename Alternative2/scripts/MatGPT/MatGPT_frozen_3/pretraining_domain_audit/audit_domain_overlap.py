#!/usr/bin/env python3
"""Compara fórmulas de dominio con desarrollo; test se usa solo por fórmula."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import ALT, REPORT_DIR, AuditError, write_json

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
SPLIT_MANIFEST = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/split_manifest.json"


def parse_formula(formula: str) -> tuple[tuple[str, int], ...]:
    """Convierte fórmula sin paréntesis en clave reducida independiente del orden.

    Args:
        formula: Fórmula textual del manifiesto histórico.

    Returns:
        Pares de elemento y estequiometría reducida ordenados.
    """
    matches = FORMULA_RE.findall(formula)
    if not matches or "".join(e + n for e, n in matches) != formula:
        raise AuditError(f"Fórmula no soportada: {formula}")
    counts = Counter({e: int(n) if n else 1 for e, n in matches})
    divisor = math.gcd(*counts.values())
    return tuple(sorted((element, amount // divisor) for element, amount in counts.items()))


def run_overlap() -> dict[str, Any]:
    """Genera solapamiento exacto y composicional con train candidato.

    Returns:
        Resumen de coincidencias; ninguna etiqueta DFT es cargada.
    """
    corpus_path = REPORT_DIR / "corpus_statistics" / "composition_records.csv"
    split_path = REPORT_DIR / "corpus_statistics" / "reconstructed_split_assignments.csv"
    if not corpus_path.is_file() or not split_path.is_file() or not SPLIT_MANIFEST.is_file():
        raise AuditError("Faltan CSV de composición/split reconstruido o manifiesto histórico de fórmulas.")
    corpus = pd.read_csv(corpus_path, usecols=["formula_reduced_key", "composition_json"])
    candidate = pd.read_csv(split_path, usecols=["formula_reduced_key", "split_candidate"])
    counts_by_key: dict[tuple[tuple[str, int], ...], Counter[str]] = {}
    for (_, row), (_, split_row) in zip(corpus.iterrows(), candidate.iterrows()):
        key = tuple(sorted((str(e), int(v)) for e, v in json.loads(row.composition_json).items()))
        record = counts_by_key.setdefault(key, Counter())
        record["corpus_rows"] += 1
        record[str(split_row.split_candidate)] += 1
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for formula, split in manifest["formula_splits"].items():
        key = parse_formula(str(formula))
        count = counts_by_key.get(key, Counter())
        same_elements = [other for other in counts_by_key if {e for e, _ in other} == {e for e, _ in key} and other != key]
        partial = [other for other in counts_by_key if set(e for e, _ in other) & set(e for e, _ in key) and other != key]
        rows.append({"target_formula": formula, "historical_split": split,
                     "target_scope": "development" if split in {"train", "val"} else "historical_test_formula_only",
                     "exact_match_full_corpus": count["corpus_rows"] > 0,
                     "exact_match_train_candidate": count["train_candidate"] > 0,
                     "exact_match_validation_candidate": count["validation"] > 0,
                     "full_corpus_rows_exact_formula": count["corpus_rows"],
                     "train_candidate_rows_exact_formula": count["train_candidate"],
                     "validation_candidate_rows_exact_formula": count["validation"],
                     "same_elements_other_stoichiometry_rows": sum(counts_by_key[item]["corpus_rows"] for item in same_elements),
                     "partial_element_overlap_formula_count": len(partial)})
    out = REPORT_DIR / "domain_overlap"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "abs3_composition_overlap.csv", index=False)
    dev = frame[frame.historical_split.isin(["train", "val"])]
    test = frame[frame.historical_split.eq("test")]
    summary = {"status": "CONFIG_REPRODUCED_EXPLORATORY", "corpus_rows_scope": "full corpus and torch.randperm reconstructed train candidate",
               "development_compositions": int(len(dev)), "development_exact_in_candidate_train": int(dev.exact_match_train_candidate.sum()),
               "development_exact_in_full_corpus": int(dev.exact_match_full_corpus.sum()),
               "development_absent_from_full_corpus": int((~dev.exact_match_full_corpus).sum()),
               "development_same_elements_different_stoichiometry": int(dev.same_elements_other_stoichiometry_rows.gt(0).sum()),
               "historical_test_compositions_checked_by_formula_only": int(len(test)),
               "historical_test_exact_in_candidate_train": int(test.exact_match_train_candidate.sum()),
               "test_labels_read": False,
               "caveat": "Split candidato; identidad/orden del CSV de pretraining no queda atestiguada por checkpoint metadata."}
    write_json(out / "composition_overlap_summary.json", summary)
    return summary

def main() -> None:
    """Ejecuta el cotejo químico contra el split candidato."""
    try:
        print(json.dumps(run_overlap(), indent=2, ensure_ascii=False))
    except (AuditError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc

if __name__ == "__main__":
    main()
