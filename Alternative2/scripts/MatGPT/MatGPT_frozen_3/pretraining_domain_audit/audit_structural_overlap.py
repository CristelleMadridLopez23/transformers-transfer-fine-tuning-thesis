#!/usr/bin/env python3
"""Comprueba duplicados exactos de SLICES para development, sin inferir geometría."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import ALT, REPORT_DIR, AuditError, sha256_file, write_json

TARGET_DATASET = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
CORPUS = ALT / "01_data/matgpt y matter/Training_dataset_alex20_with_encoding.csv"


def run_audit() -> dict[str, Any]:
    """Compara SLICES de estructuras de desarrollo contra el corpus original.

    Returns:
        Resumen de coincidencias exactas de cadenas SLICES en el corpus y train candidato.
    """
    assignments_path = REPORT_DIR / "corpus_statistics/reconstructed_split_assignments.csv"
    if not TARGET_DATASET.is_file() or not CORPUS.is_file() or not assignments_path.is_file():
        raise AuditError("Falta dataset ABS3 versionado, corpus exacto o asignación candidata.")
    target = pd.read_csv(TARGET_DATASET, usecols=["structure_id", "reduced_formula", "polymorph_family", "split", "matgpt_slices"])
    target = target[target.split.isin(["train", "val"])].copy()
    target["slices_sha256"] = target.matgpt_slices.astype(str).map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    train_assignments = pd.read_csv(assignments_path, usecols=["split_candidate"])
    if len(train_assignments) != sum(1 for _ in CORPUS.open("rb")) - 1:
        raise AuditError("Número de asignaciones no coincide con filas del corpus.")
    candidate_hashes: dict[str, set[str]] = {"full": set(), "train_candidate": set(), "validation": set()}
    # SLICES is the second field and the corpus has no embedded quoted newlines.
    corpus = pd.read_csv(CORPUS, usecols=["SLICES"], dtype={"SLICES": str})
    hashes = corpus.SLICES.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    candidate_hashes["full"] = set(hashes)
    candidate_hashes["train_candidate"] = set(hashes[train_assignments.split_candidate.eq("train_candidate")])
    candidate_hashes["validation"] = set(hashes[train_assignments.split_candidate.eq("validation")])
    target["exact_slices_full_corpus"] = target.slices_sha256.isin(candidate_hashes["full"])
    target["exact_slices_train_candidate"] = target.slices_sha256.isin(candidate_hashes["train_candidate"])
    target["exact_slices_validation_candidate"] = target.slices_sha256.isin(candidate_hashes["validation"])
    target.drop(columns=["matgpt_slices", "slices_sha256"]).to_csv(REPORT_DIR / "structural_overlap/structural_overlap.csv", index=False)
    summary = {"status": "SLICES_EXACT_MATCH_ONLY",
               "development_structures": int(len(target)),
               "exact_sequence_matches_full_corpus": int(target.exact_slices_full_corpus.sum()),
               "exact_sequence_matches_train_candidate": int(target.exact_slices_train_candidate.sum()),
               "exact_sequence_matches_validation_candidate": int(target.exact_slices_validation_candidate.sum()),
               "duplicate_full_corpus_slices": int(corpus.SLICES.duplicated().sum()),
               "corpus_sha256": sha256_file(CORPUS), "target_dataset_sha256": sha256_file(TARGET_DATASET),
               "test_structures_read": False,
               "limitation": "Exact token sequence is an encoded-structure match; no CIF or canonical geometric StructureMatcher comparison is available in the pretraining CSV."}
    write_json(REPORT_DIR / "structural_overlap/structural_overlap_summary.json", summary)
    return summary

def main() -> None:
    """Ejecuta cotejo de cadenas SLICES en estructuras de development."""
    try:
        print(json.dumps(run_audit(), indent=2, ensure_ascii=False))
    except (AuditError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc

if __name__ == "__main__":
    main()
