#!/usr/bin/env python3
"""Muestra conteos empíricos ABX3 desde SLICES parseados."""

from __future__ import annotations

import json

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR, write_json

def main() -> None:
    """Imprime resumen ABS3; no clasifica estructura como perovskita."""
    path = REPORT_DIR / "corpus_statistics" / "abs3_corpus_statistics.json"
    if not path.is_file():
        raise SystemExit("BLOCKED: primero audita el corpus exacto con audit_corpus_schema.py.")
    summary = json.loads(path.read_text(encoding="utf-8"))
    formulas = pd.read_csv(REPORT_DIR / "corpus_statistics" / "abs3_formulas.csv")
    formulas["pair_A_B"] = formulas[["element_a", "element_b"]].apply(lambda row: "-".join(sorted(row)), axis=1)
    frequencies = (formulas.groupby(["family", "pair_A_B", "formula_reduced_key"], as_index=False)
                   .agg(material_rows=("row_index", "size")))
    frequencies.to_csv(REPORT_DIR / "corpus_statistics" / "abs3_composition_frequencies.csv", index=False)
    summary["distinct_A_B_pairs_by_family"] = {
        family: int(part.pair_A_B.nunique()) for family, part in formulas.groupby("family")
    }
    full_s_rows = int(pd.read_csv(REPORT_DIR / "corpus_statistics" / "composition_records.csv",
                                  usecols=["n_s"]).n_s.gt(0).sum())
    summary["P_ABS3_given_contains_S_full_corpus"] = summary["ABS3_rows"] / full_s_rows
    train_table = pd.read_csv(REPORT_DIR / "corpus_statistics" / "abs3_counts_by_reconstructed_split.csv")
    chalcogen_table = pd.read_csv(REPORT_DIR / "corpus_statistics" / "chalcogenide_counts_by_reconstructed_split.csv")
    for family in ("ABS3", "ABSe3", "ABTe3"):
        for split in ("train_candidate", "validation"):
            found = train_table[(train_table.family == family) & (train_table.split_candidate == split)]
            summary[f"{family}_rows_{split}"] = int(found.material_rows.iloc[0]) if len(found) else 0
            summary[f"{family}_unique_formulas_{split}"] = int(found.unique_formula_keys.iloc[0]) if len(found) else 0
    train_s_rows = int(chalcogen_table.loc[(chalcogen_table.split_candidate == "train_candidate") &
                                           (chalcogen_table.measure == "contains_S"), "material_rows"].iloc[0])
    summary["P_ABS3_given_contains_S_train_candidate"] = summary["ABS3_rows_train_candidate"] / train_s_rows
    write_json(path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
