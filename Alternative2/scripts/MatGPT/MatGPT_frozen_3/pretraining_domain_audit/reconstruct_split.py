#!/usr/bin/env python3
"""Reconstruye el split candidato de MatGPT con randperm y semilla del YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import (ALT, REPORT_DIR, SOURCE_CONFIG, AuditError, read_yaml_scalars,
                          sha256_file, write_json)


def reconstruct(corpus: Path) -> dict[str, Any]:
    """Aplica torch.randperm a filas conservando el orden del CSV.

    Args:
        corpus: CSV exacto nombrado en uncond.yaml.

    Returns:
        Resumen de una reconstrucción de configuración, no prueba de exposición.
    """
    if not corpus.is_file() or corpus.name != "Training_dataset_alex20_with_encoding.csv":
        raise AuditError("Se requiere el CSV exacto Training_dataset_alex20_with_encoding.csv.")
    records_path = REPORT_DIR / "corpus_statistics" / "composition_records.csv"
    if not records_path.is_file():
        raise AuditError("Ejecuta primero run_audit.py --corpus para validar todas las cadenas SLICES.")
    records = pd.read_csv(records_path)
    n_rows = sum(1 for _ in corpus.open("rb")) - 1
    if len(records) != n_rows or records["row_index"].tolist() != list(range(n_rows)):
        raise AuditError("El inventario parseado no cubre las filas del CSV en orden; no se reconstruye split.")
    config = read_yaml_scalars(SOURCE_CONFIG)
    seed = int(config["seed"])
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_rows, generator=generator).tolist()
    n_train = int(0.9 * n_rows)
    split_by_row = ["validation"] * n_rows
    for index in permutation[:n_train]:
        split_by_row[index] = "train_candidate"
    records["split_candidate"] = split_by_row
    records.to_csv(REPORT_DIR / "corpus_statistics" / "reconstructed_split_assignments.csv", index=False)
    summary: dict[str, Any] = {"status": "CONFIG_REPRODUCED",
        "interpretation": "candidato reproducido desde el CSV disponible; identidad histórica del archivo/orden no demostrada",
        "corpus_path": str(corpus.resolve()), "corpus_sha256": sha256_file(corpus),
        "source_config_sha256": sha256_file(SOURCE_CONFIG), "seed": seed,
        "torch_version": torch.__version__, "randperm_device": "cpu",
        "rows": n_rows, "train_candidate_rows": n_train, "validation_candidate_rows": n_rows - n_train,
        "train_candidate_unique_formulas": int(records.loc[records.split_candidate.eq("train_candidate"), "formula_reduced_key"].nunique()),
        "validation_candidate_unique_formulas": int(records.loc[records.split_candidate.eq("validation"), "formula_reduced_key"].nunique()),
        "caveat": "No afirmar que estas filas fueron vistas por el checkpoint sin hash/registro de entrenamiento del corpus original."}
    write_json(REPORT_DIR / "corpus_statistics" / "reconstructed_split_summary.json", summary)
    domain_rows: list[dict[str, Any]] = []
    for split_name, part in records.groupby("split_candidate", sort=True):
        for element, column in (("S", "n_s"), ("Se", "n_se"), ("Te", "n_te")):
            selected = part[part[column].gt(0)]
            domain_rows.append({"split_candidate": split_name, "measure": f"contains_{element}",
                                "material_rows": int(len(selected)),
                                "unique_formula_keys": int(selected.formula_reduced_key.nunique())})
        any_calcogen = part[part[["n_s", "n_se", "n_te"]].gt(0).any(axis=1)]
        domain_rows.append({"split_candidate": split_name, "measure": "contains_any_S_Se_Te",
                            "material_rows": int(len(any_calcogen)),
                            "unique_formula_keys": int(any_calcogen.formula_reduced_key.nunique())})
    domain_frame = pd.DataFrame(domain_rows)
    domain_frame.to_csv(REPORT_DIR / "corpus_statistics" / "chalcogenide_counts_by_reconstructed_split.csv", index=False)
    abs3_path = REPORT_DIR / "corpus_statistics" / "abs3_formulas.csv"
    if abs3_path.is_file():
        abs3 = pd.read_csv(abs3_path).merge(records[["row_index", "split_candidate"]], on="row_index", validate="one_to_one")
        family_rows = []
        for (split_name, family), part in abs3.groupby(["split_candidate", "family"]):
            family_rows.append({"split_candidate": split_name, "family": family, "material_rows": len(part),
                                "unique_formula_keys": part.formula_reduced_key.nunique(),
                                "distinct_A_B_pairs": part[["element_a", "element_b"]].apply(lambda row: tuple(sorted(row)), axis=1).nunique()})
        pd.DataFrame(family_rows).to_csv(REPORT_DIR / "corpus_statistics" / "abs3_counts_by_reconstructed_split.csv", index=False)
    manifest_path = REPORT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_reconstruction"] = summary
    manifest["corpus_status"] = "CONFIG_REPRODUCED_CANDIDATE"
    manifest["provenance"]["certainty"] = "CONFIG_REPRODUCED"
    manifest["status"] = "PARTIAL"
    write_json(manifest_path, manifest)
    return summary


def main() -> None:
    """Valida los argumentos y guarda la reconstrucción candidata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ALT / "01_data/matgpt y matter/Training_dataset_alex20_with_encoding.csv")
    args = parser.parse_args()
    try:
        result = reconstruct(args.corpus)
    except (AuditError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
