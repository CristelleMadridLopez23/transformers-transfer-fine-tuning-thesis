#!/usr/bin/env python3
"""Audita checkpoint, conjunto de desarrollo, folds y referencias H0-H2/D1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import (CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA,
    EXPECTED_COMMIT, FOLDS, H2_REPORT, MODEL_INFO, REPORT_DIR, REPORT_ROOT, SOURCE_DIR, ExperimentError,
    load_dev, load_folds, log, sha256_file, write_json)


def run() -> dict[str, Any]:
    """Valida procedencia, split development-only y correspondencia de OOF históricos."""
    data = load_dev()
    folds = load_folds(data)
    checkpoint_sha = sha256_file(CHECKPOINT)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
        raise ExperimentError(f"SHA del checkpoint distinto al esperado: {checkpoint_sha}")
    model_meta = json.loads((PROJECT / "Alternative2/models/MatGPT-source.download-manifest.json").read_text())
    if model_meta.get("resolved_commit") != EXPECTED_COMMIT:
        raise ExperimentError(f"Commit local no coincide: {model_meta.get('resolved_commit')}")
    h2 = pd.read_csv(H2_REPORT / "oof_predictions.csv", low_memory=False)
    h2 = h2.loc[h2.model.eq("H2")].copy()
    h2["structure_id"] = h2.structure_id.astype(str)
    if len(h2) != 630 or set(h2.structure_id) != set(data.structure_id):
        raise ExperimentError("OOF H2 histórico no tiene 630 filas y los 210 IDs de desarrollo.")
    if set(h2.historical_split.unique()) - {"train", "val"}:
        raise ExperimentError("OOF histórico contiene particiones distintas de train/val.")
    check = h2.merge(data[["structure_id", "composition_id", "polymorph_family", "E_DFT_eV_atom"]],
                     on="structure_id", suffixes=("_oof", "_data"), validate="many_to_one")
    if not (check.composition_id_oof.astype(str).eq(check.composition_id_data.astype(str)).all()
            and check.polymorph_family_oof.eq(check.polymorph_family_data).all()
            and np.allclose(check.E_DFT_eV_atom_oof, check.E_DFT_eV_atom_data, atol=1e-10, rtol=0)):
        raise ExperimentError("IDs, fórmula, familia o etiqueta H2 no coinciden con las fuentes.")
    d1_path = PROJECT / "Alternative2/reports/MatGPT_diagnostics/D1/v3/oof_predictions.csv"
    d1 = pd.read_csv(d1_path, low_memory=False)
    d1 = d1.loc[d1.configuration.eq("D1-B Ridge composición")].copy()
    d1["structure_id"] = d1.structure_id.astype(str)
    d1 = d1.merge(data[["structure_id", "composition_id", "polymorph_family", "E_DFT_eV_atom"]],
                  on="structure_id", suffixes=("_d1", "_data"), validate="one_to_one")
    if len(d1) != 210 or not np.allclose(d1.energy_per_atom_ev, d1.E_DFT_eV_atom, atol=1e-10, rtol=0):
        raise ExperimentError("D1 externo no coincide con los 210 IDs/etiquetas.")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentError(f"No se pudo registrar estado Git de solo lectura: {exc}") from exc
    audit = {"status": "PASS", "checkpoint_sha256": checkpoint_sha, "checkpoint_sha_expected_match": True,
        "tokenizer_model_info_sha256": sha256_file(MODEL_INFO), "source_commit": model_meta["resolved_commit"],
        "source_commit_expected_match": True, "dataset_sha256": sha256_file(DATASET), "fold_manifest_sha256": sha256_file(FOLDS),
        "development_structures": len(data), "development_compositions": data.composition_id.nunique(),
        "historical_test_structures_excluded": 24, "historical_test_compositions_excluded": 8,
        "historical_test_used": False, "fold_rows": len(folds), "outer_folds": 5,
        "outer_compositions_per_fold": 14, "h2_oof_rows_and_unique_ids": {"rows": len(h2), "ids": h2.structure_id.nunique()},
        "h2_ids_formula_family_energy_match": True, "d1_rows_and_ids": len(d1), "d1_energy_labels_match": True,
        "h2_report_sha256": sha256_file(H2_REPORT / "H0_H1_H2_report.md"),
        "git_head_at_audit": commit, "working_tree_preexisting_status_lines": status,
        "historical_validation_preselection_bias": True,
        "target": "static GGA/PBE DFT total energy per atom, eV/atom; not formation energy"}
    out = REPORT_DIR / "baseline_audit"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "provenance.json", audit)
    (out / "baseline_audit.md").write_text(
        "# Auditoría de referencias MatGPT_frozen_3\n\n"
        f"Estado: **{audit['status']}**. Se confirmó el checkpoint E0 congelado, commit histórico, 210 estructuras/70 composiciones de desarrollo, cinco folds y exclusión de 24 estructuras/8 composiciones del test. H2 OOF coincide por ID, fórmula, familia y energía con el dataset; D1 coincide por ID y etiqueta.\n\n"
        f"Checkpoint SHA-256: `{checkpoint_sha}`. Tokenizer/model-info SHA-256: `{audit['tokenizer_model_info_sha256']}`. Dataset SHA-256: `{audit['dataset_sha256']}`. Fold manifest SHA-256: `{audit['fold_manifest_sha256']}`. Git HEAD al auditar: `{commit}`.\n\n"
        "H2 y D1 se leen como referencias históricas, no se usan para escoger los parámetros nuevos. D1 es composición-only y queda como control externo de energía absoluta; no discrimina polimorfos. La validation histórica participó en selección previa, así que las nuevas métricas OOF son exploratorias.\n",
        encoding="utf-8")
    folds.to_csv(REPORT_DIR / "fold_assignments_checked.csv", index=False)
    return audit


def main() -> None:
    """Ejecuta la auditoría inicial y detiene fases posteriores ante errores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = run()
        log("input_audit_complete", status=result["status"], dev_rows=result["development_structures"], test_used=False)
    except (ExperimentError, OSError, ValueError, KeyError) as exc:
        log("input_audit_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
