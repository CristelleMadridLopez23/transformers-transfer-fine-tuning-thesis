#!/usr/bin/env python3
"""Preflight H0/H1/H2 y congela los cinco folds E1.3 solo de desarrollo."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import ALT, CHECKPOINT, DATASET, E13_FOLDS, EXPECTED_CHECKPOINT_SHA, EXPECTED_COMMIT, FAMILIES, FOLDS, MODEL_INFO, REPORT_DIR, SCRIPT_DIR, SOURCE, SOURCE_MANIFEST, ExperimentError, load_development, log, sha256_file, verify_provenance, write_json


def run(out_dir: Path) -> dict[str, Any]:
    """Audita identidad del modelo/datos, folds, separación y política de métricas."""
    if out_dir.exists():
        unexpected = [p.name for p in out_dir.iterdir() if p.name != "fold_assignments.csv"]
        if unexpected:
            raise ExperimentError(f"Destino preflight tiene artefactos ajenos; se preservan: {unexpected}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_config = out_dir.parent / "config.yaml"
    source_config = SCRIPT_DIR / "config.yaml"
    if report_config.exists():
        if report_config.read_bytes() != source_config.read_bytes():
            raise ExperimentError(f"Ya existe config.yaml distinto; se preserva: {report_config}")
    else:
        shutil.copy2(source_config, report_config)
    identity = verify_provenance()
    data = load_development()
    if not E13_FOLDS.is_file():
        raise ExperimentError(f"Faltan folds E1.3: {E13_FOLDS}")
    raw_assignments = pd.read_csv(E13_FOLDS)
    folds = raw_assignments.loc[raw_assignments.historical_split.isin(["train", "val"])].copy()
    if folds.duplicated(["structure_id", "outer_fold"]).any() or len(folds) != 1050:
        raise ExperimentError("Folds E1.3 no contienen 210 IDs únicos por cada uno de los 5 folds.")
    if set(folds.structure_id.astype(str)) != set(data.structure_id):
        raise ExperimentError("Los folds E1.3 no corresponden exactamente a los 210 IDs development.")
    for fold in range(5):
        part = folds.loc[folds.outer_fold.eq(fold)]
        groups = part.drop_duplicates("composition_group")
        outer = set(groups.loc[groups.outer_role.eq("outer_validation"), "composition_group"])
        inner_val = set(groups.loc[groups.inner_role.eq("inner_validation"), "composition_group"])
        inner_train = set(groups.loc[groups.inner_role.eq("inner_train"), "composition_group"])
        if (len(outer), len(inner_val), len(inner_train)) != (14, 11, 45):
            raise ExperimentError(f"Conteos incorrectos fold {fold}: outer={len(outer)} val={len(inner_val)} train={len(inner_train)}")
        if outer & inner_val or outer & inner_train or inner_val & inner_train:
            raise ExperimentError(f"Fuga por composición en fold {fold}.")
        if part.groupby("composition_group").outer_role.nunique().ne(1).any():
            raise ExperimentError(f"Los tres polimorfos no viajan juntos en fold {fold}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_out = out_dir / "fold_assignments.csv"
    if fold_out.exists():
        existing = pd.read_csv(fold_out)
        if not existing.equals(folds.reset_index(drop=True)):
            raise ExperimentError("Ya existe fold_assignments.csv con contenido distinto; no se sobrescribe.")
    else:
        folds.to_csv(fold_out, index=False)
    # Strict state load through the already audited E0 loader, then check all weights frozen.
    model, stoi, block_size, _ = load_model(SOURCE, MODEL_INFO, CHECKPOINT,
                                            SOURCE / "config" / "uncond.yaml")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ExperimentError("Se detectó un parámetro entrenable en MatGPT.")
    del model
    recon_source = ALT / "reports" / "MatGPT_diagnostics" / "common" / "baseline_reconciliation.md"
    if not recon_source.is_file():
        raise ExperimentError("Falta reconciliación histórica D1-D3 de GS.")
    reconciliation = """# Reconciliación de GS antes de H0/H1/H2\n\n## Diferencia histórica 0.533 vs 0.495\n\nLos dos resultados publicados usan las mismas predicciones OOF E2-A, 210 estructuras y 70 composiciones, pero distintas reglas. **0.533** es `argmin` sin excluir composiciones casi degeneradas ni abstenciones predichas. **0.495** excluye las 3 composiciones cuyo gap DFT entre los dos mínimos es ≤1 meV/átomo, no acredita empates predichos y promedia primero dentro de fold, después entre semillas. El valor calculado directamente sobre las 67 composiciones elegibles es ≈0.493; no es el agregado fold→semilla publicado.\n\nH0/H1/H2 usarán como métrica principal la regla estricta: DFT-GS evaluable solo si el gap al segundo mínimo supera 1 meV/átomo; una predicción empatada a tolerancia de 1 meV/átomo será abstención y no acierto. Siempre se informa el denominador elegible y excluido. Pairwise accuracy excluye pares DFT empatados (|ΔE|≤1 meV/átomo); un empate predicho no cuenta como acierto. Los valores históricos 0.533 y 0.495 se conservan solo como reconciliación; no se mezclan con resultados nuevos.\n\n## Convención en este experimento\n\nSe promedian primero las tres semillas por estructura para la matriz de confusión de ensamble, de modo que cada composición y pareja física cuente una sola vez. Las tablas por ejecución reportan fold y semilla separadamente. La exactitud OOF macro para estimación principal se calcula por composición elegible; se adjuntan también las agregaciones fold→semilla para que los denominadores queden visibles. El conjunto histórico de validación ya participó en preselección previa, por lo que esta CV sigue siendo exploratoria. No se consulta ni evalúa el test histórico.\n\nFuente reconciliada: `Alternative2/reports/MatGPT_diagnostics/common/baseline_reconciliation.md`; comparación de predicciones anteriores solo como antecedente.\n"""
    (out_dir / "baseline_reconciliation.md").write_text(reconciliation, encoding="utf-8")
    fold_summary = {}
    for fold in range(5):
        part = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
        fold_summary[str(fold)] = {"outer_validation_compositions": int(part.outer_role.eq("outer_validation").sum()),
            "inner_train_compositions": int(part.inner_role.eq("inner_train").sum()),
            "inner_validation_compositions": int(part.inner_role.eq("inner_validation").sum())}
    source_paths = [DATASET, E13_FOLDS, CHECKPOINT, MODEL_INFO, SOURCE_MANIFEST, SOURCE / "config" / "uncond.yaml",
                    SOURCE / "utils" / "models.py", recon_source]
    script_paths = sorted(SCRIPT_DIR.glob("*.py"))
    provenance = {"status": "PASS", "resolved_project_root": str(ALT.resolve()), "resolved_sources": {"dataset": str(DATASET.resolve()), "e13_folds": str(E13_FOLDS.resolve()), "checkpoint": str(CHECKPOINT.resolve()), "tokenizer_vocab": str(MODEL_INFO.resolve())},
        "source_hashes_sha256": {str(path.relative_to(ALT)): sha256_file(path) for path in source_paths},
        "scripts_hashes_sha256": {path.name: sha256_file(path) for path in script_paths},
        "identity": identity, "strict_state_load": True, "backbone_trainable_parameters": 0,
        "development_structures": len(data), "development_compositions": int(data.composition_id.nunique()),
        "test_ids_in_fold_copy": 0, "test_labels_used": False, "fold_manifest_sha256": sha256_file(fold_out),
        "fold_summary": fold_summary, "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "energy_target": "static GGA/PBE DFT total energy per atom, eV/atom; not formation energy", "historical_validation_preselection_bias": True}
    write_json(out_dir / "provenance.json", provenance)
    write_json(out_dir / "integrity_checks.json", {"status": "PASS", "checkpoint_sha256_match": identity["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA,
        "commit_match": identity["source_commit"] == EXPECTED_COMMIT, "strict_load": True, "backbone_trainable_parameters": 0,
        "unique_development_ids": len(data), "development_compositions": int(data.composition_id.nunique()), "fold_rows_without_test": len(folds),
        "outer_folds": 5, "fold_summary": fold_summary, "per_composition_three_families": True,
        "historical_test_used": False, "training_started": False})
    return provenance


def main() -> None:
    """Valida carpeta de salida y ejecuta el preflight reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    try:
        result = run(args.out_dir)
        log(logging.INFO, "preflight_complete", status=result["status"], folds=5, dev=210, test=False)
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(logging.ERROR, "preflight_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
