#!/usr/bin/env python3
"""Audita preflight E1.3 o verifica entregables finales sin abrir el test histórico."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import CANDIDATES, OUT, SEEDS, E13Error, get_logger, load_inputs, log, sha256_file, write_json
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.prepare_group_folds import strict_backbone_audit

LOGGER = get_logger("matgpt_e1_3_audit")


def preflight(out: Path) -> dict[str, Any]:
    """Verifica entradas, hashes, agrupación, caches y carga estricta del backbone."""
    data, matrices, provenance = load_inputs()
    model = strict_backbone_audit()
    if set(matrices) != {"last_token", "masked_mean"}:
        raise E13Error("Se requieren ambos caches de embeddings.")
    report = {"status": "PASS", "phase": "preflight", "checkpoint": model,
        "dataset_rows": len(data), "unique_ids": int(data.structure_id.nunique()),
        "development_structures": int(data.split.isin(["train", "val"]).sum()),
        "development_compositions": int(data.loc[data.split.isin(["train", "val"]), "composition_group"].nunique()),
        "historical_test_structures_excluded": int(data.split.eq("test").sum()),
        "historical_test_compositions_excluded": int(data.loc[data.split.eq("test"), "composition_group"].nunique()),
        "cache_shapes": {key: list(value.shape) for key, value in matrices.items()},
        "cache_all_finite": {key: bool(np.isfinite(value).all()) for key, value in matrices.items()},
        "embedding_audits": provenance["embed_audits"],
        "extractor_code_verification": provenance["extractor_verification"],
        "e12_hashes": provenance["e12_hashes"],
        "historical_test_metrics_loaded": False, "test_used_for_selection": False,
        "message": "Los 234 IDs y caches se contrastan por identidad; solo 210 filas train/val pasan al preparador de CV."}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "preflight_audit.json", report)
    log(LOGGER, logging.INFO, "e13_preflight_passed", structures=len(data), development=210,
        excluded_test=24, backbone_frozen=True, out_dir=out.resolve())
    return report


def results_audit(out: Path) -> dict[str, Any]:
    """Comprueba OOF/checkpoints/resultados y confirma exclusión total del test."""
    required = [out / "fold_assignments.csv", out / "provenance.json", out / "training_manifest.json",
                out / "oof_predictions.csv", out / "metrics_per_fold_seed.csv", out / "metrics_oof_per_seed.csv",
                out / "metrics_composition.csv", out / "metrics_by_family.csv", out / "paired_comparisons.csv", out / "paired_bootstrap_ci.json",
                out / "E1_3_cross_validation_report.md", out / "evaluation_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise E13Error(f"Faltan artefactos finales: {missing}")
    training = json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((out / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(out / "fold_assignments.csv", dtype={"structure_id": str})
    oof = pd.read_csv(out / "oof_predictions.csv", dtype={"structure_id": str})
    if training.get("status") != "PASS" or training.get("completed_fits") != 45 or training.get("checkpoint_count") != 45:
        raise E13Error("No se completaron exactamente las 45 cabezas con sus checkpoints.")
    if len(training.get("runs", [])) != 45:
        raise E13Error("El manifiesto no registra archivos/hashes para las 45 ejecuciones.")
    for run in training["runs"]:
        checkpoint_path = out / run["checkpoint"]
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != run.get("checkpoint_sha256"):
            raise E13Error(f"Hash/ruta de checkpoint inconsistente: {run.get('checkpoint')}")
    if evaluation.get("status") != "PASS" or evaluation.get("test_historical_rows_used") != 0:
        raise E13Error("Evaluación incompleta o registra uso del test histórico.")
    if not oof.historical_split.isin(["train", "val"]).all() or len(oof) != 3*3*210:
        raise E13Error("OOF contiene test o un número inesperado de predicciones.")
    counts = oof.groupby(["configuration", "seed", "structure_id"]).size()
    if len(counts) != 3*3*210 or not counts.eq(1).all():
        raise E13Error("Hay duplicados/faltantes en las predicciones OOF por seed y structure_id.")
    excluded = assignments[assignments.outer_role.eq("historical_test_excluded")]
    if len(excluded) != 24 or excluded.composition_group.nunique() != 8 or set(excluded.structure_id) & set(oof.structure_id):
        raise E13Error("Exclusión del test histórico no es completa en los entregables.")
    outer_valid = assignments[assignments.outer_role.eq("outer_validation")]
    fold_counts = outer_valid.groupby("outer_fold").agg(structures=("structure_id", "size"), compositions=("composition_group", "nunique"))
    if len(fold_counts) != 5 or not fold_counts.structures.eq(42).all() or not fold_counts.compositions.eq(14).all():
        raise E13Error("fold_assignments no conserva 5 outer-validation con 42 filas/14 composiciones cada uno.")
    checkpoint_paths = sorted((out / "checkpoints").rglob("*.pt"))
    if len(checkpoint_paths) != 45:
        raise E13Error(f"Se esperaban 45 checkpoints de cabeza, se hallaron {len(checkpoint_paths)}.")
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not payload.get("backbone_frozen") or payload.get("trainable_parameters") not in {8*768 + 8 + 8 + 1, 32*768 + 32 + 32 + 1}:
            raise E13Error(f"Checkpoint no contiene únicamente la cabeza esperada: {path}")
        if set(payload.get("state_dict", {})) != {"layers.0.weight", "layers.0.bias", "layers.3.weight", "layers.3.bias"}:
            raise E13Error(f"El checkpoint incluye parámetros ajenos a la cabeza: {path}")
    figures = sorted((out / "figures").glob("*.png"))
    if len(figures) < 7:
        raise E13Error(f"Faltan figuras PNG; se esperaban al menos 7, se hallaron {len(figures)}.")
    report = {"status": "PASS", "phase": "results", "completed_fits": 45, "folds": 5,
        "seeds": list(SEEDS), "oof_rows": len(oof), "oof_structures_per_candidate_seed": 210,
        "historical_test_rows_in_oof": 0, "historical_test_groups_excluded": 8,
        "checkpoint_count": len(checkpoint_paths), "all_head_only_state_dicts": True,
        "figures_png": [path.name for path in figures], "test_read_for_metrics_or_selection": False,
        "integrity_manifest_checked": True}
    write_json(out / "integrity_checks.json", report)
    log(LOGGER, logging.INFO, "e13_results_audit_passed", fits=45, oof_rows=len(oof), figures=len(figures), test_used=False)
    return report


def main() -> None:
    """Expone fases input preflight y auditoría final de forma explícita."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "results"), required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        preflight(args.out_dir) if args.phase == "preflight" else results_audit(args.out_dir)
    except (E13Error, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "e13_audit_failed", phase=args.phase, error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
