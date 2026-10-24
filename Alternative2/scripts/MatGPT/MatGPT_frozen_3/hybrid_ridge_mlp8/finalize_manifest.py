#!/usr/bin/env python3
"""Finaliza hashes y metadatos de los artefactos H3 ya completados.

No lee datos científicos ni modifica resultados; corrige etiquetas de hash en
los manifiestos de corridas H3-C y registra archivos de procedencia/salida.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import (CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA, EXPECTED_COMMIT,
                    FEATURES, FOLDS, MODEL_INFO, REPORT_DIR, SCRIPT_DIR, SOURCE_MANIFEST,
                    write_json)


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo en bloques.

    Args:
        path: Archivo local a resumir.
    Returns:
        Digest hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize() -> dict[str, Any]:
    """Valida el checkpoint, repara metadatos de corrida y escribe el manifiesto.

    Returns:
        Manifiesto final de entradas y archivos de resultado.
    """
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("El hash del checkpoint MatGPT no coincide con E0.")
    run_paths = sorted((REPORT_DIR / "H3_C_ridge_retrained_delta/runs").glob("fold_*_seed_*/run_manifest.json"))
    if len(run_paths) != 15:
        raise RuntimeError(f"Se esperaban 15 manifiestos H3-C PASS, se encontraron {len(run_paths)}.")
    for path in run_paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "PASS" or run.get("test_used") is not False:
            raise RuntimeError(f"Corrida no PASS o test marcado como usado: {path}")
        # En la primera versión, checkpoint_sha256 contenía por error el hash
        # del estado MLP8. Conservarlo con su nombre correcto y fijar MatGPT.
        run.setdefault("structural_mlp8_sha256", run.get("checkpoint_sha256"))
        run["checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA
        run["source_commit"] = EXPECTED_COMMIT
        run["model_info_sha256"] = sha256(MODEL_INFO)
        run["source_manifest_sha256"] = sha256(SOURCE_MANIFEST)
        run["tokenizer_vocabulary_source"] = "vocabulario/token IDs del artefacto original model_info_property_0.json y cache auditado; sin tokens añadidos"
        fold_ridge = REPORT_DIR / "H3_B_ridge_fixed_delta" / "ridge_by_fold" / f"fold_{run['fold']}"
        ridge_model = fold_ridge / "ridge.joblib"
        ridge_info = fold_ridge / "ridge_metadata.json"
        if not ridge_model.is_file() or not ridge_info.is_file():
            raise RuntimeError(f"Falta componente Ridge requerido para H3-C: {fold_ridge}")
        ridge_info_data = json.loads(ridge_info.read_text(encoding="utf-8"))
        run["ridge_model_path"] = str(ridge_model.relative_to(Path.cwd()))
        run["ridge_model_sha256"] = sha256(ridge_model)
        run["ridge_metadata_sha256"] = sha256(ridge_info)
        run["ridge_alpha_selected_inner_validation"] = ridge_info_data["selected_alpha"]
        path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    expected_oof = REPORT_DIR / "H3_C_ridge_retrained_delta/oof_predictions.csv"
    if not expected_oof.is_file() or len(pd.read_csv(expected_oof)) != 630:
        raise RuntimeError("H3-C debe tener exactamente 630 predicciones OOF antes de cerrar el manifiesto.")

    inputs = {"checkpoint": CHECKPOINT, "model_info_vocab": MODEL_INFO,
              "source_manifest": SOURCE_MANIFEST, "dataset": DATASET,
              "fold_assignments": FOLDS, "feature_cache": FEATURES,
              "feature_manifest": FEATURES.with_name("feature_manifest.json")}
    input_hashes = {name: {"path": str(path.relative_to(Path.cwd())), "sha256": sha256(path)}
                    for name, path in inputs.items()}
    script_hashes = {str(path.relative_to(Path.cwd())): sha256(path)
                     for path in sorted(SCRIPT_DIR.rglob("*.py")) if "__pycache__" not in path.parts}
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(REPORT_DIR.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPORT_DIR))
        files[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    figures = sorted(str(p.relative_to(REPORT_DIR)) for p in (REPORT_DIR / "figures").glob("*.png"))
    non_png_figures = [str(p.relative_to(REPORT_DIR)) for p in (REPORT_DIR / "figures").rglob("*")
                       if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".eps", ".jpg", ".jpeg"}]
    if len(figures) != 24 or non_png_figures:
        raise RuntimeError(f"Figuras: se esperaban 24 PNG sin formatos adicionales: {len(figures)}, {non_png_figures}")
    manifest = {
        "status": "PASS", "experiment": "H3_matgpt_frozen_ridge_mean_mlp8_residual",
        "source_commit": EXPECTED_COMMIT, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "development_only": True, "test_labels_parsed": False,
        "completed": {"outer_folds": 5, "seeds": [42, 123, 2026],
                      "H3_C_training_runs": len(run_paths), "OOF_predictions_per_H3_variant": 630,
                      "figures_png": len(figures)},
        "input_hashes": input_hashes,
        "current_script_hashes_after_analysis": script_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "figures": figures, "non_png_figure_files": non_png_figures,
        "report_artifacts": files,
    }
    write_json(REPORT_DIR / "manifest.json", manifest)
    return manifest


def main() -> None:
    """Ejecuta finalización del manifiesto con salida JSON breve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = finalize()
    print(json.dumps({"status": manifest["status"], "runs": manifest["completed"]["H3_C_training_runs"],
                      "figures": manifest["completed"]["figures_png"],
                      "hashed_artifacts": len(manifest["report_artifacts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
