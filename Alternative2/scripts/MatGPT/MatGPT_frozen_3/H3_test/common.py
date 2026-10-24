#!/usr/bin/env python3
"""Rutas, lecturas seguras y validaciones del holdout final H3-TEST."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
SCRIPT_DIR = ALT / "scripts/MatGPT_frozen_3/H3_test"
REPORT_DIR = ALT / "reports/MatGPT_frozen_3/H3_test"
DATASET = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen_2/v1/fold_assignments.csv"
H2_FEATURES = ALT / "reports/MatGPT_frozen_2/v1/features/frozen_features_development.npz"
H2_OOF = ALT / "reports/MatGPT_frozen_2/v1/oof_predictions_enriched.csv"
H2_RUNS = ALT / "reports/MatGPT_frozen_2/v1/runs/H2"
R4_ROOT = ALT / "reports/MatGPT_frozen_3/run_20261023_v2/R4_layers_pooling"
R4_OOF = R4_ROOT / "oof_predictions.csv"
R4_SELECTION = R4_ROOT / "validation_selection.csv"
H3_MANIFEST = ALT / "reports/MatGPT_frozen_3/hybrid_ridge_mlp8/manifest.json"
H3_REPORT = ALT / "reports/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_scientific_report.md"
FEATURES = ALT / "reports/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
FEATURE_MANIFEST = FEATURES.with_name("feature_manifest.json")
SOURCE_ROOT = ALT / "models/MatGPT-source/mat-concat"
SOURCE_MANIFEST = ALT / "models/MatGPT-source.download-manifest.json"
CHECKPOINT = SOURCE_ROOT / "mat-uncond/weights/transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
EXPECTED_CHECKPOINT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
EXPECTED_DATASET_SHA = "0719f496c96297761ede3a7e2e7ef57aa7b62400fb0004e679a66430a70c31e8"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
FAMILY_LABELS = {"dist_perovskite": "Perovskita distorsionada", "hexagonal": "Hexagonal", "needle": "Tipo aguja"}
SEEDS = (42, 123, 2026)
OUTER_FOLDS = tuple(range(5))
TAU_MEV = 1.0
LOGGER = logging.getLogger("matgpt_h3_test")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


class H3TestError(RuntimeError):
    """Bloqueo de procedencia, fuga, incompatibilidad o evaluación."""


def log(event: str, **fields: Any) -> None:
    """Escribe un evento JSON compacto y estructurado."""
    LOGGER.info(json.dumps({"level": "INFO", "event": event, **fields}, ensure_ascii=False, default=str))


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques sin interpretar el contenido del archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe JSON UTF-8 y crea su directorio padre."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_dev_targets_safely(path: Path = DATASET) -> pd.DataFrame:
    """Lee etiquetas únicamente de train/val, filtrando antes de parsear energía.

    Args:
        path: CSV histórico que contiene partición y etiquetas.
    Returns:
        Datos de desarrollo con 210 estructuras y etiquetas DFT por átomo.
    """
    required = ("structure_id", "reduced_formula", "polymorph_family", "split", "energy_per_atom_ev", "matgpt_slices")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(required) - set(reader.fieldnames):
            raise H3TestError("El dataset no contiene las columnas históricas necesarias.")
        for row in reader:
            if row["split"] not in {"train", "val"}:
                continue
            rows.append({key: row[key] for key in required})
    data = pd.DataFrame(rows).rename(columns={"reduced_formula": "composition_id", "energy_per_atom_ev": "E_DFT_eV_atom"})
    data["structure_id"] = data.structure_id.astype(str)
    data["E_DFT_eV_atom"] = pd.to_numeric(data.E_DFT_eV_atom, errors="coerce")
    if len(data) != 210 or data.structure_id.duplicated().any() or not np.isfinite(data.E_DFT_eV_atom).all():
        raise H3TestError("Train/validation no coincide con 210 IDs únicos y etiquetas finitas.")
    if data.groupby("composition_id").size().ne(3).any():
        raise H3TestError("Cada composición de desarrollo debe conservar tres polimorfos.")
    return data.reset_index(drop=True)


def read_test_inputs_safely(path: Path = DATASET) -> pd.DataFrame:
    """Lee solo IDs, familias y SLICES de test; excluye columnas de energía.

    Args:
        path: CSV histórico del dataset.
    Returns:
        Las 24 entradas de inferencia, sin targets ni conteo atómico.
    """
    columns = ["structure_id", "reduced_formula", "polymorph_family", "split", "matgpt_slices"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame.loc[frame.split.eq("test"), columns].copy()
    frame = frame.rename(columns={"reduced_formula": "composition_id"})
    frame["structure_id"] = frame.structure_id.astype(str)
    frame["composition_id"] = frame.composition_id.astype(str)
    frame["matgpt_slices"] = frame.matgpt_slices.astype(str)
    if len(frame) != 24 or frame.structure_id.duplicated().any() or frame.composition_id.nunique() != 8:
        raise H3TestError("El holdout esperado debe tener 24 IDs y ocho composiciones.")
    for formula, group in frame.groupby("composition_id"):
        if len(group) != 3 or set(group.polymorph_family) != set(FAMILIES):
            raise H3TestError(f"Triplete/familias inválidos en test para {formula}.")
    return frame.sort_values(["composition_id", "polymorph_family"], kind="stable").reset_index(drop=True)


def read_test_labels_after_lock(path: Path = DATASET) -> pd.DataFrame:
    """Lee DFT de test solo después del lock y usando solo filas split=test.

    Args:
        path: CSV histórico original.
    Returns:
        IDs test con energía total, número de átomos y energía por átomo.
    """
    fields = ("structure_id", "split", "num_atoms", "energy_total_ev", "energy_per_atom_ev")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(fields) - set(reader.fieldnames):
            raise H3TestError("Faltan columnas de etiqueta para la evaluación bloqueada.")
        for row in reader:
            if row["split"] != "test":
                continue
            rows.append({field: row[field] for field in fields if field != "split"})
    labels = pd.DataFrame(rows)
    labels["structure_id"] = labels.structure_id.astype(str)
    for key in ("num_atoms", "energy_total_ev", "energy_per_atom_ev"):
        labels[key] = pd.to_numeric(labels[key], errors="coerce")
    labels = labels.rename(columns={"energy_total_ev": "E_DFT_total_eV", "num_atoms": "num_atoms_DFT",
                                   "energy_per_atom_ev": "E_DFT_eV_atom"})
    if len(labels) != 24 or labels.structure_id.duplicated().any() or not np.isfinite(labels.select_dtypes("number")).all().all():
        raise H3TestError("Etiquetas test incompletas, repetidas o no finitas.")
    if not np.allclose(labels.E_DFT_total_eV / labels.num_atoms_DFT, labels.E_DFT_eV_atom, atol=1e-8, rtol=1e-8):
        raise H3TestError("Energía total/número de átomos no reproduce energía por átomo.")
    return labels
