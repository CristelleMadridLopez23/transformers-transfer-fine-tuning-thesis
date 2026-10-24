#!/usr/bin/env python3
"""Carga y valida únicamente los artefactos de desarrollo del experimento H3."""
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
SCRIPT_DIR = ALT / "scripts/MatGPT_frozen_3/hybrid_ridge_mlp8"
REPORT_DIR = ALT / "reports/MatGPT_frozen_3/hybrid_ridge_mlp8"
DATASET = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen_2/v1/fold_assignments.csv"
FEATURES = ALT / "reports/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
FEATURE_MANIFEST = FEATURES.with_name("feature_manifest.json")
CHECKPOINT = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
SOURCE_MANIFEST = ALT / "models/MatGPT-source.download-manifest.json"
H2_ROOT = ALT / "reports/MatGPT_frozen_2/v1"
R4_ROOT = ALT / "reports/MatGPT_frozen_3/run_20261023_v2/R4_layers_pooling"
HISTORICAL_DEV_SHA = "0719f496c96297761ede3a7e2e7ef57aa7b62400fb0004e679a66430a70c31e8"
EXPECTED_CHECKPOINT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
SEEDS = (42, 123, 2026)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LOGGER = logging.getLogger("matgpt_h3")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


class H3Error(RuntimeError):
    """Bloqueo de integridad, alineación o protocolo del experimento."""


def log(event: str, **fields: Any) -> None:
    """Registra eventos JSON para ejecución y auditoría."""
    LOGGER.info(json.dumps({"level": "INFO", "event": event, **fields}, ensure_ascii=False, default=str))


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 en bloques sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe JSON indentado, creando solo carpetas H3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_development_without_parsing_test_targets(path: Path = DATASET) -> pd.DataFrame:
    """Lee filas train/val y omite test antes de acceder al campo de energía.

    Args:
        path: CSV histórico con columna split.
    Returns:
        Solo 210 estructuras y sus etiquetas de desarrollo.
    """
    if not path.is_file():
        raise H3Error(f"No existe el dataset: {path}")
    needed = ("structure_id", "reduced_formula", "polymorph_family", "split", "energy_per_atom_ev", "matgpt_slices")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or set(needed) - set(header):
            raise H3Error(f"Faltan columnas requeridas en dataset: {sorted(set(needed) - set(header or []))}")
        index = {name: header.index(name) for name in needed}
        for row in reader:
            if len(row) != len(header):
                raise H3Error("Fila CSV mal formada; se detiene el preflight.")
            split = row[index["split"]]
            if split not in {"train", "val"}:
                continue
            # Solo después de descartar test se extrae/convierte la etiqueta.
            rows.append({name: row[index[name]] for name in needed})
    data = pd.DataFrame(rows)
    data["structure_id"] = data.structure_id.astype(str)
    data["composition_id"] = data.reduced_formula.astype(str)
    data["E_DFT_eV_atom"] = pd.to_numeric(data.energy_per_atom_ev, errors="coerce")
    data["polymorph_family"] = data.polymorph_family.astype(str)
    if len(data) != 210 or data.structure_id.duplicated().any() or not np.isfinite(data.E_DFT_eV_atom).all():
        raise H3Error("El conjunto de desarrollo no contiene 210 IDs únicos con etiquetas finitas.")
    if data.groupby("composition_id").size().ne(3).any():
        raise H3Error("Cada composición de desarrollo debe conservar tres polimorfos.")
    if any(set(group.polymorph_family) != set(FAMILIES) for _, group in data.groupby("composition_id")):
        raise H3Error("Hay familias polimórficas incompletas en desarrollo.")
    return data.reset_index(drop=True)


def load_fold_manifest(dev_ids: set[str]) -> pd.DataFrame:
    """Carga folds existentes; no contiene etiquetas energéticas."""
    folds = pd.read_csv(FOLDS)
    folds = folds.loc[folds.historical_split.isin(["train", "val"])].copy()
    folds["structure_id"] = folds.structure_id.astype(str)
    if len(folds) != 1050 or set(folds.structure_id) != dev_ids:
        raise H3Error("El manifiesto no conserva exactamente los 210 IDs de desarrollo por fold.")
    for fold in range(5):
        part = folds.loc[folds.outer_fold.eq(fold)]
        if part.groupby("composition_group").outer_role.nunique().gt(1).any():
            raise H3Error(f"Fold {fold}: polimorfos separados entre outer train/eval.")
        unique = part.drop_duplicates("composition_group")
        if len(unique) != 70 or int(unique.outer_role.eq("outer_validation").sum()) != 14:
            raise H3Error(f"Fold {fold}: se esperaban 70 grupos, 14 en outer validation.")
    return folds


def load_features(dev_ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carga el cache auditado y alinea L03_MEAN/FINAL_END por structure_id."""
    if not FEATURES.is_file() or not FEATURE_MANIFEST.is_file():
        raise H3Error("Falta cache de features o su manifiesto.")
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("historical_test_used_or_extracted") is not False:
        raise H3Error("El manifiesto de features no certifica cache PASS sin test.")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise H3Error("El cache no corresponde al checkpoint esperado.")
    with np.load(FEATURES, allow_pickle=False) as archive:
        ids = archive["structure_id"].astype(str).tolist()
        if ids != dev_ids:
            raise H3Error("El orden de IDs del cache no coincide exactamente con desarrollo.")
        feats = {key: np.array(archive[key], dtype=np.float32, copy=True) for key in ("L03_MEAN", "FINAL_END")}
    if any(array.shape != (210, 768) or not np.isfinite(array).all() for array in feats.values()):
        raise H3Error("Features inválidas, no finitas o con dimensión distinta a 768.")
    return feats, manifest


def ordered_compositions(data: pd.DataFrame, indices: np.ndarray, feature: np.ndarray | None = None) -> tuple[list[str], np.ndarray | None, np.ndarray]:
    """Devuelve composiciones ordenadas en tripletes de familias canónicas."""
    subset = data.iloc[indices].copy()
    rank = {name: i for i, name in enumerate(FAMILIES)}
    subset["_family_order"] = subset.polymorph_family.map(rank)
    subset = subset.sort_values(["composition_id", "_family_order"], kind="stable")
    row_indices = subset.index.to_numpy()
    if len(row_indices) % 3 or subset.groupby("composition_id").size().ne(3).any():
        raise H3Error("El subconjunto no está formado por tripletes completos.")
    x = feature[row_indices].reshape(-1, 3, feature.shape[-1]) if feature is not None else None
    y = subset.E_DFT_eV_atom.to_numpy(np.float32).reshape(-1, 3)
    return subset.composition_id.iloc[::3].tolist(), x, y


def split_indices(data: pd.DataFrame, folds: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa inner-train, inner-validation y outer-eval exclusivamente por grupo."""
    part = folds.loc[folds.outer_fold.eq(fold)].set_index("structure_id")
    roles = data.structure_id.map(part.inner_role)
    outer_roles = data.structure_id.map(part.outer_role)
    train = np.flatnonzero(roles.eq("inner_train").to_numpy())
    valid = np.flatnonzero(roles.eq("inner_validation").to_numpy())
    outer = np.flatnonzero(outer_roles.eq("outer_validation").to_numpy())
    sets = [set(data.iloc[ix].composition_id) for ix in (train, valid, outer)]
    if (len(train), len(valid), len(outer)) != (135, 33, 42) or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise H3Error(f"Fold {fold}: partición inválida o con fuga composicional.")
    return train, valid, outer
