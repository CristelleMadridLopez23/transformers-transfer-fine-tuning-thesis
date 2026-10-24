#!/usr/bin/env python3
"""Rutas, procedencia, datos de desarrollo y utilidades de MatGPT_frozen_3."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ALT = ROOT / "Alternative2"
SCRIPT_DIR = ALT / "scripts" / "MatGPT_frozen_3"
REPORT_ROOT = ALT / "reports" / "MatGPT_frozen_3"
RUN_ID = "run_20261023_v2"
REPORT_DIR = REPORT_ROOT / RUN_ID
SOURCE_DIR = ALT / "models" / "MatGPT-source" / "mat-concat"
CHECKPOINT = SOURCE_DIR / "mat-uncond" / "weights" / "transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
DATASET = ALT / "reports" / "MatGPT_frozen" / "matgpt_e1_v2" / "dataset.csv"
FOLDS = ALT / "reports" / "MatGPT_frozen_2" / "v1" / "fold_assignments.csv"
H2_REPORT = ALT / "reports" / "MatGPT_frozen_2" / "v1"
EXPECTED_CHECKPOINT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
FAMILY_LABELS = {"dist_perovskite": "Perovskita distorsionada", "hexagonal": "Hexagonal", "needle": "Aguja"}
SEEDS = (42, 123, 2026)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TAU_MEV = 1.0


class ExperimentError(RuntimeError):
    """Bloqueo de integridad, procedencia o protocolo."""


class JsonFormatter(logging.Formatter):
    """Formateador JSON para mensajes de ejecución."""
    def format(self, record: logging.LogRecord) -> str:
        """Serializa evento y campos asociados."""
        return json.dumps({"level": record.levelname, "event": record.getMessage(), **getattr(record, "fields", {})}, ensure_ascii=False, default=str)


LOGGER = logging.getLogger("matgpt_frozen_3")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log(event: str, **fields: Any) -> None:
    """Escribe un evento estructurado."""
    LOGGER.info(event, extra={"fields": fields})


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 en bloques para un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON UTF-8 con indentación y crea solo su carpeta de reporte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_dev() -> pd.DataFrame:
    """Carga solo train/validation histórica y verifica tripletes completos.

    Returns:
        DataFrame con 210 estructuras de desarrollo; nunca conserva filas test.
    """
    if not DATASET.is_file():
        raise ExperimentError(f"No existe el dataset auditado: {DATASET}")
    frame = pd.read_csv(DATASET, low_memory=False)
    frame = frame.loc[frame.split.isin(["train", "val"])].copy().reset_index(drop=True)
    required = {"structure_id", "reduced_formula", "polymorph_family", "split", "energy_per_atom_ev", "matgpt_slices"}
    if required - set(frame.columns):
        raise ExperimentError(f"Faltan columnas del dataset: {sorted(required - set(frame.columns))}")
    frame["structure_id"] = frame.structure_id.astype(str)
    frame["composition_id"] = frame.reduced_formula.astype(str)
    frame["E_DFT_eV_atom"] = pd.to_numeric(frame.energy_per_atom_ev, errors="coerce")
    if len(frame) != 210 or frame.structure_id.duplicated().any() or not np.isfinite(frame.E_DFT_eV_atom).all():
        raise ExperimentError("Development no contiene exactamente 210 IDs únicos y energías finitas.")
    if frame.groupby("composition_id").size().ne(3).any():
        raise ExperimentError("Se requiere un triplete de estructuras por fórmula.")
    for formula, group in frame.groupby("composition_id"):
        if set(group.polymorph_family) != set(FAMILIES):
            raise ExperimentError(f"Familias incompletas para {formula}.")
    return frame


def load_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Carga cinco folds exactos E1.3/H0-H2, cotejando IDs y agrupamiento."""
    if not FOLDS.is_file():
        raise ExperimentError(f"Falta fold manifest histórico: {FOLDS}")
    folds = pd.read_csv(FOLDS)
    folds = folds.loc[folds.historical_split.isin(["train", "val"])].copy()
    if len(folds) != 1050 or set(folds.structure_id.astype(str)) != set(data.structure_id):
        raise ExperimentError("Los folds no cubren exactamente 210 estructuras en 5 pliegues.")
    for fold in range(5):
        part = folds.loc[folds.outer_fold.eq(fold)]
        groups = part.drop_duplicates("composition_group")
        if len(groups) != 70 or (groups.outer_role == "outer_validation").sum() != 14:
            raise ExperimentError(f"Conteo de grupos incorrecto en fold {fold}.")
        if part.groupby("composition_group").outer_role.nunique().gt(1).any():
            raise ExperimentError(f"Polimorfos separados en fold {fold}.")
    return folds


def split_indices(data: pd.DataFrame, folds: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve índices inner-train, inner-validation y outer por fórmula."""
    part = folds.loc[folds.outer_fold.eq(fold)].set_index("structure_id")
    roles = data.structure_id.map(part.inner_role)
    outer = data.structure_id.map(part.outer_role)
    train_ids = np.flatnonzero(roles.eq("inner_train").to_numpy())
    val_ids = np.flatnonzero(roles.eq("inner_validation").to_numpy())
    outer_ids = np.flatnonzero(outer.eq("outer_validation").to_numpy())
    if (len(train_ids), len(val_ids), len(outer_ids)) != (135, 33, 42):
        raise ExperimentError(f"Partición interna/externa inesperada en fold {fold}.")
    sets = [set(data.iloc[ix].composition_id) for ix in (train_ids, val_ids, outer_ids)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ExperimentError(f"Fuga por composición en fold {fold}.")
    return train_ids, val_ids, outer_ids
