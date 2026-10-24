#!/usr/bin/env python3
"""Utilidades auditables del experimento MatGPT congelado de dos cabezas."""
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
SCRIPT_DIR = ALT / "scripts" / "MatGPT_frozen_2"
REPORT_DIR = ALT / "reports" / "MatGPT_frozen_2" / "v1"
DATASET = ALT / "reports" / "MatGPT_frozen" / "matgpt_e1_v2" / "dataset.csv"
E13_FOLDS = ALT / "reports" / "MatGPT_frozen" / "matgpt_e1_3" / "fold_assignments.csv"
FOLDS = REPORT_DIR / "fold_assignments.csv"
E0_AUDIT = ALT / "reports" / "MatGPT_frozen" / "matgpt_e0" / "load_validation.json"
CHECKPOINT = ALT / "models" / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
SOURCE = ALT / "models" / "MatGPT-source" / "mat-concat"
SOURCE_MANIFEST = ALT / "models" / "MatGPT-source.download-manifest.json"
EXPECTED_CHECKPOINT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
SEEDS = (42, 123, 2026)
MODELS = ("H0", "H1", "H2")
TAU_MEV = 1.0


class ExperimentError(RuntimeError):
    """Error de protocolo, procedencia o integridad que detiene la fase."""


class JsonFormatter(logging.Formatter):
    """Serializa eventos del experimento en líneas JSON."""
    def format(self, record: logging.LogRecord) -> str:
        """Devuelve un mensaje JSON legible por máquina."""
        return json.dumps({"level": record.levelname, "message": record.getMessage(), **getattr(record, "fields", {})}, ensure_ascii=False, default=str)


def logger_for(name: str) -> logging.Logger:
    """Crea logger estructurado sin propagar al logger raíz."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = logger_for("matgpt_frozen_2")


def log(level: int, event: str, **fields: Any) -> None:
    """Escribe un evento JSON estructurado."""
    LOGGER.log(level, event, extra={"fields": fields})


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 en streaming."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Calcula SHA-256 de una secuencia serializada."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe un JSON UTF-8 y crea su carpeta destino."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_development() -> pd.DataFrame:
    """Carga solo las filas históricas train/val, descartando test de inmediato."""
    if not DATASET.is_file():
        raise ExperimentError(f"Falta dataset: {DATASET}")
    raw = pd.read_csv(DATASET, low_memory=False)
    required = {"structure_id", "reduced_formula", "polymorph_family", "split", "energy_per_atom_ev", "matgpt_slices"}
    if required - set(raw.columns):
        raise ExperimentError(f"Faltan columnas: {sorted(required - set(raw.columns))}")
    data = raw.loc[raw.split.isin(["train", "val"]), sorted(required)].copy().reset_index(drop=True)
    del raw
    data["structure_id"] = data.structure_id.astype(str)
    data["composition_id"] = data.reduced_formula.astype(str)
    data["E_DFT_eV_atom"] = pd.to_numeric(data.energy_per_atom_ev, errors="coerce")
    if len(data) != 210 or data.structure_id.duplicated().any() or not np.isfinite(data.E_DFT_eV_atom.to_numpy()).all():
        raise ExperimentError("Development debe tener 210 IDs únicos con energías DFT finitas.")
    for group, part in data.groupby("composition_id", sort=True):
        if len(part) != 3 or set(part.polymorph_family) != set(FAMILIES):
            raise ExperimentError(f"{group}: se requieren tres familias polimórficas físicas.")
        if part.E_DFT_eV_atom.max() - part.E_DFT_eV_atom.min() < 0:
            raise ExperimentError(f"Rango energético inválido para {group}.")
    return data


def load_fold_manifest(data: pd.DataFrame) -> pd.DataFrame:
    """Extrae las asignaciones exactas E1.3 sin incluir filas del test."""
    if not FOLDS.is_file():
        raise ExperimentError(f"Falta manifiesto E1.3: {FOLDS}")
    folds = pd.read_csv(FOLDS)
    folds = folds.loc[folds.historical_split.isin(["train", "val"])].copy()
    if folds.duplicated(["structure_id", "outer_fold"]).any():
        raise ExperimentError("E1.3 contiene IDs repetidos dentro del mismo fold.")
    if len(folds) != 210 * 5:
        raise ExperimentError(f"Se esperaban 1050 relaciones estructura-fold; hay {len(folds)}.")
    if set(folds.structure_id.astype(str)) != set(data.structure_id):
        raise ExperimentError("El manifiesto de folds no coincide con los 210 IDs development.")
    for fold in range(5):
        part = folds[folds.outer_fold.eq(fold)]
        role = part.drop_duplicates("composition_group")
        outer = role.loc[role.outer_role.eq("outer_validation"), "composition_group"]
        if len(outer) != 14 or role.composition_group.nunique() != 70:
            raise ExperimentError(f"El fold {fold} no tiene exactamente 14 grupos outer únicos.")
        if part.groupby("composition_group").outer_role.nunique().gt(1).any():
            raise ExperimentError(f"Polimorfos separados entre outer roles en fold {fold}.")
    return folds


def verify_provenance() -> dict[str, Any]:
    """Comprueba identidad E0 del checkpoint y commit del código original."""
    for path in (CHECKPOINT, MODEL_INFO, SOURCE_MANIFEST, E0_AUDIT, SOURCE / "config" / "uncond.yaml", SOURCE / "utils" / "models.py"):
        if not path.is_file():
            raise ExperimentError(f"Falta evidencia de procedencia: {path}")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA:
        raise ExperimentError(f"SHA checkpoint inesperado: {checkpoint_hash}")
    source_meta = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    e0 = json.loads(E0_AUDIT.read_text(encoding="utf-8"))
    if source_meta.get("resolved_commit") != EXPECTED_COMMIT or e0.get("status") != "PASS":
        raise ExperimentError("Commit histórico o auditoría E0 no PASS.")
    info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in info["stoi"].items()}
    if len(stoi) != 172 or int(info["block_size"]) != 442 or any(t not in stoi for t in ("<pad>", "<start>", "<end>")):
        raise ExperimentError("Vocabulario/special tokens no concuerdan con E0.")
    if len({stoi[t] for t in ("<pad>", "<start>", "<end>")}) != 3:
        raise ExperimentError("IDs de tokens especiales colisionan.")
    return {"checkpoint_sha256": checkpoint_hash, "source_commit": EXPECTED_COMMIT, "vocab_size": len(stoi),
            "block_size": int(info["block_size"]), "special_token_ids": {token: stoi[token] for token in ("<pad>", "<start>", "<end>")},
            "hidden_size": 768, "layers": 12, "heads": 12, "e0_status": e0["status"]}


def group_inner_indices(data: pd.DataFrame, folds: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve indices inner-train, inner-val y outer-val agrupados por fórmula."""
    assignment = folds.loc[folds.outer_fold.eq(fold)].set_index("structure_id")
    roles = data.structure_id.map(assignment.inner_role)
    outer_roles = data.structure_id.map(assignment.outer_role)
    train_idx = np.flatnonzero(roles.eq("inner_train").to_numpy())
    val_idx = np.flatnonzero(roles.eq("inner_validation").to_numpy())
    outer_idx = np.flatnonzero(outer_roles.eq("outer_validation").to_numpy())
    if not (len(train_idx) == 135 and len(val_idx) == 33 and len(outer_idx) == 42):
        raise ExperimentError(f"Fold {fold}: conteos inner/outer incorrectos.")
    groups = [set(data.iloc[ix].composition_id) for ix in (train_idx, val_idx, outer_idx)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ExperimentError(f"Fuga de composición dentro del fold {fold}.")
    return train_idx, val_idx, outer_idx
