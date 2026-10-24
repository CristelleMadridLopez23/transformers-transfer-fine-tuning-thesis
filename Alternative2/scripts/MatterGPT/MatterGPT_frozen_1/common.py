"""MatterGPT experiment paths, provenance helpers, and leakage-safe data loading."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Alternative2/models/MatterGPT"
REPORT_DIR = ROOT / "Alternative2/reports/MatterGPT/MatterGPT_frozen_1"
DATASET = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
TEST_INPUTS = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen_3/H3_test/predictions/test_inputs.csv"
SEEDS = (42, 123, 2026)
CONDITION = (-1.0, 2.0)
FAMILIES = ("dist_perovskite", "hexagonal", "needle")


def logger(name: str) -> logging.Logger:
    """Return a JSON-lines logger with a stable event field."""
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","message":"%(message)s"}')
    return logging.getLogger(name)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file, reading in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write indented, deterministic JSON and create parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def prepare_dirs() -> None:
    """Create this experiment's report subfolders without touching history."""
    for name in ("provenance", "tokenizer", "conditioning", "features", "cv", "h3a", "test", "comparisons", "figures"):
        (REPORT_DIR / name).mkdir(parents=True, exist_ok=True)


def _rows_without_test_targets(path: Path, dev_ids: set[str]) -> pd.DataFrame:
    """Read target columns only for predeclared development IDs.

    The first pass reads only identity/split metadata. The second pass skips
    physical CSV rows outside development before parsing any target values.
    """
    identity = pd.read_csv(path, usecols=["structure_id", "split"], dtype={"structure_id": str})
    excluded_line_numbers = [int(index) + 1 for index, row in identity.iterrows()
                             if row.structure_id not in dev_ids or row.split == "test"]
    # pandas skiprows counts physical lines including the header as row 0.
    skiprows = excluded_line_numbers
    selected = ["structure_id", "energy_per_atom_ev"]
    targets = pd.read_csv(path, usecols=selected, skiprows=skiprows)
    targets = targets[targets.structure_id.astype(str).isin(dev_ids)].copy()
    if len(targets) != len(dev_ids):
        raise ValueError(f"Esperaba {len(dev_ids)} targets de development, encontré {len(targets)}")
    return targets


def load_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return development metadata/labels and the exact grouped fold manifest.

    TEST structures are read only from the label-free H3 test input table. The
    DFT target parser explicitly skips every non-development source row.
    """
    if not all(path.is_file() for path in (DATASET, FOLDS, TEST_INPUTS)):
        raise FileNotFoundError("No se encontraron dataset, folds y test inputs históricos")
    fold_rows = pd.read_csv(FOLDS, dtype={"structure_id": str})
    folds = fold_rows[fold_rows.outer_role == "outer_validation"][["structure_id", "composition_group", "outer_fold"]].drop_duplicates("structure_id")
    if len(folds) != 210 or folds.structure_id.duplicated().any():
        raise ValueError("El manifiesto no produce una asignación OOF única por estructura")
    dev_ids = set(folds.structure_id.astype(str))
    test = pd.read_csv(TEST_INPUTS, dtype={"structure_id": str})
    if set(test.structure_id.astype(str)) & dev_ids:
        raise ValueError("IDs de test y development se solapan")
    usecols = ["structure_id", "reduced_formula", "polymorph_family", "split", "num_atoms", "matgpt_slices"]
    metadata = pd.read_csv(DATASET, usecols=usecols, dtype={"structure_id": str})
    metadata = metadata[metadata.structure_id.isin(dev_ids)].copy()
    if len(metadata) != len(dev_ids):
        raise ValueError("Los IDs del fold manifest no coinciden con las filas de dataset")
    targets = _rows_without_test_targets(DATASET, dev_ids)
    data = metadata.merge(targets, on="structure_id", validate="one_to_one").merge(
        folds[["structure_id", "composition_group", "outer_fold"]], on="structure_id", validate="one_to_one")
    data = data.rename(columns={"reduced_formula": "composition_id", "matgpt_slices": "slices", "energy_per_atom_ev": "energy_dft_ev_atom"})
    if data["energy_dft_ev_atom"].isna().any() or not np.isfinite(data.energy_dft_ev_atom.to_numpy(float)).all():
        raise ValueError("Energías DFT de development faltantes/no finitas")
    if data.groupby("composition_id").size().ne(3).any():
        raise ValueError("Development requiere tres polimorfos por composición")
    if data.groupby("composition_id").polymorph_family.nunique().ne(3).any():
        raise ValueError("Familias de polimorfos incompletas o duplicadas")
    return data.sort_values(["composition_id", "polymorph_family"]).reset_index(drop=True), folds


def environment() -> dict[str, str]:
    """Return selected runtime versions for provenance."""
    import sklearn
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"}


def stable_ridge_predict(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    """Evaluate a fitted Ridge with explicit rowwise sums.

    The local macOS NumPy/BLAS build emits false divide/overflow RuntimeWarnings
    in dense ``matmul`` while returning finite results. This algebraically equal
    evaluation avoids that noisy BLAS path and is checked for finite outputs.
    """
    values = np.asarray(matrix, dtype=np.float64)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    prediction = np.sum(values * coefficients[None, :], axis=1) + float(estimator.intercept_)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Ridge produjo predicciones NaN/Inf")
    return prediction
