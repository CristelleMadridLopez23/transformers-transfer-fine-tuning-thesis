"""Carga y valida datos comunes para los diagnósticos D1–D3 de MatGPT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ALT = ROOT / "Alternative2"
DATA_PATH = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS_PATH = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
TEST_IDS_PATH = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/split_manifest.json"
REPORT = ALT / "reports/MatGPT_diagnostics"


def sha256(path: Path) -> str:
    """Devuelve SHA-256 del archivo indicado."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Carga 210 estructuras y un fold manifest sin incluir test histórico.

    Returns:
        Dataset de desarrollo, filas de folds (210 por fold) y detalles de integridad.
    """
    for path in (DATA_PATH, FOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Falta la entrada requerida: {path}")
    full = pd.read_csv(DATA_PATH)
    if len(full) != 234 or full.structure_id.astype(str).duplicated().any():
        raise ValueError("El dataset debe conservar 234 structure_id únicos.")
    dev = full.loc[full.split.isin(["train", "val"])].copy().reset_index(drop=True)
    # El dataset E1 conserva reduced_formula pero no la columna derivada de grupo.
    # Para este ABS3, la fórmula reducida es la clave química canónica usada por E1.3.
    if "composition_group" not in dev.columns:
        dev["composition_group"] = dev.reduced_formula.astype(str)
    if len(dev) != 210 or dev.composition_group.nunique() != 70:
        raise ValueError("Desarrollo debe contener 210 estructuras y 70 composiciones.")
    folds = pd.read_csv(FOLDS_PATH)
    folds = folds.loc[folds.historical_split.isin(["train", "val"])].copy()
    historical_test = full.loc[full.split.eq("test")]
    if set(dev.reduced_formula.astype(str)) & set(historical_test.reduced_formula.astype(str)):
        raise ValueError("Una composición de desarrollo también aparece en test histórico.")
    fold_map = folds.drop_duplicates("structure_id").set_index("structure_id").composition_group.astype(str)
    data_groups = dev.set_index(dev.structure_id.astype(str)).reduced_formula.astype(str)
    if not data_groups.index.isin(fold_map.index).all() or not data_groups.eq(fold_map.reindex(data_groups.index).to_numpy()).all():
        raise ValueError("La fórmula reducida del dataset no coincide con composition_group de E1.3.")
    if folds.groupby("outer_fold").structure_id.nunique().to_dict() != {i: 210 for i in range(5)}:
        raise ValueError("El manifiesto E1.3 no contiene cinco folds completos de desarrollo.")
    expected = set(dev.structure_id.astype(str))
    for fold, group in folds.groupby("outer_fold"):
        if set(group.structure_id.astype(str)) != expected:
            raise ValueError(f"El fold {fold} no contiene exactamente los IDs de desarrollo.")
        for role, allowed in (("outer_role", {"outer_train", "outer_validation"}),
                              ("inner_role", {"inner_train", "inner_validation", "not_applicable"})):
            if not set(group[role]).issubset(allowed):
                raise ValueError(f"Valores inesperados en {role}, fold {fold}.")
        by_id = group.groupby("structure_id")
        if (by_id.composition_group.nunique() > 1).any():
            raise ValueError(f"Composición inconsistente dentro del fold {fold}.")
        for role in ("outer_role", "inner_role"):
            comp_role = group.groupby("composition_group")[role].nunique()
            if (comp_role > 1).any():
                raise ValueError(f"Fuga de composición entre roles {role}, fold {fold}.")
    # DFT debe tener tres estructuras físicas por composición y energías finitas.
    sizes = dev.groupby("composition_group").size()
    if not sizes.eq(3).all() or not pd.Series(dev.energy_per_atom_ev).map(pd.notna).all():
        raise ValueError("Cada composición debe tener tres etiquetas DFT válidas.")
    meta = {
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256": sha256(DATA_PATH),
        "folds_path": str(FOLDS_PATH.relative_to(ROOT)),
        "folds_sha256": sha256(FOLDS_PATH),
        "n_structures_development": len(dev),
        "n_compositions_development": int(dev.composition_group.nunique()),
        "n_polymorphs_per_composition": {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "outer_folds": sorted(int(v) for v in folds.outer_fold.unique()),
        "historical_test_used": False,
        "historical_test_structures_excluded": int(len(historical_test)),
        "historical_test_compositions_excluded": int(historical_test.reduced_formula.nunique()),
    }
    return dev, folds, meta


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Guarda JSON de forma determinista creando el directorio padre."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
