#!/usr/bin/env python3
"""Operaciones de tripletes físicos y asignaciones agrupadas E1.3."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import FAMILIES, ExperimentError, split_indices


def ordered_triplet(group: pd.DataFrame) -> pd.DataFrame:
    """Ordena filas con el orden fijo perovskita, hexagonal, aguja."""
    rank = {family: i for i, family in enumerate(FAMILIES)}
    result = group.assign(_order=group.polymorph_family.map(rank)).sort_values("_order")
    if len(result) != 3 or result.polymorph_family.tolist() != list(FAMILIES):
        raise ExperimentError("Triplete incompleto o familia duplicada.")
    return result.drop(columns="_order")


def composition_groups(data: pd.DataFrame, indices: np.ndarray) -> list[np.ndarray]:
    """Agrupa índices de filas manteniendo juntos los tres polimorfos."""
    part = data.iloc[indices]
    return [part.loc[ids].index.to_numpy(dtype=int) for _, ids in part.groupby("composition_id", sort=True).groups.items()]


def fold_rows(data: pd.DataFrame, fold_manifest: pd.DataFrame, fold: int) -> dict[str, Any]:
    """Devuelve los tres bloques por fold con validación de disjunción."""
    train, val, outer = split_indices(data, fold_manifest, fold)
    return {"fold": fold, "inner_train": train, "inner_validation": val, "outer_validation": outer,
            "train_compositions": sorted(data.iloc[train].composition_id.unique().tolist()),
            "inner_validation_compositions": sorted(data.iloc[val].composition_id.unique().tolist()),
            "outer_validation_compositions": sorted(data.iloc[outer].composition_id.unique().tolist())}


def assert_no_overlap(data: pd.DataFrame, *indices: np.ndarray) -> None:
    """Asegura que conjuntos de filas no comparten composiciones."""
    sets = [set(data.iloc[ix].composition_id) for ix in indices]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ExperimentError("Fuga de fórmula entre subconjuntos.")
