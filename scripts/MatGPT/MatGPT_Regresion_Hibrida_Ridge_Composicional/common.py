"""Utilidades verificables para combinar Ridge composicional y residuos guardados.

Las etiquetas históricas se filtran por split antes de parsear cualquier energía.
Este módulo nunca abre el test ni realiza inferencia con el Transformer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
EXPERIMENT = "MatGPT_Regresion_Hibrida_Ridge_Composicional"
REPORT = ROOT / "reports/MatGPT" / EXPERIMENT
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = (ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/"
         "reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv")
C_REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA"
C_PREDICTIONS = C_REPORT / "predicciones_outer_por_estructura.csv"
C_MANIFEST = C_REPORT / "manifest.json"
H3C_REPLAY = ROOT / "reports/MatGPT/MatGPT_H3C_Final/reproduction/replayed_predictions.csv"
H3C_MANIFEST = ROOT / "reports/MatGPT/MatGPT_H3C_Final/reproduction/reproduction_manifest.json"
COMPOSITION_CONTROL = ROOT / "reports/MatGPT/MatGPT_H3C_Final/controls/composition_seed_42/predictions.csv"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
ELEMENTS = ("Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr")
SEEDS = (42, 123, 2026)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TOLERANCE_EV_ATOM = 1e-12


def sha256(path: Path) -> str:
    """Calcula el SHA-256 de un archivo sin cargarlo completo en memoria.

    Args:
        path: Archivo existente.
    Returns:
        Hash hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_development(path: Path = DATASET) -> pd.DataFrame:
    """Lee train/val, ignorando cada fila test antes de leer sus etiquetas.

    Args:
        path: CSV histórico con columna `split`.
    Returns:
        Metadatos y energías DFT de 210 estructuras de desarrollo.
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") not in {"train", "val"}:
                continue
            atoms = int(row["num_atoms"])
            energy_total = float(row["energy_total_ev"])
            energy_atom = float(row["energy_per_atom_ev"])
            if atoms <= 0 or abs(energy_total / atoms - energy_atom) > 1e-8:
                raise ValueError(f"E_total/N inconsistente para {row.get('structure_id')}")
            rows.append({"structure_id": row["structure_id"],
                         "composition_id": row["reduced_formula"],
                         "polymorph_family": row["polymorph_family"],
                         "n_atoms": atoms,
                         "energy_total_dft_ev": energy_total,
                         "energy_per_atom_dft_ev": energy_atom,
                         "historical_split": row["split"]})
    frame = pd.DataFrame(rows)
    if (len(frame) != 210 or frame.structure_id.nunique() != 210 or
            frame.composition_id.nunique() != 70 or frame.groupby("composition_id").size().ne(3).any()):
        raise ValueError("Se requieren 210 estructuras de desarrollo, 70 fórmulas y tripletes completos")
    if set(frame.polymorph_family) != set(FAMILIES):
        raise ValueError("Las familias estructurales no coinciden con las tres familias históricas")
    return frame


def stoichiometric_fractions(formula: str) -> np.ndarray:
    """Codifica una fórmula en las 17 fracciones elementales bloqueadas.

    Args:
        formula: Fórmula reducida de desarrollo.
    Returns:
        Vector float64 ordenado según `ELEMENTS`.
    """
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens or "".join(element + count for element, count in tokens) != formula:
        raise ValueError(f"Fórmula reducida no interpretable: {formula}")
    counts: dict[str, int] = {}
    for element, count in tokens:
        if element not in ELEMENTS:
            raise ValueError(f"Elemento fuera del vocabulario bloqueado: {element} en {formula}")
        counts[element] = counts.get(element, 0) + int(count or "1")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"Conteo atómico inválido para {formula}")
    vector = np.asarray([counts.get(element, 0) / total for element in ELEMENTS], dtype=np.float64)
    if not np.isclose(vector.sum(), 1.0, atol=1e-12):
        raise ValueError(f"Fracciones estequiométricas no normalizadas: {formula}")
    return vector


def load_and_audit_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verifica procedencia e identidad y devuelve fuentes alineadas.

    Returns:
        Tupla (dataset, folds, predicciones C, auditoría de procedencia).
    Raises:
        FileNotFoundError: Si falta una fuente autorizada.
        ValueError: Si falla hash, alineación, cobertura o un control de leakage.
    """
    required = (DATASET, FOLDS, C_PREDICTIONS, C_MANIFEST, H3C_REPLAY, H3C_MANIFEST,
               COMPOSITION_CONTROL)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan fuentes históricas verificadas: " + ", ".join(missing))
    c_manifest = json.loads(C_MANIFEST.read_text(encoding="utf-8"))
    h3c_manifest = json.loads(H3C_MANIFEST.read_text(encoding="utf-8"))
    if (c_manifest.get("status") != "COMPLETE" or c_manifest.get("historical_test_used") is not False or
            c_manifest.get("preflight_status") != "PASS"):
        raise ValueError("Manifiesto del Experimento C no certifica outputs completos y test cerrado")
    c_hash = next((row["sha256"] for row in c_manifest.get("generated_outputs", [])
                   if row.get("path", "").endswith("predicciones_outer_por_estructura.csv")), None)
    if c_hash != sha256(C_PREDICTIONS):
        raise ValueError("Hash de predicciones del Experimento C no coincide con su manifiesto")
    if (h3c_manifest.get("status") != "PASS" or h3c_manifest.get("historical_test_accessed") is not False or
            sha256(H3C_REPLAY) != h3c_manifest.get("replayed_predictions_sha256")):
        raise ValueError("Reproducción histórica H3-C ausente o con hash/estado inválido")
    data = read_development()
    folds = pd.read_csv(FOLDS)
    folds = folds[folds.historical_split.isin(["train", "val"])].copy()
    if len(folds) != 1050 or folds.structure_id.nunique() != 210 or set(folds.structure_id) != set(data.structure_id):
        raise ValueError("Manifiesto histórico debe cubrir exactamente 210 estructuras en cinco pliegues")
    byid = data.set_index("structure_id")
    if not folds.apply(lambda row: row.composition_group == byid.loc[row.structure_id, "composition_id"], axis=1).all():
        raise ValueError("La fórmula de los folds no coincide con el dataset de desarrollo")
    for fold, group in folds.groupby("outer_fold"):
        outer = group.drop_duplicates("structure_id")
        train_formulas = set(group.loc[group.inner_role == "inner_train", "composition_group"])
        val_formulas = set(group.loc[group.inner_role == "inner_validation", "composition_group"])
        held_formulas = set(group.loc[group.outer_role == "outer_validation", "composition_group"])
        outer_train = set(group.loc[group.outer_role == "outer_train", "composition_group"])
        if (len(train_formulas), len(val_formulas), len(held_formulas), len(outer_train)) != (45, 11, 14, 56):
            raise ValueError(f"Fold {fold}: tamaños esperados 45/11/14 y 56 outer-train")
        if train_formulas & val_formulas or train_formulas & held_formulas or val_formulas & held_formulas:
            raise ValueError(f"Fold {fold}: fuga de composición")
        if outer.groupby("composition_group").size().ne(3).any():
            raise ValueError(f"Fold {fold}: hay tripletes incompletos")
    predictions = pd.read_csv(C_PREDICTIONS)
    if (len(predictions) != 1260 or predictions.structure_id.nunique() != 210 or
            set(predictions.condition) != {"frozen", "lora"} or set(predictions.seed) != set(SEEDS)):
        raise ValueError("Predicciones C requieren 210 IDs × dos ramas residuales × tres semillas")
    if predictions.duplicated(["condition", "fold", "seed", "structure_id"]).any():
        raise ValueError("Predicciones C contienen filas duplicadas")
    aligned = predictions.merge(data, on=["structure_id", "composition_id", "polymorph_family", "n_atoms",
                                          "energy_total_dft_ev", "energy_per_atom_dft_ev"],
                                how="left", validate="many_to_one", indicator=True)
    if len(aligned) != len(predictions) or not aligned._merge.eq("both").all():
        raise ValueError("Predicciones C no conservan IDs, composición, N o targets del desarrollo")
    valid_pairs = set(zip(folds.structure_id, folds.outer_fold))
    if not all((row.structure_id, row.fold) in valid_pairs for row in predictions.itertuples()):
        raise ValueError("Predicciones C no respetan asignación de pliegue histórica")
    # El promedio Ridge histórico de C no depende de seed ni de congelado/LoRA.
    means = predictions.groupby(["fold", "seed", "composition_id"]).ridge_mean_ev_atom.agg(["min", "max"])
    if ((means["max"] - means["min"]).abs() > 1e-6).any():
        raise ValueError("La rama Ridge original cambia entre residual Frozen/LoRA o semillas")
    residual_sums = predictions.groupby(["condition", "fold", "seed", "composition_id"]).residual_ev_atom.sum()
    if residual_sums.abs().max() > 1e-6:
        raise ValueError("Los residuos guardados no están centrados por fórmula")
    replay = pd.read_csv(H3C_REPLAY)
    frozen = predictions[predictions.condition.eq("frozen")].merge(
        replay, on=["structure_id", "composition_id", "polymorph_family", "fold", "seed"],
        suffixes=("_C", "_H3C"), validate="one_to_one")
    if len(frozen) != 630:
        raise ValueError("La comparación Frozen vs H3-C no cubre 630 predicciones")
    frozen_difference = float((frozen.energy_per_atom_pred_ev_C - frozen.energy_per_atom_pred_ev_H3C).abs().max())
    if frozen_difference > 1e-5:
        raise ValueError(f"Predicciones Frozen no reproducen H3-C: max abs {frozen_difference:g} eV/átomo")
    control = pd.read_csv(COMPOSITION_CONTROL)
    if len(control) != 210 or control.structure_id.nunique() != 210 or set(control.model) != {"composition"}:
        raise ValueError("Control composicional histórico no íntegro")
    composition_hash = sha256(COMPOSITION_CONTROL)
    audit: dict[str, Any] = {
        "dataset_sha256": sha256(DATASET), "folds_sha256": sha256(FOLDS),
        "frozen_lora_predictions_sha256": sha256(C_PREDICTIONS),
        "h3c_replayed_predictions_sha256": sha256(H3C_REPLAY),
        "existing_composition_control_sha256": composition_hash,
        "checkpoint_sha256": c_manifest.get("source_hashes_sha256", {}).get(
            "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"),
        "source_counts": {"structures": 210, "compositions": 70, "fold_assignments": 1050,
                           "C_predictions": 1260, "composition_control_predictions": 210},
        "fold_sizes": {str(int(fold)): {"inner_train_compositions": 45,
                                         "inner_validation_compositions": 11,
                                         "outer_train_compositions": 56,
                                         "outer_test_compositions": 14}
                       for fold in sorted(folds.outer_fold.unique())},
        "historical_test_accessed": False,
        "frozen_prediction_max_abs_difference_from_h3c_ev_atom": frozen_difference,
        "C_source_status": c_manifest.get("status"),
        "H3C_reproduction_status": h3c_manifest.get("status"),
    }
    return data, folds, predictions, audit


def formula_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por fórmula con un vector composicional fijo de 17 elementos.

    Args:
        data: Dataset único de desarrollo.
    Returns:
        DataFrame con fórmula y columnas fraccionarias en el orden bloqueado.
    """
    formulas = sorted(data.composition_id.unique())
    rows = [{"composition_id": formula,
             **{element: float(value) for element, value in zip(ELEMENTS, stoichiometric_fractions(formula))}}
            for formula in formulas]
    table = pd.DataFrame(rows)
    values = table.loc[:, ELEMENTS].to_numpy(float)
    if values.shape != (70, 17) or not np.isfinite(values).all() or not np.allclose(values.sum(1), 1.0):
        raise ValueError("Matriz composicional esperada: 70×17 fracciones finitas")
    return table
