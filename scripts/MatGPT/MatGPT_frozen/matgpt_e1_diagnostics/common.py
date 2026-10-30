"""Validación común y métricas para los diagnósticos congelados de MatGPT."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
ALT = ROOT / "Alternative2"
E1_REPORT = ALT / "reports" / "matgpt_e1_v2"
E1_EMBED = ALT / "data" / "embeddings" / "matgpt_e1_v2"
BACKBONE_SHA256 = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"


class DiagnosticError(RuntimeError):
    """Error de validación que debe detener una fase científica dependiente."""


class JsonFormatter(logging.Formatter):
    """Formatea eventos de logging como objetos JSON por línea."""

    def format(self, record: logging.LogRecord) -> str:
        """Devuelve un evento JSON con nivel, mensaje y metadatos.

        Args:
            record: Registro estándar de logging con nivel, mensaje y `fields`.

        Returns:
            Cadena JSON UTF-8 segura para una línea del log.
        """
        return json.dumps({"level": record.levelname, "message": record.getMessage(), **getattr(record, "fields", {})}, ensure_ascii=False, default=str)


def logger_for(name: str) -> logging.Logger:
    """Configura logging estructurado y devuelve el logger solicitado.

    Args:
        name: Nombre jerárquico del logger.

    Returns:
        Logger INFO que emite objetos JSON por línea.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emite un evento JSON con nivel, mensaje y campos científicos.

    Args:
        logger: Logger estructurado que recibe el evento.
        level: Nivel estándar de logging.
        message: Nombre breve del evento.
        **fields: Valores serializables asociados al evento.

    Returns:
        None.
    """
    logger.log(level, message, extra={"fields": fields})


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo existente por bloques.

    Args:
        path: Ruta de archivo que se va a resumir.

    Returns:
        Digest SHA-256 hexadecimal en minúsculas.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON indentado y crea el directorio padre.

    Args:
        path: Destino del archivo JSON.
        payload: Diccionario serializable con resultados reales.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_inputs(dataset_path: Path, cache_path: Path, metadata_path: Path, audit_path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Carga y valida el dataset histórico y un caché MatGPT alineado.

    Args:
        dataset_path: CSV E1 con etiquetas, IDs y partición histórica.
        cache_path: NPZ de vectores congelados de forma `[n, 768]`.
        metadata_path: CSV de identidades en el mismo orden del caché.
        audit_path: JSON con hashes y controles de extracción.

    Returns:
        Dataset ordenado, matriz float32 y auditoría asociada al caché.
    """
    required_paths = (dataset_path, cache_path, metadata_path, audit_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise DiagnosticError(f"Faltan entradas E1: {missing}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("checkpoint_sha256") != BACKBONE_SHA256:
        raise DiagnosticError("La auditoría del embedding no valida el checkpoint E0 esperado.")
    if audit.get("cache_sha256") != sha256_file(cache_path) or audit.get("metadata_sha256") != sha256_file(metadata_path):
        raise DiagnosticError("El caché o sus metadatos cambiaron desde la auditoría E1.")
    if audit.get("dataset_sha256") != sha256_file(dataset_path):
        raise DiagnosticError("El dataset no coincide con el usado para extraer embeddings E1.")
    data, meta = pd.read_csv(dataset_path, low_memory=False), pd.read_csv(metadata_path, low_memory=False)
    if data.empty or data["structure_id"].astype(str).tolist() != meta["structure_id"].astype(str).tolist():
        raise DiagnosticError("Dataset vacío o IDs/orden distintos a los metadatos del caché.")
    # El dataset E1 histórico guarda la energía como `energy_per_atom_ev`.
    # Normalizamos el nombre en memoria para los diagnósticos; el CSV y su hash
    # permanecen intactos.
    if "energy_dft_ev_atom" not in data.columns:
        if "energy_per_atom_ev" not in data.columns:
            raise DiagnosticError("El dataset debe incluir energy_dft_ev_atom o energy_per_atom_ev.")
        data["energy_dft_ev_atom"] = pd.to_numeric(data["energy_per_atom_ev"], errors="coerce")
    if not np.isfinite(data["energy_dft_ev_atom"].to_numpy(dtype=float)).all():
        raise DiagnosticError("La energía DFT contiene valores ausentes o no finitos.")
    with np.load(cache_path, allow_pickle=False) as cache:
        key = "embeddings"
        if key not in cache:
            raise DiagnosticError(f"No se encuentra la matriz {key} en {cache_path}.")
        vectors = cache[key].astype(np.float32, copy=True)
    if vectors.shape != (len(data), 768) or not np.isfinite(vectors).all():
        raise DiagnosticError(f"Forma o valores inválidos en caché: {vectors.shape}.")
    if data["structure_id"].astype(str).duplicated().any():
        raise DiagnosticError("Hay structure_id duplicados.")
    if "composition" not in data.columns and "reduced_formula" in data.columns:
        data["composition"] = data["reduced_formula"].astype(str)
    if set(data["split"].astype(str)) != {"train", "val", "test"}:
        raise DiagnosticError("El manifiesto no contiene exactamente train, val y test.")
    if data.groupby("reduced_formula")["split"].nunique().gt(1).any():
        raise DiagnosticError("Fuga de composición entre particiones.")
    expected = {"train": (186, 62), "val": (24, 8), "test": (24, 8)}
    observed = data.groupby("split").agg(rows=("structure_id", "size"), compositions=("reduced_formula", "nunique"))
    observed_counts = {str(split): (int(row.rows), int(row.compositions)) for split, row in observed.iterrows()}
    if observed_counts != expected:
        raise DiagnosticError(f"El split no coincide con el manifiesto E1 esperado {expected}; observado {observed_counts}.")
    if "matgpt_slices" in data and data.groupby("matgpt_slices")["split"].nunique().gt(1).any():
        raise DiagnosticError("La misma secuencia SLICES aparece en más de una partición.")
    return data, vectors, audit


def energy_metrics(frame: pd.DataFrame, tolerance_ev: float = 0.001) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Calcula métricas absolutas, relativas, de ranking y de amplitud.

    Args:
        frame: Filas de una partición con energía DFT y predicción en eV/átomo.
        tolerance_ev: Tolerancia para pares DFT empatados y mínimos predichos.

    Returns:
        Métricas agregadas, tabla por composición y pares intracomposición.
    """
    dft = frame["energy_dft_ev_atom"].to_numpy(dtype=float)
    pred = frame["energy_pred_ev_atom"].to_numpy(dtype=float)
    if not np.isfinite(dft).all() or not np.isfinite(pred).all():
        raise DiagnosticError("Energías no finitas al calcular métricas.")
    formula_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rel_abs: list[float] = []
    pair_correct = pair_total = gs_correct = gs_total = gs_tied = 0
    prediction_tied_pairs = total_pairs = 0
    spearman_values: list[float] = []
    for formula, group in frame.groupby("composition", sort=True):
        y = group["energy_dft_ev_atom"].to_numpy(float)
        p = group["energy_pred_ev_atom"].to_numpy(float)
        yrel, prel = y - y.min(), p - p.min()
        rel_abs.extend((np.abs(yrel - prel) * 1000).tolist())
        dft_min = np.flatnonzero(np.abs(y - y.min()) <= tolerance_ev)
        pred_min = np.flatnonzero(np.abs(p - p.min()) <= tolerance_ev)
        if len(y) >= 2:
            for i in range(len(y)):
                for j in range(i + 1, len(y)):
                    total_pairs += 1
                    prediction_tied_pairs += int(abs(p[i] - p[j]) <= tolerance_ev)
        tied = len(pred_min) > 1
        if tied:
            gs_tied += 1
        else:
            gs_total += 1
            gs_correct += int(int(pred_min[0]) in set(dft_min.tolist()))
        ranges_dft = float(np.ptp(y) * 1000)
        ranges_pred = float(np.ptp(p) * 1000)
        if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(p) > 0:
            rho = float(spearmanr(y, p).statistic)
            if np.isfinite(rho):
                spearman_values.append(rho)
        formula_rows.append({"composition": formula, "n_structures": len(group), "dft_range_mev_atom": ranges_dft,
                             "predicted_range_mev_atom": ranges_pred,
                             "range_ratio": ranges_pred / ranges_dft if ranges_dft > 1e-12 else None,
                             "ground_state_correct": None if tied else int(int(pred_min[0]) in set(dft_min.tolist())),
                             "predicted_ground_state_tied": tied})
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                dft_delta, pred_delta = y[i] - y[j], p[i] - p[j]
                if abs(dft_delta) <= tolerance_ev:
                    continue
                match = (dft_delta > 0) == (pred_delta > 0) if abs(pred_delta) > tolerance_ev else False
                pair_total += 1
                pair_correct += int(match)
                pair_rows.append({"composition": formula,
                                  "structure_id_i": str(group.iloc[i]["structure_id"]),
                                  "structure_id_j": str(group.iloc[j]["structure_id"]),
                                  "dft_delta_ev_atom": dft_delta, "predicted_delta_ev_atom": pred_delta,
                                  "signed_delta_error_mev_atom": (pred_delta - dft_delta) * 1000,
                                  "ordering_correct": int(match)})
    errors = pred - dft
    comp = pd.DataFrame(formula_rows)
    summary = {
        "n_structures": int(len(frame)), "n_compositions": int(frame["composition"].nunique()),
        "energy_mae_ev_atom": float(np.mean(np.abs(errors))), "energy_rmse_ev_atom": float(np.sqrt(np.mean(errors ** 2))),
        "relative_energy_mae_mev_atom": float(np.mean(rel_abs)) if rel_abs else None,
        "ground_state_accuracy": gs_correct / gs_total if gs_total else None,
        "ground_state_correct_compositions": gs_correct, "ground_state_evaluable_compositions": gs_total,
        "predicted_tied_compositions": gs_tied, "pairwise_ordering_accuracy": pair_correct / pair_total if pair_total else None,
        "n_pairs_evaluated": pair_total, "prediction_tied_pair_fraction": prediction_tied_pairs / total_pairs if total_pairs else None,
        "mean_composition_spearman": float(np.mean(spearman_values)) if spearman_values else None,
        "mean_dft_range_mev_atom": float(comp.dft_range_mev_atom.mean()) if len(comp) else None,
        "median_dft_range_mev_atom": float(comp.dft_range_mev_atom.median()) if len(comp) else None,
        "mean_predicted_range_mev_atom": float(comp.predicted_range_mev_atom.mean()) if len(comp) else None,
        "median_predicted_range_mev_atom": float(comp.predicted_range_mev_atom.median()) if len(comp) else None,
        "mean_range_ratio": float(comp.range_ratio.dropna().mean()) if comp.range_ratio.notna().any() else None,
        "mean_absolute_range_error_mev_atom": float(np.abs(comp.predicted_range_mev_atom - comp.dft_range_mev_atom).mean()) if len(comp) else None,
        "near_zero_predicted_ranges_lt_1mev": int((comp.predicted_range_mev_atom < 1).sum()) if len(comp) else 0,
        "tie_tolerance_ev_atom": tolerance_ev,
    }
    return summary, comp, pd.DataFrame(pair_rows)


def with_prediction(frame: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Añade predicción, error y energías relativas por composición.

    Args:
        frame: Filas de una sola partición con energía DFT e identificadores.
        predicted: Vector de energía predicha en eV/átomo, alineado por fila.

    Returns:
        Copia del DataFrame con predicción, residuo y ΔE en meV/átomo.
    """
    result = frame.copy()
    result["energy_pred_ev_atom"] = predicted.astype(float)
    result["energy_error_ev_atom"] = result.energy_pred_ev_atom - result.energy_dft_ev_atom
    result["dft_relative_energy_mev_atom"] = result.energy_dft_ev_atom.groupby(result.composition).transform(lambda x: (x - x.min()) * 1000)
    result["predicted_relative_energy_mev_atom"] = result.energy_pred_ev_atom.groupby(result.composition).transform(lambda x: (x - x.min()) * 1000)
    return result
