#!/usr/bin/env python3
"""Funciones compartidas para la validación cruzada por composición MatGPT E1.3."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ALT = ROOT / "Alternative2"
REPORTS = ALT / "reports"
OUT = REPORTS / "matgpt_e1_3"
DATASET = REPORTS / "matgpt_e1_v2" / "dataset.csv"
SPLIT_MANIFEST = REPORTS / "matgpt_e1_v2" / "split_manifest.json"
LOAD_AUDIT = REPORTS / "matgpt_e0" / "load_validation.json"
LAST_DIR = ALT / "data" / "embeddings" / "matgpt_e1_v2"
MEAN_DIR = ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1"
CHECKPOINT = ALT / "models" / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "transformer_property_0.pth"
MODEL_INFO = CHECKPOINT.with_name("model_info_property_0.json")
SOURCE = ALT / "models" / "MatGPT-source" / "mat-concat"
LAST_EXTRACTOR = ALT / "scripts" / "matgpt_e1" / "extract_embeddings.py"
MEAN_EXTRACTOR = ALT / "scripts" / "matgpt_e1_diagnostics" / "extract_masked_mean.py"
SOURCE_MANIFEST = ALT / "models" / "MatGPT-source.download-manifest.json"
EXPECTED_CHECKPOINT_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
SEEDS = (42, 123, 2026)
BOOTSTRAP_SEED = 13072026
BOOTSTRAP_REPLICATES = 2000
TIE_TOLERANCE_EV = 0.001
EXPECTED_FAMILIES = {"dist_perovskite", "hexagonal", "needle"}
E12_DIRS = {
    "last_mlp32_abs": REPORTS / "matgpt_e1_2_last_mlp32_abs_v1",
    "mean_mlp32_abs": REPORTS / "matgpt_e1_2_mean_mlp32_abs_v1",
    "last_mlp8_pair1": REPORTS / "matgpt_e1_2_last_mlp8_1",
    "mean_mlp8_pair1": REPORTS / "matgpt_e1_2_mean_mlp8_1",
}
CANDIDATES: dict[str, dict[str, Any]] = {
    "matgpt_e13_mlp8_mean_pair1": {"pooling": "masked_mean", "head": "mlp8", "width": 8, "pairwise_lambda": 1.0, "hypothesis": "Selección del estado fundamental"},
    "matgpt_e13_mlp8_last_pair1": {"pooling": "last_token", "head": "mlp8", "width": 8, "pairwise_lambda": 1.0, "hypothesis": "Diferencias de energía relativa"},
    "matgpt_e13_mlp32_mean_abs": {"pooling": "masked_mean", "head": "mlp32", "width": 32, "pairwise_lambda": 0.0, "hypothesis": "Energía absoluta"},
}


class E13Error(RuntimeError):
    """Fallo de integridad o de protocolo que detiene E1.3."""


class JsonFormatter(logging.Formatter):
    """Formato JSON por línea para logs reproducibles."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa nivel, evento y campos extra del registro."""
        return json.dumps({"level": record.levelname, "message": record.getMessage(), **getattr(record, "fields", {})}, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    """Construye un logger JSON sin propagación al root logger."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Escribe un evento JSON con campos estructurados."""
    logger.log(level, message, extra={"fields": fields})


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques para no cargar artefactos grandes completos."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON UTF-8 indentado y crea su directorio padre."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def canonical_formula(formula: str) -> str:
    """Normaliza fórmula elemental completa y reduce proporciones estequiométricas.

    Acepta fórmulas planas de símbolos químicos y cantidades decimales; no
    adivina paréntesis, cargas o tokens desconocidos, y detiene el preflight
    para evitar agrupar composiciones de forma ambigua.
    """
    text = str(formula).replace(" ", "")
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", text))
    if not text or "".join(match.group(0) for match in matches) != text:
        raise E13Error(f"Fórmula no compatible con canonización plana: {formula!r}")
    amounts: dict[str, float] = {}
    for match in matches:
        element, raw = match.groups()
        amounts[element] = amounts.get(element, 0.0) + (float(raw) if raw else 1.0)
    if not amounts or any(value <= 0 or not np.isfinite(value) for value in amounts.values()):
        raise E13Error(f"Composición química inválida: {formula!r}")
    scale = min(amounts.values())
    reduced = {element: value / scale for element, value in amounts.items()}
    parts = []
    for element in sorted(reduced):
        value = reduced[element]
        count = int(round(value))
        if abs(value - count) < 1e-7:
            suffix = "" if count == 1 else str(count)
        else:
            suffix = f"{value:.8g}"
        parts.append(element + suffix)
    return "".join(parts)


def _require_files(paths: list[Path]) -> None:
    """Falla con una lista legible si falta cualquier entrada crítica."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise E13Error(f"Faltan entradas críticas de E1.3: {missing}")


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Valida procedencia/datasets/cachés y devuelve solo los 234 embeddings reales.

    Returns:
        Dataset ordenado, cachés ``last_token`` y ``masked_mean`` y procedencia.
    """
    files = [DATASET, SPLIT_MANIFEST, LOAD_AUDIT, CHECKPOINT, MODEL_INFO, SOURCE_MANIFEST,
             LAST_DIR / "matgpt_uncond_last_token.npz", LAST_DIR / "embedding_metadata.csv", LAST_DIR / "embedding_audit.json",
             MEAN_DIR / "matgpt_uncond_masked_mean.npz", MEAN_DIR / "embedding_metadata.csv", MEAN_DIR / "embedding_audit.json",
             SOURCE / "config" / "uncond.yaml", SOURCE / "utils" / "models.py", LAST_EXTRACTOR, MEAN_EXTRACTOR]
    for directory in E12_DIRS.values():
        files.extend([directory / "metrics.csv", directory / "predictions_train_val.csv", directory / "config.yaml", directory / "run_manifest.json"])
    _require_files(files)
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise E13Error("El hash del checkpoint MatGPT no coincide con el auditado en E0.")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("resolved_commit") != EXPECTED_COMMIT:
        raise E13Error("El código local MatGPT no corresponde al commit E0 fijado.")
    e0 = json.loads(LOAD_AUDIT.read_text(encoding="utf-8"))
    if e0.get("status") != "PASS" or e0.get("provenance", {}).get("checkpoint_sha256", EXPECTED_CHECKPOINT_SHA) != EXPECTED_CHECKPOINT_SHA:
        raise E13Error("La auditoría E0 no tiene estado PASS o registra otro checkpoint.")
    dataset = pd.read_csv(DATASET, low_memory=False)
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != sha256_file(DATASET):
        raise E13Error("El dataset no coincide con su split_manifest histórico.")
    required = {"structure_id", "reduced_formula", "energy_per_atom_ev", "split", "polymorph_family", "matgpt_slices"}
    if required - set(dataset.columns):
        raise E13Error(f"Columnas faltantes en dataset E1: {sorted(required - set(dataset.columns))}")
    if len(dataset) != 234 or dataset.structure_id.astype(str).duplicated().any():
        raise E13Error("Se esperaban 234 structure_id únicos en el dataset E1.")
    dataset["composition_group"] = dataset.reduced_formula.astype(str).map(canonical_formula)
    dataset["energy_dft_ev_atom"] = pd.to_numeric(dataset.energy_per_atom_ev, errors="coerce")
    if not np.isfinite(dataset.energy_dft_ev_atom.to_numpy(float)).all():
        raise E13Error("El dataset contiene energía DFT ausente o no finita.")
    if "energy_consistency_error_ev_atom" in dataset.columns:
        consistency = pd.to_numeric(dataset.energy_consistency_error_ev_atom, errors="coerce").to_numpy(float)
        if not np.isfinite(consistency).all() or np.max(np.abs(consistency)) > 1e-5:
            raise E13Error("La auditoría de consistencia de energía por átomo excede 1e-5 eV/átomo o no es finita.")
    expected_splits = {"train": (186, 62), "val": (24, 8), "test": (24, 8)}
    observed = dataset.groupby("split").agg(rows=("structure_id", "size"), groups=("composition_group", "nunique"))
    counts = {str(key): (int(row.rows), int(row.groups)) for key, row in observed.iterrows()}
    if counts != expected_splits:
        raise E13Error(f"Split histórico inesperado: {counts}; esperado {expected_splits}.")
    if dataset.groupby("composition_group").split.nunique().gt(1).any():
        raise E13Error("Hay composición química compartida entre splits históricos.")
    for group, part in dataset.groupby("composition_group"):
        families = set(part.polymorph_family.astype(str))
        if len(part) != 3 or families != EXPECTED_FAMILIES:
            raise E13Error(f"{group}: se requieren exactamente 3 polimorfos físicos, uno por familia; hallado {len(part)} {families}.")
    dev_groups = set(dataset.loc[dataset.split.isin(["train", "val"]), "composition_group"])
    test_groups = set(dataset.loc[dataset.split.eq("test"), "composition_group"])
    dev_ids = set(dataset.loc[dataset.split.isin(["train", "val"]), "structure_id"].astype(str))
    test_ids = set(dataset.loc[dataset.split.eq("test"), "structure_id"].astype(str))
    if len(dev_groups) != 70 or len(test_groups) != 8 or len(dev_ids) != 210 or dev_groups & test_groups or dev_ids & test_ids:
        raise E13Error("development/test no satisface 70/8 composiciones y 210/24 estructuras disjuntas.")

    vectors: dict[str, np.ndarray] = {}
    meta_frames: list[pd.DataFrame] = []
    embed_audits: dict[str, Any] = {}
    extractor_verification: dict[str, dict[str, Any]] = {}
    for pooling, folder, name, extractor in (("last_token", LAST_DIR, "matgpt_uncond_last_token.npz", LAST_EXTRACTOR),
                                              ("masked_mean", MEAN_DIR, "matgpt_uncond_masked_mean.npz", MEAN_EXTRACTOR)):
        cache_path, meta_path, audit_path = folder / name, folder / "embedding_metadata.csv", folder / "embedding_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS" or audit.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
            raise E13Error(f"Auditoría no PASS para {pooling}.")
        recorded_extractor_hash = audit.get("extractor_script_sha256")
        current_extractor_hash = sha256_file(extractor)
        if recorded_extractor_hash and recorded_extractor_hash != current_extractor_hash:
            raise E13Error(f"El extractor de {pooling} cambió desde que se generó el caché.")
        # Algunas auditorías E1.2 antiguas no guardaron el hash del script
        # extractor. En ese caso no se inventa una coincidencia histórica:
        # la procedencia disponible es el hash del caché y metadatos, la
        # referencia al mismo caché en los manifiestos de E1.2 y la
        # comprobación semántica del código de pooling más abajo.
        extractor_verification[pooling] = {
            "recorded_sha256": recorded_extractor_hash,
            "current_sha256": current_extractor_hash,
            "status": "hash_match" if recorded_extractor_hash else "historical_hash_not_recorded_semantics_checked",
        }
        if audit.get("cache_sha256") != sha256_file(cache_path) or audit.get("metadata_sha256") != sha256_file(meta_path) or audit.get("dataset_sha256") != sha256_file(DATASET):
            raise E13Error(f"Hashes de caché/metadatos/dataset no coinciden para {pooling}.")
        policy = str(audit.get("pooling", "")).lower()
        if pooling == "last_token" and "<end>" not in policy:
            raise E13Error("La auditoría last_token no acredita pooling del estado <end>.")
        if pooling == "masked_mean" and not all(token in policy for token in ("start", "end", "pad")):
            raise E13Error("La auditoría masked_mean no declara exclusión de start/end/pad.")
        extractor_source = extractor.read_text(encoding="utf-8")
        if "generate_square_subsequent_mask" not in extractor_source or "ln_f" not in extractor_source:
            raise E13Error(f"El extractor {pooling} no muestra máscara causal y hidden state posterior a ln_f.")
        if pooling == "last_token" and '"<end>"' not in extractor_source:
            raise E13Error("El extractor last-token no referencia explícitamente el token <end>.")
        meta = pd.read_csv(meta_path, low_memory=False)
        if meta.structure_id.astype(str).tolist() != dataset.structure_id.astype(str).tolist():
            raise E13Error(f"Orden/IDs de metadatos no coinciden con dataset en {pooling}.")
        with np.load(cache_path, allow_pickle=False) as cache:
            matrix = cache["embeddings"].astype(np.float32, copy=True)
        if matrix.shape != (234, 768) or not np.isfinite(matrix).all():
            raise E13Error(f"Caché {pooling} debe ser finite [234,768], hallado {matrix.shape}.")
        vectors[pooling] = matrix
        meta_frames.append(meta)
        embed_audits[pooling] = audit
    # Confirmar identidad semántica, etiqueta y split además del orden físico.
    for meta in meta_frames:
        for column in ("structure_id", "reduced_formula", "split", "energy_per_atom_ev", "polymorph_family", "matgpt_slices"):
            if column not in meta or meta[column].astype(str).tolist() != dataset[column].astype(str).tolist():
                raise E13Error(f"Metadata de embedding no coincide con el dataset en {column}.")

    # Confirma que los cuatro resultados históricos incluyen las configuraciones citadas.
    expected_configs = {
        "last_mlp32_abs": ("last_token", "mlp32", 0.0),
        "mean_mlp32_abs": ("masked_mean", "mlp32", 0.0),
        "last_mlp8_pair1": ("last_token", "mlp8", 1.0),
        "mean_mlp8_pair1": ("masked_mean", "mlp8", 1.0),
    }
    e12_hashes: dict[str, dict[str, str]] = {}
    for key, folder in E12_DIRS.items():
        frame = pd.read_csv(folder / "metrics.csv")
        run_manifest_path = folder / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        pooling, head, lam = expected_configs[key]
        valid = frame[(frame.pooling == pooling) & (frame.head_type == head) & (frame.pairwise_lambda.astype(float) == lam)]
        if valid.empty or not {"train", "val"}.issubset(set(valid.split.astype(str))):
            raise E13Error(f"Resultados {key} no documentan train/val para el control esperado.")
        if set(valid.seed.astype(str)) != {str(seed) for seed in SEEDS}:
            raise E13Error(f"Resultados {key} no cubren exactamente las semillas {SEEDS}.")
        hp = run_manifest.get("configured_hyperparameters", {})
        expected_settings = (hp.get("optimizer") == "AdamW" and hp.get("learning_rate") == 0.001 and
            hp.get("weight_decay") == 0.0001 and hp.get("max_epochs") == 200 and
            hp.get("early_stopping_patience") == 20 and hp.get("compositions_per_batch") == 8 and
            hp.get("dropout") == 0.1 and hp.get("mlp_hidden_units") == [int(head.removeprefix("mlp"))] and
            hp.get("pairwise_lambda_grid") == [lam] and set(hp.get("seeds", [])) == set(SEEDS) and
            run_manifest.get("dataset_sha256") == sha256_file(DATASET) and
            run_manifest.get("backbone_sha256") == EXPECTED_CHECKPOINT_SHA and
            run_manifest.get("selection_uses_test") is False and run_manifest.get("test_labels_read_for_metrics") is False and
            run_manifest.get("early_stopping") == "minimum validation normalized training objective; validation labels only" and
            "calculated from train only" in str(run_manifest.get("loss_scales", "")))
        if not expected_settings:
            raise E13Error(f"Run manifest E1.2 {key} difiere del entrenamiento/candidato efectivo esperado.")
        expected_embed_hash = embed_audits[pooling]["cache_sha256"]
        if run_manifest.get("embedding_sha256") != expected_embed_hash:
            raise E13Error(f"Run manifest E1.2 {key} referencia otro caché de embeddings.")
        e12_hashes[key] = {name: sha256_file(folder / name) for name in ("metrics.csv", "predictions_train_val.csv", "config.yaml", "run_manifest.json")}

    # Detecta secuencias repetidas sin descartar las observaciones.
    return dataset, vectors, {"e0_audit": e0, "embed_audits": embed_audits,
                             "extractor_verification": extractor_verification, "e12_hashes": e12_hashes,
                             "source_manifest": source, "manifest": manifest}


def load_fold_assignments(path: Path = OUT / "fold_assignments.csv") -> pd.DataFrame:
    """Carga el manifiesto largo, con una fila por estructura/fold de desarrollo."""
    if not path.is_file():
        raise E13Error(f"Primero prepara folds: no existe {path}")
    frame = pd.read_csv(path, dtype={"structure_id": str, "composition_group": str})
    dev = frame[frame.historical_split.isin(["train", "val"])]
    test = frame[frame.historical_split.eq("test")]
    if (frame.empty or frame.duplicated(["structure_id", "outer_fold"]).any() or len(dev) != 1050 or
            dev.groupby("structure_id").outer_fold.nunique().ne(5).any() or len(test) != 24 or
            test.structure_id.duplicated().any() or not test.outer_fold.eq(-1).all()):
        raise E13Error("fold_assignments debe tener 5 roles por cada uno de 210 IDs de desarrollo y 24 IDs de test excluidos.")
    return frame


def group_metrics(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calcula métricas pooled y macro por composición, nunca cruza fórmulas."""
    rows: list[dict[str, Any]] = []
    pair_correct = pair_total = 0
    delta_errors_pooled: list[float] = []
    for group, part in frame.groupby("composition_group", sort=True):
        dft = part.energy_dft_ev_atom.to_numpy(float)
        pred = part.energy_pred_ev_atom.to_numpy(float)
        if len(part) != 3 or not np.isfinite(dft).all() or not np.isfinite(pred).all():
            raise E13Error(f"Predicciones/energías inválidas o grupo incompleto: {group}.")
        dft_rel, pred_rel = dft - dft.min(), pred - pred.min()
        delta_errors_pooled.extend((np.abs(pred_rel-dft_rel)*1000).tolist())
        dft_min = np.abs(dft - dft.min()) <= TIE_TOLERANCE_EV
        pred_min = np.abs(pred - pred.min()) <= TIE_TOLERANCE_EV
        pred_tie = bool(pred_min.sum() > 1)
        dft_tie = bool(dft_min.sum() > 1)
        # Si DFT empata, elegir cualquiera de sus mínimos cuenta como correcto;
        # si ML empata, la cabeza no selecciona un estado y se marca indefinida.
        gs_ok: bool | None = None if pred_tie else bool(np.flatnonzero(pred_min)[0] in set(np.flatnonzero(dft_min).tolist()))
        local_correct = local_total = 0
        near_tie_pairs = 0
        for i in range(len(dft)):
            for j in range(i + 1, len(dft)):
                dd, dp = dft[i] - dft[j], pred[i] - pred[j]
                if abs(dd) <= TIE_TOLERANCE_EV:
                    near_tie_pairs += 1
                    continue
                local_total += 1
                local_correct += int(abs(dp) > TIE_TOLERANCE_EV and ((dd > 0) == (dp > 0)))
        pair_correct += local_correct
        pair_total += local_total
        dft_range, pred_range = float(np.ptp(dft) * 1000), float(np.ptp(pred) * 1000)
        range_error = pred_range - dft_range
        rho = None
        if np.ptp(dft) > 0 and np.ptp(pred) > 0:
            from scipy.stats import spearmanr
            candidate = float(spearmanr(dft, pred).statistic)
            rho = candidate if np.isfinite(candidate) else None
        rows.append({"composition_group": group, "n_structures": len(part), "absolute_mae_ev_atom": float(np.mean(np.abs(pred-dft))),
                     "absolute_rmse_ev_atom": float(np.sqrt(np.mean((pred-dft)**2))),
                     "delta_e_mae_mev_atom": float(np.mean(np.abs(pred_rel-dft_rel))*1000),
                     "ground_state_accuracy": None if gs_ok is None else int(gs_ok), "ground_state_evaluable": gs_ok is not None,
                     "dft_tied": dft_tie, "predicted_tied": pred_tie, "pairwise_correct": local_correct, "pairwise_eligible": local_total,
                     "dft_pairs_within_tie_tolerance": near_tie_pairs,
                     "pairwise_accuracy": local_correct/local_total if local_total else None,
                     "dft_range_mev_atom": dft_range, "predicted_range_mev_atom": pred_range,
                     "range_error_mev_atom": range_error, "range_error_sign": int(np.sign(range_error)),
                     "range_ratio": pred_range/dft_range if dft_range > 1e-12 else None,
                     "collapsed_lt_1mev": pred_range < 1.0, "spearman": rho,
                     "polymorph_family_mae": json.dumps({str(family): float(np.mean(np.abs((group_part.energy_pred_ev_atom-group_part.energy_dft_ev_atom))))
                         for family, group_part in part.groupby("polymorph_family")}, ensure_ascii=False)})
    by_group = pd.DataFrame(rows)
    errors = frame.energy_pred_ev_atom.to_numpy(float) - frame.energy_dft_ev_atom.to_numpy(float)
    valid_gs = by_group.ground_state_accuracy.dropna()
    valid_pair = by_group.pairwise_eligible.sum()
    pooled = {"n_structures": int(len(frame)), "n_compositions": int(by_group.composition_group.nunique()),
              "n_pairs_eligible": int(valid_pair), "absolute_mae_ev_atom_pooled": float(np.mean(np.abs(errors))),
              "absolute_rmse_ev_atom_pooled": float(np.sqrt(np.mean(errors**2))),
              "absolute_mae_ev_atom_macro": float(by_group.absolute_mae_ev_atom.mean()),
              "absolute_rmse_ev_atom_macro": float(by_group.absolute_rmse_ev_atom.mean()),
              "delta_e_mae_mev_atom_pooled": float(np.mean(delta_errors_pooled)),
              "delta_e_mae_mev_atom_macro": float(by_group.delta_e_mae_mev_atom.mean()),
              "ground_state_accuracy": float(valid_gs.mean()) if len(valid_gs) else None,
              "ground_state_evaluable_compositions": int(len(valid_gs)),
              "pairwise_ordering_accuracy": float(by_group.pairwise_correct.sum()/valid_pair) if valid_pair else None,
              "pairwise_macro_accuracy": float(by_group.pairwise_accuracy.dropna().mean()) if by_group.pairwise_accuracy.notna().any() else None,
              "collapsed_compositions_lt_1mev": int(by_group.collapsed_lt_1mev.sum()),
              "predicted_tied_compositions": int(by_group.predicted_tied.sum()),
              "dft_tied_compositions": int(by_group.dft_tied.sum()),
              "dft_pairs_within_tie_tolerance": int(by_group.dft_pairs_within_tie_tolerance.sum()),
              "range_error_mev_atom_macro": float(by_group.range_error_mev_atom.mean()),
              "range_error_median_mev_atom": float(by_group.range_error_mev_atom.median()),
              "range_error_std_mev_atom": float(by_group.range_error_mev_atom.std(ddof=1)) if len(by_group) > 1 else 0.0,
              "delta_e_mae_median_mev_atom": float(by_group.delta_e_mae_mev_atom.median()),
              "range_ratio_median": float(by_group.range_ratio.median()) if by_group.range_ratio.notna().any() else None,
              "range_ratio_std": float(by_group.range_ratio.std(ddof=1)) if by_group.range_ratio.notna().sum() > 1 else None,
              "spearman_macro_defined": float(by_group.spearman.dropna().mean()) if by_group.spearman.notna().any() else None,
              "tie_tolerance_ev_atom": TIE_TOLERANCE_EV}
    return pooled, by_group

