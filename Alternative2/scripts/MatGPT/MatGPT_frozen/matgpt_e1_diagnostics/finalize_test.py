#!/usr/bin/env python3
"""Evalúa en test únicamente las configuraciones ya bloqueadas con validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, E1_REPORT, DiagnosticError, energy_metrics, json_write, load_inputs, log, logger_for, sha256_file, with_prediction
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.train_heads import Head

LOGGER = logger_for("matgpt_e1_diagnostics")


def evaluate(selection_path: Path, dataset_path: Path, last_cache: Path, last_metadata: Path, last_audit: Path,
             mean_cache: Path, mean_metadata: Path, mean_audit: Path, output: Path, tolerance: float) -> dict[str, Any]:
    """Evalúa lista de checkpoints bloqueados y exporta predicciones y estadísticas test.

    Args:
        selection_path: CSV con candidato, checkpoint y motivo de selección por validation.
        dataset_path: Dataset inmutable con split histórico.
        last_cache, last_metadata, last_audit: Caché E1 de estado `<end>`.
        mean_cache, mean_metadata, mean_audit: Caché masked mean auditado.
        output: Directorio nuevo para resultados test.
        tolerance: Tolerancia intracomposición en eV/átomo.

    Returns:
        Resumen de las configuraciones evaluadas.
    """
    if not selection_path.is_file():
        raise DiagnosticError(f"Falta archivo de selección bloqueada: {selection_path}")
    selection = pd.read_csv(selection_path, low_memory=False)
    required = {"run_dir", "checkpoint_file", "configuration", "pooling", "seed", "selection_reason"}
    if selection.empty or not required.issubset(selection.columns):
        raise DiagnosticError(f"CSV de selección debe contener {sorted(required)} y al menos una fila.")
    if selection["selection_reason"].isna().any() or selection["selection_reason"].astype(str).str.strip().eq("").any():
        raise DiagnosticError("Cada candidato requiere una justificación predefinida basada solo en train/validation.")
    if not set(selection.pooling.astype(str)) <= {"last_token", "masked_mean"}:
        raise DiagnosticError("pooling debe ser last_token o masked_mean.")
    if output.exists() and any(output.iterdir()):
        raise DiagnosticError(f"El destino no está vacío: {output}; use una nueva versión para preservar resultados.")
    output.mkdir(parents=True, exist_ok=True)
    cache_info = {"last_token": (last_cache, last_metadata, last_audit), "masked_mean": (mean_cache, mean_metadata, mean_audit)}
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    formula_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    selected_meta: list[dict[str, Any]] = []
    test_labels_read_after_selection = True
    for row in selection.itertuples(index=False):
        run_dir = Path(str(row.run_dir)).expanduser()
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        checkpoint_path = run_dir / str(row.checkpoint_file)
        if not checkpoint_path.is_file():
            raise DiagnosticError(f"No existe checkpoint seleccionado: {checkpoint_path}")
        cache_path, meta_path, audit_path = cache_info[str(row.pooling)]
        data, vectors, audit = load_inputs(dataset_path, cache_path, meta_path, audit_path)
        test_mask = data.split.astype(str).eq("test").to_numpy()
        if not test_mask.any():
            raise DiagnosticError("No existen filas test en el manifiesto histórico.")
        if checkpoint_path.suffix == ".pt":
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload.get("dataset_sha256") != sha256_file(dataset_path) or payload.get("embedding_cache_sha256") != sha256_file(cache_path):
                raise DiagnosticError(f"Checkpoint no corresponde a dataset/caché seleccionados: {checkpoint_path}")
            if payload.get("backbone_sha256", payload.get("backbone_checkpoint_sha256")) != audit["checkpoint_sha256"]:
                raise DiagnosticError("El checkpoint fue entrenado con otro backbone.")
            if str(payload.get("configuration")) != str(row.configuration) or str(payload.get("seed")) != str(row.seed):
                raise DiagnosticError("La configuración o semilla declarada no coincide con el checkpoint MLP.")
            if str(row.pooling) not in str(payload.get("configuration", "")):
                raise DiagnosticError("El pooling declarado no coincide con la configuración MLP guardada.")
            model = Head(int(payload["hidden_dim"]))
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            with torch.inference_mode():
                pred = model(torch.from_numpy(vectors[test_mask])).numpy() * float(payload["target_scale_train_ev_atom"]) + float(payload["target_mean_train_ev_atom"])
            label = str(payload.get("configuration", checkpoint_path.stem))
            seed = payload.get("seed")
            best_epoch = payload.get("best_epoch")
        elif checkpoint_path.suffix == ".joblib":
            bundle = joblib.load(checkpoint_path)
            if bundle.get("dataset_sha256") != sha256_file(dataset_path) or bundle.get("embedding_cache_sha256") != sha256_file(cache_path):
                raise DiagnosticError("Checkpoint Ridge no corresponde a dataset/caché seleccionados.")
            if bundle.get("backbone_sha256") != audit["checkpoint_sha256"]:
                raise DiagnosticError("Checkpoint Ridge refiere un backbone distinto.")
            if str(checkpoint_path.stem) != str(row.configuration) or str(row.seed) != "not_applicable":
                raise DiagnosticError("La configuración o semilla declarada no coincide con el checkpoint Ridge.")
            if bundle.get("pooling") != str(row.pooling):
                raise DiagnosticError("El pooling declarado no coincide con el caché usado por Ridge.")
            model, scaler = bundle["model"], bundle["scaler"]
            pred = model.predict(scaler.transform(vectors[test_mask]))
            label, seed, best_epoch = checkpoint_path.stem, "not_applicable", None
        else:
            raise DiagnosticError(f"Extensión de checkpoint desconocida: {checkpoint_path}")
        test = with_prediction(data.loc[test_mask].copy(), np.asarray(pred))
        stats, by_formula, pairs = energy_metrics(test, tolerance)
        test["configuration"] = label
        test["pooling"] = str(row.pooling)
        test["seed"] = seed
        predictions.append(test)
        by_formula["configuration"] = label
        by_formula["pooling"] = str(row.pooling)
        formula_rows.append(by_formula)
        if not pairs.empty:
            pairs["configuration"] = label
            pairs["pooling"] = str(row.pooling)
            pair_rows.append(pairs)
        metric_rows.append({"configuration": label, "pooling": str(row.pooling), "seed": seed, "split": "test",
                            "best_epoch": best_epoch, **stats,
                            "selection_reason_validation_only": str(row.selection_reason),
                            "checkpoint_path": str(checkpoint_path)})
        selected_meta.append({"configuration": label, "pooling": str(row.pooling), "seed": seed,
                              "checkpoint_sha256": sha256_file(checkpoint_path), "embedding_cache_sha256": sha256_file(cache_path),
                              "selection_reason": str(row.selection_reason)})
    pd.concat(predictions, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "test_metrics.csv", index=False)
    pd.concat(formula_rows, ignore_index=True).to_csv(output / "test_by_composition.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(output / "test_pairwise.csv", index=False) if pair_rows else None
    result = {"status": "FINAL_TEST_EVALUATION", "selection_file": str(selection_path),
              "selection_file_sha256": sha256_file(selection_path), "selection_rows": selected_meta,
              "dataset_sha256": sha256_file(dataset_path), "n_test_structures": int(metric_rows[0]["n_structures"]),
              "n_candidate_evaluations": len(metric_rows),
              "test_labels_read_after_explicit_selection": test_labels_read_after_selection,
              "test_previously_inspected_in_historical_e1": True,
              "warning": "El test histórico ya había sido inspeccionado en E1; los resultados finales son descriptivos y no deben usarse para seleccionar o ajustar candidatos."}
    json_write(output / "final_test_manifest.json", result)
    return result


def main() -> None:
    """Valida argumentos y ejecuta una sola evaluación final de test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="CSV bloqueado tras validation; columnas run_dir, checkpoint_file, configuration, pooling, seed, selection_reason.")
    parser.add_argument("--dataset", type=Path, default=E1_REPORT / "dataset.csv")
    parser.add_argument("--last-cache", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "matgpt_uncond_last_token.npz")
    parser.add_argument("--last-metadata", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "embedding_metadata.csv")
    parser.add_argument("--last-audit", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_v2" / "embedding_audit.json")
    parser.add_argument("--mean-cache", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "matgpt_uncond_masked_mean.npz")
    parser.add_argument("--mean-metadata", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "embedding_metadata.csv")
    parser.add_argument("--mean-audit", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "embedding_audit.json")
    parser.add_argument("--out-dir", type=Path, default=ALT / "reports" / "matgpt_e1_final_test_v1")
    parser.add_argument("--tie-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    try:
        result = evaluate(args.selection, args.dataset, args.last_cache, args.last_metadata, args.last_audit,
                          args.mean_cache, args.mean_metadata, args.mean_audit, args.out_dir, args.tie_tolerance)
        log(LOGGER, logging.INFO, "final_test_evaluation_complete", candidates=len(result["selection_rows"]), structures=result["n_test_structures"], report=str(args.out_dir))
    except (DiagnosticError, OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "final_test_evaluation_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
