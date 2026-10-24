#!/usr/bin/env python3
"""Resume validation sin puntuación compuesta y marca candidatos no dominados."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, DiagnosticError, log, logger_for

LOGGER = logger_for("matgpt_e1_diagnostics")
OBJECTIVES = {"energy_mae_ev_atom": "min", "relative_energy_mae_mev_atom": "min",
              "ground_state_accuracy": "max", "pairwise_ordering_accuracy": "max",
              "mean_absolute_range_error_mev_atom": "min"}


def markdown_table(frame: pd.DataFrame) -> str:
    """Convierte una tabla pequeña a Markdown sin depender de tabulate.

    Args:
        frame: Tabla con columnas de métricas y candidaturas.

    Returns:
        Tabla Markdown escapada para insertar en el reporte.
    """
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pareto(metrics: pd.DataFrame, collapse_threshold_mev: float) -> pd.DataFrame:
    """Calcula medias, dispersiones y candidatos no dominados.

    Args:
        metrics: CSV tabular con una fila por configuración, semilla y split.
        collapse_threshold_mev: Umbral en meV/átomo para marcar colapso del rango.

    Returns:
        Una fila por configuración con objetivos medios, DE y estado Pareto.
    """
    validation = metrics[metrics.split.astype(str).eq("val")].copy()
    if validation.empty:
        raise DiagnosticError("metrics.csv no contiene filas de validación.")
    required = set(OBJECTIVES) | {"configuration", "mean_predicted_range_mev_atom"}
    if not required.issubset(validation.columns):
        raise DiagnosticError(f"Faltan campos para el análisis multiobjetivo: {sorted(required-set(validation.columns))}")
    values = validation.groupby("configuration", dropna=False).agg(
        **{f"{column}_mean": (column, "mean") for column in OBJECTIVES},
        **{f"{column}_std": (column, "std") for column in OBJECTIVES},
        mean_predicted_range_mev_atom=("mean_predicted_range_mev_atom", "mean"),
        n_seed_rows=("seed", "nunique"), head_type=("head_type", "first"), pooling=("pooling", "first"),
        loss_type=("loss_type", "first"), pairwise_lambda=("pairwise_lambda", "first")).reset_index()
    frontier = []
    for index, candidate in values.iterrows():
        collapsed = float(candidate.mean_predicted_range_mev_atom) < collapse_threshold_mev
        dominated = False
        for other_index, other in values.iterrows():
            if other_index == index or float(other.mean_predicted_range_mev_atom) < collapse_threshold_mev:
                continue
            no_worse = True
            strictly_better = False
            for metric, direction in OBJECTIVES.items():
                a, b = candidate[f"{metric}_mean"], other[f"{metric}_mean"]
                if pd.isna(a) or pd.isna(b):
                    no_worse = False
                    break
                if direction == "min":
                    no_worse &= float(b) <= float(a) + 1e-12
                    strictly_better |= float(b) < float(a) - 1e-12
                else:
                    no_worse &= float(b) >= float(a) - 1e-12
                    strictly_better |= float(b) > float(a) + 1e-12
            if no_worse and strictly_better:
                dominated = True
                break
        frontier.append({"pareto_nondominated": bool(not collapsed and not dominated),
                         "predicted_range_collapse": bool(collapsed),
                         "collapse_threshold_mev_atom": collapse_threshold_mev})
    return pd.concat([values, pd.DataFrame(frontier)], axis=1).sort_values(["pareto_nondominated", "energy_mae_ev_atom_mean"], ascending=[False, True])


def main() -> None:
    """Exporta tabla de frontera de Pareto y reporte de validation en Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--collapse-threshold-mev", type=float, default=1.0)
    args = parser.parse_args()
    try:
        source = args.experiment_dir / "metrics.csv"
        if not source.is_file() or args.collapse_threshold_mev < 0:
            raise DiagnosticError("Falta metrics.csv o el umbral de colapso es negativo.")
        result = pareto(pd.read_csv(source), args.collapse_threshold_mev)
        output = args.experiment_dir
        result.to_csv(output / "validation_pareto.csv", index=False)
        frontier = result[result.pareto_nondominated]
        header = ["# Diagnóstico de validación E1.1/E1.2", "", "La selección usa exclusivamente validation. Test no se consulta.",
                  "La frontera no asigna una puntuación global: conserva compromisos entre MAE absoluta, MAE relativa, exactitud de estado fundamental, ordenamiento de pares y error de rango.",
                  f"Se marca como colapso un rango predicho medio menor que {args.collapse_threshold_mev:g} meV/átomo.", ""]
        table_cols = ["configuration", "head_type", "pooling", "loss_type", "pairwise_lambda", "energy_mae_ev_atom_mean",
                      "relative_energy_mae_mev_atom_mean", "ground_state_accuracy_mean", "pairwise_ordering_accuracy_mean",
                      "mean_predicted_range_mev_atom", "mean_absolute_range_error_mev_atom_mean", "pareto_nondominated", "predicted_range_collapse"]
        header += [markdown_table(result[table_cols]), "", "No se declara ganador automático. Revise también dispersión entre semillas, MAE por familia, curvas de entrenamiento/validación, rangos por composición y predicciones colapsadas.",
                   "El conjunto test histórico ya fue inspeccionado durante E1; utilícelo solo para evaluación final descriptiva una vez bloqueadas las configuraciones."]
        (output / "validation_report.md").write_text("\n".join(header) + "\n", encoding="utf-8")
        log(LOGGER, logging.INFO, "validation_pareto_complete", candidates=len(result), nondominated=len(frontier), report=str(output / "validation_report.md"))
    except (DiagnosticError, OSError, ValueError, KeyError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "validation_pareto_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
