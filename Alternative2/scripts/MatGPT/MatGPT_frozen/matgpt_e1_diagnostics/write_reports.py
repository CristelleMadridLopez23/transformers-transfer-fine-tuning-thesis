#!/usr/bin/env python3
"""Genera reportes Markdown a partir de artefactos reales E1.1/E1.2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, DiagnosticError, log, logger_for

LOGGER = logger_for("matgpt_e1_diagnostics")


def md_table(frame: pd.DataFrame) -> str:
    """Serializa resultados tabulares a Markdown sin tabulate.

    Args:
        frame: DataFrame de métricas para insertar en el reporte.

    Returns:
        Texto Markdown con celdas escapadas.
    """
    cols = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def require(path: Path) -> pd.DataFrame:
    """Carga un CSV existente o detiene el reporte con error descriptivo.

    Args:
        path: Archivo CSV de resultados requerido.

    Returns:
        DataFrame de métricas leído con pandas.
    """
    if not path.is_file():
        raise DiagnosticError(f"Falta artefacto requerido para el reporte: {path}")
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    """Compila resultados separados por partición y notas metodológicas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-1", type=Path, required=True)
    parser.add_argument("--e1-2-last", type=Path, required=True)
    parser.add_argument("--e1-2-mean", type=Path, required=True)
    parser.add_argument("--embedding-summary", type=Path, required=True)
    parser.add_argument("--final-test", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ALT / "reports" / "matgpt_e1_diagnostics_v1")
    args = parser.parse_args()
    try:
        e11 = require(args.e1_1 / "metrics.csv")
        e11_frontier = require(args.e1_1 / "validation_pareto.csv")
        last = require(args.e1_2_last / "metrics.csv")
        mean = require(args.e1_2_mean / "metrics.csv")
        geometry = json.loads(args.embedding_summary.read_text(encoding="utf-8"))
        output = args.out_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise DiagnosticError(f"El directorio de reportes ya contiene archivos: {output}; elige una versión nueva.")
        output.mkdir(parents=True, exist_ok=True)
        key_metrics = ["energy_mae_ev_atom", "relative_energy_mae_mev_atom", "ground_state_accuracy", "pairwise_ordering_accuracy",
                       "mean_predicted_range_mev_atom", "mean_absolute_range_error_mev_atom"]
        validation = e11[e11.split.eq("val")]
        agg = validation.groupby(["configuration", "head_type", "loss_type", "pairwise_lambda"], dropna=False)[key_metrics].agg(["mean", "std"]).reset_index()
        e11_path = output / "E1_1_report.md"
        e11_path.write_text("\n".join([
            "# E1.1 — Diagnóstico de cabezas y pérdidas", "",
            f"Configuraciones observadas: {validation.configuration.nunique()}; filas semilla/configuración en validation: {len(validation)}.",
            "La selección de época usó el objetivo de validación normalizado. La frontera de Pareto informa compromisos; no hay puntuación compuesta ni ganador automático.", "",
            "## Métricas de validación (media y desviación entre semillas)", "",
            md_table(agg), "", "## Candidatos no dominados y control de colapso", "",
            md_table(e11_frontier[e11_frontier.pareto_nondominated].copy()), "",
            "El reporte incluye MAE por familia, curvas y predicciones train/validation en los CSV/PNG del directorio del experimento.",
            "Las energías son DFT-GGA estáticas por átomo (eV/átomo), no energías de formación. La procedencia local no identifica el funcional exacto ni los pseudopotenciales.",
            "El test histórico ya había sido inspeccionado durante E1 y no se usó para seleccionar estos candidatos.", ""]), encoding="utf-8")

        comparisons = []
        for pooling, frame in (("last_token", last), ("masked_mean", mean)):
            val = frame[frame.split.eq("val")]
            if val.empty:
                raise DiagnosticError(f"No hay validation para pooling {pooling}.")
            for column in key_metrics:
                comparisons.append({"pooling": pooling, "metric": column, "mean_across_seed_rows": val[column].mean(),
                                    "std_across_seed_rows": val[column].std(), "n_seed_rows": val[column].notna().sum()})
        e12_path = output / "E1_2_report.md"
        e12_path.write_text("\n".join([
            "# E1.2 — Diagnóstico de pooling y representaciones congeladas", "",
            "## Métricas de validation", "", md_table(pd.DataFrame(comparisons)), "",
            "## Geometría de embeddings", "",
            md_table(pd.DataFrame([{"pooling": key, **{metric: value for metric, value in summary.items() if metric in (
                "n_structures", "n_compositions", "intra_cosine_mean", "inter_cosine_mean", "intra_euclidean_mean",
                "inter_euclidean_mean", "spearman_cosine_vs_abs_dft_delta", "spearman_euclidean_vs_abs_dft_delta",
                "spearman_embedding_norm_vs_sequence_length", "n_duplicate_slices_groups")}}
                for key, summary in geometry.items() if isinstance(summary, dict)])), "",
            "Similitud y distancia describen geometría; no prueban por sí solas que el embedding contenga o no información física. Las asociaciones con energía y longitud son descriptivas.",
            "Las representaciones se extrajeron después de ln_f y antes de lm_head; el backbone permaneció congelado. El estado oculto promedio excluyó <start>, <end> y <pad>.", ""]), encoding="utf-8")

        gap_rows = []
        train = e11[e11.split.eq("train")].groupby("configuration").energy_mae_ev_atom.mean()
        val = e11[e11.split.eq("val")].groupby("configuration").energy_mae_ev_atom.mean()
        for name in sorted(set(train.index) & set(val.index)):
            gap_rows.append({"configuration": name, "train_mae_ev_atom": train[name], "validation_mae_ev_atom": val[name],
                             "validation_minus_train_mae_ev_atom": val[name] - train[name]})
        by_head = validation.groupby("head_type")[key_metrics].mean().reset_index()
        by_lambda = validation[validation.head_type.str.startswith("mlp")].groupby(["head_type", "pairwise_lambda"])[key_metrics].mean().reset_index()
        pooling_val = pd.concat([last[last.split.eq("val")].assign(pooling="last_token"), mean[mean.split.eq("val")].assign(pooling="masked_mean")], ignore_index=True)
        pooling_summary = pooling_val.groupby("pooling")[key_metrics].agg(["mean", "std"]).reset_index()
        comparative = [
            "# Diagnóstico comparativo MatGPT E1, E1.1 y E1.2", "",
            "Este documento resume los artefactos disponibles. Train, validation y test se presentan separados. Las cifras no implican causalidad.", "",
            "## A. ¿La MLP original presenta sobreajuste?", "",
            "La tabla muestra la diferencia descriptiva entre MAE train y validation por configuración. Interprétela junto con curvas de entrenamiento, variación entre semillas y errores por composición.", "",
            md_table(pd.DataFrame(gap_rows)), "",
            "## B. ¿Reducir la capacidad mejora el rendimiento?", "", md_table(by_head), "",
            "Compare Ridge, MLP8 y MLP32 por separado en MAE absoluta, MAE relativa, ranking y rangos; ningún promedio de criterios sustituye esos compromisos.", "",
            "## C. ¿La pérdida pairwise corrige la amplitud energética?", "", md_table(by_lambda), "",
            "Revise también error firmado de diferencias y rangos por composición. Una menor MAE relativa acompañada de rangos casi nulos no demuestra una buena discriminación.", "",
            "## D. ¿Masked mean pooling es más útil que <end>?", "", md_table(pooling_summary), "",
            "La comparación debe mantener cabeza, lambda, semillas, checkpoint, IDs y particiones constantes. Consulte la geometría de embeddings y las figuras de paridad para el mismo candidato seleccionado con validation.", "",
            "## E. ¿Los embeddings congelados distinguen polimorfos?", "",
            md_table(pd.DataFrame([{"pooling": key, "similitud_coseno_intra": value.get("intra_cosine_mean"),
                "similitud_coseno_inter_muestreada": value.get("inter_cosine_mean"),
                "rho_coseno_vs_delta_DFT": value.get("spearman_cosine_vs_abs_dft_delta"),
                "grupos_SLICES_duplicadas": value.get("n_duplicate_slices_groups")} for key, value in geometry.items() if isinstance(value, dict)])), "",
            "Las distancias y correlaciones son evidencia geométrica descriptiva, no prueba suficiente de ausencia o presencia de información energética.", "",
            "## F. Consideración de E2", "",
            "Estos artefactos comparan cabezas y pooling con el backbone congelado; no ejecutan E2 ni demuestran causalmente que el preentrenamiento sea responsable. La evaluación test es descriptiva porque sus etiquetas ya fueron inspeccionadas durante E1. No se justifica iniciar E2 automáticamente: cualquier hipótesis posterior requeriría una revisión de validation, controles y límites de trazabilidad DFT.", "",
            "## Límites del protocolo", "",
            "Las etiquetas son energías totales DFT-GGA estáticas por átomo, no energías de formación. Los metadatos locales no documentan el funcional exacto, pseudopotenciales ni correcciones. Ningún peso del Transformer se actualizó.", ""]
        if args.final_test is not None:
            test = require(args.final_test / "test_metrics.csv")
            comparative += ["## Evaluación final de test (descriptiva; separada de validation)", "", md_table(test), "",
                            "Los resultados de test se calcularon después del registro de selección y no deben usarse para ajustar o volver a seleccionar configuraciones.", ""]
        else:
            comparative += ["## Evaluación test", "", "Pendiente: no se proporcionó el directorio de la evaluación final bloqueada.", ""]
        (output / "E1_comparative_report.md").write_text("\n".join(comparative), encoding="utf-8")
        log(LOGGER, logging.INFO, "reports_written", paths=[str(e11_path), str(e12_path), str(output / "E1_comparative_report.md")])
    except (DiagnosticError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        log(LOGGER, logging.ERROR, "report_generation_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
