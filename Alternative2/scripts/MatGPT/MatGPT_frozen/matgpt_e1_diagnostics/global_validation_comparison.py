#!/usr/bin/env python3
"""Compara baseline E1 y candidatos bloqueados E1.1/E1.2 en validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, DiagnosticError, energy_metrics, log, logger_for
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.plot_results import note, save_figure, style

LOGGER = logger_for("matgpt_e1_diagnostics")


def select_candidate(path: Path, name: str, configuration: str | None, pooling: str | None) -> pd.DataFrame:
    """Carga únicamente filas validation de un baseline/configuración.

    Args:
        path: CSV histórico o de predicciones train/validation.
        name: Nombre legible para errores de auditoría.
        configuration: Identificador exacto del candidato; None para E1 histórico.
        pooling: Filtro last_token/masked_mean; None para el baseline E1.

    Returns:
        Predicciones de validation del candidato, sin filas de test.
    """
    if not path.is_file():
        raise DiagnosticError(f"No existe archivo de predicciones: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "split" not in frame or "energy_dft_ev_atom" not in frame or "energy_pred_ev_atom" not in frame:
        raise DiagnosticError(f"Formato de predicciones no reconocido: {path}")
    validation = frame[frame.split.astype(str).eq("val")].copy()
    if configuration is not None:
        if "configuration" not in validation:
            raise DiagnosticError(f"No hay columna configuration para seleccionar {name}.")
        validation = validation[validation.configuration.astype(str).eq(configuration)]
    if pooling is not None:
        if "pooling" not in validation:
            raise DiagnosticError(f"No hay columna pooling para seleccionar {name}.")
        validation = validation[validation.pooling.astype(str).eq(pooling)]
    if validation.empty:
        raise DiagnosticError(f"No hay filas de validation para candidato {name}.")
    return validation


def summarize(frame: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    """Calcula métricas validation por semilla y media por estructura.

    Args:
        frame: Predicciones validation de una o varias semillas.
        model_name: Etiqueta para identificar el experimento comparado.

    Returns:
        Métricas por semilla y, si hay réplicas, métrica sobre predicción media.
    """
    if "seed" not in frame:
        frame = frame.assign(seed="historical_single_run")
    results: list[dict[str, Any]] = []
    for seed, group in frame.groupby("seed", dropna=False, sort=True):
        local = group.drop_duplicates("structure_id", keep="first")
        metrics, _, _ = energy_metrics(local)
        results.append({"model": model_name, "seed": str(seed), "split": "val", **metrics})
    if len(results) > 1:
        averaged = frame.groupby(["structure_id", "composition", "polymorph_family"], as_index=False).agg(
            energy_dft_ev_atom=("energy_dft_ev_atom", "first"), energy_pred_ev_atom=("energy_pred_ev_atom", "mean"))
        metrics, _, _ = energy_metrics(averaged)
        results.append({"model": model_name, "seed": "mean_across_seeds", "split": "val_seed_mean_predictions", **metrics})
    return results


def markdown_table(frame: pd.DataFrame) -> str:
    """Convierte resultados resumidos a Markdown sin dependencias auxiliares.

    Args:
        frame: Tabla de métricas por semilla o agregado.

    Returns:
        Tabla Markdown escapada.
    """
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("—" if pd.isna(value) else str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    """Exporta tabla y figuras PNG de la comparación global por validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-predictions", type=Path, default=ALT / "reports" / "matgpt_e1_v2_eval_v2" / "predictions.csv")
    parser.add_argument("--e1-1-predictions", type=Path, required=True)
    parser.add_argument("--e1-1-configuration", required=True)
    parser.add_argument("--e1-2-last-predictions", type=Path, required=True)
    parser.add_argument("--e1-2-last-configuration", required=True)
    parser.add_argument("--e1-2-mean-predictions", type=Path, required=True)
    parser.add_argument("--e1-2-mean-configuration", required=True)
    parser.add_argument("--out-dir", type=Path, default=ALT / "reports" / "matgpt_e1_global_comparison_v1")
    args = parser.parse_args()
    try:
        output = args.out_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise DiagnosticError(f"El directorio ya contiene resultados: {output}; use una versión nueva.")
        output.mkdir(parents=True, exist_ok=True)
        candidates = [
            ("E1 MLP histórica", args.e1_predictions, None, None),
            ("E1.1 seleccionado", args.e1_1_predictions, args.e1_1_configuration, None),
            ("E1.2 token final", args.e1_2_last_predictions, args.e1_2_last_configuration, "last_token"),
            ("E1.2 masked mean", args.e1_2_mean_predictions, args.e1_2_mean_configuration, "masked_mean"),
        ]
        frames: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, Any]] = []
        for name, path, config, pooling in candidates:
            frame = select_candidate(path, name, config, pooling)
            frames[name] = frame
            rows.extend(summarize(frame, name))
        reference_ids = sorted(frames["E1 MLP histórica"].structure_id.astype(str).tolist())
        reference_labels = frames["E1 MLP histórica"].set_index("structure_id").energy_dft_ev_atom
        for name, frame in frames.items():
            if sorted(frame.structure_id.astype(str).tolist()) != reference_ids:
                raise DiagnosticError(f"Los structure_id de validation no coinciden entre E1 y {name}.")
            labels = frame.set_index("structure_id").energy_dft_ev_atom.reindex(reference_labels.index)
            if not np.allclose(labels.to_numpy(float), reference_labels.to_numpy(float), rtol=0, atol=1e-10):
                raise DiagnosticError(f"Las etiquetas DFT difieren respecto a E1 para {name}.")
        table = pd.DataFrame(rows)
        table.to_csv(output / "global_validation_metrics.csv", index=False)
        metrics = ("energy_mae_ev_atom", "relative_energy_mae_mev_atom")
        colors = {"E1 MLP histórica":"#697386", "E1.1 seleccionado":"#D05A47",
                  "E1.2 token final":"#3569A8", "E1.2 masked mean":"#C14E80"}
        style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        for ax, metric, ylabel, title in zip(axes, metrics, ("MAE (eV/átomo)", "MAE de ΔE (meV/átomo)"),
                ("Energía absoluta", "Energía relativa intracomposición")):
            grouped = table[(table.split == "val") & (table.seed != "mean_across_seeds")].groupby("model")[metric].agg(["mean", "std"]).reindex(colors.keys())
            ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), capsize=4,
                   color=[colors[name] for name in grouped.index])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
            ax.legend(handles=[Patch(facecolor=colors[name], label=name) for name in grouped.index], title="Configuración comparada", fontsize=7)
        fig.suptitle("Comparación global de E1, E1.1 y E1.2 — validation")
        fig.tight_layout(rect=(0.02, 0.17, 0.98, 0.90))
        n = frames["E1 MLP histórica"].structure_id.nunique()
        comps = frames["E1 MLP histórica"].composition.nunique()
        note(fig, f"Validation: N={n} estructuras estáticas y {comps} composiciones, IDs comunes requeridos. Barras = media entre semillas; bigotes = ±1 DE. E1 es una corrida histórica; para E1.2 se comparan candidatos elegidos solo con train/validation. Test excluido.")
        save_figure(fig, output / "global_validation_comparison.png")
        plt.close(fig)
        note_rows = ["# Comparación global en validation", "", "Las métricas de esta tabla usan exclusivamente validation; ninguna etiqueta test se usa para selección.", "",
                     "## Métricas por semilla", "", markdown_table(table), "",
                     "## Nota de comparabilidad", "",
                     "Las filas E1.1/E1.2 deben corresponder a las configuraciones documentadas en sus manifiestos. La comparación mantiene los IDs históricos; la media de predicciones entre semillas se reporta aparte en `global_validation_metrics.csv`.",
                     "El conjunto test fue previamente inspeccionado durante E1; esta comparación no es evidencia de generalización ciega.", ""]
        (output / "global_validation_report.md").write_text("\n".join(note_rows), encoding="utf-8")
        log(LOGGER, logging.INFO, "global_comparison_complete", configurations=len(candidates), validation_rows=len(table), report=str(output))
    except (DiagnosticError, OSError, ValueError, KeyError) as exc:
        log(LOGGER, logging.ERROR, "global_comparison_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
