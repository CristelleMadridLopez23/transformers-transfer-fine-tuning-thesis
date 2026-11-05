"""Grafica el MAE de energía DFT por átomo según la época para ABS3.

Lee el replay diagnóstico que registra el MAE energético de train y validación
en cada época; no reemplaza los resultados OOF canónicos.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

LOGGER = logging.getLogger("abs3.energy_mae_curves")
MODELS = ("MatGPT", "MatterGPT")
CONDITIONS = ("F0", "F4")
COLORS = {"F0": "#2878B5", "F4": "#E76F51"}


def project_root() -> Path:
    """Encuentra la raíz del proyecto a partir de AGENTS.md.

    Returns:
        Ruta absoluta a la raíz del repositorio.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file():
            return parent
    raise FileNotFoundError("No se encontró AGENTS.md")


def parse_args() -> argparse.Namespace:
    """Valida las rutas configurables del replay y de salida.

    Returns:
        Argumentos parseados.
    """
    root = project_root()
    base = root / "experiments/abs3_matgpt_mattergpt_original_f4_71/diagnostics/mae_curve_replay"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=base / "training_mae_per_epoch.csv")
    parser.add_argument("--runs", type=Path, default=base / "runs")
    parser.add_argument("--output", type=Path,
                        default=root / "images/abs3_matgpt_mattergpt_original_f4_71/curvas_mae_energetico_epoca.png")
    return parser.parse_args()


def load_data(history_path: Path, runs_path: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], list[dict[str, Any]]]]:
    """Carga MAE por época y checkpoints seleccionados; valida 15 corridas por grupo.

    Args:
        history_path: CSV de métricas energéticas por época.
        runs_path: JSON de resumen de cada corrida diagnóstica.

    Returns:
        Historial validado y resúmenes por extractor/condición.
    """
    if not history_path.is_file() or not runs_path.is_dir():
        raise FileNotFoundError("El replay de MAE energético aún no terminó o faltan sus archivos")
    history = pd.read_csv(history_path)
    required = {"model", "condition", "fold", "seed", "epoch", "train_minibatch_energy_mae_ev_atom",
                "inner_validation_full_energy_mae_ev_atom", "warmup_inner_validation_energy_mae_ev_atom"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Faltan campos del MAE energético: {sorted(missing)}")
    selections: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        for condition in CONDITIONS:
            sub = history[(history.model == model) & (history.condition == condition)]
            n_runs = sub[["fold", "seed"]].drop_duplicates().shape[0]
            if n_runs != 15:
                raise ValueError(f"{model} {condition}: historial parcial ({n_runs}/15 corridas)")
            records = []
            for path in sorted(runs_path.glob(f"{model}_{condition}_fold*_seed*.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                records.append({"best_epoch": int(item["best_epoch"]),
                                "best_checkpoint_phase": str(item["best_checkpoint_phase"])})
            if len(records) != 15:
                raise ValueError(f"{model} {condition}: resúmenes incompletos ({len(records)}/15)")
            selections[(model, condition)] = records
    values = history[["train_minibatch_energy_mae_ev_atom", "inner_validation_full_energy_mae_ev_atom"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Hay MAE energético no finito")
    return history, selections


def plot(history: pd.DataFrame, selections: dict[tuple[str, str], list[dict[str, Any]]], output: Path) -> None:
    """Crea paneles de MAE de energía por época con conteo de corridas activas.

    Args:
        history: MAE medido en entrenamiento y validación por época.
        selections: Época/fase elegida por inner-validation.
        output: Ruta del PNG a guardar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.3), sharex=True, sharey=True)
    all_values: list[float] = []
    for model in MODELS:
        part = history[history.model == model]
        for condition in CONDITIONS:
            rows = part[part.condition == condition]
            for column in ("train_minibatch_energy_mae_ev_atom", "inner_validation_full_energy_mae_ev_atom"):
                stat = rows.groupby("epoch")[column].agg(["mean", "std"])
                all_values.extend((stat["mean"] + stat["std"].fillna(0)).to_numpy(dtype=float) * 1000)
    y_max = max(all_values) * 1.12
    for ax, model in zip(axes, MODELS):
        part = history[history.model == model]
        warmup = part.groupby(["fold", "seed"])["warmup_inner_validation_energy_mae_ev_atom"].first().dropna()
        warmup_mev = float(warmup.mean() * 1000)
        ax.axhline(warmup_mev, color="#555555", linestyle=(0, (1, 2)), linewidth=1.3)
        count_ax = ax.twinx()
        for condition in CONDITIONS:
            rows = part[part.condition == condition]
            color = COLORS[condition]
            for column, style in (("train_minibatch_energy_mae_ev_atom", "-"),
                                  ("inner_validation_full_energy_mae_ev_atom", "--")):
                stat = rows.groupby("epoch")[column].agg(["mean", "std", "count"])
                x = stat.index.to_numpy(dtype=int)
                mean = stat["mean"].to_numpy(dtype=float) * 1000
                sd = stat["std"].fillna(0).to_numpy(dtype=float) * 1000
                ax.plot(x, mean, color=color, linestyle=style, linewidth=1.9)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
            n_active = rows.groupby("epoch").apply(lambda g: g[["fold", "seed"]].drop_duplicates().shape[0])
            count_ax.step(n_active.index.to_numpy(dtype=int), n_active.to_numpy(dtype=int), where="mid",
                          color=color, linestyle=":", linewidth=1.15, alpha=0.75)
        count_ax.set_ylim(0, 16)
        count_ax.set_yticks([0, 5, 10, 15])
        count_ax.set_ylabel("Corridas fold-semilla activas (n)", color="#555555")
        count_ax.tick_params(axis="y", labelcolor="#555555")
        ax.set_title(f"{model}: SLICES + geometría")
        ax.set_xlim(0, 72)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Época local después del warmup")
        ax.grid(True, alpha=0.23)
        selection_text = " · ".join(
            f"{cond}: warmup {sum(r['best_checkpoint_phase']=='warmup' for r in selections[(model, cond)])}/15, "
            f"mediana época {np.median([r['best_epoch'] for r in selections[(model, cond)]]):.0f}"
            for cond in CONDITIONS)
        ax.text(0.02, 0.97, "Checkpoint elegido por validación — " + selection_text,
                transform=ax.transAxes, va="top", ha="left", fontsize=8.4,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90,
                      "edgecolor": "#bbbbbb"})
    axes[0].set_ylabel("MAE de energía absoluta por átomo (meV/átomo)")
    fig.suptitle("ABS₃: MAE energético por época — MLP8, SLICES + geometría",
                 fontsize=16, y=0.97)
    fig.legend(handles=[
        Line2D([0], [0], color=COLORS["F0"], linestyle="-", label="F0 train"),
        Line2D([0], [0], color=COLORS["F0"], linestyle="--", label="F0 validación"),
        Line2D([0], [0], color=COLORS["F4"], linestyle="-", label="F4 train"),
        Line2D([0], [0], color=COLORS["F4"], linestyle="--", label="F4 validación"),
        Line2D([0], [0], color="#555555", linestyle=(0, (1, 2)), label="MAE validación del warmup"),
        Line2D([0], [0], color="#555555", linestyle=":", label="Corridas activas (eje derecho)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=True)
    note = ("E_pred = μ_Ridge-M2 + δ_pred; se compara con energía DFT por átomo. "
            "210 estructuras · 70 composiciones · 3 polimorfos/composición · 5 pliegues × 3 semillas; "
            "por pliegue: 45 fórmulas de entrenamiento y 11 de validación. Se promedian 15 corridas fold-semilla; "
            "train es promedio ponderado de predicciones de minilotes en modo entrenamiento y validación es el conjunto completo. "
            "La referencia horizontal es el MAE de validación del warmup compartido. Bandas = ±1 DE entre corridas activas (no IC); "
            "n baja por parada temprana. Checkpoints elegidos solo con inner-validation; outer/test no se muestra ni selecciona.")
    fig.text(0.06, 0.025, note, ha="left", va="bottom", fontsize=8.3, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.79, bottom=0.17, wspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Figura MAE energético guardada en %s", output)


def main() -> None:
    """Valida replay y genera figura PNG."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    history, selections = load_data(args.history, args.runs)
    plot(history, selections, args.output)


if __name__ == "__main__":
    main()
