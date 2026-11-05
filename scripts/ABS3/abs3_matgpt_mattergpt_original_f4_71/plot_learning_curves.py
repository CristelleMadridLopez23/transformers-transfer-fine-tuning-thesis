"""Grafica curvas OOF de entrenamiento y validación para F0 y F4 en ABS3.

Las curvas resumen las corridas por pliegue y semilla; las pérdidas de
entrenamiento son promedios de minilotes y validación evalúa el conjunto
inner-validation completo en cada época.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

LOGGER = logging.getLogger("abs3.learning_curves")
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
    raise FileNotFoundError("No se encontró AGENTS.md para determinar la raíz del proyecto")


def parse_args() -> argparse.Namespace:
    """Valida argumentos de entrada y salida.

    Returns:
        Argumentos con las rutas del historial y la figura PNG.
    """
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path,
                        default=root / "experiments/abs3_matgpt_mattergpt_original_f4_71/train_history.csv")
    parser.add_argument("--runs", type=Path,
                        default=root / "experiments/abs3_matgpt_mattergpt_original_f4_71/runs")
    parser.add_argument("--output", type=Path,
                        default=root / "images/abs3_matgpt_mattergpt_original_f4_71/curvas_entrenamiento_f0_f4.png")
    return parser.parse_args()


def load_history(path: Path) -> pd.DataFrame:
    """Carga y valida el historial de entrenamiento final.

    Args:
        path: CSV con pérdidas por época y corrida.

    Returns:
        Historial F0/F4 con claves y métricas verificadas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe el historial: {path}")
    frame = pd.read_csv(path)
    required = {"model", "condition", "fold", "seed", "epoch", "train_minibatch_smoothl1",
                "inner_validation_full_smoothl1", "warmup_inner_validation_smoothl1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en historial: {sorted(missing)}")
    frame = frame[frame.condition.isin(CONDITIONS)].copy()
    if frame.empty or not np.isfinite(frame[["train_minibatch_smoothl1",
                                             "inner_validation_full_smoothl1"]].to_numpy()).all():
        raise ValueError("El historial F0/F4 está vacío o contiene pérdidas no finitas")
    for model in MODELS:
        for condition in CONDITIONS:
            n_runs = frame[(frame.model == model) & (frame.condition == condition)][
                ["fold", "seed"]].drop_duplicates().shape[0]
            if n_runs != 15:
                raise ValueError(f"{model} {condition}: se esperaban 15 corridas, se hallaron {n_runs}")
    return frame


def load_selections(run_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Lee la época/fase elegida por validación para las 60 corridas.

    Args:
        run_dir: Directorio de resúmenes JSON por corrida.

    Returns:
        Resúmenes agrupados por modelo y condición.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de corridas: {run_dir}")
    import json

    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        for condition in CONDITIONS:
            rows = []
            for path in sorted(run_dir.glob(f"{model}_{condition}_fold*_seed*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"best_epoch": int(payload["best_epoch"]),
                             "best_checkpoint_phase": str(payload["best_checkpoint_phase"])})
            if len(rows) != 15:
                raise ValueError(f"{model} {condition}: se esperaban 15 resúmenes, se hallaron {len(rows)}")
            selected[(model, condition)] = rows
    return selected


def plot_curves(history: pd.DataFrame, selections: dict[tuple[str, str], list[dict[str, Any]]],
                output: Path) -> None:
    """Dibuja curvas medias, dispersión entre corridas y número activo.

    Args:
        history: Pérdidas por época para F0 y F4.
        selections: Checkpoints elegidos por inner-validation.
        output: Destino de la figura PNG.
    """
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                         "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.3), sharex=True, sharey=True)
    all_values: list[float] = []
    for model in MODELS:
        model_rows = history[history.model == model]
        for condition in CONDITIONS:
            rows = model_rows[model_rows.condition == condition]
            for column in ("train_minibatch_smoothl1", "inner_validation_full_smoothl1"):
                grouped = rows.groupby("epoch")[column].agg(["mean", "std", "count"])
                all_values.extend((grouped["mean"] + grouped["std"].fillna(0)).tolist())
    y_top = max(all_values) * 1.12
    for ax, model in zip(axes, MODELS):
        model_rows = history[history.model == model]
        # Shared head warmup is shown as the validation reference at branch start.
        warmup = (model_rows.groupby(["fold", "seed"])["warmup_inner_validation_smoothl1"]
                  .first().dropna())
        warmup_mean = float(warmup.mean())
        ax.axhline(warmup_mean, color="#555555", linestyle=(0, (1, 2)), linewidth=1.3,
                   label=f"Validación de warmup compartido: {warmup_mean:.3f}")
        count_ax = ax.twinx()
        for condition in CONDITIONS:
            rows = model_rows[model_rows.condition == condition]
            color = COLORS[condition]
            for column, style, label in (
                    ("train_minibatch_smoothl1", "-", f"{condition}: train, promedio de minilotes"),
                    ("inner_validation_full_smoothl1", "--", f"{condition}: validación completa")):
                stats = rows.groupby("epoch")[column].agg(["mean", "std", "count"])
                x = stats.index.to_numpy(dtype=int)
                mean = stats["mean"].to_numpy(dtype=float)
                sd = stats["std"].fillna(0).to_numpy(dtype=float)
                ax.plot(x, mean, color=color, linestyle=style, linewidth=1.9, label=label)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
            counts = rows.groupby("epoch")[["fold", "seed"]].apply(lambda values: values.drop_duplicates().shape[0])
            count_ax.step(counts.index.to_numpy(dtype=int), counts.to_numpy(dtype=int), where="mid",
                          color=color, linestyle=":", linewidth=1.15, alpha=0.75)
        count_ax.set_ylim(0, 16)
        count_ax.set_yticks([0, 5, 10, 15])
        count_ax.tick_params(axis="y", labelcolor="#555555")
        count_ax.set_ylabel("Corridas fold-semilla activas (n)", color="#555555")
        ax.set_title(f"{model}: SLICES + geometría")
        ax.set_xlim(0, 72)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Época local después del warmup")
        ax.grid(True, alpha=0.23)
        ax.text(0.02, 0.97,
                "Checkpoint elegido por validación — " +
                " · ".join(f"{cond}: warmup {sum(r['best_checkpoint_phase']=='warmup' for r in selections[(model, cond)])}/15, "
                           f"mediana época {np.median([r['best_epoch'] for r in selections[(model, cond)]]):.0f}"
                           for cond in CONDITIONS),
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88,
                      "edgecolor": "#bbbbbb"})
    axes[0].set_ylabel("Pérdida SmoothL1 del residuo normalizado")
    fig.suptitle("ABS₃: curvas de entrenamiento y validación — MLP8 con F0 y F4",
                 fontsize=16, y=0.97)
    fig.legend(handles=[
        Line2D([0], [0], color=COLORS["F0"], linestyle="-", label="F0 train"),
        Line2D([0], [0], color=COLORS["F0"], linestyle="--", label="F0 validación"),
        Line2D([0], [0], color=COLORS["F4"], linestyle="-", label="F4 train"),
        Line2D([0], [0], color=COLORS["F4"], linestyle="--", label="F4 validación"),
        Line2D([0], [0], color="#555555", linestyle=(0, (1, 2)), label="Validación del warmup compartido"),
        Line2D([0], [0], color="#555555", linestyle=":", label="Corridas activas (eje derecho)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=True)
    note = ("210 estructuras · 70 composiciones · 3 polimorfos/composición · 5 pliegues × 3 semillas. "
            "Cada pliegue: entrenamiento interno 45 composiciones y validación interna 11; se muestran 15 corridas fold-semilla por condición. "
            "Train = media de pérdidas por minilote en entrenamiento; validación = pérdida del conjunto completo. "
            "La línea del warmup es la validación inicial compartida; época 0 en la selección significa que ganó el warmup. "
            "Las bandas muestran ±1 desviación estándar entre corridas activas (no IC); n disminuye por parada temprana. "
            "No se muestra ni se usa la evaluación outer para seleccionar checkpoints.")
    fig.text(0.06, 0.025, note, ha="left", va="bottom", fontsize=8.4, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.79, bottom=0.17, wspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Figura guardada en %s", output)


def main() -> None:
    """Genera las curvas de aprendizaje en PNG."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    history = load_history(args.history)
    selections = load_selections(args.runs)
    plot_curves(history, selections, args.output)


if __name__ == "__main__":
    main()
