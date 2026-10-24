#!/usr/bin/env python3
"""Grafica las curvas MLP8 de MatterGPT asociadas al híbrido M3-A.

M3-A combina la media de Ridge M4 con los residuos centrados aprendidos por
las cabezas MLP8 de M2. La historia disponible guarda la validación de M2,
que selecciona los checkpoints; M3-A no tuvo un entrenamiento ni validación
por época independiente.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "Alternative2/reports/MatterGPT/MatterGPT_frozen_1"
DEFAULT_HISTORY = REPORT / "cv/training_history.csv"
DEFAULT_RUNS = REPORT / "cv/runs/M2"
DEFAULT_OUTPUT = REPORT / "figures_v2"
LOGGER = logging.getLogger("mattergpt_m3a_curves")
REQUIRED_COLUMNS = {"model", "fold", "seed", "epoch", "train_total", "inner_val_absolute_mae_ev_atom"}


def configure_logging() -> None:
    """Activa logging estructurado con registros JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


def parse_args() -> argparse.Namespace:
    """Lee y valida rutas configurables para historial, corridas y salida.

    Returns:
        Argumentos de línea de comandos validados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="CSV histórico de curvas por época.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="Manifiestos de M2 por fold y semilla.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directorio nuevo de salida PNG.")
    args = parser.parse_args()
    if not args.history.is_file():
        parser.error(f"No existe el historial: {args.history}")
    if not args.runs_dir.is_dir():
        parser.error(f"No existe el directorio de corridas M2: {args.runs_dir}")
    return args


def load_history(path: Path) -> pd.DataFrame:
    """Carga solo las curvas M2 relevantes y valida sus observaciones.

    Args:
        path: CSV combinado de historiales MatterGPT.
    Returns:
        Curvas M2 ordenadas por fold, semilla y época.
    """
    full = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(full.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el historial: {sorted(missing)}")
    frame = full.loc[full.model.eq("M2")].copy()
    if frame.empty:
        raise ValueError("El CSV no contiene filas del modelo M2.")
    frame["inner_val_absolute_mae_ev_atom"] = pd.to_numeric(frame.inner_val_absolute_mae_ev_atom, errors="coerce")
    frame["train_total"] = pd.to_numeric(frame.train_total, errors="coerce")
    if not np.isfinite(frame[["train_total", "inner_val_absolute_mae_ev_atom"]].to_numpy(dtype=float)).all():
        raise ValueError("Hay pérdidas o MAE no finitas en las filas M2.")
    if frame.duplicated(["fold", "seed", "epoch"]).any():
        raise ValueError("Hay duplicados fold/semilla/época en M2.")
    frame["fold"] = pd.to_numeric(frame.fold, errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame.seed, errors="raise").astype(int)
    frame["epoch"] = pd.to_numeric(frame.epoch, errors="raise").astype(int)
    if frame.epoch.lt(1).any() or (frame.train_total < 0).any() or (frame.inner_val_absolute_mae_ev_atom < 0).any():
        raise ValueError("Épocas o métricas M2 fuera del dominio permitido.")
    for (fold, seed), group in frame.groupby(["fold", "seed"]):
        epochs = group.epoch.to_numpy()
        if not np.array_equal(epochs, np.arange(1, len(epochs) + 1)):
            raise ValueError(f"Épocas faltantes o desordenadas en fold={fold}, semilla={seed}.")
    return frame.sort_values(["fold", "seed", "epoch"], kind="stable").reset_index(drop=True)


def load_best_epochs(history: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """Valida checkpoint y criterio de selección para cada corrida M2.

    Args:
        history: Historial M2 ya validado.
        runs_dir: Directorio con `training.json` de cada M2 fold/semilla.
    Returns:
        Tabla de épocas guardadas por early stopping en inner-validation.
    """
    selected: list[dict[str, Any]] = []
    for (fold, seed), group in history.groupby(["fold", "seed"], sort=True):
        path = runs_dir / f"fold_{fold}_seed_{seed}" / "training.json"
        if not path.is_file():
            raise ValueError(f"Falta metadata de checkpoint: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        best_epoch = int(metadata["best_epoch"])
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError(f"Semilla no coincide en {path}")
        if metadata.get("inner_val_absolute_mae_ev_atom") is None:
            raise ValueError(f"Falta criterio de selección de validación en {path}")
        logged_best = float(group.loc[group.epoch.eq(best_epoch), "inner_val_absolute_mae_ev_atom"].iloc[0])
        if not np.isclose(logged_best, float(metadata["inner_val_absolute_mae_ev_atom"]), rtol=1e-6, atol=1e-9):
            raise ValueError(f"MAE guardado y CSV no concuerdan para fold={fold}, semilla={seed}")
        if not np.isclose(logged_best, group.inner_val_absolute_mae_ev_atom.min(), rtol=1e-6, atol=1e-9):
            raise ValueError(f"La época del manifiesto no minimiza inner-validation en fold={fold}, semilla={seed}")
        selected.append({"fold": fold, "seed": seed, "best_epoch": best_epoch})
    return pd.DataFrame(selected)


def summarize(history: pd.DataFrame, column: str, scale: float = 1.0) -> pd.DataFrame:
    """Calcula media, DE muestral y cantidad de corridas activas por época.

    Args:
        history: Observaciones por corrida y época.
        column: Métrica que se resume.
        scale: Factor de conversión aplicado antes de resumir.
    Returns:
        Resumen por época, sin imputar corridas que ya pararon.
    """
    frame = history.assign(_metric=history[column].astype(float) * scale)
    return frame.groupby("epoch")._metric.agg(mean="mean", std="std", n="count").reset_index()


def draw(history: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> Path:
    """Dibuja objetivos de entrenamiento y MAE de validación en paneles separados.

    Args:
        history: Curvas M2 por fold/semilla/época.
        selected: Épocas elegidas mediante MAE de inner-validation.
        output_dir: Directorio de revisión; la figura histórica no se toca.
    Returns:
        Ruta del PNG exportado.
    """
    train = summarize(history, "train_total")
    valid = summarize(history, "inner_val_absolute_mae_ev_atom", 1000.0)
    if not np.array_equal(train.epoch.to_numpy(), valid.epoch.to_numpy()):
        raise ValueError("Las series M2 no comparten las mismas épocas registradas.")
    median_best = float(selected.best_epoch.median())
    q1, q3 = (float(value) for value in selected.best_epoch.quantile([.25, .75]))
    colors = {"train": "#2878b5", "valid": "#c83c61", "count": "#5b6573"}
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 3.2, .9]})
    for ax, stats, color, label, ylabel in (
        (axes[0], train, colors["train"], "Objetivo total M2 (media + residuo)", "SmoothL1 normalizada (adimensional)"),
        (axes[1], valid, colors["valid"], "MAE absoluta de M2 en inner-validation", "MAE (meV/átomo)"),
    ):
        x = stats.epoch.to_numpy(dtype=int)
        mean = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, mean, color=color, marker="o", markersize=3.2, linewidth=2.3, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.17, linewidth=0,
                        label="±1 desviación estándar entre corridas")
        ax.axvspan(q1, q3, color="#6b7280", alpha=.08, label="Rango intercuartílico de épocas elegidas")
        ax.axvline(median_best, color="#454545", linestyle="--", linewidth=1.25,
                   label=f"Mediana de mejor época: {median_best:.0f}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.22)
        ax.legend(loc="best", fontsize=8.5, frameon=True)
    count = train[["epoch", "n"]]
    axes[2].step(count.epoch, count.n, where="mid", color=colors["count"], linewidth=1.5)
    axes[2].scatter(count.epoch, count.n, color=colors["count"], s=10, zorder=3)
    axes[2].set_ylabel("Corridas\nactivas")
    axes[2].set_xlabel("Época local de MLP8")
    axes[2].set_yticks([1, int(count.n.max())])
    axes[2].set_ylim(0, int(count.n.max()) + 1)
    axes[2].grid(axis="y", alpha=.22)
    fig.suptitle("MatterGPT M3-A (Ridge + MLP8) · entrenamiento M2 y validación usada para el checkpoint", fontsize=14, y=.99)
    fig.text(
        .07, .01,
        "15 corridas (5 folds por composición × 3 semillas). Por fold: inner-train 45 composiciones/135 estructuras; "
        "inner-validation 11 composiciones/33 estructuras. Panel superior: objetivo de minibatches M2, suma de las "
        "pérdidas SmoothL1 normalizadas de las cabezas de media y residuo; cada minibatch pesa igual. Panel central: "
        "MAE absoluta de la salida completa M2, evaluada sobre inner-validation en eV/átomo y expresada aquí en "
        "meV/átomo. Las métricas tienen escalas/unidades distintas y se muestran en paneles separados. Bandas: "
        "±1 desviación estándar entre corridas; "
        "n activa cambia con early stopping. La época se elige por mínimo de MAE M2 en inner-validation. M3-A sustituye "
        "después la cabeza de media M2 por Ridge M4 y conserva los residuos M2; no existe una curva M3-A independiente. "
        "No se usó outer-fold ni test para early stopping.",
        ha="left", va="bottom", fontsize=8.2, color="#333333", wrap=True,
    )
    fig.tight_layout(rect=(0, .105, 1, .965))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "01_mattergpt_m3a_training_validation_curves.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> int:
    """Valida procedencia y escribe la figura nueva en PNG.

    Returns:
        Código de salida cero al completar la figura.
    """
    configure_logging()
    args = parse_args()
    try:
        manifest = json.loads((REPORT / "cv/cv_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or "M2" not in manifest.get("models", []):
            raise ValueError("El manifiesto CV no certifica PASS con M2.")
        history = load_history(args.history)
        runs = set(zip(history.fold, history.seed))
        if len(runs) != 15 or history.fold.nunique() != 5 or history.seed.nunique() != 3:
            raise ValueError(f"Se esperaban 15 corridas M2; encontradas {len(runs)}.")
        selected = load_best_epochs(history, args.runs_dir)
        output = draw(history, selected, args.output_dir)
        LOGGER.info(json.dumps({"event": "figure_written", "path": str(output), "rows": len(history),
                                "runs": len(runs), "epochs": int(history.epoch.max())}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        LOGGER.exception("figure_generation_failed error=%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
