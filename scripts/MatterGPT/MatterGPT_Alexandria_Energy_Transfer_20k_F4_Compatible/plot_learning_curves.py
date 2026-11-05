#!/usr/bin/env python3
"""Grafica las curvas históricas MatterGPT Alexandria F4 SLICES-only.

Train muestra la pérdida media de minibatch registrada por época y validation
la SmoothL1 calculada sobre el conjunto completo de 1,718 estructuras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXPERIMENT = "mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset"
DATA = ROOT / "experiments" / EXPERIMENT
IMAGE_DIR = ROOT / "images" / EXPERIMENT


def load_history(path: Path) -> pd.DataFrame:
    """Carga y valida las curvas de warmup/F4 del entrenamiento SLICES-only.

    Args:
        path: CSV de historial original.
    Returns:
        Tabla ordenada con contador global de época.
    Raises:
        ValueError: Si faltan columnas, etapas o valores finitos.
    """
    frame = pd.read_csv(path)
    required = {"stage", "epoch", "train_smoothl1", "validation_smoothl1"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas del historial: {sorted(required-set(frame.columns))}")
    if set(frame.stage) != {"warmup_frozen", "finetune_f4"}:
        raise ValueError(f"Etapas inesperadas: {sorted(frame.stage.unique())}")
    warm = frame.loc[frame.stage.eq("warmup_frozen")].copy()
    f4 = frame.loc[frame.stage.eq("finetune_f4")].copy()
    if len(warm) != 20 or warm.epoch.tolist() != list(range(1, 21)):
        raise ValueError("Warmup histórico no contiene 20 épocas ordenadas")
    if f4.epoch.tolist() != list(range(1, len(f4) + 1)) or f4.empty:
        raise ValueError("Épocas F4 ausentes o desordenadas")
    for column in ("train_smoothl1", "validation_smoothl1"):
        if not pd.notna(frame[column]).all():
            raise ValueError(f"Valores faltantes en {column}")
    frame = frame.sort_values(["stage", "epoch"]).copy()
    frame["global_epoch"] = frame.apply(
        lambda row: int(row.epoch) if row.stage == "warmup_frozen" else 20 + int(row.epoch), axis=1)
    return frame.sort_values("global_epoch").reset_index(drop=True)


def draw_curves(history: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    """Genera una figura PNG autocontenida en español.

    Args:
        history: Curvas históricas validadas.
        summary: Resumen con época de checkpoint seleccionado y tamaños de split.
        output: Ruta final de la figura PNG.
    Returns:
        None.
    """
    warm = history.loc[history.stage.eq("warmup_frozen")]
    f4 = history.loc[history.stage.eq("finetune_f4")]
    best = int(summary["metrics"]["selected_epoch"])
    if best < 1 or best > len(f4):
        raise ValueError(f"Época seleccionada inválida para F4: {best}")
    best_row = f4.loc[f4.epoch.eq(best)].iloc[0]
    global_best = 20 + best
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.axvspan(0.5, 20.5, color="#e8eef5", alpha=0.8, label="Warmup: MatterGPT congelado")
    ax.axvspan(20.5, 20.5 + len(f4), color="#f9eee4", alpha=0.55, label="F4: bloques 8–11 entrenables")
    ax.plot(warm.global_epoch, warm.train_smoothl1, color="#277da1", lw=1.8,
            marker="o", ms=3, label="Train: media de minibatches")
    ax.plot(warm.global_epoch, warm.validation_smoothl1, color="#f9844a", lw=1.8,
            marker="o", ms=3, label="Validación: conjunto completo")
    ax.plot(f4.global_epoch, f4.train_smoothl1, color="#277da1", lw=1.8,
            marker="o", ms=3, label="Train F4: media de minibatches")
    ax.plot(f4.global_epoch, f4.validation_smoothl1, color="#f9844a", lw=1.8,
            marker="o", ms=3, label="Validación F4: conjunto completo")
    ax.axvline(global_best, color="#6a4c93", lw=1.7, ls="--",
               label=f"Checkpoint seleccionado: F4 época {best}")
    ax.scatter([global_best], [float(best_row.validation_smoothl1)], s=70, color="#6a4c93", zorder=5)
    ax.annotate(f"Mejor validación = {float(best_row.validation_smoothl1):.5f}\nF4 época {best}",
                xy=(global_best, float(best_row.validation_smoothl1)), xytext=(-18, 28),
                textcoords="offset points", ha="right", fontsize=9, color="#513879",
                arrowprops={"arrowstyle": "->", "color": "#6a4c93"})
    ax.axvline(20.5, color="#555555", lw=1, ls=":")
    ax.text(10.5, 0.985, "Warmup", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.text(20.5 + len(f4)/2, 0.985, "F4", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.set_title("MatterGPT SLICES-only en Alexandria: curvas de aprendizaje F4", fontsize=15, pad=14)
    ax.set_xlabel("Época global (warmup 1–20; F4 empieza después del warmup)")
    ax.set_ylabel("SmoothL1 normalizada de energía DFT por átomo (adimensional)")
    ax.set_xlim(0.5, 20.5 + len(f4) + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True, fontsize=8.5, ncol=2)
    train_n = int(summary["train_rows"]); val_n = int(summary["validation_rows"])
    note = (f"Train: {train_n:,} estructuras; validación: {val_n:,}. Train = pérdida media de minibatches; "
            "validación = cálculo completo por época. La validación selecciona el checkpoint; no se usó test independiente. "
            "SLICES-only: sin descriptores geométricos; condición MatterGPT [1.0, 2.0].")
    fig.text(0.08, 0.015, note, ha="left", va="bottom", fontsize=8.3, wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Valida artefactos históricos y exporta la curva de aprendizaje PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA / "historial_entrenamiento.csv")
    parser.add_argument("--summary", type=Path, default=DATA / "summary.json")
    parser.add_argument("--output", type=Path, default=IMAGE_DIR / "curvas_aprendizaje_slices_only.png")
    args = parser.parse_args()
    for path in (args.history, args.summary):
        if not path.is_file():
            parser.error(f"No existe artefacto histórico requerido: {path}")
    history = load_history(args.history)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED" or summary.get("training_executed") is not True:
        parser.error("El resumen no acredita un entrenamiento SLICES-only completado")
    draw_curves(history, summary, args.output)
    print(f"Figura guardada: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
