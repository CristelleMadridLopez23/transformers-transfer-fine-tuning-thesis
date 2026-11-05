"""Genera figuras de la transferencia F4 de Alexandria y contexto ABS3 validado.

Lee historiales y métricas existentes, sin entrenar modelos ni modificar resultados.
Escribe únicamente PNG en images/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("matgpt_alexandria_figures")


def sha256_file(path: Path) -> str:
    """Devuelve SHA-256 de un archivo.

    Args:
        path: Archivo de entrada.

    Returns:
        Hash hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Carga y valida que el JSON de métricas sea un objeto.

    Args:
        path: Ruta al JSON.

    Returns:
        Contenido JSON como diccionario.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Falta el archivo de métricas: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}")
    return data


def plot_learning_curves(history_path: Path, out_path: Path) -> None:
    """Traza pérdidas de entrenamiento y validación para warmup y F4.

    Args:
        history_path: CSV original por época.
        out_path: PNG de salida.
    """
    frame = pd.read_csv(history_path)
    required = {"stage", "epoch", "train_smoothl1", "validation_smoothl1"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Columnas ausentes en {history_path}: {required - set(frame.columns)}")
    if frame[list(required - {"stage"})].isna().any().any():
        raise ValueError("Hay pérdidas o índices de época nulos en el historial")
    warm = frame[frame.stage == "warmup_frozen"].sort_values("epoch")
    f4 = frame[frame.stage == "finetune_f4"].sort_values("epoch")
    if warm.empty or f4.empty:
        raise ValueError("El historial debe incluir warmup_frozen y finetune_f4")

    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    warm_x = warm.epoch.to_numpy(dtype=float)
    f4_x = 20 + f4.epoch.to_numpy(dtype=float)
    ax.plot(warm_x, warm.train_smoothl1, color="#376996", lw=1.8,
            label="Entrenamiento · encoder congelado (20 épocas)")
    ax.plot(warm_x, warm.validation_smoothl1, color="#376996", lw=1.8, ls="--",
            label="Validación · encoder congelado")
    ax.plot(f4_x, f4.train_smoothl1, color="#c35a31", lw=1.8,
            label="Entrenamiento · F4 (bloques 8–11 + cabeza)")
    ax.plot(f4_x, f4.validation_smoothl1, color="#c35a31", lw=1.8, ls="--",
            label="Validación · F4")
    best_idx = int(f4.validation_smoothl1.to_numpy().argmin())
    best = f4.iloc[best_idx]
    best_x = 20 + float(best.epoch)
    ax.scatter([best_x], [best.validation_smoothl1], marker="*", s=190,
               color="#a21d25", zorder=5, label=f"Mejor validación: época F4 {int(best.epoch)}")
    ax.axvline(20.5, color="#555555", ls=":", lw=1.2)
    ax.text(20.9, ax.get_ylim()[1] * 0.95, "Inicio de F4", color="#444444", va="top")
    ax.set(title="MatGPT · transferencia energética Alexandria (20k): pérdida por época",
           xlabel="Época acumulada (20 de warmup + época local F4)",
           ylabel="SmoothL1 normalizada (menor es mejor)")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    note = ("20.000 fórmulas para entrenamiento y 2.000 para validación; una semilla (42). "
            "Cada punto es una época completa; línea discontinua = validación completa. "
            "F4 terminó las 100 épocas; selección por mínima pérdida de validación. "
            "La validación pertenece a Alexandria y no es el test ABS₃.")
    fig.text(0.01, -0.015, note, ha="left", va="top", fontsize=8.4, wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_transfer_validation(metrics: dict[str, Any], out_path: Path) -> None:
    """Compara F4 y Ridge composicional en la misma validación Alexandria.

    Args:
        metrics: JSON de métricas del entrenamiento local.
        out_path: PNG de salida.
    """
    trained = metrics.get("training", {})
    pred = trained.get("prediction_metrics", {})
    ridge = trained.get("composition_ridge_alpha_1_baseline", {})
    if not pred or not ridge:
        raise ValueError("El JSON no contiene ambas métricas de validación")
    labels = ["Ridge composicional\n(alpha=1)", "MatGPT F4\n20k + ajuste parcial"]
    mae = [ridge["mae_mev_atom"], pred["mae_mev_atom"]]
    rmse = [ridge["rmse_ev_atom"] * 1000, pred["rmse_ev_atom"] * 1000]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), constrained_layout=True)
    colors = ["#8b949e", "#c35a31"]
    for ax, values, title, ylabel in [
        (axes[0], mae, "Error absoluto medio", "MAE (meV/átomo)"),
        (axes[1], rmse, "Raíz del error cuadrático medio", "RMSE (meV/átomo)"),
    ]:
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
    fig.suptitle("Validación Alexandria: F4 frente a Ridge composicional", fontsize=14)
    note = ("Misma validación: 2.000 fórmulas; energía DFT por átomo. Menor es mejor. "
            "F4: encoder MatGPT parcialmente ajustado (bloques 8–11) tras 20 épocas de warmup; "
            "Ridge usa solo composición. No hay predicciones guardadas para MatGPT congelado "
            "en estos mismos 2.000 casos; por tanto, esta figura no compara F4 contra SLICES congelado "
            "ni contra el modelo con geometría. No es una evaluación ABS₃ externa.")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8.4, wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_abs3_context(metrics: dict[str, Any], out_path: Path) -> None:
    """Compara condiciones congeladas con M2 en el OOF histórico ABS3.

    Args:
        metrics: Auditoría de la comparación M2 MatGPT/MatterGPT.
        out_path: PNG de salida.
    """
    if metrics.get("status") != "PASS":
        raise ValueError("La fuente de comparación ABS3 no está en estado PASS")
    all_metrics = metrics.get("metrics_estimand_A", {})
    keys = ["RETRAINED_M2_C0", "RETRAINED_M2_C1", "MatterGPT_F0", "MatterGPT_F0_geometry"]
    names = ["MatGPT\nSLICES-only", "MatGPT\nSLICES + geometría", "MatterGPT\nSLICES-only", "MatterGPT\nSLICES + geometría"]
    if any(k not in all_metrics for k in keys):
        raise ValueError("Faltan condiciones M2 en la fuente de métricas ABS3")
    colors = ["#376996", "#63a1c7", "#8e5d9f", "#bd86c9"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4), constrained_layout=True)
    panels = [
        ("mae_EN_mev_atom", "MAE energía absoluta", "meV/átomo"),
        ("mae_delta_signed_mev_atom", "MAE ΔE firmado entre pares", "meV/átomo"),
        ("pairwise_accuracy_eligible", "Exactitud pairwise", "Proporción correcta"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        vals = [float(all_metrics[k][key]) for k in keys]
        bars = ax.bar(range(4), vals, color=colors, width=0.72)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), names, fontsize=8.5)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        if key == "pairwise_accuracy_eligible":
            ax.set_ylim(0, 1)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=8)
        else:
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    fig.suptitle("ABS₃: modelos con encoder congelado y Ridge estequiométrico M2", fontsize=14)
    note = ("OOF de desarrollo: 70 composiciones, 210 estructuras y 591 pares elegibles; "
            "cinco pliegues externos y tres semillas, agregación histórica por composición (estimando A). "
            "ΔE son diferencias firmadas intrafórmula; no se mezclan composiciones. "
            "Exactitud pairwise usa |ΔE_DFT| > 5 meV/átomo y cuenta predicciones no resueltas como incorrectas. "
            "Esta evaluación ABS₃ es contexto comparativo independiente de la validación Alexandria 20k; "
            "no mide la transferencia F4.")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8.2, wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Define y valida rutas configurables relativas al proyecto.

    Returns:
        Argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Carga fuentes verificadas, crea figuras y registra procedencia.

    Returns:
        Código de salida 0 si todas las figuras fueron creadas.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    args = parse_args()
    root = args.project_root.resolve()
    transfer_dir = root / "experiments/matgpt_alexandria_energy_transfer_20k_f4_local_mps"
    abs3_dir = root / "experiments/matgpt_f0_stoichiometric_ridge_m2_comparison"
    out_dir = (args.output_dir or root / "images/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k").resolve()
    history = transfer_dir / "historial_entrenamiento.csv"
    transfer_metrics_path = transfer_dir / "metrics.json"
    abs3_metrics_path = abs3_dir / "metrics.json"
    try:
        transfer_metrics = load_json(transfer_metrics_path)
        abs3_metrics = load_json(abs3_metrics_path)
        outputs = {
            "curvas_entrenamiento_validacion_f4.png": lambda p: plot_learning_curves(history, p),
            "validacion_alexandria_f4_vs_ridge.png": lambda p: plot_transfer_validation(transfer_metrics, p),
            "contexto_abs3_slices_congelado_vs_geometria.png": lambda p: plot_abs3_context(abs3_metrics, p),
        }
        for name, build in outputs.items():
            target = out_dir / name
            build(target)
            LOGGER.info("Figura guardada: %s", target)
        manifest = {
            "created_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "outputs": {name: str((out_dir / name).relative_to(root)) for name in outputs},
            "sources": {
                str(history.relative_to(root)): sha256_file(history),
                str(transfer_metrics_path.relative_to(root)): sha256_file(transfer_metrics_path),
                str(abs3_metrics_path.relative_to(root)): sha256_file(abs3_metrics_path),
            },
            "resource_note": "Postproceso CPU; sin inferencia ni entrenamiento; solo generación PNG.",
            "environment": {"python": __import__("sys").version.split()[0], "matplotlib": plt.matplotlib.__version__, "pandas": pd.__version__},
        }
        (transfer_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        LOGGER.info("Manifiesto guardado; postproceso %.2f s", manifest["elapsed_seconds"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.exception("No se pudieron generar las figuras: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
