#!/usr/bin/env python3
"""Replica las curvas train/validación H3-C desde historiales reales.

Genera una figura PNG versionada con las pérdidas registradas para las 15
corridas (cinco folds agrupados por composición y tres semillas). No reentrena
modelos ni modifica historiales, predicciones o figuras históricas.
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


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/comparative_analysis/H3C_training_curves_data.csv"
DEFAULT_RUNS = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_C_ridge_retrained_delta/runs"
DEFAULT_OUTPUT = ROOT / "Alternative2/reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/figures_v2"
REQUIRED_COLUMNS = {
    "fold", "seed", "epoch", "train_loss_delta", "inner_validation_loss_delta",
    "scale_delta_ev_atom",
}
LOGGER = logging.getLogger("h3c_training_curves_v2")


def _configure_logging() -> None:
    """Configura logging estructurado en JSON para ejecución reproducible."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


def _parse_args() -> argparse.Namespace:
    """Valida argumentos de entrada y rutas configurables.

    Returns:
        Argumentos con las rutas al CSV, manifiestos y PNG de salida.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV histórico de curvas por época.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="Directorio con manifiestos PASS fold/semilla.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directorio versionado para PNG.")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"No existe el historial: {args.input}")
    if not args.runs.is_dir():
        parser.error(f"No existe el directorio de corridas: {args.runs}")
    return args


def _load_history(path: Path) -> pd.DataFrame:
    """Carga y valida los historiales existentes sin completar datos faltantes.

    Args:
        path: CSV con una observación por corrida y época.
    Returns:
        Historial numérico ordenado por fold, semilla y época.
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el historial: {sorted(missing)}")
    if frame.empty or frame[["fold", "seed", "epoch"]].isna().any().any():
        raise ValueError("El historial está vacío o tiene identificadores fold/semilla/época nulos.")
    numeric = ["train_loss_delta", "inner_validation_loss_delta", "scale_delta_ev_atom"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("El historial contiene pérdidas o escalas no finitas.")
    if frame.duplicated(["fold", "seed", "epoch"]).any():
        raise ValueError("Hay filas duplicadas por fold, semilla y época.")
    if frame.epoch.lt(1).any() or frame.scale_delta_ev_atom.le(0).any():
        raise ValueError("Épocas o escalas inválidas.")
    if frame.groupby(["fold", "seed"]).epoch.apply(lambda values: values.is_monotonic_increasing is False).any():
        raise ValueError("Las épocas no son crecientes dentro de una corrida.")
    return frame.sort_values(["fold", "seed", "epoch"], kind="stable").reset_index(drop=True)


def _load_selected_epochs(runs_dir: Path, expected: set[tuple[int, int]]) -> pd.DataFrame:
    """Lee los checkpoints elegidos en inner-validation para cada corrida.

    Args:
        runs_dir: Directorio con un subdirectorio fold_<n>_seed_<n> por corrida.
        expected: Pares fold/semilla presentes en el CSV.
    Returns:
        Tabla con la época mínima de inner-validation por corrida.
    """
    records: list[dict[str, Any]] = []
    for fold, seed in sorted(expected):
        manifest_path = runs_dir / f"fold_{fold}_seed_{seed}" / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Falta manifiesto para fold={fold}, semilla={seed}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("test_used") is not False:
            raise ValueError(f"Manifiesto no PASS o test consultado: {manifest_path}")
        if (int(manifest.get("fold", -1)), int(manifest.get("seed", -1))) != (fold, seed):
            raise ValueError(f"IDs fold/semilla no coinciden en {manifest_path}")
        records.append({"fold": fold, "seed": seed, "best_epoch": int(manifest["best_epoch"])})
    return pd.DataFrame.from_records(records)


def _curve_stats(history: pd.DataFrame, column: str) -> pd.DataFrame:
    """Resume media, desviación estándar muestral y n activa por época.

    Args:
        history: Historial validado de todas las corridas.
        column: Nombre de la pérdida que se resume.
    Returns:
        Estadísticas por época usando solo corridas con esa época registrada.
    """
    return history.groupby("epoch")[column].agg(mean="mean", std="std", n="count").reset_index()


def make_figure(history: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> Path:
    """Dibuja curvas comparables y selecciona salida PNG versionada.

    Args:
        history: Historial train/inner-validation por fold, semilla y época.
        selected: Épocas seleccionadas por early stopping en inner-validation.
        output_dir: Directorio de salida exclusivo para esta revisión.
    Returns:
        Ruta del PNG generado.
    """
    train = _curve_stats(history, "train_loss_delta")
    valid = _curve_stats(history, "inner_validation_loss_delta")
    if not np.array_equal(train.epoch.to_numpy(), valid.epoch.to_numpy()):
        raise ValueError("Train y validación no comparten las mismas épocas registradas.")

    color_train, color_valid = "#2878b5", "#c83c61"
    fig, (ax, count_ax) = plt.subplots(
        2, 1, figsize=(12.5, 8), sharex=True, gridspec_kw={"height_ratios": [4.2, 1]},
    )
    for stats, color, label in (
        (train, color_train, "Entrenamiento: media de minibatches"),
        (valid, color_valid, "Inner-validation: evaluación completa"),
    ):
        x = stats.epoch.to_numpy(dtype=int)
        mean = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, mean, color=color, linewidth=2.5, marker="o", markersize=3.5, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.17, linewidth=0,
                        label="±1 DE entre corridas" if label.startswith("Entrenamiento") else None)

    median_epoch = float(selected.best_epoch.median())
    q1, q3 = (float(value) for value in selected.best_epoch.quantile([.25, .75]))
    ax.axvspan(q1, q3, color="#6b7280", alpha=.09, label="Rango intercuartílico de época elegida")
    ax.axvline(median_epoch, color="#454545", linestyle="--", linewidth=1.4,
               label=f"Mediana de mejor época: {median_epoch:.0f}")
    ax.set_ylabel("SmoothL1 normalizada del residuo centrado (adimensional)")
    ax.set_title("H3-C · evolución de entrenamiento y validación de MLP8")
    ax.grid(alpha=.22)
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=1)

    count = train[["epoch", "n"]]
    count_ax.step(count.epoch, count.n, where="mid", color="#5b6573", linewidth=1.7)
    count_ax.scatter(count.epoch, count.n, color="#5b6573", s=12, zorder=3)
    count_ax.set_ylabel("Corridas\nactivas")
    count_ax.set_xlabel("Época local de MLP8")
    count_ax.set_yticks(sorted(set([1, int(count.n.max())])))
    count_ax.set_ylim(0, int(count.n.max()) + 1)
    count_ax.grid(axis="y", alpha=.22)

    n_runs = history[["fold", "seed"]].drop_duplicates().shape[0]
    n_folds = history.fold.nunique()
    n_seeds = history.seed.nunique()
    n_struct_train, n_struct_val = 135, 33
    scales = history.scale_delta_ev_atom
    note = (
        f"{n_runs} corridas ({n_folds} folds agrupados por composición × {n_seeds} semillas); "
        f"por fold: train 45 composiciones/135 estructuras, inner-validation 11 composiciones/33 estructuras. "
        "Azul: media por época de pérdidas promedio de minibatches train (cada minibatch pesa igual); "
        "rojo: pérdida sobre toda inner-validation. Ambas son SmoothL1 de residuos centrados, "
        "normalizadas con escala calculada solo en inner-train (rango de escala observado "
        f"{scales.min():.3f}–{scales.max():.3f} eV/átomo). Bandas: ±1 desviación estándar entre corridas; "
        "n cambia al detenerse cada corrida. Checkpoint = mínimo de inner-validation; "
        "línea y franja grises = mediana e intervalo intercuartílico de épocas seleccionadas. "
        "No se usó outer-validation ni test para early stopping."
    )
    fig.text(.07, .01, note, ha="left", va="bottom", fontsize=8.2, color="#333333", wrap=True)
    fig.tight_layout(rect=(0, .095, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_h3c_training_validation_curves.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    """Ejecuta validación de procedencia y exporta la figura PNG.

    Returns:
        Código de salida cero si la figura quedó escrita correctamente.
    """
    _configure_logging()
    args = _parse_args()
    try:
        history = _load_history(args.input)
        expected = set(zip(history.fold.astype(int), history.seed.astype(int)))
        if len(expected) != 15 or history.fold.nunique() != 5 or history.seed.nunique() != 3:
            raise ValueError(f"Se esperaban 15 corridas (5 folds × 3 semillas); encontrados {len(expected)}.")
        selected = _load_selected_epochs(args.runs, expected)
        if set(zip(selected.fold, selected.seed)) != expected:
            raise ValueError("Los manifiestos no cubren exactamente las corridas del historial.")
        result = make_figure(history, selected, args.output_dir)
        LOGGER.info(json.dumps({"event": "figure_written", "path": str(result), "rows": len(history),
                                "runs": len(expected), "epochs": int(history.epoch.max())}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.exception("figure_generation_failed error=%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
