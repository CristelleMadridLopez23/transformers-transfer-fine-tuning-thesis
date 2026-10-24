#!/usr/bin/env python3
"""Render development-only E2-T1 trajectory and token-collision PNG diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / "Alternative2/reports/MatGPT_finetuning/E2-T1"


def render(report_dir: Path) -> None:
    """Create three PNG figures from audited T1 CSV outputs.

    Args:
        report_dir: E2-T1 output with frame, trajectory, and collision tables.
    Returns:
        None; writes figures/energy_by_step.png, token_coverage.png, and collisions.png.
    """
    frames = pd.read_csv(report_dir / "frame_tokenization.csv", low_memory=False)
    trajectories = pd.read_csv(report_dir / "trajectory_manifest.csv", low_memory=False)
    collisions = pd.read_csv(report_dir / "collision_summary.csv", low_memory=False)
    output = report_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for _, group in frames.groupby("trajectory_id", sort=False):
        group = group.sort_values("ionic_step")
        ax.plot(group.ionic_step, group.delta_to_last_frame_mev_atom, color="#457B9D", alpha=.12, lw=.7)
    median = frames.groupby("ionic_step").delta_to_last_frame_mev_atom.median().sort_index()
    ax.plot(median.index, median.values, color="#D1495B", lw=2.5, label="Mediana entre trayectorias")
    ax.axhline(0, color="black", lw=1, ls="--", label="Energía del último frame de cada trayectoria")
    ax.set(xlabel="Paso iónico", ylabel="Energía relativa al último frame (meV/átomo)",
           title="E2-T1: evolución energética dentro de relajaciones")
    ax.legend()
    ax.grid(True, alpha=.25)
    fig.text(.01, .01, f"Trazos finos: {frames.trajectory_id.nunique()} trayectorias de desarrollo; cada línea se resta a su propio último frame.\n"
             f"{len(frames):,} frames train+validation. Los valores pueden subir o bajar durante la relajación.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "energy_by_step.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.hist(trajectories.unique_token_ratio.dropna(), bins=np.linspace(0, 1, 21), color="#2A9D8F", edgecolor="white")
    ax.set(xlabel="Secuencias de IDs únicas / frames válidos (fracción)", ylabel="Número de trayectorias",
           title="E2-T1: cobertura de tokens por trayectoria")
    ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, f"N={len(trajectories)} trayectorias enlazadas a train+validation; cada trayectoria pesa una vez; 20 bins iguales de ancho 0.05.\n"
             "La fracción no mide diversidad geométrica que SLICES no codifique.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "token_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    positive = collisions.collision_range_mev_atom.to_numpy(float) if len(collisions) else np.asarray([])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive):
        bins = np.geomspace(max(float(positive.min()), 1e-3), float(positive.max()), 30)
        ax.hist(positive, bins=bins, color="#E07A24", edgecolor="white")
        ax.set_xscale("log")
    ax.set(xlabel="Rango DFT entre frames con IDs idénticos (meV/átomo; eje log)", ylabel="Grupos de IDs repetidos",
           title="E2-T1: colisiones de secuencias tokenizadas")
    ax.grid(True, which="both", axis="both", alpha=.2)
    fig.text(.01, .01, f"N={len(collisions)} grupos repetidos de IDs en train+validation; bins geométricos.\n"
             "Rango = máximo−mínimo dentro del grupo. Cada grupo define una cota de ambigüedad para un predictor de solo tokens.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "collisions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse report location and render audited figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    render(args.report_dir)


if __name__ == "__main__":
    main()
