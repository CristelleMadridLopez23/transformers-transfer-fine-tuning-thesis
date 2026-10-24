#!/usr/bin/env python3
"""Compare the permitted T2 pilot with the exactly matched Aug2 fold/seed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
SHARED = ALT / "scripts/MatGPT_finetuning/matgpt_e2_extended"
sys.path.insert(0, str(SHARED))
from evaluate_aug import bootstrap_pair, composition_metrics  # noqa: E402


def evaluate(t2_path: Path, aug2_path: Path, output: Path) -> None:
    """Generate pilot-only static metrics and paired composition bootstrap.

    Args:
        t2_path: T2 pilot OOF CSV.
        aug2_path: Full Aug2 CV OOF CSV containing the matched fold/seed.
        output: New report directory.
    Returns:
        None; writes paired CSV, JSON, Markdown, and PNG.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"No se sobrescribe un reporte T2 existente: {output}")
    t2 = pd.read_csv(t2_path)
    aug2 = pd.read_csv(aug2_path)
    t2 = t2[t2.fold.eq(0) & t2.seed.eq(42)].copy()
    aug2 = aug2[aug2.fold.eq(0) & aug2.seed.eq(42)].copy()
    if len(t2) != 42 or len(aug2) != 42:
        raise RuntimeError(f"Se requieren 42 predicciones outer emparejadas: T2={len(t2)}, Aug2={len(aug2)}")
    left = composition_metrics(t2, "E2-T2-pilot")
    right = composition_metrics(aug2, "E2-Aug2").drop(columns="configuration")
    paired = left.merge(right, on="composition_group", suffixes=("_t2", "_aug2"), validate="one_to_one")
    if len(paired) != 14:
        raise RuntimeError(f"El fold emparejado debe contener 14 composiciones; hay {len(paired)}")
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "pilot_composition_metrics.csv", index=False)
    ci = bootstrap_pair(left, right.assign(configuration="E2-Aug2"), "E2-T2-pilot - E2-Aug2", n_boot=2000)
    payload = {"status": "PILOT_ONLY", "fold": 0, "seed": 42, "n_compositions": 14,
               "bootstrap_seed": 13072026, "bootstrap_replicates": 2000,
               "historical_test_used": False, "confidence_intervals": ci,
               "interpretation": "Exploratorio; un único fold y semilla no permite concluir una mejora ni justifica E2 full."}
    (output / "pilot_paired_bootstrap.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    ax.scatter(paired.delta_e_mae_mev_atom_aug2, paired.delta_e_mae_mev_atom_t2,
               color="#8055A6", s=55, alpha=.85, label="Una composición ABS₃")
    lo = min(paired.delta_e_mae_mev_atom_aug2.min(), paired.delta_e_mae_mev_atom_t2.min())
    hi = max(paired.delta_e_mae_mev_atom_aug2.max(), paired.delta_e_mae_mev_atom_t2.max())
    ax.plot([lo, hi], [lo, hi], "k--", label="Igualdad: y = x")
    ax.set(xlabel="Aug2 MAE de ΔE (meV/átomo)", ylabel="T2 piloto MAE de ΔE (meV/átomo)",
           title="E2-T2: comparación exploratoria en outer-validation")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.text(.01, .01, "14 composiciones, fold 0 / semilla 42; 42 estructuras estáticas outer.\n"
             "Puntos bajo y=x favorecen T2 en ΔE MAE; el piloto no fue una selección de hiperparámetros.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "t2_pilot_deltae.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    t2_metrics = left[["absolute_mae_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "range_error_mev_atom"]].mean()
    aug2_metrics = right[["absolute_mae_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "range_error_mev_atom"]].mean()
    report = ["# E2-T2 — piloto de supervisión de trayectorias", "",
              "Estado: PILOT_ONLY. T1 LIMITED permite un único piloto; no autoriza CV completa. Se comparan predicciones estáticas outer del fold 0 / semilla 42 sobre los mismos 14 grupos (42 estructuras). El test histórico no se usó.", "",
              "| Métrica macro por composición | E2-Aug2 | E2-T2 piloto |", "|---|---:|---:|"]
    units = {"absolute_mae_mev_atom": "MAE absoluta (meV/átomo)", "delta_e_mae_mev_atom": "MAE ΔE (meV/átomo)",
             "ground_state_accuracy": "Exactitud GS", "pairwise_accuracy": "Pairwise accuracy", "range_error_mev_atom": "Error de rango (meV/átomo)"}
    for key, label in units.items():
        report.append(f"| {label} | {aug2_metrics[key]:.4f} | {t2_metrics[key]:.4f} |")
    report += ["", "Bootstrap pareado por las 14 composiciones (2,000 remuestras, semilla 13072026), IC exploratorios. No interpretar cambios como una mejora generalizable por contar solo con un fold y una semilla. Las métricas principales siguen siendo estáticas; no se evalúa estado fundamental usando frames de relajación.", ""]
    (output / "E2_T2_pilot_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    """Parse matched pilot paths and render the paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2-predictions", type=Path, required=True)
    parser.add_argument("--aug2-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.t2_predictions, args.aug2_predictions, args.out)


if __name__ == "__main__":
    main()
