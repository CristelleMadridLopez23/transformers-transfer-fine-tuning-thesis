"""Calcula Spearman y Kendall por triplete físico H3-C sin nuevos ajustes."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from reproduce import project_root, sha256

LOG = logging.getLogger("h3c_rank_correlations")


def calculate(root: Path) -> dict[str, object]:
    """Lee predicciones archivadas; exporta correlaciones por fórmula y resumen.

    Args:
        root: Raíz del proyecto.
    Returns:
        Manifiesto con conteos y SHA-256 de los CSV.
    """
    target = root / "reports/MatGPT/MatGPT_H3C_Final/analysis"
    data = pd.read_csv(target / "predictions_by_structure.csv")
    rows: list[dict[str, object]] = []
    for (model, seed, composition), group in data.groupby(["model", "seed", "composition_id"], sort=True):
        if len(group) != 3 or group.structure_id.nunique() != 3:
            raise ValueError(f"Triplete inválido: {model}, {seed}, {composition}")
        dft = group.energy_per_atom_dft_ev.to_numpy(float)
        pred = group.energy_per_atom_pred_ev.to_numpy(float)
        if not np.isfinite(dft).all() or not np.isfinite(pred).all():
            raise ValueError(f"Energías no finitas: {composition}")
        tied = np.ptp(pred) < 1e-10
        rho = float(spearmanr(dft, pred).statistic) if not tied else float("nan")
        tau = float(kendalltau(dft, pred).statistic) if not tied else float("nan")
        rows.append({"model": model, "seed": int(seed), "composition_id": composition,
                     "fold": int(group.fold.iloc[0]), "n_polymorphs": 3,
                     "prediction_tie": tied, "spearman": rho, "kendall": tau})
    out = pd.DataFrame(rows)
    if len(out) != 630 or out.groupby(["model", "seed"]).size().ne(70).any():
        raise ValueError("Conteos de composiciones inesperados")
    detail = target / "rank_correlations_by_formula.csv"
    out.to_csv(detail, index=False)
    summary = out.groupby(["model", "seed"], as_index=False).agg(
        formulas=("composition_id", "size"), prediction_ties=("prediction_tie", "sum"),
        spearman_mean_defined=("spearman", "mean"), kendall_mean_defined=("kendall", "mean"),
        defined_formulas=("spearman", "count"))
    summary_path = target / "rank_correlations_summary.csv"
    summary.to_csv(summary_path, index=False)
    result: dict[str, object] = {"status": "PASS", "triplet_seed_rows": len(out),
                                 "detail_sha256": sha256(detail), "summary_sha256": sha256(summary_path),
                                 "historical_test_accessed": False}
    (target / "rank_correlations_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    LOG.info(json.dumps({"event": "rank_correlations", **result}))
    return result


def main() -> None:
    """Valida argumentos y ejecuta análisis de solo lectura sobre predicciones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=project_root())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    calculate(args.repo.resolve())


if __name__ == "__main__":
    main()
