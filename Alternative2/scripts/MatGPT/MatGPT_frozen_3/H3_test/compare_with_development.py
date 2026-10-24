#!/usr/bin/env python3
"""Compara métricas OOF de desarrollo con el ensemble final histórico test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import REPORT_DIR, H3TestError, write_json


def compare() -> dict[str, Any]:
    """Alinea métricas H2/H3-A sin mezclar particiones ni estimadores.

    Returns:
        Tablas de desarrollo OOF y test, con diferencias descriptivas.
    """
    dev_path = REPORT_DIR.parent / "hybrid_ridge_mlp8/comparative_analysis/comparative_metrics.csv"
    test_path = REPORT_DIR / "metrics/test_metrics_summary.csv"
    integrity_path = REPORT_DIR / "metrics/evaluation_integrity.json"
    if not all(path.is_file() for path in (dev_path, test_path, integrity_path)):
        raise H3TestError("Primero completar evaluación H3-TEST para comparar desarrollo/test.")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS" or integrity.get("test_used_for_selection") is not False:
        raise H3TestError("La evaluación test no está cerrada en PASS.")
    dev = pd.read_csv(dev_path).set_index("model")
    test = pd.read_csv(test_path).set_index("model")
    rows = []
    mapping = {
        "MAE absoluto, macro (meV/átomo)": ("MAE_absolute_mev_atom_macro", "MAE_absolute_mev_atom_macro_composition"),
        "MAE ΔE anclada a DFT-GS (meV/átomo)": ("MAE_delta_GS_anchored_mev_atom_macro", "MAE_delta_DFT_GS_anchored_mev_atom_macro"),
        "MAE ΔE own-min (meV/átomo)": ("MAE_delta_own_min_legacy_mev_atom_macro", "MAE_delta_own_min_mev_atom_macro"),
        "Ground-state accuracy": ("GS_accuracy", "GS_accuracy"),
        "Pairwise accuracy pooled": ("Pairwise_accuracy_pooled", "pairwise_accuracy"),
    }
    for model in ("H2", "H3-A"):
        for label, (dev_col, test_col) in mapping.items():
            dev_value = float(dev.loc[model, dev_col]) if dev_col in dev.columns else float("nan")
            test_value = float(test.loc[model, test_col]) if test_col in test.columns else float("nan")
            rows.append({"model": model, "metric": label, "development_OOF": dev_value,
                         "development_N_compositions": 70, "test_ensemble": test_value,
                         "test_N_compositions": 8,
                         "descriptive_test_minus_dev": test_value-dev_value,
                         "estimators_identical": False,
                         "interpretation": "Descriptive only: development is outer OOF; test averages five fold models and three H2 seeds."})
    output = pd.DataFrame(rows)
    out_dir = REPORT_DIR / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "development_vs_test.csv", index=False)
    note = {"status": "PASS", "comparison_unit": "composition", "development": "70-composition OOF",
            "test": "8-composition ensemble over 5 historical folds and 3 H2 seeds",
            "same_estimator": False, "no_test_reselection": True,
            "warning": "A difference between development and test is not by itself proof of domain shift or overfitting.",
            "rows": len(output)}
    write_json(out_dir / "comparison_protocol.json", note)
    return {"table": output, "protocol": note}


def main() -> None:
    """Guarda comparación descriptiva tras el scoring bloqueado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = compare()
        print(json.dumps(result["protocol"], ensure_ascii=False))
    except (H3TestError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"H3-TEST development comparison FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
