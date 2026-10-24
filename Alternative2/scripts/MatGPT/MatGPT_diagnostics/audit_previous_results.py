#!/usr/bin/env python3
"""Audita predicciones históricas E2-A/E2-Aug y explica la regla de ranking."""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, sha256, write_json

LOG = logging.getLogger("matgpt_diagnostics.reconcile")
BASE = ALT / "reports/MatGPT_finetuning"


def composition_rows(frame: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """Resume cada composición; strict aplica abstención ante degeneración/empate.

    Args:
        frame: Predicciones OOF para un modelo y semilla.
        strict: Omite GS si el gap DFT o el empate predicho es <=1 meV/átomo.
    Returns:
        Una fila de métricas por composición con flags de elegibilidad.
    """
    rows = []
    for keys, group in frame.groupby(["seed", "fold", "composition_group"], sort=True):
        y = group.energy_per_atom_ev.to_numpy(float) * 1000.0
        p = group.energy_pred_ev_atom.to_numpy(float) * 1000.0
        if len(y) != 3:
            raise ValueError(f"Composición {keys[2]} no tiene tres polimorfos.")
        order_y = y.argsort(); order_p = p.argsort()
        dft_gap = float(y[order_y[1]] - y[order_y[0]])
        pred_gap = float(p[order_p[1]] - p[order_p[0]])
        dft_de = y - y.min(); pred_de = p - p.min()
        pairs = [(i, j) for i in range(3) for j in range(i + 1, 3) if abs(y[i] - y[j]) > 1.0]
        pairwise = sum((y[i] - y[j]) * (p[i] - p[j]) > 0 for i, j in pairs) / len(pairs) if pairs else float("nan")
        # Una degeneración DFT sale del denominador. Una predicción empatada
        # permanece en el denominador y cuenta como no acierto.
        eligible = dft_gap > 1.0
        rows.append({
            "seed": int(keys[0]), "fold": int(keys[1]), "composition_group": str(keys[2]),
            "dft_ground_gap_mev_atom": dft_gap, "pred_ground_gap_mev_atom": pred_gap,
            "eligible_gs_strict": bool(eligible), "predicted_tie_non_discriminative": bool(pred_gap <= 1.0),
            "gs_legacy_argmin": float(int(order_y[0] == order_p[0])),
            "gs_strict": float(int(order_y[0] == order_p[0] and pred_gap > 1.0)) if eligible else float("nan"),
            "delta_e_mae_mev_atom": float(abs(pred_de - dft_de).mean()),
            "pairwise_accuracy": float(pairwise), "n_eligible_pairs": len(pairs),
        })
    return pd.DataFrame(rows)


def run() -> None:
    """Recalcula métricas bajo ambas reglas sobre OOF originales."""
    inputs = {
        "E2-A-partial_ft": BASE / "matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv",
        "E2-Aug1": BASE / "E2-Aug1/development_only_v2/cv/cpu/oof_predictions.csv",
        "E2-Aug2": BASE / "E2-Aug2/development_only_v2/cv/cpu/oof_predictions.csv",
    }
    out = REPORT / "common"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    provenance = {}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró OOF histórico: {path}")
        data = pd.read_csv(path)
        data = data.loc[data.configuration.isin(["partial_ft", name])].copy()
        if data.structure_id.astype(str).str.contains("test", case=False).any():
            raise ValueError(f"La entrada {name} contiene test; se detiene la auditoría.")
        if data.groupby("seed").composition_group.nunique().nunique() != 1:
            raise ValueError(f"Cobertura OOF inconsistente en {name}.")
        comp = composition_rows(data, strict=True)
        comp.insert(0, "configuration", name)
        comp.to_csv(out / f"{name.replace('-', '_')}_composition_metrics.csv", index=False)
        for seed, group in comp.groupby("seed"):
            fold_macro = float(group.groupby("fold").gs_strict.mean().mean())
            all_metrics.append({
                "configuration": name, "seed": int(seed),
                "n_compositions_oof": int(group.composition_group.nunique()),
                "n_gs_eligible": int(group.eligible_gs_strict.sum()),
                "gs_legacy_all_compositions": float(group.gs_legacy_argmin.mean()),
                "gs_strict_eligible_only": float(group.gs_strict.mean()),
                "gs_strict_fold_macro": fold_macro,
                "delta_e_mae_mev_atom": float(group.delta_e_mae_mev_atom.mean()),
                "pairwise_accuracy": float(group.pairwise_accuracy.mean()),
            })
        provenance[name] = {"path": str(path.relative_to(ALT.parent)), "sha256": sha256(path),
                            "rows": len(data), "seeds": sorted(int(x) for x in data.seed.unique()),
                            "folds": sorted(int(x) for x in data.fold.unique()),
                            "structures_per_seed": sorted(int(v) for v in data.groupby("seed").structure_id.nunique().unique())}
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(out / "baseline_reconciliation_metrics.csv", index=False)
    partial = summary[summary.configuration.eq("E2-A-partial_ft")]
    historic = float(partial.gs_legacy_all_compositions.mean())
    strict = float(partial.gs_strict_fold_macro.mean())
    n_all = int(partial.n_compositions_oof.iloc[0])
    n_eligible = int(partial.n_gs_eligible.mean())
    text = f"""# Reconciliación de métricas OOF E2-A y E2-Aug

## Qué originó 0.533 frente a 0.495

Las cifras usan las mismas predicciones OOF de E2-A partial fine-tuning: 210 estructuras, 70 composiciones de desarrollo, cinco folds y tres semillas. Ninguna usa test. La tabla histórica en `E2_report.md` aplicó `argmin` a cada triplete y siempre eligió un polimorfo, incluso cuando la diferencia DFT del mínimo era de 1 meV/átomo o menor, o la predicción empataba. Su media es **{historic:.3f}** sobre las {n_all} composiciones.

La comparación posterior usa una regla de ranking más prudente: solo considera pares con diferencia DFT >1 meV/átomo; GS excluye composiciones cuyo gap entre los dos mínimos DFT sea ≤1 meV/átomo. Una predicción empatada en el mínimo cuenta como **no acierto** y se reporta como no discriminativa, sin elegir el primer polimorfo. Con esa cobertura, GS promediado primero por fold y luego por semilla es **{strict:.3f}**, de 67 composiciones elegibles de 70 por semilla. Si se agrupan directamente las 67 composiciones elegibles, sin promediar folds por separado, el resultado es aproximadamente 0.493. El 0.495 publicado usa la primera agregación; el 0.533 antiguo usa `argmin` sobre todas las composiciones, incluso las DFT casi empatadas. Por tanto, la diferencia es la definición del denominador/ponderación, no un checkpoint distinto.

**Convención para D1–D3:** mantener esta regla estricta para cualquier comparación de ranking; reportar siempre cuántas composiciones/pares son elegibles. Mantener el `argmin` histórico solo como columna de trazabilidad, nunca como métrica principal. Las métricas de MAE absoluto y ΔE se calculan sobre tripletes completos.

## Procedencia y cobertura

| Entrada | Filas/semilla | Semillas | Folds | Composiciones/semilla |
|---|---:|---|---|---:|
"""
    for item in provenance.values():
        text += f"| {item['path'].split('/')[-4]} | {item['structures_per_seed'][0]} | {item['seeds']} | {item['folds']} | 70 |\n"
    text += "\nLas medias recalculadas por semilla, denominadores y hashes están en `baseline_reconciliation_metrics.csv` y `provenance.json`. E2-Aug1/E2-Aug2 son comparables en los mismos 70 grupos y folds. La comparación histórica de modelos fue seleccionada usando la validation de E1.1/E1.2; este análisis de desarrollo no elimina ese sesgo. Los ocho grupos históricos de test permanecieron excluidos.\n"
    (out / "baseline_reconciliation.md").write_text(text)
    write_json(out / "provenance.json", provenance)
    LOG.info("reconciliation_complete gs_legacy=%.3f gs_strict=%.3f eligible≈%d/%d", historic, strict, n_eligible, n_all)


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
