"""Consolida procedencia, métricas y recursos del experimento ABS3 real."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import sklearn
import pymatgen

LOG = logging.getLogger("geometry_finalize")


def sha256(path: Path) -> str:
    """Calcula SHA-256 de cada fuente o resultado."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(root: Path) -> dict[str, Any]:
    """Exige artefactos reales, salidas apareadas y pruebas antes de informar."""
    root = root.resolve()
    report = root / "reports/MatGPT/MatGPT_SLICES_Geometria"
    results = report / "results"
    link = json.loads((report / "structures_development.manifest.json").read_text())
    export = json.loads((report / "geometry_cell_development.manifest.json").read_text())
    audit_path = report / "staging_repo/reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.numeric_audit.json"
    cache_audit = json.loads(audit_path.read_text())
    experiment = json.loads((results / "manifest_experimento.json").read_text())
    if (link["status"], export["status"], cache_audit["status"], experiment["status"]) != (
        "PASS", "PASS", "PASS_NUMERIC", "PASS"):
        raise ValueError("Algún gate crítico no pasó")
    for name in ("preflight_recovered_resources.json", "preflight_before_train_resources.json", "tests_final_resources.json"):
        if json.loads((report / name).read_text())["exit_code"] != 0:
            raise ValueError(f"Gate fallido: {name}")
    geometry = pd.read_csv(report / "geometry_cell_development.csv")
    if len(geometry) != 210 or geometry.structure_id.nunique() != 210 or geometry.geometry_source.nunique() != 1:
        raise ValueError("Geometría incompleta o fuentes heterogéneas")
    dataset = root / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
    with dataset.open(newline="", encoding="utf-8-sig") as stream:
        ids = {row["structure_id"] for row in csv.DictReader(stream) if row["split"] in ("train", "val")}
    if set(geometry.structure_id) != ids:
        raise ValueError("La geometría no coincide exactamente con los IDs de desarrollo")
    predictions = pd.read_csv(results / "predicciones_oof_por_estructura.csv")
    if len(predictions) != 1260 or set(predictions.structure_id) != ids:
        raise ValueError("Predicciones incompletas")
    labels = predictions.groupby("structure_id").agg(n=("n_atoms", "nunique"),
        total=("energy_total_dft_ev", "nunique"), atom=("energy_per_atom_dft_ev", "nunique"))
    if (labels != 1).any().any():
        raise ValueError("Target o tamaño difiere entre condiciones/semillas")
    for (_, _), part in predictions.groupby(["condition", "seed"]):
        if len(part) != 210 or set(part.structure_id) != ids:
            raise ValueError("Condiciones sin muestras apareadas")
    training = pd.read_csv(results / "seleccion_y_entrenamiento_por_fold.csv")
    if len(training) != 30 or set(training.parameters) != {6161, 6217} or (training.wall_seconds <= 0).any():
        raise ValueError("No constan 30 corridas apareadas con parámetros y tiempos")
    summary = pd.read_csv(results / "metricas/resumen_comparativo.csv").set_index("condition")
    ci = pd.read_csv(results / "metricas/intervalos_confianza_bootstrap_por_composicion.csv").set_index("metric")
    hist = pd.read_csv(root / "reports/MatGPT/MatGPT_H3C_Final/analysis/summary_by_seed.csv")
    prior = hist[hist.model == "pretrained"].mean(numeric_only=True)
    fig_dir = results / "figuras"
    figures = sorted(fig_dir.glob("*.png"))
    if len(figures) != 8:
        raise ValueError(f"Se esperaban 8 PNG; hay {len(figures)}")
    resource_rows = []
    for path in sorted(report.glob("*resources.json")):
        entry = json.loads(path.read_text())
        resource_rows.append({"run": path.stem, "wall_seconds": entry["wall_seconds"],
                              "user_cpu_seconds": entry["user_cpu_seconds"],
                              "system_cpu_seconds": entry["system_cpu_seconds"],
                              "peak_rss_bytes": entry["peak_rss_bytes"],
                              "exit_code": entry["exit_code"], "log": entry["log"]})
    resources = report / "resources_summary.csv"
    pd.DataFrame(resource_rows).to_csv(resources, index=False)
    p, g = summary.loc["slices_only"], summary.loc["slices_plus_cell_geometry"]
    pair_ci = ci.loc["pair_mae_mev_atom"]
    residual_ci = ci.loc["residual_mae_mev_atom"]
    rank_ci = ci.loc["pairwise_accuracy_gt5"]
    geometry["reduced_formula"] = geometry.structure_id.map(
        pd.read_csv(dataset, usecols=["structure_id", "reduced_formula"]).drop_duplicates("structure_id").set_index("structure_id").reduced_formula)
    variation = geometry.groupby("reduced_formula")[
        ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg",
         "gamma_reduced_deg", "volume_per_atom_ang3"]].agg(lambda column: column.max() - column.min())
    varying = int((variation.max(axis=1) > 1e-7).sum())
    provenance = f"""# Auditoría de procedencia geométrica ABS₃

[HECHO VERIFICADO] Los 210 IDs físicos train/val del CSV H3-C se enlazaron por `source_index`, `config_id` y `source_key` al archivo `data/abs3/only_hex_new/configurations_only_hex_new.extxyz`. Su SHA-256 `{link['source_sha256']}` coincide con el manifiesto de preprocesamiento. Se comprobaron además N, especies en orden, celda, posiciones, periodicidad y composición reducida contra el dataset. Ninguna correspondencia se infirió por nombre o proximidad química.

[HECHO VERIFICADO] El EXTXYZ contiene 8 898 frames. Solo se parsearon las 210 entradas train/val autorizadas; los demás frames se saltaron sin interpretar su comentario ni energía. El JSONL posee exactamente 210 `Structure.as_dict()` y no exporta etiquetas energéticas. Fuente geométrica común: `{link['geometry_source']}`. Los metadatos indican cálculo `static`, `endpoint_role=static_ground_truth` y limpieza de endpoints. El manifiesto `only_hex_new/manifest.json` declara copia byte a byte de frames retenidos y retención de estáticos.

[RESULTADO EXPERIMENTAL] El CSV de celda posee 210 filas, 70 fórmulas y una sola política de procedencia; {varying} fórmulas tienen variación en al menos un descriptor entre sus tres polimorfos. JSONL SHA-256 `{link['jsonl_sha256']}`; descriptores SHA-256 `{export['geometry_csv_sha256']}`.

[LIMITACIÓN] `static` no certifica por sí solo que la geometría esté plenamente relajada; el historial de relajación y la política XC/U/correcciones por registro no quedaron demostrados aquí. Los descriptores contienen solo forma/ángulos de celda reducida y volumen por átomo; omiten posiciones internas. El test histórico no se leyó.

[HECHO VERIFICADO] Las dependencias H3-C faltaban del árbol activo. Se recuperaron cinco Ridge y el CSV de folds desde el commit Git registrado en `staging_repo/recovery_manifest.json`, sin reintroducir rutas borradas. El caché de embeddings se reconstruyó desde checkpoint MatGPT original. Su hash binario `{cache_audit['cache_sha256']}` difiere del histórico `{cache_audit['historical_cache_sha256']}`; el cotejo numérico con distancias de embeddings históricos y medias de los cinco Ridge pasó (`max |Δμ|={cache_audit['max_ridge_mean_difference_ev_atom']:.2e} eV/átomo`). Se conservan ambos hashes y esta limitación.
"""
    scientific = f"""# SLICES frente a SLICES + geometría real de celda en ABS₃

## Diseño y gate

[HECHO VERIFICADO] `preflight.py` terminó PASS tras el enlace exacto de 210 geometrías, la recuperación aislada de folds/Ridge y la auditoría numérica del caché. Cinco folds agrupados por las mismas 70 fórmulas, tres semillas (42, 123, 2026), dos condiciones apareadas: 1 260 predicciones de 210 estructuras físicas. No son 1 260 muestras independientes. Test histórico cerrado.

[HECHO VERIFICADO] MatGPT permaneció congelado. Los cinco Ridge H3-C históricos se cargaron sin reentrenarlos. Ambas condiciones entrenaron MLP8 con igual inicialización, orden de minibatches y selección por validación interna. La fusión añade siete descriptores de celda solo a la primera proyección: `Linear(768,8)` pasa a `Linear(775,8)`, **56 pesos** adicionales; 6 161 → 6 217 parámetros. Ancho 8, GELU, dropout, pérdida y target DFT por átomo se conservaron. La geometría se estandarizó exclusivamente con las 45 fórmulas inner-train de cada fold.

## Resultados en validación de desarrollo

| Métrica | SLICES solo | SLICES + celda | Diferencia fusión − control |
|---|---:|---:|---:|
| MAE energía por átomo (meV/átomo) | {p.mae_energy_per_atom_mev:.2f} | {g.mae_energy_per_atom_mev:.2f} | {g.mae_energy_per_atom_mev-p.mae_energy_per_atom_mev:+.2f} |
| MAE residuo intrafórmula (meV/átomo) | {p.mae_residual_intraformula_mev:.2f} | {g.mae_residual_intraformula_mev:.2f} | {g.mae_residual_intraformula_mev-p.mae_residual_intraformula_mev:+.2f} |
| MAE pares firmados (meV/átomo) | {p.mae_delta_pairs_mev:.2f} | {g.mae_delta_pairs_mev:.2f} | {g.mae_delta_pairs_mev-p.mae_delta_pairs_mev:+.2f} |
| Estado fundamental elegible (%, 67 fórmulas/semilla) | {100*p.ground_state_eligible_accuracy:.1f} | {100*g.ground_state_eligible_accuracy:.1f} | {100*(g.ground_state_eligible_accuracy-p.ground_state_eligible_accuracy):+.1f} pp |
| Orden de pares >5 meV/átomo (%, 197 pares/semilla) | {100*p.pairwise_accuracy_gt5:.1f} | {100*g.pairwise_accuracy_gt5:.1f} | {100*(g.pairwise_accuracy_gt5-p.pairwise_accuracy_gt5):+.1f} pp |

[RESULTADO EXPERIMENTAL] Bootstrap apareado por 70 fórmulas, promediando primero las tres semillas: ΔMAE de pares {pair_ci.difference_mean:+.2f} meV/átomo (IC 95 % [{pair_ci.ci95_lower:+.2f}, {pair_ci.ci95_upper:+.2f}]); ΔMAE residual {residual_ci.difference_mean:+.2f} [{residual_ci.ci95_lower:+.2f}, {residual_ci.ci95_upper:+.2f}]; Δacierto de pares {100*rank_ci.difference_mean:+.2f} pp [{100*rank_ci.ci95_lower:+.2f}, {100*rank_ci.ci95_upper:+.2f}]. Los tres intervalos incluyen cero.

[HECHO VERIFICADO] El control SLICES solo replica de cerca H3-C histórico: MAE absoluto {p.mae_energy_per_atom_mev:.2f} frente a {prior.atom_mae_mev:.2f}; MAE de pares {p.mae_delta_pairs_mev:.2f} frente a {prior.signed_pair_mae_mev:.2f} meV/átomo. Es un reentrenamiento de la MLP8 para el control apareado; no se sustituyó el Ridge ni se usó test para selección.

[INFERENCIA] La celda aporta una tendencia favorable pequeña en MAE de diferencias y orden, pero la incertidumbre no permite afirmar una mejora reproducible en estas 70 fórmulas. La salida relativa sigue comprimida (pendiente de pares {p.delta_slope:.3f} → {g.delta_slope:.3f}, lejos de 1). Añadir geometría no equivale a demostrar que sus siete descriptores causen la mejora: también cambian 56 parámetros de entrada.

[LIMITACIÓN] La procedencia es estática con endpoint limpiado, sin certificar relajación completa ni homogeneidad XC/U. El caché reconstruido pasó cotejo numérico pero no igualdad SHA-256. Las composiciones de desarrollo ya guiaron investigaciones anteriores; estos intervalos son exploratorios, no prueba externa ciega. No se utilizó el test histórico.

## Reproducción y recursos

Los archivos `structures_development.jsonl`, `geometry_cell_development.csv`, `staging_repo/recovery_manifest.json`, `results/manifest_experimento.json`, `results/metricas/` y `results/figuras/` conservan IDs, predicciones, parámetros, hashes y gráficos PNG. `resources_summary.csv` registra tiempo CPU, reloj y RSS por fase; `results/seleccion_y_entrenamiento_por_fold.csv` registra tiempo y RSS observado de las 30 corridas.

Entorno: Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}, pymatgen disponible; CPU con dos threads, sin GPU.
"""
    provenance_path = report / "geometry_provenance_audit.md"
    science_path = report / "geometry_scientific_report.md"
    provenance_path.write_text(provenance, encoding="utf-8")
    science_path.write_text(scientific, encoding="utf-8")
    result = {"status": "EXPLORATORY_PASS", "physical_structures": 210, "compositions": 70,
              "conditions": 2, "folds": 5, "seeds": [42, 123, 2026], "prediction_rows": 1260,
              "geometry_source": link["geometry_source"], "geometry_csv_sha256": sha256(report / "geometry_cell_development.csv"),
              "jsonl_sha256": sha256(report / "structures_development.jsonl"),
              "predictions_sha256": sha256(results / "predicciones_oof_por_estructura.csv"),
              "feature_cache_policy": "numeric parity, binary hash differs", "historical_test_accessed": False,
              "ridge_retrained": False, "backbone_finetuned": False,
              "figure_png_hashes": {path.name: sha256(path) for path in figures},
              "report_hashes": {path.name: sha256(path) for path in (provenance_path, science_path)},
              "resources_summary_sha256": sha256(resources)}
    (report / "final_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    LOG.info(json.dumps({"event": "geometry_finalized", "status": result["status"], "figures": len(figures)}))
    return result


def main() -> None:
    """Valida raíz y consolida el experimento sin nuevas corridas de modelo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    finalize(args.repo_root)


if __name__ == "__main__":
    main()
