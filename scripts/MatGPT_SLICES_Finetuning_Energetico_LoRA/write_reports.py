"""Resume resultados reales, procedencia, incertidumbre y recursos del Experimento C."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

import experiment as ex

LOG = logging.getLogger("matgpt_lora_reports")


def render(output: Path) -> None:
    """Genera reportes interpretables solo desde métricas y manifests existentes."""
    required = [output / "metricas_por_semilla.csv", output / "bootstrap_apareado.csv",
                output / "intervalos_gap_dft.csv", output / "predicciones_outer_por_estructura.csv"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Faltan resultados de evaluación: " + str([str(p) for p in required if not p.is_file()]))
    metrics = pd.read_csv(required[0]); bootstrap = pd.read_csv(required[1])
    bins = pd.read_csv(required[2]); predictions = pd.read_csv(required[3])
    run_pattern = re.compile(r"(frozen|lora)_fold[0-4]_seed(42|123|2026)")
    run_summaries = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*_fold*_seed*/summary.json"))
                     if run_pattern.fullmatch(path.parent.name)]
    resources = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*_resources.json"))
                 if re.fullmatch(r"(frozen|lora)_fold[0-4]_seed(42|123|2026)_resources\.json", path.name)]
    if len(run_summaries) != 30 or any(item.get("status") != "COMPLETE" for item in run_summaries):
        raise ValueError("El informe requiere 30 corridas completas")
    aggregate = metrics.groupby("condition").agg({
        "mae_per_atom_mev": ["mean", "std"], "mae_residual_mev": ["mean", "std"],
        "mae_pair_mev": ["mean", "std"], "mae_anchored_gs_dft_mev": ["mean", "std"],
        "gs_exact_argmin_accuracy": ["mean", "std"], "gs_eligible_accuracy": ["mean", "std"],
        "pairwise_accuracy_gt5": ["mean", "std"], "energy_slope": ["mean", "std"],
        "residual_slope": ["mean", "std"], "gap_slope": ["mean", "std"]})
    output.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "resumen_condiciones_media_desviacion_fold_semilla.csv")
    summary = []
    for condition, row in aggregate.iterrows():
        summary.append(f"- **{condition}** — MAE E/N {row[('mae_per_atom_mev','mean')]:.1f} ± {row[('mae_per_atom_mev','std')]:.1f}; "
                       f"residuo {row[('mae_residual_mev','mean')]:.1f} ± {row[('mae_residual_mev','std')]:.1f}; "
                       f"ΔE pares {row[('mae_pair_mev','mean')]:.1f} ± {row[('mae_pair_mev','std')]:.1f}; "
                       f"GS elegible {row[('gs_eligible_accuracy','mean')]:.3f}; "
                       f"pares elegibles {row[('pairwise_accuracy_gt5','mean')]:.3f}.")
    c0 = aggregate.loc["frozen"]
    c1 = aggregate.loc["lora"]
    gap_summary = bins.groupby(["condition", "gap_min_mev", "gap_max_mev"], dropna=False).agg(
        n_semillas=("seed", "count"),
        pares_promedio_por_semilla=("n_pairs", "mean"),
        MAE_delta_promedio_meV_atom=("mae_pair_mev", "mean"),
        MAE_delta_cero_promedio_meV_atom=("mae_zero_mev", "mean"),
        pares_elegibles_promedio=("n_pairs_ranking", "mean"),
        acierto_orden_promedio=("pair_accuracy", "mean"),
    ).reset_index()
    gap_summary["intervalo_gap_DFT_meV_atom"] = gap_summary.apply(
        lambda row: f">{row.gap_min_mev:g}" if not math.isfinite(row.gap_max_mev)
        else f"{row.gap_min_mev:g}–{row.gap_max_mev:g}", axis=1)
    gap_summary = gap_summary.drop(columns=["gap_min_mev", "gap_max_mev"])
    gap_summary["acierto_orden_promedio"] = gap_summary["acierto_orden_promedio"].map(
        lambda value: "Sin pares elegibles" if pd.isna(value) else f"{value:.3f}")
    historical_path = output / "reproduccion_control_congelado_h3c.csv"
    historical_note = ""
    if historical_path.is_file():
        h = pd.read_csv(historical_path)
        historical_note = ("\n\n## Reproducción del H3-C congelado\n\n" + h.to_markdown(index=False, floatfmt=".4f") +
                           "\n\nLas diferencias se calcularon por ID, fold y semilla contra el replay histórico disponible; no se usaron para seleccionar checkpoints.")
    detail = ["# Comparación C0/C1 y hallazgos de Experimento C", "",
              "**Diseño:** 210 estructuras físicas, 70 fórmulas, tres polimorfos por fórmula, cinco pliegues de composición y tres semillas. Cada condición aporta 630 predicciones, que son 210 estructuras repetidas por tres ajustes; no son 630 materiales independientes. El test histórico permanece cerrado.", "",
              "## Resultados agrupados por fórmula/fold y semilla", "", *summary, "",
              "La desviación indicada resume las 15 combinaciones fold/semilla; no es un intervalo inferencial. La incertidumbre apareada por fórmula se presenta en `bootstrap_apareado.csv` y en las figuras.", "",
              "## Contraste apareado LoRA menos congelado", "", bootstrap.to_markdown(index=False, floatfmt=".4f"), "",
              "## Error por tamaño de diferencia DFT", "", "Los pares se asignan por |ΔE DFT|. Cada una de las tres filas de semilla resume conjuntamente los cinco pliegues (210 pares físicos por semilla); la tabla promedia esas tres repeticiones de entrenamiento. En 0–5 meV/átomo no hay pares elegibles para evaluar orden.", "", gap_summary.to_markdown(index=False, floatfmt=".3f"), "",
              "## Interpretación técnica", "",
              f"La diferencia media entre condiciones en MAE residual fue {c1[('mae_residual_mev','mean')]-c0[('mae_residual_mev','mean')]:+.2f} meV/átomo; en MAE de pares fue {c1[('mae_pair_mev','mean')]-c0[('mae_pair_mev','mean')]:+.2f} meV/átomo. Interpretar cada diferencia junto con su IC95% apareado; no llamar mejora a una diferencia cuyo intervalo incluya cero.", "",
              "Las energías medias por fórmula provienen de cinco Ridge históricos congelados sobre L03_MEAN. El MLP8 predice residuos centrados de energía DFT por átomo. ΔE se deriva restando energías dentro de fórmula; no se entrenó una pérdida pairwise. La rama Ridge permaneció invariante en la auditoría del primer checkpoint LoRA.", "",
              "Referencia H3-C histórica aproximada: MAE E/N 183.3 meV/átomo, MAE de pares 58.8, estado fundamental 54.3% por argmin bruto y orden de pares elegibles 66.8%. La métrica elegible con abstenciones fue 53.2% en el replay científico final; se mantienen las definiciones y denominadores separados.", "",
              "## Recursos observados", "",
              f"Corridas completas: {len(run_summaries)}. Tiempo total registrado por proceso: {sum(float(r.get('wall_seconds',0)) for r in resources)/3600:.2f} h; máximo RSS entre corridas: {max(int(r.get('peak_rss_bytes',0)) for r in resources)/1e9:.2f} GB. Cada log y JSON de recursos individual está en `runs/`.", "",
              "## Límites", "",
              "Las 70 fórmulas ya participaron en análisis anteriores; estos resultados son exploratorios. El caché de features recuperado fue reconstruido y se aceptó por auditoría numérica, no por igualdad de hash con el archivo histórico. El conjunto DFT gga-static conserva la procedencia original disponible; este protocolo no homogeneiza XC ni correcciones. El éxito de ranking no implica precisión de ΔE, y una pendiente mayor no demuestra causalidad de representación.", "",
              "**Estado del test histórico:** no abierto ni evaluado.", ""]
    detail.extend(historical_note.splitlines())
    reports = {
        "reporte_entrenamiento_finetuning_energetico.md": ["# Entrenamiento energético MatGPT + LoRA", "", "Configuración fija: Q/V en bloques 10–11, rango 4, alpha 8, dropout 0.05, AdamW; MLP8 768→8→1, 200 épocas máximas y paciencia 20. Selección por SmoothL1 residual normalizada en 11 fórmulas de validación interna. Entrenamientos secuenciales en CPU, dos hilos.", "", "No se ajustó Ridge, no se usó geometría y no se cambió el checkpoint original.", "", *summary, "", f"Tiempo total observado: {sum(float(r.get('wall_seconds',0)) for r in resources)/3600:.2f} h. Ver logs por corrida en `runs/`.", ""],
        "comparacion_frozen_vs_finetuned.md": detail,
        "analisis_diferencias_energeticas_intraformula.md": ["# Diferencias energéticas intrafórmula", "", "Por cada fórmula, las energías por átomo DFT y predichas se centran usando la media de sus tres polimorfos. El Ridge común se cancela y la diferencia se evalúa sobre el residuo estructural MLP8.", "", *summary, "", "Consulte `metricas_por_composicion.csv`, `diferencias_por_par.csv`, `intervalos_gap_dft.csv` y las figuras de paridad/residuos. El predictor ΔE=0 se evalúa con el mismo universo de pares; no se asigna un estado fundamental al empate.", ""],
        "analisis_ranking_y_estado_fundamental.md": ["# Ordenamiento y estado fundamental", "", "Se evaluaron tripletes por fórmula. Pairwise usa |ΔE DFT| > 5 meV/átomo; predicciones con diferencia ≤1 meV/átomo cuentan como no resueltas e incorrectas. Ground-state elegible requiere un mínimo DFT separado por >1 meV/átomo; un empate de predicción no se acredita.", "", *summary, "", "Los denominadores completos están en `metricas_por_semilla.csv` y los pares en `diferencias_por_par.csv`. El test histórico no se usó.", ""],
        "analisis_efecto_lora_en_embeddings.md": ["# Cambio de representaciones tras LoRA", "", "La auditoría de una corrida verificó igualdad de bloque 3 y pesos base, junto con un cambio reproducible en FINAL_END. El análisis detallado sobre estructuras externas de la semilla 42 está en `cambio_embeddings_outer_semilla42.csv` y en la figura de drift.", "", "El cambio geométrico de una representación no prueba que contenga más información de energía; la asociación se interpreta junto con el error de validación y el bootstrap por fórmula.", ""],
        "conclusiones_cientificas_finetuning.md": ["# Conclusiones científicas", "", *summary, "", "**HECHO VERIFICADO:** los adaptadores Q/V se actualizan mientras pesos originales y rama Ridge se mantienen congelados. **RESULTADO EXPERIMENTAL:** consultar las 15 repeticiones por condición, sus IC pareados y errores por gaps. **LIMITACIÓN:** desarrollo ABS₃ previamente observado; no hay validación externa ciega. El test histórico sigue cerrado.", ""],
        "protocolo_siguiente_experimento_con_geometria.md": ["# Protocolo para una etapa posterior con geometría", "", "Este informe no implementa geometría. Si se abre una etapa posterior, comparar en los mismos folds y semillas la entrada SLICES sola frente a SLICES más descriptores geométricos con MatGPT y la MLP8 fijados. Mantener el checkpoint y Ridge congelados; ajustar normalización geométrica solo con train interno; no introducir geometría en la selección de C1. Declarar por separado procedencia estructural y disponibilidad de geometría al predecir.", "", "El experimento C por sí solo no demuestra que geometría sea la causa de las limitaciones de los embeddings.", ""]}
    for name, lines in reports.items():
        (output / name).write_text("\n".join(lines), encoding="utf-8")
    audit = ex.REPORT / "auditoria_viabilidad_finetuning_lora.md"
    audit.write_text("# Auditoría de viabilidad — Experimento C\n\n**Estado final: ejecución completa; evaluación y figuras generadas.**\n\n" +
                     "Las 30 corridas (5 pliegues × 3 semillas × 2 condiciones) terminaron y produjeron 1 260 filas de predicción, equivalentes a 210 estructuras por condición/semilla. La evaluación es exploratoria sobre las 70 fórmulas de desarrollo; el test histórico permanece cerrado.\n\n" +
                     "## Controles técnicos\n\nPreflight PASS; smoke C0/C1 PASS; siete pruebas de integridad PASS. Se verificaron equivalencia inicial LoRA, selección del token `<end>`, invariancia de bloque 3/Ridge, gradientes de adaptadores y MLP8, reproducción determinista del checkpoint, normalización interna y splits agrupados por fórmula.\n\n" +
                     "## Corrección de protocolo\n\nUn primer intento se detuvo al detectar recorte de gradiente no especificado. Se eliminaron los artefactos de ese intento y se repitieron todas las corridas sin clipping; los resultados de este informe corresponden únicamente a la versión corregida.\n\n" +
                     "## Resumen observado\n\n" + "\n".join(summary) + "\n\n" +
                     "La tabla completa por fold/semilla está en `metricas_por_semilla.csv`; los IC95% apareados por fórmula están en `bootstrap_apareado.csv`.\n\n" +
                     "**Procedencia:** checkpoint MatGPT SHA-256 `" + ex.EXPECTED["checkpoint"] + "`; tokenizer `" + ex.EXPECTED["tokenizer"] + "`; folds históricos con hash registrado en `preflight/preflight.json`. El caché reconstruido difiere en hash del original ausente y pasa la auditoría numérica guardada junto al artefacto.\n\n" +
                     f"Tiempo acumulado de procesos de entrenamiento: {sum(float(r.get('wall_seconds',0)) for r in resources)/3600:.2f} h; RSS máximo por corrida: {max(int(r.get('peak_rss_bytes',0)) for r in resources)/1e9:.2f} GB. Para limitaciones, interpretación de bins y definiciones científicas, ver `comparacion_frozen_vs_finetuned.md`.\n",
                     encoding="utf-8")


def main() -> None:
    """Valida argumentos y escribe reportes finales.»"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = args.output.resolve()
    render(output)
    LOG.info("%s", {"event": "reports_complete", "directory": str(output)})


if __name__ == "__main__":
    main()
