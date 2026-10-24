#!/usr/bin/env python3
"""Compone el reporte Markdown desde salidas auditablemente producidas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR, sha256_file, write_json


def load_json(path: Path) -> dict[str, Any]:
    """Carga JSON requerido por el reporte.

    Args:
        path: JSON de salida de auditoría.

    Returns:
        Contenido JSON como diccionario.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> Path:
    """Genera conclusiones limitadas al alcance y la certeza de las fuentes.

    Returns:
        Ruta del informe Markdown.
    """
    corpus = load_json(REPORT_DIR / "corpus_statistics/corpus_schema.json")
    chalc = load_json(REPORT_DIR / "corpus_statistics/chalcogenide_statistics.json")
    abx = load_json(REPORT_DIR / "corpus_statistics/abs3_corpus_statistics.json")
    split = load_json(REPORT_DIR / "corpus_statistics/reconstructed_split_summary.json")
    overlap = load_json(REPORT_DIR / "domain_overlap/composition_overlap_summary.json")
    structural = load_json(REPORT_DIR / "structural_overlap/structural_overlap_summary.json")
    error_summary = load_json(REPORT_DIR / "error_correlations/analysis_summary.json")
    corr = pd.read_csv(REPORT_DIR / "error_correlations/spearman_bootstrap.csv")
    seen = pd.read_csv(REPORT_DIR / "error_correlations/seen_vs_unseen_descriptive.csv")
    elements = pd.read_csv(REPORT_DIR / "corpus_statistics/element_frequencies.csv")
    element_counts = elements.set_index("element").to_dict(orient="index")
    train_domain = pd.read_csv(REPORT_DIR / "corpus_statistics/chalcogenide_counts_by_reconstructed_split.csv")
    abx_split = pd.read_csv(REPORT_DIR / "corpus_statistics/abs3_counts_by_reconstructed_split.csv")
    formulas = pd.read_csv(REPORT_DIR / "domain_overlap/abs3_composition_overlap.csv")
    manifest = load_json(REPORT_DIR / "manifest.json")
    manifest.update({"status": "PARTIAL", "corpus_sha256": corpus["sha256"],
                     "corpus_rows": corpus["rows"], "split_reconstruction_status": split["status"],
                     "composition_overlap_status": overlap["status"],
                     "structural_overlap_status": structural["status"],
                     "error_association_status": error_summary["status"],
                     "figures": sorted(p.name for p in (REPORT_DIR / "figures").glob("*.png")),
                     "test_labels_read": False})
    manifest["corpus_schema_sha256"] = sha256_file(REPORT_DIR / "corpus_statistics/corpus_schema.json")
    write_json(REPORT_DIR / "manifest.json", manifest)

    def atom_rows(el: str, split_name: str) -> tuple[int, int]:
        row = train_domain[(train_domain.split_candidate == split_name) & (train_domain.measure == f"contains_{el}")].iloc[0]
        return int(row.material_rows), int(row.unique_formula_keys)

    def abx_row(family: str, scope: str) -> tuple[int, int]:
        if scope == "full":
            suffix = {"ABS3": "ABS3", "ABSe3": "ABSe3", "ABTe3": "ABTe3"}[family]
            return int(abx[f"{suffix}_rows"]), int(abx[f"{suffix}_unique_formulas"])
        row = abx_split[(abx_split.family == family) & (abx_split.split_candidate == scope)]
        return (int(row.material_rows.iloc[0]), int(row.unique_formula_keys.iloc[0])) if len(row) else (0, 0)

    corr_train = corr[corr.exposure_measure == "train_candidate_rows_exact_formula"]
    error_lines = []
    for model in ["H0", "H2", "R3-DROP30", "R4-SELECTED-ABSOLUTE"]:
        abs_row = corr_train[(corr_train.model == model) & (corr_train.outcome == "MAE_absolute_mev_atom")].iloc[0]
        rel_row = corr_train[(corr_train.model == model) & (corr_train.outcome == "MAE_delta_GS_anchored_mev_atom")].iloc[0]
        error_lines.append(f"| {model} | {abs_row.spearman_rho:.2f} [{abs_row.bootstrap_95_ci_low:.2f}, {abs_row.bootstrap_95_ci_high:.2f}] | {rel_row.spearman_rho:.2f} [{rel_row.bootstrap_95_ci_low:.2f}, {rel_row.bootstrap_95_ci_high:.2f}] |")
    seen_lines = []
    for model in ["H0", "H2", "R3-DROP30", "R4-SELECTED-ABSOLUTE"]:
        row = seen[(seen.model == model) & (seen.metric == "MAE_absolute_mev_atom")].iloc[0]
        seen_lines.append(f"| {model} | {row.seen_mean:.1f} | {row.unseen_mean:.1f} | {row.mean_difference_seen_minus_unseen:.1f} [{row.bootstrap_95_ci_low:.1f}, {row.bootstrap_95_ci_high:.1f}] |")
    dev = formulas[formulas.historical_split.isin(["train", "val"])]
    test_formula = formulas[formulas.historical_split.eq("test")]
    element_lines = []
    for element in ["S", "Se", "Te", "Ba", "Sr", "Ca", "Ti", "Zr", "Hf", "Sn", "Pb", "V", "Nb"]:
        row = element_counts.get(element, {"atom_occurrences": 0, "material_rows": 0, "unique_formula_keys": 0})
        element_lines.append(f"| {element} | {int(row['atom_occurrences']):,} | {int(row['material_rows']):,} | {int(row['unique_formula_keys']):,} |")
    table_abx = []
    for family in ["ABS3", "ABSe3", "ABTe3"]:
        full = abx_row(family, "full"); train = abx_row(family, "train_candidate"); val = abx_row(family, "validation")
        table_abx.append(f"| {family} | {full[0]} ({full[1]}) | {train[0]} ({train[1]}) | {val[0]} ({val[1]}) |")

    text = f"""# Auditoría del dominio de preentrenamiento de MatGPT

## Estado: PARTIAL — exposición química cuantificada; procedencia efectiva del split no demostrada

## Resumen

El CSV disponible coincide por nombre con el archivo que declara la configuración unconditional. Contiene **{corpus['rows']:,} registros**, con **{corpus['valid_slices']:,} cadenas SLICES válidas** y cero nulos en las columnas. El SHA-256 del CSV es `{corpus['sha256']}`. El vocabulario/checkpoint y el commit de código local coinciden con los valores esperados; el checkpoint conserva el SHA-256 `{manifest['provenance']['checkpoint_sha256']}`.

La configuración fija semilla **20260910**, y el código local reproduce `torch.randperm` seguido de un corte 90:10. Aplicado al archivo disponible, el split candidato tiene **{split['train_candidate_rows']:,} filas de train** y **{split['validation_candidate_rows']:,} de validación**. Lo llamamos *candidato* porque los metadatos del checkpoint no identifican el hash ni el orden exacto del CSV de entrenamiento; no se puede asegurar que estas filas sean las vistas históricamente.

## Calcógenos en el CSV completo

| Elemento | Ocurrencias atómicas | Registros con elemento | Fórmulas reducidas únicas |
|---|---:|---:|---:|
{chr(10).join(element_lines)}

De **{corpus['rows']:,}** registros válidos, **{chalc['rows_with_any_S_Se_Te']:,} ({100*chalc['fraction_with_any_S_Se_Te']:.2f}%)** contienen al menos uno de S, Se o Te. Las categorías S, Se y Te se superponen; no deben sumarse como categorías exclusivas. En el train candidato hay {atom_rows('S','train_candidate')[0]:,} registros con S, {atom_rows('Se','train_candidate')[0]:,} con Se, {atom_rows('Te','train_candidate')[0]:,} con Te y {int(train_domain[(train_domain.split_candidate=='train_candidate') & (train_domain.measure=='contains_any_S_Se_Te')].material_rows.iloc[0]):,} con cualquier calcógeno. Son conteos elementales; **no clasifican cada compuesto como sulfuro iónico**.

## Estequiometrías ABX₃

Conteos de registros y entre paréntesis número de fórmulas reducidas distintas:

| Familia por composición | CSV completo | Train candidato | Validación candidata |
|---|---:|---:|---:|
{chr(10).join(table_abx)}

La frecuencia de ABS3 es **{100*abx['ABS3_rows']/corpus['rows']:.4f}%** del CSV completo y **{100*abx['P_ABS3_given_contains_S_full_corpus']:.3f}%** entre los registros con S. En el train candidato, ABS3 representa **{100*abx['ABS3_rows_train_candidate']/split['train_candidate_rows']:.4f}%** de las filas, o **{100*abx['P_ABS3_given_contains_S_train_candidate']:.3f}%** entre las filas que contienen S. Esta fórmula solo indica composición; no demuestra una estructura perovskita. No se obtuvo evidencia para clasificar topologías geométricas del corpus.

## Solapamiento con ABS₃ de la tesis

El manifiesto histórico contiene 70 composiciones de desarrollo y ocho de test. El cotejo químico usó fórmulas y asignaciones de split; **no cargó etiquetas de energía del test**.

- **11/70** fórmulas de desarrollo tienen al menos un registro exacto en el train candidato; **12/70** aparecen en el CSV completo. Una fórmula adicional aparece solo en la validación candidata.
- **49/70** no tienen coincidencia exacta, pero sí aparece al menos una composición con el mismo conjunto de elementos y otra estequiometría.
- **9/70** no tienen coincidencia exacta ni otra fórmula con ese mismo conjunto de elementos.
- En el control permitido por fórmula del test histórico, **2/8** aparecen en el train candidato. Esto documenta posible exposición composicional; no se consultaron sus energías ni se usaron para selección.
- Entre las **210 estructuras de desarrollo**, hubo **0 coincidencias exactas de cadena SLICES** con el corpus. La tabla de pretraining no contiene CIF; no se pudo verificar equivalencia geométrica con `StructureMatcher`. Cero coincidencias textuales no demuestra ausencia de topologías parecidas.

## Exposición candidata frente a errores OOF de desarrollo

El cruce usa exclusivamente 70 composiciones de desarrollo y resultados OOF ya guardados para H0, H2, R3-DROP30 y R4-SELECTED-ABSOLUTE. La exposición es el número de filas exactas por fórmula en el **train candidato** reconstruido. Cada intervalo es bootstrap percentil del 95% por composición, **2,000 remuestras**, semilla `{error_summary['bootstrap_seed']}`.

| Modelo | Spearman: exposición vs MAE absoluto | Spearman: exposición vs MAE ΔE DFT-GS anclada |
|---|---:|---:|
{chr(10).join(error_lines)}

La asociación con MAE absoluto es positiva en H0/H2/R3, mientras que R4 es cercana a cero; la asociación con el error de ΔE incluye cero en los cuatro casos. El grupo con coincidencia exacta (11 fórmulas) tampoco muestra menor MAE absoluto promedio:

| Modelo | MAE absoluto exacta vista | MAE absoluto sin match exacto | Diferencia (vista − no vista), IC bootstrap 95% |
|---|---:|---:|---:|
{chr(10).join(seen_lines)}

Esto no demuestra que la exposición empeore el modelo. Los grupos son pequeños, el split es candidato, la dificultad química puede confundir la asociación, y R1–R4 fueron seleccionados usando parte de development. Las correlaciones son exploratorias y no causales. No encontramos señal de que más ejemplos de fórmula exacta reduzcan el error de ranking; para H0/H2/R3 los rangos de ΔE no muestran relación precisa con frecuencia.

## Lectura de las hipótesis

- **H-DOM1 (calcógenos ausentes/subrepresentados):** rechazada en su versión de ausencia: 26.87% de filas contienen S/Se/Te. No se evalúa si son subrepresentados respecto de una distribución de referencia adecuada.
- **H-DOM2 (pocos ABS₃):** compatible con los conteos: 48 registros ABS3 en el CSV completo, 41 en el train candidato y 33 fórmulas distintas en ese train. La exposición de las composiciones objetivo exactas es baja (11/70 candidatas), aunque la exposición química por elementos es amplia.
- **H-DOM3 (falta de diversidad topológica):** no resuelta. Hay variación en registros por composición, pero el CSV no aporta CIF ni clasificación polimórfica comparable. Las cadenas SLICES exactas objetivo no se repiten.
- **H-DOM4 (objetivo generativo no optimizado para ΔE):** sigue siendo plausible, no probado por esta auditoría. La exposición exacta no se asocia de forma clara con el error de ΔE; el objetivo de preentrenamiento y la supervisión energética pueden ser determinantes, junto con representación, cabeza y tamaño de muestra.

## Conclusión científica

Hay exposición química sustancial a calcógenos en el corpus disponible, pero poca coincidencia exacta con las composiciones ABS3 objetivo en el train candidato y ninguna repetición exacta de las SLICES de desarrollo. Los datos son compatibles con **cambio de dominio composicional a nivel de fórmula**, pero no prueban ausencia de estructuras/topologías parecidas ni que el domain shift cause el MAE observado. Las asociaciones por fórmula no apuntan a menor error al aumentar la frecuencia exacta; por tanto, el resultado favorece investigar el objetivo de preentrenamiento y qué información estructural retiene el embedding, sin descartar que una adaptación de dominio futura pueda ayudar.

**No recomendaría todavía un reentrenamiento completo.** El siguiente paso probatorio sería verificar con el proveedor el hash y orden del CSV histórico, y obtener CIF/metadatos estructurales del corpus para comparar prototipos. Si se confirma baja cobertura de ABS3 específica, evaluar después una adaptación causal SLICES controlada, sin mezclarla con el benchmark energético.

## Alcance y limitaciones

1. `CONFIG_REPRODUCED`, no `VERIFIED`: el checkpoint no trae hash del corpus ni manifiesto de filas; la semilla no acredita el archivo exacto usado.
2. El CSV carece de CIF y de energía DFT para esta tarea. No se calculó equivalencia geométrica ni se interpretó `e_form` como energía de polimorfos.
3. El parser lee especies del bloque inicial de sitios y valida que las aristas posteriores sean tripletes con índices y vectores válidos. Aceptó todas las filas. Los conteos químicos no identifican valencia o carácter iónico.
4. La validación de patrones OOF reutiliza métricas del estudio R1–R4 sobre el mismo desarrollo donde se hizo selección; no es una evaluación confirmatoria.
5. El test histórico se usó únicamente para cotejo de fórmula como control de posible contaminación. Sus etiquetas DFT no se leyeron.
6. No se entrenó ni modificó MatGPT.

## Figuras

Todas las figuras son PNG creadas desde las tablas de esta auditoría. Incluyen denominadores y límites de interpretación en las notas.

{chr(10).join(f'- ![Figura: {p.stem}](figures/{p.name})' for p in sorted((REPORT_DIR / 'figures').glob('*.png')))}

## Artefactos reproducibles

- [Manifiesto](manifest.json)
- [Procedencia](provenance/provenance.json)
- [Esquema del corpus](corpus_statistics/corpus_schema.json)
- [Estadísticas de calcógenos](corpus_statistics/chalcogenide_statistics.json)
- [Estadísticas ABX₃](corpus_statistics/abs3_corpus_statistics.json)
- [Asignación candidata torch.randperm](corpus_statistics/reconstructed_split_assignments.csv)
- [Solapamiento composicional](domain_overlap/abs3_composition_overlap.csv)
- [Solapamiento SLICES exacto](structural_overlap/structural_overlap.csv)
- [Correlaciones OOF por composición](error_correlations/spearman_bootstrap.csv)

## Referencias

- MatGPT, código oficial y configuración unconditional: [repositorio](https://github.com/devalab/matgpt), [configuración](https://github.com/devalab/matgpt/blob/main/mat-concat/config/uncond.yaml), [entrenamiento](https://github.com/devalab/matgpt/blob/main/mat-concat/train.py). La auditoría usa el commit local fijado `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- Dataset asociado por la documentación del proyecto: [Figshare DOI 10.6084/m9.figshare.34330655](https://doi.org/10.6084/m9.figshare.34330655). El archivo local se verificó por nombre/configuración y SHA; no hay hash de publicación contrastado.
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- Witman y Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D).
- MatterGPT Zenodo es un dataset distinto: [doi:10.5281/zenodo.16879675](https://doi.org/10.5281/zenodo.16879675); no se usó como sustituto.
"""
    report_path = REPORT_DIR / "Pretraining_Domain_Audit.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    """Construye el informe final desde artefactos presentes."""
    print(build_report())


if __name__ == "__main__":
    main()
