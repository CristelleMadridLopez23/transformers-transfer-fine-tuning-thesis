#!/usr/bin/env python3
"""Resume resultados D1/D3 en informes explicativos con límites claros."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import platform

import sklearn
import torch

import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development, sha256, write_json

LOG=logging.getLogger("matgpt_diagnostics.reports")


def run() -> None:
    """Genera reportes D1 y D3 y bootstrap pareado de D1 vs E2-A."""
    d1=REPORT/"D1/v3";d3=REPORT/"D3/v5";common=REPORT/"common"
    metrics=pd.read_csv(d1/"metrics.csv")
    summary=metrics.groupby("configuration").agg(
        mae_mean=("absolute_mae_mev_atom","mean"),mae_sd_folds=("absolute_mae_mev_atom","std"),
        compmean_mae=("composition_mean_absolute_error_mev_atom","mean"),
        r2_mean=("absolute_r2_pooled_structures","mean"),delta_mae=("delta_e_mae_mev_atom","mean"),
        delta_sd=("delta_e_mae_mev_atom","std"),gs=("ground_state_accuracy","mean"),
        pairwise=("pairwise_accuracy","mean"),n_gs=("n_gs_eligible","mean")).reset_index()
    paired=pd.read_csv(d1/"paired_e2a_comparison.csv")
    ridge=paired[paired.configuration.eq("D1-B Ridge composición")]
    values=ridge.groupby("composition_group").d1_minus_e2a_mae_mev_atom.mean().to_numpy()
    rng=np.random.default_rng(17042026)
    draws=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    bootstrap={"comparison":"D1-B Ridge formula minus E2-A partial_ft mean prediction",
               "metric":"difference in composition-macro absolute MAE (meV/atom); negative favors Ridge",
               "mean_difference":float(values.mean()),"median_difference":float(np.median(values)),
               "ci95_percentile":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
               "n_compositions":int(len(values)),"resamples":2000,"seed":17042026,
               "note":"Each composition contributes its three polymorphs together; exploratory paired interval."}
    write_json(common/"D1_vs_E2A_paired_bootstrap.json",bootstrap)
    table="| Baseline | MAE absoluto (meV/átomo), media ± DE entre folds | MAE de medias por fórmula (meV/átomo) | MAE ΔE (meV/átomo), media ± DE folds | R² absoluto medio por fold | GS / pairwise |\n|---|---:|---:|---:|---:|---:|\n"
    for _,r in summary.iterrows():
        gs="no discriminativo" if pd.isna(r.gs) else f"{r.gs:.3f} / {r.pairwise:.3f}"
        table+=f"| {r.configuration} | {r.mae_mean:.1f} ± {r.mae_sd_folds:.1f} | {r.compmean_mae:.1f} | {r.delta_mae:.2f} ± {r.delta_sd:.2f} | {r.r2_mean:.3f} | {gs} |\n"
    data,_,_=load_development()
    energy=data.energy_per_atom_ev.to_numpy(float)
    comp_mean=data.groupby("reduced_formula").energy_per_atom_ev.transform("mean").to_numpy(float)
    between=float(np.var(comp_mean));within=float(np.mean((energy-comp_mean)**2));fraction=between/(between+within)
    pd.DataFrame([{"variance_between_composition_means_ev2_atom2":between,
                   "mean_within_composition_variance_ev2_atom2":within,
                   "total_variance_ev2_atom2":float(np.var(energy)),
                   "fraction_between":fraction,"fraction_within":1-fraction,
                   "n_structures":len(data),"n_compositions":data.reduced_formula.nunique()}]).to_csv(d1/"variance_decomposition.csv",index=False)
    d1text=f"""# D1 — ¿Cuánto explica la composición sin ver la estructura?

## En palabras sencillas

Ridge recibe únicamente la fórmula química. Para una fórmula dada, los tres polimorfos reciben exactamente las mismas features y, por ello, la misma predicción. El modelo puede acertar la escala absoluta al comparar una composición con otra, pero no tiene cómo decidir cuál de los tres polimorfos es más estable. Ese empate es la respuesta esperada de un baseline sin estructura.

## Datos y protocolo

Se usaron 210 energías estáticas DFT-PBE por átomo, de 70 composiciones de desarrollo con tres polimorfos cada una. Se conservaron los cinco folds agrupados por composición. Dentro de cada fold, Ridge eligió α entre {0.01, 0.1, 1, 10, 100} exclusivamente con inner-validation; el scaler se ajustó con inner-train. El Random Forest pequeño usó parámetros fijos (200 árboles, profundidad máxima 4, mínimo 3 muestras por hoja). La media de train predice la media de las etiquetas inner-train. No se incluyó el identificador de composición como feature y no se consultó el test histórico.

Features de composición: fracciones por elemento, número de elementos/átomos reducidos, estadísticos de número atómico y electronegatividad de Pauling disponibles. No se usaron energías, rankings, familia polimórfica, SLICES ni geometría como entrada.

## Resultados

{table}
El MAE absoluto de Ridge queda cerca de **{summary.loc[summary.configuration.eq('D1-B Ridge composición'),'mae_mean'].iloc[0]:.1f} meV/átomo**. Comparado por las mismas composiciones con E2-A partial_ft (predicción promediada entre sus tres semillas), la diferencia promedio Ridge − E2-A es **{values.mean():.1f} meV/átomo**; IC bootstrap pareado 95% **[{bootstrap['ci95_percentile'][0]:.1f}, {bootstrap['ci95_percentile'][1]:.1f}]**, 2,000 remuestras de 70 composiciones, semilla 17042026. Negativo favorece Ridge en error absoluto. Este intervalo es exploratorio, no elimina que configuraciones MatGPT se preseleccionaron con la validation histórica.

La fracción descriptiva de varianza entre medias de composición es **{fraction:.1%}**; la variación dentro de composición corresponde a **{1-fraction:.1%}**. Cada composición conserva los tres valores DFT reales. Esta descomposición no es causal: la variación entre fórmulas y la estructura están relacionadas en estos 70 grupos.

Para el baseline constante y Ridge, el rango predicho dentro de cada fórmula es cero. Por eso su ranking polimórfico es no discriminativo: el MAE de ΔE (que resulta aproximadamente el rango DFT medio dividido por tres) no debe leerse como capacidad de seleccionar un estado fundamental. Los valores GS y pairwise se conservan en cero/abstención explícita según la regla y no se interpretan como éxitos.

## Cómo leer las figuras

- `d1_paridad_composicion.png`: cada punto es una estructura OOF; DFT está en x y predicción en y. La diagonal discontinua es el ideal; rosa es OLS con intercepto libre sobre todos los puntos mostrados. En cada panel se muestran N, m, b, R² y MAE.
- `d1_ridge_vs_matgpt.png`: compara las mismas 210 estructuras. La gran diferencia de error absoluto muestra que, en este conjunto, la fórmula informa mucho sobre el nivel de energía. No prueba que las features sean universales ni transfiere a otras composiciones.
- `d1_error_por_composicion.png`: cada caja resume 70 errores macro por composición; los puntos externos son fórmulas individuales.
- `d1_varianza_entre_dentro.png`: separa descriptivamente dispersión entre medias de fórmulas y desviaciones de los polimorfos respecto a su propia media.

**Lectura científica:** en energía absoluta este benchmark da una señal química fuerte; el mal resultado absoluto de E2-A no se explica diciendo simplemente que la composición carece de información. La pérdida/head/calibración de MatGPT requieren diagnóstico. D1 confirma que el ranking necesita una fuente de señal estructural adicional, porque el baseline químico empata cada triplete por construcción.

## Archivos reproducibles

OOF, métricas, features, elección de α, regresiones OLS y provenance se guardan en este directorio. El test histórico no se usó. Los folds y validation histórica sí fueron usados para investigación previa de candidatas, así que estas auditorías de desarrollo no eliminan el sesgo histórico de selección.
"""
    (d1/"D1_report.md").write_text(d1text)
    # D3 summary and pairwise signal.
    summary=json.loads((d3/"d3_summary.json").read_text())
    corr=summary["spearman_vs_abs_dft_pair_gap"]
    collision=pd.read_csv(d3/"collision_analysis.csv")
    counts=collision.n_slices_strings.value_counts().to_dict()
    geom=pd.read_csv(d3/"geometry_descriptors.csv")
    values={key:f"ρ={value['spearman_rho']:.3f}, p ingenuo={value['p_value_exploratory']:.3g}" for key,value in corr.items()}
    d3text=f"""# D3 — ¿SLICES y MatGPT distinguen los polimorfos estáticos?

## En palabras sencillas

Se preguntaron tres cosas distintas: ¿cambia la cadena SLICES?, ¿cambia el embedding que produce el checkpoint?, y ¿esas diferencias siguen la diferencia de energía DFT? Que dos entradas se distingan no significa que el modelo sepa cuál es más estable.

## Auditoría de datos y modelo

Se analizaron los 210 casos de desarrollo: 70 composiciones y 3 estructuras estáticas por composición. El test histórico estuvo excluido. En **{(collision.n_slices_strings==3).sum()} de 70 composiciones**, los tres SLICES son diferentes; lo mismo ocurre para los tres hashes de input IDs. No hay colisión exacta de cadena/IDs entre los tres polimorfos estáticos en este subconjunto, ni composición con los tres strings iguales. El campo `topology_sha256` también se exportó, pero esta auditoría no ejecutó un algoritmo independiente de isomorfismo de grafos: por eso no declara equivalencia topológica a partir de la cadena solamente.

Los embeddings congelados son del checkpoint original unconditional, cargado con `strict=True`; SHA-256 **`{summary['embedding_audit']['checkpoint_sha256']}`**. El modelo se mantuvo en eval con cero parámetros entrenables. El vector `<end>` tiene 768 dimensiones; también se calculó masked-mean excluyendo `<start>`, `<end>` y padding. Al añadir padding derecho, el máximo cambio del vector `<end>` fue {summary['embedding_audit']['padding_end_max_abs_error']:.2e}, prácticamente cero. No se añadió energía a los tokens.

La geometría auxiliar se calculó desde la celda y las posiciones estáticas: volumen/átomo, densidad, distribución global y por par de especies de distancias periódicas, y número medio de vecinos por átomo bajo cortes de 2.5, 3.0, 3.5 y 4.0 Å. Esos descriptores no usan energía, fuerzas ni tensión. Las distancias y volúmenes no dependen del orden de sitios ni de una rotación rígida. Los cortes son descriptores exploratorios, no una definición química universal de coordinación. No se calcularon ángulos/entornos SOAP en esta pasada.

## Relación entre distancia de representación y diferencia energética

En los **210 pares intracomposición** (tres pares por fórmula), las correlaciones Spearman exploratorias entre distancia y diferencia absoluta DFT fueron: `<end>` euclidiana {values['end_embedding_euclidean']}; `<end>` coseno {values['end_embedding_cosine']}; masked-mean euclidiana {values['mean_embedding_euclidean']}; masked-mean coseno {values['mean_embedding_cosine']}; descriptor geométrico L1 {values['geometry_descriptor_l1']}.

Las correlaciones de masked-mean y geometría son positivas pero modestas; la de `<end>` es casi nula. La distancia geométrica L1 promedia diferencias absolutas solo en las features finitas compartidas por ambos miembros del par; la fracción se calcula sobre features presentes en al menos una de las dos estructuras. La cobertura se exporta aparte porque algunos pares de especies no tienen vecinos a ≤6 Å en una estructura. Los p-valores son descriptivos sin corrección por multiplicidad ni ajuste por dependencia: tres pares de la misma composición no son observaciones independientes. No se interpretan como pruebas de significancia ni de desempeño predictivo. Una representación puede separar estructuras y aun así ordenar mal su energía.

## Cómo leer las figuras

- `d3_distincion_slices.png`: cuenta cuántas cadenas de SLICES diferentes hay en los tres polimorfos de cada composición. En este conjunto, las 70 tienen tres cadenas distintas.
- `d3_distancias_y_deltae.png`: cada punto es un par de polimorfos de la misma fórmula; eje x es diferencia DFT absoluta en meV/átomo. Los paneles muestran distancia de embeddings y una métrica L1 geométrica de features mixtas. La nota recuerda que 210 pares se agrupan en 70 composiciones.

## Conexión con E2-T1

E2-T1 ya auditó 12,486 frames de relajación en desarrollo: 4,033 secuencias de IDs únicas y 1,148 de 1,677 grupos repetidos con rango energético ≥1 meV/átomo. Ese resultado describe configuraciones intermedias; no se reproduce ni se extrapola a los estáticos. Aquí no se encontró colisión exacta entre cadenas estáticas de los tres polimorfos, pero aun así la similitud geométrica/energética de los frames de relajación puede crear colisiones a lo largo de las trayectorias.

## Interpretación y siguiente experimento

D3 descarta una explicación simple de “los estáticos tienen exactamente el mismo input SLICES”: los 3 polimorfos por composición tienen cadenas e IDs distintos. Sin embargo, los embeddings `<end>` preentrenados no muestran una relación apreciable con |ΔE| en este análisis, y las asociaciones masked-mean/geométricas son solo exploratorias. Esto deja abierta una limitación de pooling/objetivo/cabeza, y también la posibilidad de que geometría aporte señal adicional. **D3 por sí solo no justifica afirmar causalidad ni que D4 vaya a mejorar.** Sí ofrece fundamento razonable para diseñar D4 como comparación controlada, si D2 confirma primero que el objetivo energético no es la explicación principal.

## Referencias

- SLICES, representación periódica invertible e invariante: Xiao et al., *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7). El artículo define SLICES como codificación de composición y topología; D3 aquí prueba si las diferencias geométricas estáticas se reflejan en cadenas/embeddings del dataset.
- MatGPT: [código oficial](https://github.com/devalab/matgpt), commit local E0 `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- SOAP: [documentación DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html). SOAP se menciona como alternativa publicada de descriptor geométrico local; no fue usado aquí (DScribe no está instalado).
- MPtrj/CHGNet: [documentación del conjunto MPtrj](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj) y Deng et al. (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3). Contexto de potenciales que aprenden energía/fuerzas/tensión desde estructuras; no es comparación de rendimiento en esta auditoría.
- PyTorch: [mecánicas de Autograd e inference mode](https://docs.pytorch.org/docs/stable/notes/autograd). El modo inference evita autograd; los embeddings de D3 solo se usan para análisis, no para optimizar cabezas.
- MatFold, protocolo de partición por composición: Witman y Schindler, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D). Referencia metodológica; los folds utilizados provienen del manifiesto E1.3 del proyecto.

## Procedencia

Ver `d3_summary.json`, `static_slices_inventory.csv`, `collision_analysis.csv`, `frozen_embeddings.csv`, `embedding_pair_distances.csv` y `geometry_descriptors.csv`. Los resultados son del conjunto de desarrollo únicamente; validation histórica participó en preselecciones E1, lo que limita comparaciones confirmatorias.
"""
    (d3/"D3_report.md").write_text(d3text)
    # Hash/presence manifest for reproducibility.
    d2cv=REPORT/"D2/cv_v1"
    d2_integrity=d2cv/"integrity_checks_cv.json"
    d2_pass=d2_integrity.is_file() and json.loads(d2_integrity.read_text()).get("status")=="PASS"
    d2_status=("PASS científico; PARTIAL de procedencia: hash de trainer no capturado al inicio"
               if d2_pass else "PARTIAL")
    manifest={"status":{"D1":"PASS","D2":d2_status,"D3":"PASS"},
              "historical_test_used":False,"dataset_sha256":sha256(ALT/"reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"),
              "folds_sha256":sha256(ALT/"reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"),
              "checkpoint_sha256":summary["embedding_audit"]["checkpoint_sha256"],"source_commit":"a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e",
              "model_info_sha256":sha256(ALT/"models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"),
              "matgpt_model_source_sha256":sha256(ALT/"models/MatGPT-source/mat-concat/utils/models.py"),
              "D1_script_sha256":sha256(Path(__file__).with_name("d1_composition_baselines.py")),
              "D3_script_sha256":sha256(Path(__file__).with_name("d3_static_representation.py")),
              "D2_script_sha256":sha256(Path(__file__).with_name("d2_centered_training.py")),
              "D2_oof_sha256":sha256(d2cv/"oof_predictions_cv.csv") if d2_pass else None,
              "D2_integrity_sha256":sha256(d2_integrity) if d2_pass else None,
              "D2_summary_script_sha256":sha256(Path(__file__).with_name("summarize_d2.py")),
              "D2_audit_script_sha256":sha256(Path(__file__).with_name("audit_outputs.py")),
              "D2_fits_completed":30 if d2_pass else None,
              "D2_training_wall_seconds":json.loads(d2_integrity.read_text()).get("sum_fit_wall_seconds") if d2_pass else None,
              "D2_exact_trainer_source_hash_captured_at_run_start":False if d2_pass else None,
              "D2_test_used":False,
              "E2A_oof_sha256":sha256(ALT/"reports/MatGPT_finetuning/matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv"),
              "python_version":platform.python_version(),"torch_version":torch.__version__,"sklearn_version":sklearn.__version__,
              "bootstrap":bootstrap}
    write_json(common/"experiment_manifest.json",manifest)
    LOG.info("reports_written D1 D2=%s D3", "PASS scientific / provenance caveat" if d2_pass else "PARTIAL")


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run()


if __name__=="__main__":
    main()
