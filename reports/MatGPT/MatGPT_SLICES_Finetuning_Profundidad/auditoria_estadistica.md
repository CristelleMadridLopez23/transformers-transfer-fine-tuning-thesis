# Auditoría estadística del fine-tuning de MatGPT por profundidad

## Estado

**PASS en integridad de los datos; discrepancias históricas confirmadas en agregación y métricas.** El MAE ΔE histórico mezcla estimandos A/B, los contadores pairwise no se reconcilian consistentemente y la fórmula histórica de ΔE anclada es incorrecta. Las correcciones están calculadas en este informe y los CSV adjuntos. No se entrenaron modelos ni se accedió a etiquetas del test histórico.

## Integridad de datos

- Estado: PASS; 210 estructuras físicas, 70 composiciones, 3 semillas; 2520 filas de predicción y 2520 filas de pares (4 condiciones × 210 estructuras/pares únicos × 3 semillas).
- Hay 210 pares físicos únicos y 197 elegibles con |ΔE_DFT| > 5 meV/átomo. La orientación recalculada es B−A; se verificó ΔE=1000·(E_B/N_B−E_A/N_A).
- Las etiquetas e IDs coinciden con train/val; no se leyeron targets de test. F0 coincide con el archivo congelado histórico: True. Ridge permanece idéntico entre condiciones; rango máximo 0 eV/átomo.
- Cada fórmula tiene 3 polimorfos, 3 pares, 3 semillas y una única evaluación outer. No hay registros OOF duplicados ni IDs de test.

## Dos estimandos distintos

**A — corrida individual promedio:** para cada fórmula se calcula el MAE de sus tres pares por semilla, se promedian las tres semillas y luego se promedian las 70 fórmulas. Mide el rendimiento medio de corridas individuales; es el estimando primario recomendado.
**B — predictor ensemble:** se promedian primero las tres energías predichas por estructura, se forman las diferencias y se calcula el error. Mide el rendimiento de un ensemble de tres modelos. Debido a |·| no coincide con A.
En el JSON histórico, `mae_energy_mev_atom` y el bootstrap de energía usan B, pero `mae_delta_signed_pairs_mev_atom` corresponde a A. El bootstrap de ΔE usa B. Por eso las diferencias directas del resumen y los puntos del bootstrap anteriores no podían coincidir.

### Reproducción de la discrepancia MAE ΔE

Histórico por condición (estimando A): F0 58.844; F2 58.557; F4 57.268; F8 55.248 meV/átomo.
Enfoque B reconstruido: F0 54.426; F2 56.111; F4 54.867; F8 53.125 meV/átomo.
Los contrastes directos A son F2−F0 = −0.287, F4−F0 = −1.576 y F8−F0 = −3.597 meV/átomo. El bootstrap histórico (+1.685, +0.441, −1.301) es B, no A; se reproduce desde las predicciones promediadas. No es redondeo ni pérdida de pares.
IC95% bootstrap A, F2−F0: -0.287 [-2.797, +2.270]; F4−F0: -1.576 [-4.009, +0.777]; F8−F0: -3.597 [-5.780, -1.366] meV/átomo.
IC95% bootstrap B, F8−F0: -1.301 [-3.528, +0.923] meV/átomo.

### Predictor ΔE=0

El baseline tiene MAE de 60.470 meV/átomo con los mismos 210 pares físicos; F0 A tiene 58.844, diferencia −1.626. El valor histórico bootstrap −6.045 vuelve a ser B: predicciones promediadas frente al cero, con estimación observada B reproducida y muestreo pareado por fórmula.
Para el estimando primario A, F0−cero: -1.626 [-8.574, +4.881] meV/átomo; F8−cero: -5.223 [-12.619, +2.000] meV/átomo. Ambos IC incluyen cero. El bootstrap histórico B para F0 es -6.045 [-13.322, +1.125] meV/átomo.
El baseline cero conserva exactamente los mismos pares DFT, sin filtro adicional. Su exactitud de ranking no está definida porque todas sus predicciones son empates.

## ΔE anclada al estado fundamental DFT

El código histórico calculó `pred_i − DFT_g` en vez de `pred_i − pred_g`, donde g es la estructura mínima DFT. Al restar luego el gap DFT, el error resultante es exactamente `pred_i − DFT_i`; por eso su MAE coincide con el MAE absoluto E/N. Es un error de fórmula/etiquetado, no una identidad física.
La métrica corregida usa el mismo polimorfo g seleccionado por DFT en ambos ejes: (pred_i−pred_g)−(DFT_i−DFT_g). La métrica primaria promedia los dos polimorfos no ancla; el cálculo incluyendo el ancla (cuyo error es exactamente cero) queda también reportado. Los cambios comunes por fórmula se cancelan exactamente.
No se hallaron empates exactos DFT que alteren la selección por `argmin`; las casi degeneraciones se tratan según los umbrales de elegibilidad (1 meV/átomo para GS, 5 para pares).

## Ranking y estado fundamental

La exactitud pairwise principal cuenta predicciones no resueltas (|ΔE_pred|≤1 meV/átomo) como incorrectas entre los pares DFT elegibles; también se reporta la exactitud condicionada a pares resueltos. El denominador físico elegible es 197 por semilla. F4 tiene pares no resueltos, por lo que exactitud sobre elegibles y exactitud entre resueltos difieren. Ground state exacto se calcula por argmin y los empates predichos no cuentan como acierto.
Las exactitudes A promedian dentro de fórmula las tres semillas; por ello pueden ser fraccionarias y no deben transformarse en conteos enteros. Se incluyen denominadores y métricas por semilla en `metricas_recalculadas.csv`.

La auditoría también detecta que los campos históricos `pairwise_correct`, `pairwise_resolved_denominator` y `pairwise_unresolved` no se reproducen consistentemente como conteos de A ni de B. El denominador elegible de 197 sí se confirma. Las razones de exactitud pueden coincidir con A aunque esos contadores no; se deben citar los conteos reconstruidos desde pares y semillas, no los contadores históricos.

## Métricas primarias recalculadas (A)

- MAE E/N (meV/átomo): F0 52.019; F2 52.634; F4 51.842; F8 51.244
- MAE residual intrafórmula (meV/átomo): F0 33.754; F2 34.019; F4 33.160; F8 31.914
- MAE ΔE de tres pares (meV/átomo): F0 58.844; F2 58.557; F4 57.268; F8 55.248
- MAE ΔE anclada corregida, dos estructuras no ancla (meV/átomo): F0 62.289; F2 61.722; F4 60.377; F8 58.345
En `bootstrap_recalculado.csv` se incluyen IC apareados por composición para MAE absoluta, residual, ΔE, exactitud GS y pairwise, además de los contrastes contra ΔE=0. Las semillas permanecen agrupadas dentro de cada fórmula.

## Impacto científico

1. ¿F8 mejora ΔE frente a F0? Para el estimando primario A, la reducción sí queda respaldada por este bootstrap: -3.597 [-5.780, -1.366] meV/átomo; el IC excluye cero. Para el predictor ensemble B el contraste no es concluyente: -1.301 [-3.528, +0.923]. Esta evidencia es interna a las 70 fórmulas de desarrollo ya exploradas, no una evaluación externa.
2. ¿F4 mejora GS? La diferencia exacta A es +0.048 [-0.000, +0.095]; el IC percentil incluye cero (límite inferior 0), por lo que no demuestra una mejora al 95%. Esto no demuestra equivalencia.
3. ¿Algún modelo supera a ΔE=0? F0−cero A es -1.626 [-8.574, +4.881] y F8−cero A es -5.223 [-12.619, +2.000]; ambos IC incluyen cero, sin evidencia concluyente frente al baseline.
4. El contraste F8−F0 mejora el MAE ΔE bajo A, pero F2−F0 y F4−F0 no son concluyentes, y F8−F4 tampoco: esto no establece una tendencia monotónica ni prueba que aumentar profundidad sea causalmente mejor. El resultado depende del estimando y de las semillas agregadas.
5. Deben corregirse las conclusiones que mezclaron MAE A con IC B y retirar la métrica histórica de ΔE anclada como si fuera una diferencia válida.
6. El conjunto ya fue explorado en experimentos previos; los intervalos son incertidumbre interna por composición, no validación externa ciega.

## Reconciliación histórica

| condition   | metric_name                        |   historical_value |   recalculated_A |   recalculated_B | status   | cause                                                                                                                                                                |
|:------------|:-----------------------------------|-------------------:|-----------------:|-----------------:|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| F0          | mae_delta_signed_pairs_mev_atom    |            58.8441 |          58.8441 |          54.4259 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F0          | mae_delta_zero_baseline_mev_atom   |            60.4704 |          60.4704 |          60.4704 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F0          | mae_delta_anchored_dft_gs_mev_atom |            50.4792 |          41.5262 |          39.1581 | ERROR    | El código histórico restó E_DFT del ancla a E_pred, por lo que reproduce el error absoluto por estructura; no es una diferencia predicha respecto al ancla predicha. |
| F0          | pairwise_correct                   |           131.0000 |         131.6667 |         136.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F0          | pairwise_eligible_denominator      |           197.0000 |         197.0000 |         197.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F0          | pairwise_resolved_denominator      |           197.0000 |         194.6667 |         197.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F0          | pairwise_unresolved                |             0.0000 |           2.3333 |           0.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F2          | mae_delta_signed_pairs_mev_atom    |            58.5567 |          58.5567 |          56.1111 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F2          | mae_delta_zero_baseline_mev_atom   |            60.4704 |          60.4704 |          60.4704 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F2          | mae_delta_anchored_dft_gs_mev_atom |            51.3079 |          41.1480 |          40.0422 | ERROR    | El código histórico restó E_DFT del ancla a E_pred, por lo que reproduce el error absoluto por estructura; no es una diferencia predicha respecto al ancla predicha. |
| F2          | pairwise_correct                   |           130.0000 |         130.6667 |         136.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F2          | pairwise_eligible_denominator      |           197.0000 |         197.0000 |         197.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F2          | pairwise_resolved_denominator      |           197.0000 |         195.0000 |         191.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F2          | pairwise_unresolved                |             0.0000 |           2.0000 |           6.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F4          | mae_delta_signed_pairs_mev_atom    |            57.2679 |          57.2679 |          54.8668 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F4          | mae_delta_zero_baseline_mev_atom   |            60.4704 |          60.4704 |          60.4704 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F4          | mae_delta_anchored_dft_gs_mev_atom |            51.0935 |          40.2516 |          38.9601 | ERROR    | El código histórico restó E_DFT del ancla a E_pred, por lo que reproduce el error absoluto por estructura; no es una diferencia predicha respecto al ancla predicha. |
| F4          | pairwise_correct                   |           134.0000 |         135.0000 |         138.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F4          | pairwise_eligible_denominator      |           197.0000 |         197.0000 |         197.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F4          | pairwise_resolved_denominator      |           196.0000 |         194.3333 |         195.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F4          | pairwise_unresolved                |             1.0000 |           2.6667 |           2.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F8          | mae_delta_signed_pairs_mev_atom    |            55.2475 |          55.2475 |          53.1252 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F8          | mae_delta_zero_baseline_mev_atom   |            60.4704 |          60.4704 |          60.4704 | PASS     | Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A).                                                                                                |
| F8          | mae_delta_anchored_dft_gs_mev_atom |            50.5748 |          38.8969 |          37.9570 | ERROR    | El código histórico restó E_DFT del ancla a E_pred, por lo que reproduce el error absoluto por estructura; no es una diferencia predicha respecto al ancla predicha. |
| F8          | pairwise_correct                   |           135.0000 |         135.3333 |         141.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F8          | pairwise_eligible_denominator      |           197.0000 |         197.0000 |         197.0000 | PASS     | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F8          | pairwise_resolved_denominator      |           197.0000 |         194.6667 |         193.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |
| F8          | pairwise_unresolved                |             0.0000 |           2.3333 |           4.0000 | ERROR    | El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.                                     |

## Reproducibilidad y archivos

Predicciones de entrada SHA-256: `2f76b17586dfa1ba33686e44823f3539afe2ffe053162799b1ab38071d832722`; pares: `8077f6aedd281ca0a2ea5e6575d83d84fcc5c5bbe5de31a67ed85c643237fd64`.
Código: `scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/audit_statistics.py`. Tests: `scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/tests/test_statistical_audit.py`. Resultados CSV y gráficos PNG se guardan bajo `experiments/MatGPT_SLICES_Finetuning_Profundidad/` e `images/MatGPT_SLICES_Finetuning_Profundidad/`.
Figuras: [comparación de agregación](../../../images/MatGPT_SLICES_Finetuning_Profundidad/comparacion_agregacion_mae_y_bootstrap.png) y [contrastes e intervalos](../../../images/MatGPT_SLICES_Finetuning_Profundidad/reconciliacion_intervalos_confianza_finetuning.png).

### Limitaciones

Solo hay 70 composiciones ya exploradas; tres semillas son repeticiones de entrenamiento, no 630 materiales. Se conservaron las definiciones DFT archivadas, sin reetiquetar ni homogeneizar la procedencia gga-static. No se consultó el test histórico, no se reentrenó y los outer folds son desarrollo, no una estimación ciega externa.
