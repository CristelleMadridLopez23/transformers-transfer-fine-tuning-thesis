# H3-TEST — evaluación final de H3-A en el holdout ABS₃

**Estado: PASS** para la evaluación bloqueada de H2 y H3-A. Se evaluaron 24 estructuras estáticas, agrupadas en 8 composiciones con tres polimorfos por composición. No se reentrenaron modelos, no se ajustaron parámetros con el test y no se consultó el test durante la selección previa. Las cifras de test se generaron una sola vez después de guardar las predicciones pre-etiquetas y verificar sus hashes.

## Pregunta y modelos

H3-A combina, dentro de cada fórmula, el nivel medio de energía de Ridge R4 con las desviaciones centradas que predice H2:

\[
\hat E_{c,i}^{H3-A}=\hat\mu_c^{R4}+\hat\delta_{c,i}^{H2}.
\]

H2 y H3-A usan el mismo MatGPT unconditional congelado, cinco folds históricos y tres semillas. Para cada una de las 24 entradas, se promediaron las predicciones de los 15 pares fold×semilla. Por tanto, este resultado mide el ensemble fijado para el holdout y no una nueva validación cruzada sobre el test.

R4 no conservaba estimadores serializados. Se reconstruyeron los cinco pipelines históricos `StandardScaler + Ridge(solver=lsqr, tol=1e-8)` con sus alphas ya seleccionados `[10, 0.1, 10, 10, 10]`, ajustando solo con los conjuntos inner-train registrados. Cada pipeline reprodujo sus predicciones OOF archivadas; el mayor error fue menor que \(5\times10^{-8}\) eV/átomo. No se volvió a elegir alpha.

## Integridad y bloqueo

- MatGPT checkpoint SHA-256: `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414` (coincide con el esperado); commit fuente local: `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- Dataset histórico: SHA-256 `0719f496c96297761ede3a7e2e7ef57aa7b62400fb0004e679a66430a70c31e8`.
- Se verificaron 210 estructuras/70 composiciones de desarrollo, 24/8 de test y disyunción completa de IDs y fórmulas. Cada composición test conserva las tres familias físicas.
- Se encontró y cargó estrictamente el conjunto congelado de 15 cabezas H2 y 5 Ridge R4. Ningún entrenamiento H2 incluye una composición test.
- La extracción posterior al lock obtuvo `L03_MEAN` (bloque 3, media de tokens SLICES antes de `ln_f`), `FINAL_END` (estado final normalizado en la posición real de `<end>`) y `FINAL_MEAN` (media final normalizada de tokens SLICES). El backbone quedó en `eval`, sin gradientes; prueba de padding derecho: error máximo `1.43e-6`.
- Se bloquearon primero las 24 entradas sin columnas energéticas. Se guardaron 360 predicciones fold×semilla antes de acceder a targets. El replay independiente de H2 sobre las 210 filas OOF tuvo error máximo `7.04e-7` eV/átomo, menor que la tolerancia bloqueada `1e-6`.
- La evaluación comprobó el SHA del dataset, las entradas, el lock y las predicciones pre-etiquetas antes de leer las 24 etiquetas. `test_used_for_selection=false`; predicciones no se regeneraron tras leerlas.
- Se conservaron tres locks previos supersedidos, todos anteriores a abrir targets. Dos registran un fallo de importación/precisión corregido antes de inferencia; el lock final conserva las mismas entradas, modelos y métricas fijadas. Tras puntuar, solo se corrigieron visualizaciones y el mapeo de columnas de la comparación descriptiva desarrollo/test; predicciones y CSV de métricas no cambiaron. El hash del generador gráfico revisado está registrado en `figures/figures_manifest.json`.
- Se observaron avisos numéricos de sklearn al evaluar Ridge; las predicciones se comprobaron finitas, y tanto el replay OOF como las igualdades algebraicas pasaron. Los avisos se conservan como una nota de ejecución y no se suprimieron.

## Resultados principales

Todas las energías absolutas se expresan por átomo. MAE y RMSE se expresan en meV/átomo. El MAE macro y pooled coinciden porque cada una de las ocho fórmulas aporta exactamente tres estructuras. Los intervalos Wilson son diagnósticos binomiales; las fórmulas son la unidad física independiente.

| Métrica | H2 | H3-A |
|---|---:|---:|
| MAE absoluto, macro (meV/átomo) | 330.408 | **133.545** |
| RMSE absoluto pooled (meV/átomo) | 434.064 | **151.998** |
| Sesgo predicción−DFT (meV/átomo) | −80.186 | +59.170 |
| R² predictivo absoluto | 0.779 | **0.973** |
| R² OLS descriptivo | 0.845 | 0.978 |
| Pendiente OLS; intercepto (meV/átomo) | 0.623; −1960.307 | 1.003; +75.063 |
| MAE de media composicional (meV/átomo) | 330.408 | **132.119** |
| MAE de residuo estructural centrado (meV/átomo) | 19.025 | 19.025 |
| MAE ΔE anclada al mínimo DFT (meV/átomo) | 24.208 | 24.208 |
| MAE ΔE own-min histórica (meV/átomo) | 21.930 | 21.930 |
| Error absoluto medio del rango (meV/átomo) | 32.272 | 32.272 |
| Ground-state correcto | 5/8 = 62.5% | 5/8 = 62.5% |
| Intervalo Wilson 95% de GS | 30.6–86.3% | 30.6–86.3% |
| Pairwise correcto | 17/24 = 70.8% | 17/24 = 70.8% |
| Intervalo Wilson 95% pairwise | 50.8–85.1% | 50.8–85.1% |
| SignedPairwiseMAE (meV/átomo) | 33.025 | 33.025 |

La mejora de MAE absoluto H3-A−H2 es −196.862 meV/átomo en el ensemble (−59.6%). El residuo y el ranking no mejoran con H3-A: al sumar el mismo escalar a los tres polimorfos no cambia su orden ni sus diferencias. Esa igualdad es una propiedad algebraica esperada, no evidencia de que H3-A haya aprendido un mejor ranking.

### De dónde procede el error absoluto

El error está casi todo en el nivel medio de la composición. La descomposición cuadrática por triplete satisface `MSE_total = MSE_mu + MSE_residual` con error numérico máximo de `6.57e-10` meV²/átomo² para H2 y `2.72e-10` para H3-A.

| MSE promedio por composición (meV²/átomo²) | H2 | H3-A |
|---|---:|---:|
| Total | 188,411.434 | 23,103.340 |
| Nivel medio \(\mu\) | 187,815.153 | 22,507.058 |
| Residuo centrado \(\delta\) | 596.281 | 596.281 |

Así, aproximadamente 99.7% del MSE total H2 y 97.4% del MSE H3-A corresponde al término de media. El MAE del residuo (19.025 meV/átomo) es un diagnóstico separado: los MAE no se suman exactamente como los MSE.

La distribución por composición no es uniforme. H3-A redujo el MAE en 5 de 8 fórmulas; empeoró en GePbS3, GeS3Zn (ligeramente) y PbS3V. Los casos con mayor MAE absoluto H3-A fueron GeS3Zn (228.99), GePbS3 (196.66), HfS3Sr (187.21) y PbS3V (152.29) meV/átomo. En MgPbS3 el MAE bajó de 965.68 a 33.58; esa mejora grande pesa mucho en la muestra pequeña y merece atención en un holdout nuevo.

### Ranking y pares

Las ocho composiciones tuvieron brecha DFT de estado fundamental mayor que 1 meV/átomo, por lo que las ocho fueron elegibles para GS. Ninguna pareja física tuvo empate DFT dentro de ±1 meV/átomo; se evaluaron los 24 pares canónicos. La matriz GS tuvo 3 composiciones cuyo GS DFT era perovskita: se predijo una hexagonal y dos tipo aguja; los 5 GS DFT tipo aguja se predijeron correctamente. No hubo GS hexagonal en este holdout. En la matriz pairwise, 17 pares acertaron; los errores fueron 6 inversiones en la clase `i_lower` y 1 en `j_lower`. No hubo abstenciones ni empates predichos.

Para la diferencia pareada H3-A−H2, bootstrap por composición con 20,000 remuestras y semilla `13072026` dio:

| Diferencia H3-A−H2 | Estimación | IC bootstrap 95% |
|---|---:|---:|
| MAE absoluto (meV/átomo) | −196.29 | [−436.99, −10.51] |
| MAE de media (meV/átomo) | −197.71 | [−441.27, −10.51] |
| MAE residuo centrado (meV/átomo) | ≈0 | ≈[0, 0] |
| MAE ΔE anclada DFT-GS (meV/átomo) | ≈0 | ≈[0, 0] |
| Exactitud GS | 0 | [0, 0] |
| Exactitud pairwise | 0 | [0, 0] |

Los intervalos son exploratorios: solo hay ocho grupos y el resultado absoluto está influido por unas pocas fórmulas. Los intervalos nulos de residuo/ranking reflejan que H3-A reutiliza literalmente esas predicciones de H2.

### Semillas y comparación con development

Por cada uno de los 15 modelos fold×semilla, el MAE macro medio fue 347.98±45.65 meV/átomo para H2 y 145.90±21.65 para H3-A (media±desviación estándar entre esas variantes; no son 15 muestras independientes). GS medio por variante fue 0.717±0.088 y pairwise 0.739±0.078 para ambos; el rango por variante fue 0.50–0.875 GS y 0.583–0.833 pairwise. El resultado principal sigue siendo el ensemble con una predicción por estructura.

| OOF desarrollo (70 fórmulas) | H2 | H3-A |
|---|---:|---:|
| MAE absoluto (meV/átomo) | 352.525 | 129.102 |
| MAE ΔE anclada DFT-GS (meV/átomo) | 39.277 | 39.277 |
| MAE ΔE own-min histórica (meV/átomo) | 36.488 | 36.488 |
| GS accuracy | 59.7% | 59.7% |
| Pairwise accuracy pooled | 67.5% | 67.5% |

El test mostró MAE absoluto H3-A de 133.545 frente a 129.102 en OOF de desarrollo; no son estimadores idénticos (OOF por composición frente al promedio de los cinco folds y tres semillas sobre el mismo holdout). Esta diferencia es descriptiva y no prueba por sí sola sobreajuste ni cambio de dominio. H2/H3-A fueron preseleccionados a partir de investigación previa en development; el test histórico ya no es completamente ciego al historial científico global.

## Interpretación

En estas ocho composiciones reservadas, el reemplazo de la media H2 por la media absoluta de R4 mejora con claridad la energía absoluta, y la mejora se explica por reducir el error composicional común. La capacidad estructural de H2 se conserva: el error residual permanece en 19.025 meV/átomo y el ranking no cambia. Esto apoya la complementariedad de las cabezas para separar nivel medio y diferencias polimórficas, pero no muestra que H3-A refine la selección del polimorfo fundamental. Un GS de 62.5% con intervalo amplio y pairwise de 70.8% todavía deja errores de ordenamiento; no es suficiente para afirmar precisión fiable en selección de fonones sin validación física adicional.

## Figuras y datos

Se generaron 18 figuras PNG, sin SVG/PDF/JPG, con notas de muestra y definición. En paridad, DFT está en x, predicción en y; se muestran `y=x`, OLS libre, pendiente, intercepto, R² OLS, R² predictivo y MAE. Las matrices muestran conteos y porcentaje por fila. Los pares usan siempre el orden (perovskita, hexagonal), (perovskita, aguja), (hexagonal, aguja).

- [Energía absoluta](figures/01_absolute_energy_parity.png)
- [Media composicional](figures/02_composition_mean_parity.png)
- [Residuo estructural centrado](figures/03_structural_residual_parity.png)
- [ΔE anclada a DFT-GS](figures/04_dft_gs_anchored_delta_parity.png)
- [ΔE own-min histórica](figures/05_own_min_delta_parity.png)
- [Matriz de estado fundamental](figures/06_ground_state_confusion.png)
- [Matriz pairwise](figures/07_pairwise_confusion.png)
- [18 figuras y scorecard](figures/18_test_energy_ranking_scorecard.png)

Predicciones pre-etiquetas: `predictions/test_predictions_prelabels.csv` y `predictions/test_ensemble_predictions_prelabels.csv`. Métricas por composición: `metrics/test_metrics_by_composition.csv`. Métricas y denominadores: `metrics/test_metrics_summary.csv`, `metrics/test_pairwise_results.csv`, `metrics/test_bootstrap_and_intervals.json`. El manifiesto de integridad es `metrics/evaluation_integrity.json`; procedencia y lock están en `provenance/`.

## Límites y siguiente experimento

Este resultado no demuestra generalización a otras químicas, energías relajadas, fuerzas, tensiones ni fonones. Ridge corrige una media absoluta que el MLP8 H2 deja desplazada, mientras que las diferencias se heredan de la misma cabeza H2. Para afirmar selección polimórfica fiable hace falta un holdout externo nuevo o una campaña DFT/phonon con incertidumbre y muestreo que cubra mejor las fórmulas con errores grandes; no debe usarse este test para volver a elegir el modelo.

## Referencias

- MatGPT, repositorio oficial y código local histórico: https://github.com/devalab/matgpt (commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`).
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023): https://doi.org/10.1038/s41467-023-42870-7.
- Witman y Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025): https://doi.org/10.1039/D4DD00250D.
- scikit-learn, [Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression), [Ridge API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) y [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
