# H3 — Ridge para la energía media y MLP8 para diferencias entre polimorfos

## Estado y alcance

**Estado: PASS para el protocolo de desarrollo; resultado exploratorio, no validación externa.** Se completaron el preflight, H3-A, H3-B y las 15 corridas H3-C (5 folds × 3 semillas). Cada conjunto OOF contiene 210 estructuras de 70 composiciones. Las 8 composiciones del test histórico y sus etiquetas no se leyeron ni entraron en el caché. El experimento respeta la exclusión del test, pero sus combinaciones se motivaron usando análisis anteriores del mismo desarrollo; no debe interpretarse como estimación independiente.

La pregunta fue si una representación intermedia de MatGPT puede predecir mejor la media de energía de una composición mientras un embedding final conserva la información necesaria para ordenar sus polimorfos. MatGPT se mantuvo congelado. No se incluyeron fórmulas como variables, geometría, trayectorias ni energías de formación.

## Procedencia y controles

- Checkpoint MatGPT unconditional: SHA-256 `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414` (coincide con el valor esperado).
- Código histórico: commit `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`; el manifiesto local de descarga y los hashes de `train.py`, `utils/dataset.py`, `utils/models.py` y `config/uncond.yaml` concuerdan. Se usó el checkout local fijado, no el `HEAD` público.
- Cache de desarrollo: `L03_MEAN` (salida cruda del bloque 3, antes de `ln_f`) y `FINAL_END` (estado final, después de `ln_f`, en el token `<end>` real), cada uno de dimensión 768 y 210 filas. La auditoría registró carga estricta, modo eval, pesos congelados, embeddings finitos y ausencia de tokens energéticos. El control de padding a derecha tuvo error absoluto máximo `1.43e-6`.
- H2: se reejecutaron los 15 checkpoints estructurales sobre el cache verificado; sus 630 residuos concordaron con el OOF archivado dentro de `2.69e-7 eV/átomo`.
- Datos: DFT estática GGA/PBE, energía total por átomo, eV/átomo. Se usaron 210 estructuras, 70 composiciones y tres polimorfos por composición. Group folds existentes: 5 folds, 14 composiciones de evaluación externa por fold. Semillas MLP8: 42, 123, 2026.
- No se volvió a calcular DFT. D1 se trata solo como control composition-only externo, no como componente H3.
- La auditoría de dominio de preentrenamiento anterior indicó posibles coincidencias composicionales, sin evidencia de coincidencia exacta de SLICES. No se usó esa auditoría para cambiar folds ni excluir estructuras.

## Modelos y evaluación

H3-A suma la media energética de R4 a los residuos H2 centrados; no entrena parámetros. H3-B ajusta un `StandardScaler + Ridge` en el promedio de los tres `L03_MEAN` por composición y suma los residuos H2 sin modificarlos. H3-C usa la misma parte Ridge y reentrena una MLP8 sobre `FINAL_END` para los residuos centrados. Para Ridge se seleccionó `alpha` con MAE de inner-validation, entre `[0.01, 0.1, 1, 10, 100, 1000]`, y luego se reajustó con el entrenamiento externo. Para MLP8 se usó 768→8→1, GELU, dropout 0.1, AdamW (LR 1e-3, weight decay 1e-4), SmoothL1 normalizada con escala estimada solo en inner-train, máximo 200 épocas y paciencia 20. H3-C terminó 15 corridas, con 21–32 épocas cada una.

Se fijaron cinco folds agrupados por composición. Dentro de cada outer-train se reservaron 11 grupos para early stopping y se entrenó con 45; el outer-validation tenía 14 grupos. Ridge usó los 56 grupos del outer-train después de seleccionar alpha. No hubo ajuste con las 14 composiciones externas.

Los alpha Ridge seleccionados en los cinco folds fueron `100, 10, 10, 10, 0.01`, respectivamente. La diferencia entre folds muestra que la regularización se eligió localmente con inner-validation, no con errores outer.

Las métricas absolutas y de residuos de la tabla son macro por composición, usando el ensamble de las tres semillas cuando corresponde. ΔE anclada usa el polimorfo de menor energía DFT como referencia común y conserva valores negativos predichos. GS excluye 3 composiciones con gap DFT ≤1 meV/átomo (`N=67`). Pairwise usa las 3 parejas físicas canónicas por composición, tolerancia 1 meV/átomo; el denominador principal excluye empates DFT (`206` parejas elegibles de 210). La matriz diagnóstica conserva empates DFT. Las semillas no se cuentan como composiciones adicionales.

## Resultados OOF de desarrollo

Errores y ΔE en meV/átomo; GS y pairwise son proporciones. Para Ridge composition-only, el ranking es abstención porque todas las energías de los tres polimorfos son iguales.

| Modelo | MAE absoluto | RMSE absoluto, pooled | MAE media μ | MAE residuo centrado | MAE ΔE anclada DFT-GS | MAE ΔE own-min | GS accuracy (`N=67`) | Pairwise (`N=206`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 composition-only | 59.29 | 81.12 | 46.96 | 35.07 | 44.20 | 44.20 | abstiene en 67/67 | abstiene en 206/206 |
| H0 | 729.08 | 834.42 | 729.08 | 30.63 | 37.99 | 35.25 | 56.7% | 67.5% |
| H2 | 352.53 | 446.42 | 351.29 | 32.01 | 39.28 | 36.49 | 59.7% | 67.5% |
| R4 selected-absolute | 153.27 | 215.55 | 125.55 | 87.92 | 99.49 | 101.08 | 35.8% | 53.4% |
| H3-A | **129.10** | 189.30 | **125.55** | 32.01 | 39.28 | 36.49 | 59.7% | 67.5% |
| H3-B | 182.60 | 258.41 | 180.18 | 32.01 | 39.28 | 36.49 | 59.7% | 67.5% |
| H3-C | 182.88 | 258.57 | 180.18 | **31.21** | **39.16** | **36.31** | 58.2% | 69.4% |

R² predictivo absoluto fue 0.947 para H3-A, 0.901 para H3-B, 0.901 para H3-C, 0.704 para H2, 0.931 para R4, 0.990 para D1 y −0.033 para H0. D1 obtiene buen error absoluto porque las energías medias de composición son informativas; al asignar una predicción idéntica a cada polimorfo no discrimina estructura y su ranking queda indefinido.

### Variabilidad por fold y semilla

La siguiente tabla resume primero las 15 métricas fold×semilla, luego muestra por separado la desviación entre medias de fold (semillas promediadas dentro de cada fold) y entre medias de semilla (folds promediados dentro de cada semilla). Los valores GS/pairwise aquí son promedio por corrida; difieren del ensamble por composición mostrado arriba.

| Modelo | MAE por corrida (media ± DE entre folds) | DE MAE entre semillas | GS por corrida (media; DE entre folds) | Pairwise por corrida (media; DE entre folds) |
|---|---:|---:|---:|---:|
| H2 | 354.51 ± 80.78 | 16.84 | 59.7% ± 10.5 pp | 67.7% ± 6.3 pp |
| H3-A | 129.13 ± 43.97 | 0.07 | 59.7% ± 10.5 pp | 67.7% ± 6.3 pp |
| H3-B | 182.64 ± 45.38 | 0.14 | 59.7% ± 10.5 pp | 67.7% ± 6.3 pp |
| H3-C | 183.30 ± 45.14 | 0.49 | 53.2% ± 13.9 pp | 66.4% ± 7.6 pp |

La dispersión entre composiciones/folds es mucho mayor que la variación por semilla en los MAE. H3-C presenta más variación por fold y un promedio GS por corrida menor que H2, aunque la comparación pareada de ensambles tiene un intervalo que incluye cero. Esto apunta a heterogeneidad entre composiciones y evita interpretar una única media como mejora uniforme.

### ¿De dónde viene el error absoluto de H0?

La identidad de MSE por composición se verificó numéricamente: `MSE_total = MSE_media + MSE_residual`, con discrepancia máxima `9.13e-10 meV²/átomo²`. Para H0, el MSE total medio es `696,255.72 meV²/átomo²`, de los cuales `694,466.31` corresponden al error de la media y `1,789.41` al residuo centrado. Su MAE medio absoluto de `729.08 meV/átomo` y el error residual centrado de `30.63 meV/átomo` muestran que el gran error absoluto está dominado por el offset composicional. Los MAE no son aditivos; la descomposición exacta se refiere a MSE.

H3-A, B y C reducen el error medio frente a H2 al reemplazar su predictor de nivel global. H3-C reduce modestamente el error residual frente a H2 (32.01 a 31.21 meV/átomo). No se observa evidencia de que ese cambio estructural mejore ΔE o GS de manera consistente.

## Comparaciones pareadas e incertidumbre

Bootstrap percentil pareado de 2,000 remuestras por composición, manteniendo unido el triplete; semilla 13072026. Diferencias expresadas como candidato menos referencia, por lo que valores negativos de MAE significan mejora. Los intervalos son exploratorios.

| Comparación | Δ MAE absoluto (IC 95%) | Δ MAE μ (IC 95%) | Δ MAE residuo (IC 95%) | Δ MAE ΔE anclada (IC 95%) | Δ GS (IC 95%) | Δ pairwise (IC 95%) |
|---|---:|---:|---:|---:|---:|---:|
| H3-A − H2 | −223.42 [−292.95, −153.42] | −225.74 [−293.46, −160.69] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| H3-B − H2 | −169.92 [−236.39, −101.79] | −171.12 [−240.86, −105.32] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| H3-C − H2 | −169.64 [−234.35, −103.05] | −171.12 [−234.68, −99.70] | −0.80 [−2.92, 1.22] | −0.12 [−2.63, 2.58] | −0.015 [−0.060, 0.030] | +0.024 [−0.019, 0.071] |
| H3-C − H3-B | +0.28 [−0.85, 1.59] | ≈0 [≈0, ≈0] | −0.80 [−2.91, 1.37] | −0.12 [−2.78, 2.64] | −0.015 [−0.060, 0.030] | +0.024 [−0.014, 0.071] |
| H3-C − R4 | +29.61 [1.12, 58.25] | +54.63 [26.95, 82.47] | −56.71 [−67.95, −44.89] | −60.33 [−75.48, −44.44] | +0.224 [0.075, 0.373] | +0.160 [0.064, 0.252] |

La tabla indica que H3-C **no mejora de manera demostrable** el ranking frente a H2: sus intervalos pareados de GS, pairwise y ΔE incluyen cero. H3-A y H3-B conservan exactamente el residuo H2; sus métricas relativas y ranking son iguales por identidad algebraica. H3-C conserva casi la misma cabeza Ridge que H3-B, pero al reentrenar la MLP8 no logra mejorar con claridad el ordenamiento. R4 tiene mejor MAE absoluto que H3-C en este benchmark; H3-C, a diferencia de R4, conserva una cabeza que distingue polimorfos, aunque esa ventaja no debe convertirse en una afirmación de superioridad general.

## Auditorías matemáticas y estabilidad

- H3-A y H3-B: diferencia máxima frente a residuos H2 reconstruidos `8.9e-16 eV/átomo`; la exactitud GS, el pairwise y las dos definiciones de ΔE quedan invariantes.
- Para H3-C, la media de sus tres energías reconstruidas coincide con Ridge y Ridge se cancela en toda diferencia entre polimorfos.
- Los pesos entrenables pertenecen a la MLP8 de 6,161 parámetros; MatGPT tiene cero parámetros entrenables.
- Las curvas de entrenamiento corresponden a train e inner-validation por fold y semilla; se registra la época seleccionada. No se usó outer-fold para early stopping.
- Las métricas por fold×semilla, desviaciones entre folds y semillas, errores por composición, pares y composiciones de mayor error se encuentran en `comparative_analysis/`.
- Las 4 pruebas unitarias sintéticas pasan. Verifican centrado, invariancia a permutación, cancelación de Ridge, referencia DFT-GS y dirección de parejas.

## Interpretación

Los resultados apoyan una conclusión acotada: la cabeza energética de H0 deja un error de nivel composicional enorme; reemplazarla por una regresión entrenada sobre una representación media mejora mucho el MAE absoluto sin alterar el ranking si los residuos se conservan. En H3-A esa mejora es mayor porque importa directamente las predicciones medias R4 ya seleccionadas. H3-B/C muestran que una cabeza Ridge ajustada solo en los grupos de entrenamiento también reduce el MAE respecto de H2, aunque con menos ganancia que H3-A. La cabeza MLP8 reentrenada de H3-C produce una mejora pequeña en el error residual medio, pero con intervalos que incluyen ausencia de mejora y sin cambio concluyente en ΔE/GS.

Esto no prueba que MatGPT contenga toda la información energética ni que el comportamiento se transfiera a composiciones externas. Las 70 composiciones son pocas, y el diseño usa el desarrollo que participó en selección previa de H2/R4. H3-A es además una recombinación de modelos elegidos anteriormente, no un predictor entrenado desde cero. La limitación de inferencia para H3 es que requiere las tres estructuras candidatas de cada composición para formar `z_c` y centrar residuos. Se necesita un holdout externo nuevo o validación anidada completa antes de una conclusión confirmatoria.

## Archivos principales

- Preflight: `provenance/preflight.json`.
- Predicciones: `H3_A_recombination/oof_predictions.csv`, `H3_B_ridge_fixed_delta/oof_predictions.csv`, `H3_C_ridge_retrained_delta/oof_predictions.csv`.
- Comparación/estadística: `comparative_analysis/comparative_metrics.csv`, `paired_bootstrap_ci.json`, `paired_comparisons.csv`, tablas por fold/semilla y composición.
- 24 gráficos: carpeta `figures/`; todos en PNG y generados desde OOF, sin extraer datos de imágenes.
- Manifiesto de archivos y hashes: `manifest.json`.

## Referencias

- MatGPT, código fuente histórico local: https://github.com/devalab/matgpt
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023): https://doi.org/10.1038/s41467-023-42870-7
- Witman & Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025): https://doi.org/10.1039/D4DD00250D
- Scikit-learn Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-learn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
