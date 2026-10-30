# Comparación C0/C1 y hallazgos de Experimento C

**Diseño:** 210 estructuras físicas, 70 fórmulas, tres polimorfos por fórmula, cinco pliegues de composición y tres semillas. Cada condición aporta 630 predicciones, que son 210 estructuras repetidas por tres ajustes; no son 630 materiales independientes. El test histórico permanece cerrado.

## Resultados agrupados por fórmula/fold y semilla

- **frozen** — MAE E/N 183.3 ± 0.5; residuo 33.8 ± 0.7; ΔE pares 58.8 ± 1.0; GS elegible 0.532; pares elegibles 0.668.
- **lora** — MAE E/N 182.8 ± 0.3; residuo 33.5 ± 0.2; ΔE pares 58.3 ± 0.2; GS elegible 0.517; pares elegibles 0.658.

La desviación indicada resume las 15 combinaciones fold/semilla; no es un intervalo inferencial. La incertidumbre apareada por fórmula se presenta en `bootstrap_apareado.csv` y en las figuras.

## Contraste apareado LoRA menos congelado

| metric                          |   difference_lora_minus_frozen |   ci95_lower |   ci95_upper |   n_formulas |   seeds_averaged_first |
|:--------------------------------|-------------------------------:|-------------:|-------------:|-------------:|-----------------------:|
| absolute_mae_mev_atom           |                        -0.5199 |      -1.2736 |       0.0962 |           70 |                      3 |
| composition_mean_error_mev_atom |                        -0.0000 |      -0.0000 |       0.0000 |           70 |                      3 |
| residual_mae_mev_atom           |                        -0.2152 |      -1.4522 |       0.9672 |           70 |                      3 |
| pair_mae_mev_atom               |                        -0.5659 |      -2.7969 |       1.7794 |           70 |                      3 |
| gs_exact_argmin                 |                        -0.0143 |      -0.0619 |       0.0333 |           70 |                      3 |
| pairwise_accuracy_gt5           |                        -0.0102 |      -0.0415 |       0.0203 |           70 |                      3 |
| gs_eligible_accuracy            |                        -0.0149 |      -0.0667 |       0.0354 |           70 |                      3 |

## Error por tamaño de diferencia DFT

Los pares se asignan por |ΔE DFT|. Cada una de las tres filas de semilla resume conjuntamente los cinco pliegues (210 pares físicos por semilla); la tabla promedia esas tres repeticiones de entrenamiento. En 0–5 meV/átomo no hay pares elegibles para evaluar orden.

| condition   |   n_semillas |   pares_promedio_por_semilla |   MAE_delta_promedio_meV_atom |   MAE_delta_cero_promedio_meV_atom |   pares_elegibles_promedio | acierto_orden_promedio   | intervalo_gap_DFT_meV_atom   |
|:------------|-------------:|-----------------------------:|------------------------------:|-----------------------------------:|---------------------------:|:-------------------------|:-----------------------------|
| frozen      |            3 |                       13.000 |                        32.932 |                              2.001 |                      0.000 | Sin pares elegibles      | 0–5                          |
| frozen      |            3 |                       17.000 |                        33.788 |                              7.847 |                     17.000 | 0.549                    | 5–10                         |
| frozen      |            3 |                       41.000 |                        42.131 |                             16.804 |                     41.000 | 0.528                    | 10–25                        |
| frozen      |            3 |                       46.000 |                        41.171 |                             35.755 |                     46.000 | 0.638                    | 25–50                        |
| frozen      |            3 |                       56.000 |                        57.583 |                             72.267 |                     56.000 | 0.750                    | 50–100                       |
| frozen      |            3 |                       37.000 |                       121.862 |                            166.452 |                     37.000 | 0.793                    | >100                         |
| lora        |            3 |                       13.000 |                        38.303 |                              2.001 |                      0.000 | Sin pares elegibles      | 0–5                          |
| lora        |            3 |                       17.000 |                        30.736 |                              7.847 |                     17.000 | 0.588                    | 5–10                         |
| lora        |            3 |                       41.000 |                        41.090 |                             16.804 |                     41.000 | 0.545                    | 10–25                        |
| lora        |            3 |                       46.000 |                        43.703 |                             35.755 |                     46.000 | 0.587                    | 25–50                        |
| lora        |            3 |                       56.000 |                        53.386 |                             72.267 |                     56.000 | 0.744                    | 50–100                       |
| lora        |            3 |                       37.000 |                       122.523 |                            166.452 |                     37.000 | 0.775                    | >100                         |

## Interpretación técnica

La diferencia media entre condiciones en MAE residual fue -0.22 meV/átomo; en MAE de pares fue -0.57 meV/átomo. Interpretar cada diferencia junto con su IC95% apareado; no llamar mejora a una diferencia cuyo intervalo incluya cero.

Las energías medias por fórmula provienen de cinco Ridge históricos congelados sobre L03_MEAN. El MLP8 predice residuos centrados de energía DFT por átomo. ΔE se deriva restando energías dentro de fórmula; no se entrenó una pérdida pairwise. La rama Ridge permaneció invariante en la auditoría del primer checkpoint LoRA.

Referencia H3-C histórica aproximada: MAE E/N 183.3 meV/átomo, MAE de pares 58.8, estado fundamental 54.3% por argmin bruto y orden de pares elegibles 66.8%. La métrica elegible con abstenciones fue 53.2% en el replay científico final; se mantienen las definiciones y denominadores separados.

## Recursos observados

Corridas completas: 30. Tiempo total registrado por proceso: 1.12 h; máximo RSS entre corridas: 2.02 GB. Cada log y JSON de recursos individual está en `runs/`.

## Límites

Las 70 fórmulas ya participaron en análisis anteriores; estos resultados son exploratorios. El caché de features recuperado fue reconstruido y se aceptó por auditoría numérica, no por igualdad de hash con el archivo histórico. El conjunto DFT gga-static conserva la procedencia original disponible; este protocolo no homogeneiza XC ni correcciones. El éxito de ranking no implica precisión de ΔE, y una pendiente mayor no demuestra causalidad de representación.

**Estado del test histórico:** no abierto ni evaluado.



## Reproducción del H3-C congelado

|      seed |   n_structures |   mae_prediction_difference_mev_atom |   max_abs_prediction_difference_mev_atom |   mae_new_mev_atom |   mae_historical_mev_atom |
|----------:|---------------:|-------------------------------------:|-----------------------------------------:|-------------------:|--------------------------:|
|   42.0000 |       210.0000 |                               0.0001 |                                   0.0007 |           183.8634 |                  183.8634 |
|  123.0000 |       210.0000 |                               0.0001 |                                   0.0007 |           182.9997 |                  182.9997 |
| 2026.0000 |       210.0000 |                               0.0001 |                                   0.0007 |           183.0235 |                  183.0235 |

Las diferencias se calcularon por ID, fold y semilla contra el replay histórico disponible; no se usaron para seleccionar checkpoints.