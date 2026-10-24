# E1.3 — Validación cruzada agrupada por composición

## Estado y alcance

E1.3 evalúa 3 cabezas congeladas con GroupKFold de 5 pliegues sobre 210 estructuras / 70 composiciones de desarrollo. Se excluyeron completamente las 24 estructuras / 8 composiciones del test histórico. Se completaron 45 ejecuciones (3 candidatos × 5 folds × 3 semillas).

Las configuraciones A/B/C fueron preseleccionadas usando las mismas 8 composiciones de validation histórica E1.1/E1.2. Esta CV mide sensibilidad y robustez; no elimina el sesgo de selección. Para estimación independiente se requiere un holdout externo nuevo o nested CV completo. El test histórico fue observado en E1, no es ciego, y no se consultó aquí.

E1.2 incluyó análisis geométricos exploratorios de los 234 embeddings; cualquier correlación que incluyera test está registrada como exploratoria y no guió folds ni selección en E1.3.

## Protocolo congelado

GroupKFold asigna grupos por fórmula química completa; cada composición aparece una vez en outer-validation. En cada outer-training se separaron 11 de 56 grupos (20%, 33 estructuras) para inner-validation determinista (semilla base 13072026 + índice fold). Las 45 ejecuciones optimizan pesos solo con inner-train. Objetivo/checkpoint se elige por mínimo objetivo normalizado de inner-validation, con paciencia 20 y máximo 200 épocas. Escalas de energía y diferencias de pares se ajustan exclusivamente con inner-train. MatGPT permanece congelado; solo se entrenan las cabezas. Las predicciones outer se calculan una vez al final de cada ejecución.

## Resultados

MAE absoluta pooled se pondera por estructura; MAE macro y ΔE MAE se promedian por composición. GS excluye grupos con empate DFT o predicho a 1 meV/átomo. Exactitud por pares omite pares DFT empatados bajo la misma tolerancia; las predicciones ML empatadas no aciertan un orden. Las semillas se resumen por separado y no cuentan como composiciones independientes.

| Candidato | MAE macro fold media ± DE (eV/átomo) | ΔE MAE fold media ± DE (meV/átomo) | GS fold media ± DE | Pares fold media ± DE | MAE OOF entre semillas media ± DE | ΔE MAE OOF entre semillas media ± DE |
|---|---:|---:|---:|---:|---:|---:|
| matgpt_e13_mlp32_mean_abs | 0.2202 ± 0.0322 | 138.37 ± 39.47 | 0.327 ± 0.120 | 0.495 ± 0.089 | 0.2202 ± 0.0071 | 138.37 ± 0.82 |
| matgpt_e13_mlp8_last_pair1 | 0.7299 ± 0.0827 | 37.01 ± 6.41 | 0.552 ± 0.150 | 0.663 ± 0.063 | 0.7299 ± 0.0036 | 37.01 ± 1.31 |
| matgpt_e13_mlp8_mean_pair1 | 0.7292 ± 0.0848 | 37.67 ± 7.15 | 0.486 ± 0.150 | 0.585 ± 0.075 | 0.7292 ± 0.0046 | 37.67 ± 0.68 |

Medianas y outliers por composición (tabla `metrics_composition.csv`, valores macro entre seeds):

| Candidato | Mediana MAE comp. (eV/átomo) | Mediana MAE ΔE (meV/átomo) | Outliers MAE Tukey (comp×seed) | Colapsos <1 meV/átomo (comp×seed) |
|---|---:|---:|---:|---:|
| matgpt_e13_mlp8_mean_pair1 | 0.7009 | 32.01 | 0 | 0 |
| matgpt_e13_mlp8_last_pair1 | 0.7172 | 30.84 | 0 | 0 |
| matgpt_e13_mlp32_mean_abs | 0.1984 | 115.15 | 9 | 0 |

## Incertidumbre pareada y outliers

Bootstrap pareado por composición: 2,000 réplicas; semilla 13072026; cada remuestra toma grupos completos con sus tres polimorfos. Intervalos porcentiles exploratorios, no garantía de superioridad.

| Comparación (izquierda − derecha) | Métrica | Diferencia media pareada | IC bootstrap 95% |
|---|---|---:|---:|
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp8_last_pair1 | delta_e_mae_mev_atom | 0.6601 | [-2.24, 3.732] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp8_last_pair1 | ground_state_accuracy | -0.07381 | [-0.169, 0.02143] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp8_last_pair1 | pairwise_accuracy | -0.08175 | [-0.1483, -0.01288] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp8_last_pair1 | absolute_range_error_mev_atom | 6.085 | [-0.8374, 13.2] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp32_mean_abs | delta_e_mae_mev_atom | -100.7 | [-123.4, -80.41] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp32_mean_abs | ground_state_accuracy | 0.1548 | [0.02857, 0.2786] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp32_mean_abs | pairwise_accuracy | 0.08968 | [-0.001684, 0.1804] |
| matgpt_e13_mlp8_mean_pair1__vs__matgpt_e13_mlp32_mean_abs | absolute_range_error_mev_atom | -159.9 | [-207.9, -115.2] |
| matgpt_e13_mlp8_last_pair1__vs__matgpt_e13_mlp32_mean_abs | delta_e_mae_mev_atom | -101.4 | [-121.9, -82.2] |
| matgpt_e13_mlp8_last_pair1__vs__matgpt_e13_mlp32_mean_abs | ground_state_accuracy | 0.2286 | [0.1071, 0.3452] |
| matgpt_e13_mlp8_last_pair1__vs__matgpt_e13_mlp32_mean_abs | pairwise_accuracy | 0.1714 | [0.08452, 0.2516] |
| matgpt_e13_mlp8_last_pair1__vs__matgpt_e13_mlp32_mean_abs | absolute_range_error_mev_atom | -166 | [-216.7, -123.5] |

Consultar también `paired_bootstrap_ci.json`, `paired_comparisons.csv` y `metrics_composition.csv`; outliers absolutos marcados con regla Tukey Q3 + 1.5×IQR.

## Criterios interpretativos predeclarados

- A favorece masked mean para mínimos solo si supera de forma consistente a B en GS/orden por pares, no presenta colapso y los intervalos pareados respaldan la diferencia.
- B favorece last-token para energías relativas si reduce consistentemente ΔE MAE/error de rango.
- C puede tener menor MAE absoluta y peor ranking; se conservan como objetivos distintos.
- Cambios de signo entre folds o intervalos amplios implican evidencia insuficiente; no se fuerza un ganador.
- Los errores no se atribuyen únicamente a la representación: también dependen de cabeza, loss, protocolo y tamaño del conjunto.

## Artefactos y limitaciones

Figuras PNG: `figures/`. Métricas: `metrics_per_fold_seed.csv`, `metrics_oof_per_seed.csv`, `metrics_composition.csv`; predicciones: `oof_predictions.csv`. La etiqueta es energía total estática GGA por átomo, no energía de formación. No se ejecutó E2.
