# Energía intrafórmula y magnitud del gap

[HECHO VERIFICADO] Para cada fórmula, `e_i = Etotal_i/N_i`; se calculan por separado las medias DFT y predicha, y los residuos `e_i − media(e_c)`. La media DFT solo entra en métricas posteriores, jamás en inferencia. Los pares se forman exclusivamente dentro de la misma fórmula.

[RESULTADO EXPERIMENTAL] H3-C: error de media por fórmula 180.2 meV/átomo, error de residuo 33.8, MAE de pares 58.8, ΔE anclado al mismo estado fundamental DFT 41.5. Pendiente residual 0.308 y razón de dispersión 0.656; esto cuantifica compresión intrafórmula de la salida, no colapso del embedding.

| Magnitud de ΔE DFT (meV/átomo) | Pares físicos/semilla | MAE H3-C | MAE ΔE=0 | Acierto de orden elegible |
|---|---:|---:|---:|---:|
| 0–5 | 13 | 32.9 | 2.0 | no definido |
| 5–10 | 17 | 33.8 | 7.8 | 54.9% |
| 10–25 | 41 | 42.1 | 16.8 | 52.8% |
| 25–50 | 46 | 41.2 | 35.8 | 63.8% |
| 50–100 | 56 | 57.6 | 72.3 | 75.0% |
| >100 | 37 | 121.9 | 166.5 | 79.3% |

[HECHO VERIFICADO] El umbral de elegibilidad principal de esta auditoría es >5 meV/átomo: 197 pares por semilla. El análisis histórico de H3-C usa una tolerancia de 1 meV/átomo y otro denominador; ambas tablas se preservan en `analysis/`. Los intervalos con menos de diez pares se marcan en `delta_bins.csv`.

[RESULTADO EXPERIMENTAL] Diferencia H3-C menos aleatorio en MAE de pares: -3.38 [-8.72, +1.89] meV/átomo, intervalo que incluye cero. Frente a ΔE=0, la reducción media es solo 1.6 meV/átomo. No confundir cancelación de offset composicional con precisión de diferencias.

[RESULTADO EXPERIMENTAL] En los 70 tripletes de cada semilla, el promedio de Spearman/Kendall fue 0.379/0.343, 0.357/0.324 y 0.407/0.352 para H3-C preentrenado (semillas 42, 123, 2026). El control aleatorio obtuvo 0.221/0.200, 0.093/0.095 y 0.200/0.171. El control de composición empata en todos los tripletes; ambas correlaciones quedan indefinidas, sin adjudicarle orden. El detalle por fórmula está en `analysis/rank_correlations_by_formula.csv`.

[LIMITACIÓN] El código histórico conserva rutas internas anteriores a una reorganización del repositorio. La reproducción de solo lectura usa rutas actuales explícitas y recarga los checkpoints originales; no modifica el código ni los pesos históricos. Los resultados de desarrollo no son una validación externa ciega.

