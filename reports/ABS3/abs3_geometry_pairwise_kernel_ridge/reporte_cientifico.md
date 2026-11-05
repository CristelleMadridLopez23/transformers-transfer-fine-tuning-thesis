# ABS₃: Kernel Ridge no lineal sobre geometría para ΔE polimórfica

## Pregunta y protocolo

Se comparó el Ridge pairwise lineal geométrico histórico (C0) con Kernel Ridge RBF sobre el mismo vector de siete descriptores (C1). C2 reaplica las predicciones de C1 con un único factor escalar positivo por fold, estimado solo con predicciones de validación interna. No se entrenaron Transformers ni se modificaron Ridge M2.

La cohorte contiene 70 composiciones ABS₃, 210 estructuras y tres polimorfos por composición. Se reutilizaron los cinco folds agrupados históricos (45 train, 11 validación y 14 outer por fold). El StandardScaler se ajustó solo sobre las 135 estructuras inner-train y se aplicó sin reajuste a validación y outer. La unidad de bootstrap es la composición completa. No se accedió al test independiente. La geometría proviene de celdas finales ya relajadas por DFT.

El KRR usa el kernel de diferencias inducido por un kernel RBF estructural: `K_pair((i,j),(k,l)) = k(j,l)-k(j,k)-k(i,l)+k(i,k)`. Se probaron 12 combinaciones (multiplicadores gamma 0.25/1/4 y lambda 0.001/0.01/0.1/1). La selección usa MAE de ΔE de validación; empates se rompen por ranking, mayor lambda y menor gamma. El refit utiliza inner-train+validación tras seleccionar, conservando el scaler train-only. C0 son las predicciones históricas GEO y sus métricas se reprodujeron desde los CSV originales.

## Resultados OOF

Energía y errores en meV/átomo. `ΔE MAE` es el estimando A: macro-promedio por fórmula de los tres pares firmados; RMSE es global sobre observaciones. Pairwise exige `|ΔE_DFT|>5`; `|ΔE_pred|≤1` es no resuelto e incorrecto. Ground state exige gap DFT al segundo mínimo >1; el orden completo excluye fórmulas con algún empate DFT ≤1.

| Condición | Energy MAE | Energy RMSE global | ΔE MAE A | ΔE RMSE global | Pairwise | Ground state | Triplete completo | Pendiente OLS | R² predictivo ΔE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge pairwise lineal GEO | 47.55 | 65.48 | 49.78 | 70.65 | 136/197 (69.04%) | 42/67 (62.69%) | 19/66 (28.79%) | 0.293 | 0.321 |
| KRR-RBF sin calibrar | 48.02 | 66.02 | 50.53 | 72.14 | 136/197 (69.04%) | 42/67 (62.69%) | 22/66 (33.33%) | 0.265 | 0.292 |
| KRR-RBF con calibración positiva | 47.68 | 66.19 | 49.57 | 72.62 | 136/197 (69.04%) | 42/67 (62.69%) | 22/66 (33.33%) | 0.227 | 0.283 |

El baseline GEO tiene 47.55 meV/átomo de Energy MAE, 49.78 de ΔE MAE, 136/197 (69.04%) pairwise, 42/67 (62.69%) ground state y 19/66 (28.79%) tripletes exactos. KRR sin calibrar no mejora Energy MAE ni ΔE MAE; mantiene la exactitud pairwise y ground state, aunque aumenta el orden completo exacto a 22/66. C2 reduce ΔE MAE puntualmente a 49.57, pero no alcanza 90% de ranking ni supera el baseline en energía absoluta.

## Hiperparámetros y calibración por fold

| Fold | Multiplicador γ | γ | λ | MAE ΔE validación | Factor positivo C2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.0786897 | 1 | 37.62 | 0.879 |
| 1 | 4 | 0.306506 | 1 | 50.45 | 0.705 |
| 2 | 1 | 0.0770841 | 1 | 51.62 | 0.795 |
| 3 | 0.25 | 0.0190084 | 1 | 38.32 | 0.755 |
| 4 | 0.25 | 0.0192985 | 1 | 60.13 | 1.024 |

## Bootstrap apareado

10 000 réplicas percentiles por composición (semilla 20261103); diferencias C1/C2 − C0. Negativo favorece KRR para errores y positivo para exactitudes. Los intervalos describen OOF de desarrollo, no una prueba externa.

| Contraste | Métrica | Diferencia | IC95% |
|---|---|---:|---:|
| C1_KRR_RBF − C0_GEO_LINEAR | energy_mae_mev_atom | 0.469 meV/átomo | [-0.543, 1.540] |
| C1_KRR_RBF − C0_GEO_LINEAR | deltaE_mae_estimand_A_mev_atom | 0.750 meV/átomo | [-2.256, 3.732] |
| C1_KRR_RBF − C0_GEO_LINEAR | pairwise_accuracy_pct | -1.523 puntos porcentuales | [-7.369, 4.041] |
| C1_KRR_RBF − C0_GEO_LINEAR | groundstate_accuracy_pct | 0.000 puntos porcentuales | [-8.571, 8.571] |
| C1_KRR_RBF − C0_GEO_LINEAR | full_triplet_accuracy_pct | 4.545 puntos porcentuales | [-6.557, 16.176] |
| C2_KRR_RBF_CALIBRATED − C0_GEO_LINEAR | energy_mae_mev_atom | 0.129 meV/átomo | [-0.934, 1.256] |
| C2_KRR_RBF_CALIBRATED − C0_GEO_LINEAR | deltaE_mae_estimand_A_mev_atom | -0.217 meV/átomo | [-3.136, 2.705] |
| C2_KRR_RBF_CALIBRATED − C0_GEO_LINEAR | pairwise_accuracy_pct | -1.523 puntos porcentuales | [-7.369, 4.041] |
| C2_KRR_RBF_CALIBRATED − C0_GEO_LINEAR | groundstate_accuracy_pct | 0.000 puntos porcentuales | [-8.571, 8.571] |
| C2_KRR_RBF_CALIBRATED − C0_GEO_LINEAR | full_triplet_accuracy_pct | 4.545 puntos porcentuales | [-6.557, 16.176] |

## Error por magnitud del gap DFT

| Condición | Intervalo | n pares | ΔE MAE (meV/átomo) | ΔE RMSE global (meV/átomo) |
|---|---|---:|---:|---:|
| Ridge pairwise lineal GEO | 0–5 | 13 | 10.63 | 13.94 |
| Ridge pairwise lineal GEO | 5–10 | 17 | 24.21 | 31.74 |
| Ridge pairwise lineal GEO | 10–25 | 41 | 21.41 | 26.19 |
| Ridge pairwise lineal GEO | 25–50 | 46 | 39.61 | 46.91 |
| Ridge pairwise lineal GEO | 50–100 | 56 | 52.55 | 63.07 |
| Ridge pairwise lineal GEO | >100 | 37 | 115.17 | 135.20 |
| KRR-RBF sin calibrar | 0–5 | 13 | 17.10 | 21.43 |
| KRR-RBF sin calibrar | 5–10 | 17 | 30.57 | 36.04 |
| KRR-RBF sin calibrar | 10–25 | 41 | 22.84 | 27.76 |
| KRR-RBF sin calibrar | 25–50 | 46 | 35.62 | 43.94 |
| KRR-RBF sin calibrar | 50–100 | 56 | 53.97 | 67.18 |
| KRR-RBF sin calibrar | >100 | 37 | 115.46 | 136.72 |
| KRR-RBF con calibración positiva | 0–5 | 13 | 13.88 | 18.00 |
| KRR-RBF con calibración positiva | 5–10 | 17 | 25.55 | 31.07 |
| KRR-RBF con calibración positiva | 10–25 | 41 | 18.22 | 22.86 |
| KRR-RBF con calibración positiva | 25–50 | 46 | 32.68 | 40.40 |
| KRR-RBF con calibración positiva | 50–100 | 56 | 52.92 | 64.41 |
| KRR-RBF con calibración positiva | >100 | 37 | 123.79 | 143.13 |

## Interpretación y límites

La función no lineal regularizada no supera de forma general al Ridge pairwise lineal geométrico. C1 presenta pendiente OLS menor (más compresión) que GEO; la escala positiva elegida en C2 también deja la pendiente por debajo de 1 y no cambia signos de ranking. El aumento de exactitud de triplete de C1 es puntual y no basta para afirmar una mejora general; revisar sus intervalos apareados. La meta ≥90% no se alcanza. Los resultados son OOF sobre desarrollo, con 70 composiciones, y no sustituyen el test histórico cerrado.

Las condiciones se seleccionaron por validación interna. C2 es exploratoria y comparte el mismo conjunto de validación para la selección de kernel y la escala; no debe presentarse como resultado confirmatorio. Las celdas ya relajadas por DFT limitan la interpretación como predicción sin cálculos estructurales previos.

Recursos medidos: 3.70 s de pared, 3.57 s CPU, RAM máxima 438288384 bytes, 1 hilo CPU; sin GPU/MPS.
