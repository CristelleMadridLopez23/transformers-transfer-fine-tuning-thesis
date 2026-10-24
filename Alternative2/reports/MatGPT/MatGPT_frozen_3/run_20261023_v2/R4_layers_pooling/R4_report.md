# R4 — capa y pooling

**Estado: PASS.**

R4 comparó END/MEAN de bloques 3, 6, 9 y 12 en salida cruda pre-ln_f, más controles finales post-ln_f END/MEAN. Ridge alpha y representación se escogieron por separado en inner-validation para E, μ y δ; la evaluación outer incluye solo candidatos elegidos internamente y controles finales fijos. H2/R4-SELECTED-2COMP combina las capas elegidas para μ y δ. La cabeza MLP secundaria R4-B no se ejecutó: no es necesaria para completar el diagnóstico lineal y queda como extensión condicionada.

Energías en meV/átomo. H0/H1/H2 y las cabezas tienen 3 semillas: la métrica principal promedia predicciones por Structure ID y recalcula cada fórmula una vez. Ridge es determinista y no se duplica artificialmente por semilla. MAE E solo aplica a predictores de energía total; MAE μ y δ se reportan para cabezas de componente. GS excluye gap DFT ≤1 meV/átomo; abstención no es acierto. Pairwise excluye empates DFT de ±1 meV/átomo. Todas las cifras son OOF development; test histórico excluido. La selección previa en validation histórica vuelve exploratorios los intervalos.

## Métricas

| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4-SELECTED-ABSOLUTE | absolute | 153.27 | 125.55 | 87.92 | 99.49 | 101.08 | 35.8% (67) | 53.4% (206) | 160.53 |
| R4-SELECTED-MU | mu | NA | 213.61 | NA | NA | NA | NA% (0) | NA% (0) | 90.71 |
| R4-SELECTED-DELTA | delta | NA | NA | 32.88 | 41.06 | 37.81 | 50.7% (67) | 61.2% (206) | 51.24 |
| R4-SELECTED-2COMP | mu_plus_delta | 217.54 | 213.61 | 32.88 | 41.06 | 37.81 | 50.7% (67) | 61.2% (206) | 51.24 |
| R4-FIXED-FINAL_END | mu_plus_delta | 507.78 | 507.32 | 30.44 | 38.07 | 35.63 | 56.7% (67) | 68.4% (206) | 56.47 |
| R4-FIXED-FINAL_MEAN | mu_plus_delta | 230.06 | 227.70 | 30.81 | 38.11 | 35.69 | 55.2% (67) | 63.1% (206) | 56.66 |

## Lectura

Las diferencias absolutas y relativas deben leerse juntas. Una predicción mejor de μ no implica mejor ordenamiento de polimorfos. La interpretación por pares y fórmulas, con intervalos pareados, está en el informe combinado.
