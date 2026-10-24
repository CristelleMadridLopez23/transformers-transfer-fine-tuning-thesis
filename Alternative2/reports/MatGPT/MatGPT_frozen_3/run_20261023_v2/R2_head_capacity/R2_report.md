# R2 — capacidad de las cabezas

**Estado: PASS.**

R2-small reduce ambas MLP de ancho 8 a ancho 4; R2-reference conserva la arquitectura H2 de ancho 8. Se mantiene el mismo masked-mean para μ, h_end para δ, pérdidas SmoothL1 normalizadas, AdamW, LR, WD, folds, semillas y selección interna. Parámetros entrenables: 6,162 frente a 12,322. H2 histórico se muestra como contexto; la referencia R2-REF8 fue reentrenada con el runner nuevo.

Energías en meV/átomo. H0/H1/H2 y las cabezas tienen 3 semillas: la métrica principal promedia predicciones por Structure ID y recalcula cada fórmula una vez. Ridge es determinista y no se duplica artificialmente por semilla. MAE E solo aplica a predictores de energía total; MAE μ y δ se reportan para cabezas de componente. GS excluye gap DFT ≤1 meV/átomo; abstención no es acierto. Pairwise excluye empates DFT de ±1 meV/átomo. Todas las cifras son OOF development; test histórico excluido. La selección previa en validation histórica vuelve exploratorios los intervalos.

## Métricas

| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2-SMALL4 | absolute | 337.51 | 336.55 | 32.58 | 40.36 | 36.89 | 58.2% (67) | 67.0% (206) | 51.35 |
| R2-REF8 | absolute | 352.53 | 351.29 | 32.01 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |

## Lectura

Las diferencias absolutas y relativas deben leerse juntas. Una predicción mejor de μ no implica mejor ordenamiento de polimorfos. La interpretación por pares y fórmulas, con intervalos pareados, está en el informe combinado.
