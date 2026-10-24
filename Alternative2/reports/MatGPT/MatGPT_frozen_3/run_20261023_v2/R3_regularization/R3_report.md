# R3 — regularización de las cabezas

**Estado: PASS.**

R2-REF8 actúa como baseline compartido. R3-DROP cambia dropout a 0.30; R3-WD cambia weight decay a 0.01; R3-EARLY mantiene regularización y cambia paciencia de 20 a 8 épocas. Se conserva el criterio H2 de suma de SmoothL1 internas normalizadas para seleccionar checkpoints y no se consulta outer-validation para parar.

Energías en meV/átomo. H0/H1/H2 y las cabezas tienen 3 semillas: la métrica principal promedia predicciones por Structure ID y recalcula cada fórmula una vez. Ridge es determinista y no se duplica artificialmente por semilla. MAE E solo aplica a predictores de energía total; MAE μ y δ se reportan para cabezas de componente. GS excluye gap DFT ≤1 meV/átomo; abstención no es acierto. Pairwise excluye empates DFT de ±1 meV/átomo. Todas las cifras son OOF development; test histórico excluido. La selección previa en validation histórica vuelve exploratorios los intervalos.

## Métricas

| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2-REF8 | absolute | 352.53 | 351.29 | 32.01 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |
| R3-DROP30 | absolute | 367.96 | 366.57 | 30.23 | 37.51 | 34.92 | 62.7% (67) | 70.9% (206) | 50.97 |
| R3-WD1E2 | absolute | 352.78 | 351.57 | 32.00 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |
| R3-EARLY8 | absolute | 419.37 | 417.79 | 32.06 | 39.51 | 36.98 | 59.7% (67) | 69.4% (206) | 52.90 |

## Lectura

Las diferencias absolutas y relativas deben leerse juntas. Una predicción mejor de μ no implica mejor ordenamiento de polimorfos. La interpretación por pares y fórmulas, con intervalos pareados, está en el informe combinado.
