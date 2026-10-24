# R1 — sondas lineales MatGPT

**Estado: PASS.**

R1-A predice energía absoluta desde h_end; R1-B predice μ usando la media de tres embeddings; R1-C predice desviaciones DFT centradas y vuelve a centrar las tres predicciones; R1-D suma sondas lineales independientes de μ y δ. Los cuatro usan StandardScaler de inner-train y Ridge alpha seleccionado por MAE inner-validation. R1-B no discrimina polimorfos; R1-C no predice el nivel absoluto. D1 composición-only es un control externo y no entra en estas ecuaciones.

Energías en meV/átomo. H0/H1/H2 y las cabezas tienen 3 semillas: la métrica principal promedia predicciones por Structure ID y recalcula cada fórmula una vez. Ridge es determinista y no se duplica artificialmente por semilla. MAE E solo aplica a predictores de energía total; MAE μ y δ se reportan para cabezas de componente. GS excluye gap DFT ≤1 meV/átomo; abstención no es acierto. Pairwise excluye empates DFT de ±1 meV/átomo. Todas las cifras son OOF development; test histórico excluido. La selección previa en validation histórica vuelve exploratorios los intervalos.

## Métricas

| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1-A-ABS | absolute | 530.49 | 439.85 | 301.58 | 353.34 | 386.60 | 41.8% (67) | 54.9% (206) | 675.05 |
| R1-B-MU | mu | NA | 507.32 | NA | NA | NA | NA% (0) | NA% (0) | 90.71 |
| R1-C-DELTA | delta | NA | NA | 30.44 | 38.07 | 35.63 | 56.7% (67) | 68.4% (206) | 56.47 |
| R1-D-COMBINED | mu_plus_delta | 507.78 | 507.32 | 30.44 | 38.07 | 35.63 | 56.7% (67) | 68.4% (206) | 56.47 |

## Lectura

Las diferencias absolutas y relativas deben leerse juntas. Una predicción mejor de μ no implica mejor ordenamiento de polimorfos. La interpretación por pares y fórmulas, con intervalos pareados, está en el informe combinado.
