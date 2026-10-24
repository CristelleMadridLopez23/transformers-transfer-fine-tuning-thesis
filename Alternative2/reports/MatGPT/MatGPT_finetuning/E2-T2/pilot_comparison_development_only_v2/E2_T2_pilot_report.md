# E2-T2 — piloto de supervisión de trayectorias

Estado: PILOT_ONLY. T1 LIMITED permite un único piloto; no autoriza CV completa. Se comparan predicciones estáticas outer del fold 0 / semilla 42 sobre los mismos 14 grupos (42 estructuras). El test histórico no se usó.

| Métrica macro por composición | E2-Aug2 | E2-T2 piloto |
|---|---:|---:|
| MAE absoluta (meV/átomo) | 720.2096 | 731.8507 |
| MAE ΔE (meV/átomo) | 39.4374 | 38.2994 |
| Exactitud GS | 0.3846 | 0.5385 |
| Pairwise accuracy | 0.6310 | 0.7143 |
| Error de rango (meV/átomo) | 54.5396 | 56.3853 |

Bootstrap pareado por las 14 composiciones (2,000 remuestras, semilla 13072026), IC exploratorios. No interpretar cambios como una mejora generalizable por contar solo con un fold y una semilla. Las métricas principales siguen siendo estáticas; no se evalúa estado fundamental usando frames de relajación.
