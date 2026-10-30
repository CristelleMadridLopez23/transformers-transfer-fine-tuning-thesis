# Ordenamiento y estado fundamental

Se evaluaron tripletes por fórmula. Pairwise usa |ΔE DFT| > 5 meV/átomo; predicciones con diferencia ≤1 meV/átomo cuentan como no resueltas e incorrectas. Ground-state elegible requiere un mínimo DFT separado por >1 meV/átomo; un empate de predicción no se acredita.

- **frozen** — MAE E/N 183.3 ± 0.5; residuo 33.8 ± 0.7; ΔE pares 58.8 ± 1.0; GS elegible 0.532; pares elegibles 0.668.
- **lora** — MAE E/N 182.8 ± 0.3; residuo 33.5 ± 0.2; ΔE pares 58.3 ± 0.2; GS elegible 0.517; pares elegibles 0.658.

Los denominadores completos están en `metricas_por_semilla.csv` y los pares en `diferencias_por_par.csv`. El test histórico no se usó.
