# Diferencias energéticas intrafórmula

Por cada fórmula, las energías por átomo DFT y predichas se centran usando la media de sus tres polimorfos. El Ridge común se cancela y la diferencia se evalúa sobre el residuo estructural MLP8.

- **frozen** — MAE E/N 183.3 ± 0.5; residuo 33.8 ± 0.7; ΔE pares 58.8 ± 1.0; GS elegible 0.532; pares elegibles 0.668.
- **lora** — MAE E/N 182.8 ± 0.3; residuo 33.5 ± 0.2; ΔE pares 58.3 ± 0.2; GS elegible 0.517; pares elegibles 0.658.

Consulte `metricas_por_composicion.csv`, `diferencias_por_par.csv`, `intervalos_gap_dft.csv` y las figuras de paridad/residuos. El predictor ΔE=0 se evalúa con el mismo universo de pares; no se asigna un estado fundamental al empate.
