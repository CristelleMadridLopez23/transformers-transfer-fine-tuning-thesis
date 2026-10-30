# Verificación de invariancia de ranking

La rama composicional aporta una constante por fórmula. Para cada fórmula esa constante se cancela en los tres gaps; los residuos MLP8 usados como entrada no cambian.

**Resultado:** PASS. Se comprobaron 30 combinaciones de rama residual, fold y semilla, 70 fórmulas y 210 pares por comparación.

Máxima diferencia absoluta de ΔE: 8.88e-13 meV/átomo; máxima diferencia de MAE residual: 0 meV/átomo.

La MAE absoluta puede cambiar porque el intercepto por fórmula cambia; la MAE residual, los gaps firmados, el ranking y el estado fundamental deben permanecer idénticos. Se reportan ambas propiedades por separado. El predictor solo composicional empata los tres polimorfos; no se le asigna un estado fundamental ni un acierto de ranking.
