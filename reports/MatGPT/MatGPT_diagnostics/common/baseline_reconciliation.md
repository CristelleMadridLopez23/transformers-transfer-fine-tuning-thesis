# Reconciliación de métricas OOF E2-A y E2-Aug

## Qué originó 0.533 frente a 0.495

Las cifras usan las mismas predicciones OOF de E2-A partial fine-tuning: 210 estructuras, 70 composiciones de desarrollo, cinco folds y tres semillas. Ninguna usa test. La tabla histórica en `E2_report.md` aplicó `argmin` a cada triplete y siempre eligió un polimorfo, incluso cuando la diferencia DFT del mínimo era de 1 meV/átomo o menor, o la predicción empataba. Su media es **0.533** sobre las 70 composiciones.

La comparación posterior usa una regla de ranking más prudente: solo considera pares con diferencia DFT >1 meV/átomo; GS excluye composiciones cuyo gap entre los dos mínimos DFT sea ≤1 meV/átomo. Una predicción empatada en el mínimo cuenta como **no acierto** y se reporta como no discriminativa, sin elegir el primer polimorfo. Con esa cobertura, GS promediado primero por fold y luego por semilla es **0.495**, de 67 composiciones elegibles de 70 por semilla. Si se agrupan directamente las 67 composiciones elegibles, sin promediar folds por separado, el resultado es aproximadamente 0.493. El 0.495 publicado usa la primera agregación; el 0.533 antiguo usa `argmin` sobre todas las composiciones, incluso las DFT casi empatadas. Por tanto, la diferencia es la definición del denominador/ponderación, no un checkpoint distinto.

**Convención para D1–D3:** mantener esta regla estricta para cualquier comparación de ranking; reportar siempre cuántas composiciones/pares son elegibles. Mantener el `argmin` histórico solo como columna de trazabilidad, nunca como métrica principal. Las métricas de MAE absoluto y ΔE se calculan sobre tripletes completos.

## Procedencia y cobertura

| Entrada | Filas/semilla | Semillas | Folds | Composiciones/semilla |
|---|---:|---|---|---:|
| matgpt_e2 | 210 | [42, 123, 2026] | [0, 1, 2, 3, 4] | 70 |
| development_only_v2 | 210 | [42, 123, 2026] | [0, 1, 2, 3, 4] | 70 |
| development_only_v2 | 210 | [42, 123, 2026] | [0, 1, 2, 3, 4] | 70 |

Las medias recalculadas por semilla, denominadores y hashes están en `baseline_reconciliation_metrics.csv` y `provenance.json`. E2-Aug1/E2-Aug2 son comparables en los mismos 70 grupos y folds. La comparación histórica de modelos fue seleccionada usando la validation de E1.1/E1.2; este análisis de desarrollo no elimina ese sesgo. Los ocho grupos históricos de test permanecieron excluidos.
