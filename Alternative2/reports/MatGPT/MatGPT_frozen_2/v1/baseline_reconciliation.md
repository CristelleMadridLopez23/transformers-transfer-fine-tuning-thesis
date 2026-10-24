# Reconciliación de GS antes de H0/H1/H2

## Diferencia histórica 0.533 vs 0.495

Los dos resultados publicados usan las mismas predicciones OOF E2-A, 210 estructuras y 70 composiciones, pero distintas reglas. **0.533** es `argmin` sin excluir composiciones casi degeneradas ni abstenciones predichas. **0.495** excluye las 3 composiciones cuyo gap DFT entre los dos mínimos es ≤1 meV/átomo, no acredita empates predichos y promedia primero dentro de fold, después entre semillas. El valor calculado directamente sobre las 67 composiciones elegibles es ≈0.493; no es el agregado fold→semilla publicado.

H0/H1/H2 usarán como métrica principal la regla estricta: DFT-GS evaluable solo si el gap al segundo mínimo supera 1 meV/átomo; una predicción empatada a tolerancia de 1 meV/átomo será abstención y no acierto. Siempre se informa el denominador elegible y excluido. Pairwise accuracy excluye pares DFT empatados (|ΔE|≤1 meV/átomo); un empate predicho no cuenta como acierto. Los valores históricos 0.533 y 0.495 se conservan solo como reconciliación; no se mezclan con resultados nuevos.

## Convención en este experimento

Se promedian primero las tres semillas por estructura para la matriz de confusión de ensamble, de modo que cada composición y pareja física cuente una sola vez. Las tablas por ejecución reportan fold y semilla separadamente. La exactitud OOF macro para estimación principal se calcula por composición elegible; se adjuntan también las agregaciones fold→semilla para que los denominadores queden visibles. El conjunto histórico de validación ya participó en preselección previa, por lo que esta CV sigue siendo exploratoria. No se consulta ni evalúa el test histórico.

Fuente reconciliada: `Alternative2/reports/MatGPT_diagnostics/common/baseline_reconciliation.md`; comparación de predicciones anteriores solo como antecedente.
