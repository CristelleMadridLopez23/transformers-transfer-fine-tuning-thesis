# SLICES frente a SLICES + geometría real de celda en ABS₃

## Diseño y gate

[HECHO VERIFICADO] `preflight.py` terminó PASS tras el enlace exacto de 210 geometrías, la recuperación aislada de folds/Ridge y la auditoría numérica del caché. Cinco folds agrupados por las mismas 70 fórmulas, tres semillas (42, 123, 2026), dos condiciones apareadas: 1 260 predicciones de 210 estructuras físicas. No son 1 260 muestras independientes. Test histórico cerrado.

[HECHO VERIFICADO] MatGPT permaneció congelado. Los cinco Ridge H3-C históricos se cargaron sin reentrenarlos. Ambas condiciones entrenaron MLP8 con igual inicialización, orden de minibatches y selección por validación interna. La fusión añade siete descriptores de celda solo a la primera proyección: `Linear(768,8)` pasa a `Linear(775,8)`, **56 pesos** adicionales; 6 161 → 6 217 parámetros. Ancho 8, GELU, dropout, pérdida y target DFT por átomo se conservaron. La geometría se estandarizó exclusivamente con las 45 fórmulas inner-train de cada fold.

## Resultados en validación de desarrollo

| Métrica | SLICES solo | SLICES + celda | Diferencia fusión − control |
|---|---:|---:|---:|
| MAE energía por átomo (meV/átomo) | 183.16 | 182.39 | -0.78 |
| MAE residuo intrafórmula (meV/átomo) | 33.49 | 32.60 | -0.88 |
| MAE pares firmados (meV/átomo) | 58.60 | 56.27 | -2.32 |
| Estado fundamental elegible (%, 67 fórmulas/semilla) | 54.7 | 54.2 | -0.5 pp |
| Orden de pares >5 meV/átomo (%, 197 pares/semilla) | 66.3 | 68.7 | +2.4 pp |

[RESULTADO EXPERIMENTAL] Bootstrap apareado por 70 fórmulas, promediando primero las tres semillas: ΔMAE de pares -2.32 meV/átomo (IC 95 % [-5.11, +0.17]); ΔMAE residual -0.88 [-2.32, +0.56]; Δacierto de pares +2.37 pp [-0.68, +5.41]. Los tres intervalos incluyen cero.

[HECHO VERIFICADO] El control SLICES solo replica de cerca H3-C histórico: MAE absoluto 183.16 frente a 183.30; MAE de pares 58.60 frente a 58.84 meV/átomo. Es un reentrenamiento de la MLP8 para el control apareado; no se sustituyó el Ridge ni se usó test para selección.

[INFERENCIA] La celda aporta una tendencia favorable pequeña en MAE de diferencias y orden, pero la incertidumbre no permite afirmar una mejora reproducible en estas 70 fórmulas. La salida relativa sigue comprimida (pendiente de pares 0.260 → 0.351, lejos de 1). Añadir geometría no equivale a demostrar que sus siete descriptores causen la mejora: también cambian 56 parámetros de entrada.

[LIMITACIÓN] La procedencia es estática con endpoint limpiado, sin certificar relajación completa ni homogeneidad XC/U. El caché reconstruido pasó cotejo numérico pero no igualdad SHA-256. Las composiciones de desarrollo ya guiaron investigaciones anteriores; estos intervalos son exploratorios, no prueba externa ciega. No se utilizó el test histórico.

## Reproducción y recursos

Los archivos `structures_development.jsonl`, `geometry_cell_development.csv`, `staging_repo/recovery_manifest.json`, `results/manifest_experimento.json`, `results/metricas/` y `results/figuras/` conservan IDs, predicciones, parámetros, hashes y gráficos PNG. `resources_summary.csv` registra tiempo CPU, reloj y RSS por fase; `results/seleccion_y_entrenamiento_por_fold.csv` registra tiempo y RSS observado de las 30 corridas.

Entorno: Python 3.10.21, PyTorch 2.8.0, NumPy 2.2.6, scikit-learn 1.7.2, pymatgen disponible; CPU con dos threads, sin GPU.
