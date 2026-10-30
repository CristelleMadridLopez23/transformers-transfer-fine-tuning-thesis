# MatGPT E1: backbone congelado + regresión energética

## Estado

`PARTIAL`: entrenamiento y evaluación se completaron, pero la procedencia local no especifica el funcional DFT ni sus correcciones; verificarlo antes de interpretar energías absolutas.

## Datos

- Estructuras aceptadas: 234; test: 24 estructuras en 8 composiciones.
- Target: energía total DFT estática dividida entre número de átomos (eV/átomo); no es energía de formación.
- Etiqueta de origen: `gga-static`; procedencia exacta de XC/correcciones no presente en archivos locales.
- Split histórico por composición, heredado del dataset; no se dividió al azar por fila.

## Modelo y evaluación

- MatGPT unconditional congelado; pooling en token `<end>` posterior a `ln_f`, dimensión 768.
- MLP seleccionada en época 86 mediante MAE de validation.
- Test MLP: MAE 0.517940 eV/átomo; RMSE 0.689964 eV/átomo.
- Test baseline media: MAE 0.854271; RMSE 0.980287 eV/átomo.
- Test linear probe: MAE 0.820028; RMSE 0.951206 eV/átomo.
- Ground-state accuracy: 3/8 composiciones evaluables; 0 con empate predicho quedan sin selección evaluable.
- Pairwise ordering accuracy: 12/24 pares; empates DFT o predichos de hasta 1 meV/átomo excluidos.

## Límites

- La selección de estado base compara únicamente las estructuras incluidas de la misma composición; no mide estabilidad frente al convex hull ni estabilidad experimental.
- No se verificó solapamiento entre las estructuras de evaluación y el corpus de preentrenamiento Alex-20.
- Pooling causal en un único token; resultados sujetos al tamaño y composición del split.
- El estado permanece PARTIAL hasta verificar la procedencia DFT y sus convenciones energéticas.

## Artefactos

- `dataset_audit.json`, `split_manifest.json`, `embedding_audit.json`, `training_history.csv`, `metrics_by_split.csv`.
- `predictions.csv`, `polymorph_metrics.csv`, `pairwise_comparisons.csv`, `best_mlp.pt`, `best_linear.pt`.
- Figuras científicas PNG en `figures/`; no se exportan PDF ni SVG.
