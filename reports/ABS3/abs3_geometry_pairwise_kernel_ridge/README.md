# ABS₃ Geometry Pairwise Kernel Ridge

KRR-RBF geometry-only, comparado con las predicciones OOF históricas GEO. No entrena extractores ni Ridge M2.

Ejecución: `python scripts/ABS3/abs3_geometry_pairwise_kernel_ridge/run_krr.py`

Artefactos principales: `config.yaml`, `preflight.json`, `kernel_hyperparameters.csv`, `oof_predictions_structures.csv`, `oof_predictions_pairs.csv`, `metrics.json`, `bootstrap_paired.csv`, `resources.json` y `reporte_cientifico.md`. C1 es la comparación primaria; C2 es calibración positiva exploratoria. El test independiente permanece cerrado.
