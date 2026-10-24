# MatGPT congelado: H0/H1/H2

Este experimento comprueba si dos cabezas separadas sobre el mismo MatGPT original congelado pueden representar el nivel energético medio de una composición y las diferencias entre sus tres polimorfos. Todo el código está aquí y los resultados se guardan en `Alternative2/reports/MatGPT_frozen_2/v1/`.

## Modelos

- **H0:** control histórico; h_end → MLP 768→8→1, pérdida Huber absoluta normalizada más pérdida pairwise E1.3 (λ=1).
- **H1:** una MLP8 predice la media desde el promedio de los tres h_end; otra MLP8 predice residuos por estructura. Se centran los residuos del triplete y la energía es μ+δ.
- **H2:** misma arquitectura que H1, pero la cabeza de media recibe el promedio de los tres masked-mean. La cabeza residual conserva h_end.

MatGPT permanece en evaluación, con carga estricta de pesos. Solo se actualizan las cabezas. h_end se toma en el índice real de `<end>`, tras `ln_f` y antes de `lm_head`; masked-mean excluye `<start>`, `<end>` y `<pad>`. No se entregan energías al modelo. La inferencia de H1/H2 requiere las tres estructuras candidatas de una composición.

## Protocolo y fuentes

Se reutiliza el mapa exacto de GroupKFold E1.3 y sus inner holdouts por composición: 5 folds, 70 fórmulas, 210 estructuras de desarrollo; 14 fórmulas outer, 11 inner-validation y 45 inner-train por fold; semillas 42, 123 y 2026. El test histórico (24 estructuras, 8 composiciones) se elimina antes de extraer vectores o etiquetas útiles. Las predicciones de validation histórica ya influyeron en selección previa, por lo que los resultados actuales son exploratorios.

H0 replica el código E1.3: AdamW, LR 1e-3, weight decay 1e-4, lotes de 8 composiciones, SmoothL1 β=1, escalado de energía con media/std de inner-train, escala pairwise igual al RMS de diferencias DFT de inner-train con piso 0.001 eV/átomo, máximo 200 épocas, paciencia 20, mínimo objetivo de inner-validation. H1/H2 conservan esos hiperparámetros de optimización/parada; sus dos escalas SmoothL1 se estiman por separado solo con inner-train y las pérdidas de media y residuo tienen pesos fijos 1:1.

La etiqueta es energía total estática GGA/PBE por átomo en eV/átomo, no energía de formación. El SHA-256 esperado del checkpoint es `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`; el código de MatGPT corresponde al commit `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.

## Ejecución por etapas

Usa el entorno `transformers-transfer-fine-tune-tesis` con PyTorch, NumPy, pandas, scikit-learn y matplotlib. En este Mac el Python de `dmc` aborta al cargar dos runtimes OpenMP; por eso los comandos emplean el intérprete del entorno del experimento, sin activar `KMP_DUPLICATE_LIB_OK`.

Desde la raíz del proyecto:

```bash
PY=/Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python
$PY -u Alternative2/scripts/MatGPT_frozen_2/prepare_experiment.py
$PY -u Alternative2/scripts/MatGPT_frozen_2/extract_frozen_features.py --device cpu
$PY -m unittest discover -s Alternative2/scripts/MatGPT_frozen_2/tests -v
$PY -u Alternative2/scripts/MatGPT_frozen_2/run_all.py --phase smoke
$PY -u Alternative2/scripts/MatGPT_frozen_2/run_all.py --phase pilot
$PY -u Alternative2/scripts/MatGPT_frozen_2/run_all.py --phase full
$PY -u Alternative2/scripts/MatGPT_frozen_2/audit_cv.py --phase all
```

Smoke ejecuta fold 0 / semilla 42 para H0/H1/H2 con el protocolo final; pilot añade las semillas 123 y 2026 para fold 0; full reanuda esas corridas y completa las otras cuatro particiones. Así el total final es 45 corridas, no una suma de pilotos duplicados. `run_all.py --phase all` ejecuta las etapas consecutivamente. Las corridas completadas se reanudan desde sus checkpoints y predicciones; un destino incompleto con manifiesto conflictivo se detiene para evitar sobrescritura.

## Salidas principales

`reports/MatGPT_frozen_2/v1/` contiene `provenance.json`, `integrity_checks.json`, `fold_assignments.csv`, `features/embeddings_manifest.json`, checkpoints y manifiestos por corrida, OOF sin y con campos evaluativos, métricas por composición/fold/semilla/familia, las dos convenciones ΔE, descomposición H0, matrices GS/pairwise, bootstrap pareado, estadísticas OLS, figuras PNG/SVG/PDF y el reporte científico. Los SVG/PDF se generan por solicitud explícita del protocolo H0/H1/H2; los resultados científicos proceden solo de predicciones CSV reales.

Los scripts principales son `prepare_experiment.py`, `extract_frozen_features.py`, `train_grouped_cv.py`, `evaluate_energy_conventions.py`, `audit_cv.py` y `run_all.py`. `config.yaml` conserva hiperparámetros fijados; las funciones compartidas viven en `common.py`.

## Referencias

- MatGPT: https://github.com/devalab/matgpt (commit histórico local).
- SLICES: Xiao et al., *Nature Communications* 14, 7027 (2023), https://doi.org/10.1038/s41467-023-42870-7.
- MatFold: Witman & Schindler, *Digital Discovery* 4, 625–635 (2025), https://doi.org/10.1039/D4DD00250D.
- PBE: Perdew, Burke & Ernzerhof, *Physical Review Letters* 77, 3865 (1996), https://doi.org/10.1103/PhysRevLett.77.3865.
- SmoothL1Loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html.
- Matrices de confusión: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html.
- pymatgen `ComputedStructureEntry`: https://pymatgen.org/pymatgen.core.html.
