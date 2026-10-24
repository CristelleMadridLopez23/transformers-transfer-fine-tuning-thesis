# MatGPT_frozen_3: sondas lineales, capacidad y regularización

Este paquete implementa R1–R4 con el MatGPT unconditional de E0 congelado. Solo lee las 210 estructuras y 70 composiciones de development. No extrae ni consulta el test histórico. Los cinco folds agrupados por composición se reutilizan de E1.3/H2; cada selección de Ridge, capa, pooling y época ocurre en inner-validation.

## Experimentos

- **R1:** Ridge para energía absoluta, media composicional μ y residuo estructural δ; también combina μ+δ. La grilla α es fija y se selecciona por inner-validation. `StandardScaler` se ajusta en inner-train.
- **R2:** compara dos cabezas MLP con ancho 4 frente a ancho 8.
- **R3:** compara dropout 0.30, weight decay 0.01 y paciencia 8 contra la referencia ancho 8.
- **R4:** compara END y MEAN de los bloques 3, 6, 9 y 12 antes de `ln_f`, junto con controles END/MEAN finales después de `ln_f`. La capa/pooling se selecciona por target dentro de inner-validation. El MLP opcional R4-B no se ejecutó.

Los cinco modelos de cabezas (R2/R3) comparten el protocolo H2: AdamW, LR 1e-3, batches de ocho composiciones, hasta 200 épocas, SmoothL1 normalizada y selección por suma de las pérdidas internas de μ y δ. Solo se entrenan las dos cabezas; sus parámetros son 6,162 para ancho 4 y 12,322 para ancho 8. Las seeds son 42, 123 y 2026.

## Ejecución

Use el entorno que contiene PyTorch, pandas, scikit-learn, SciPy y Matplotlib. Defina el intérprete del proyecto y una carpeta de configuración de Matplotlib:

```bash
PY=/Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python
export MPLCONFIGDIR=/tmp/mpl-matgpt-frozen3
```

Antes de iniciar una corrida nueva, asigne un `run_id` nuevo en `common.py` y `config.yaml`. Esto evita sobrescribir resultados anteriores. Desde la raíz del repositorio:

```bash
$PY -u Alternative2/scripts/MatGPT_frozen_3/audit_inputs.py
$PY -u Alternative2/scripts/MatGPT_frozen_3/freeze_and_extract.py --device cpu
$PY -m unittest discover -s Alternative2/scripts/MatGPT_frozen_3/tests -v
$PY -u Alternative2/scripts/MatGPT_frozen_3/ridge_probes.py
$PY -u Alternative2/scripts/MatGPT_frozen_3/train_heads.py --models R2-SMALL4 R2-REF8 R3-DROP30 R3-WD1E2 R3-EARLY8 --folds 0 --seeds 42 --device cpu
$PY -u Alternative2/scripts/MatGPT_frozen_3/train_heads.py --models R2-SMALL4 R2-REF8 R3-DROP30 R3-WD1E2 R3-EARLY8 --folds 0 --device cpu
$PY -u Alternative2/scripts/MatGPT_frozen_3/train_heads.py --models R2-SMALL4 R2-REF8 R3-DROP30 R3-WD1E2 R3-EARLY8 --device cpu
$PY -u Alternative2/scripts/MatGPT_frozen_3/evaluate_oof.py --bootstrap 2000
$PY -u Alternative2/scripts/MatGPT_frozen_3/make_png_figures.py
$PY -u Alternative2/scripts/MatGPT_frozen_3/compare_r1_r4.py
$PY -u Alternative2/scripts/MatGPT_frozen_3/audit_outputs.py
```

La primera orden de entrenamiento es el smoke (un fold y una seed); la segunda completa el piloto del fold 0 con las tres seeds; la tercera reanuda éxitos y completa los 75 pares configuración×fold×seed. `run_one` no repite corridas con estado PASS y conserva las corridas incompletas para auditoría.

## Archivos

- `common.py`, `data_and_folds.py`: rutas configurables y split agrupado.
- `freeze_and_extract.py`: cache END/MEAN por capa y auditorías de padding.
- `ridge_probes.py`, `train_heads.py`: R1/R4 y R2/R3.
- `metrics.py`, `evaluate_oof.py`, `paired_statistics.py`: métricas, matrices y bootstrap por composición.
- `make_png_figures.py`, `compare_r1_r4.py`: figuras e informes. Se exportan exclusivamente PNG.
- `tests/`: pruebas unitarias matemáticas, folds, checkpoint, extracción y extensiones gráficas.

Los resultados ejecutados están en `Alternative2/reports/MatGPT_frozen_3/run_20261023_v2/`. Consulte su informe R1–R4 antes de interpretar cualquier métrica. La selección histórica de arquitecturas/pooling con parte de validation implica que estos resultados son exploratorios, no una confirmación independiente.

## Referencias

- MatGPT: https://github.com/devalab/matgpt (commit histórico local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`).
- SLICES: Xiao et al. (2023), https://doi.org/10.1038/s41467-023-42870-7.
- MatFold: Witman & Schindler (2025), https://doi.org/10.1039/D4DD00250D.
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.
- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html.
- PyTorch: https://docs.pytorch.org/docs/stable/notes/autograd.html.
- PBE: Perdew, Burke & Ernzerhof (1996), https://doi.org/10.1103/PhysRevLett.77.3865.
