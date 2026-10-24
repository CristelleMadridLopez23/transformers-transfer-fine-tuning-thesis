# MatGPT E1: extractor congelado y MLP

Este directorio contiene el flujo E1 completo. Los scripts no cambian el código, vocabulario, configuración ni pesos de MatGPT. El extractor usa los estados de `ln_f` en el token `<end>` y antes de `lm_head`; la cabeza de regresión entrena de forma independiente sobre los embeddings cacheados.

## Datos identificados

La entrada por defecto es `Alternative2/01_data/processed/only_hex_new_csv_slices/configurations_slices.csv`. Se seleccionan registros estáticos cuya composición reducida tiene estequiometría ternaria 1:1:3 con azufre, y se reutiliza el split histórico agrupado por fórmula. La auditoría local encontró 234 candidatos ABS3 estáticos; los 234 tienen etiquetas y SLICES compatibles con MatGPT. Se excluyen dos fórmulas binarias. La distribución aceptada es train 183 filas/62 composiciones, validation 24/8 y test 23/8. El script vuelve a calcular estos datos y aborta ante fugas.

Para MatGPT se usa la columna `SLICES`, que conserva vectores direccionales como tokens completos (`ooo`, `o+o`). `slices_model_input` está preparado para otro tokenizer y divide esos vectores carácter por carácter; no se debe pasar al vocabulario de MatGPT. La carpeta inicial `matgpt_e1` conserva la auditoría fallida que detectó esta diferencia; la corrección escribe en una versión nueva `matgpt_e1_v2`.

La etiqueta es `energy_per_atom_ev`: energía total DFT estática dividida por número de átomos, en eV/átomo. No es energía de formación. La procedencia identifica `gga-static`, pero no especifica el funcional XC exacto ni las correcciones energéticas. El informe final debe quedar `PARTIAL` hasta verificar esos detalles en la procedencia de los cálculos de origen.

## Requisitos

Ejecuta desde la raíz del repositorio con el entorno Python que tenga PyTorch, NumPy, pandas, PyYAML, matplotlib y SciPy instalados. E1 valida el SHA-256 E0 antes de preparar datos, extraer embeddings, entrenar y evaluar. La selección del modelo usa únicamente validation; el script de entrenamiento no abre el test. La evaluación consume el test una sola vez.

## Ejecución manual, por etapas

Desde la raíz del repositorio, corre estos comandos individualmente y en este orden:

```bash
python -u Alternative2/scripts/matgpt_e1/prepare_dataset.py
```

```bash
python -u Alternative2/scripts/matgpt_e1/extract_embeddings.py --device cpu
```

```bash
python -u Alternative2/scripts/matgpt_e1/train_regressor.py --device cpu
```

```bash
python -u Alternative2/scripts/matgpt_e1/evaluate.py --device cpu --model-dir Alternative2/reports/matgpt_e1_v2 --report-dir Alternative2/reports/matgpt_e1_v2_eval_v3
```

Los datos y resultados del entrenamiento están en `Alternative2/reports/matgpt_e1_v2/`; la evaluación escribe en `Alternative2/reports/matgpt_e1_v2_eval_v3/`, y los embeddings están en `Alternative2/data/embeddings/matgpt_e1_v2/`. Las carpetas `matgpt_e1_v2/` y `matgpt_e1_v2_eval_v2/` conservan las tablas parciales de intentos anteriores. Ningún comando borra resultados anteriores.

Si una evaluación alcanzó a guardar `metrics.json` pero falló al crear las figuras o el informe, se puede completar esa carpeta sin volver a ejecutar predicciones ni consultar el test otra vez:

```bash
python -u Alternative2/scripts/matgpt_e1/evaluate.py --device cpu --model-dir Alternative2/reports/matgpt_e1_v2 --report-dir Alternative2/reports/matgpt_e1_v2_eval_v2 --finalize-existing
```

## Archivos principales

- `dataset.csv`, `dataset_audit.json`, `split_manifest.json`: datos supervisados filtrados y procedencia/split.
- `matgpt_uncond_last_token.npz`, `embedding_metadata.csv`, `embedding_audit.json`: embeddings, asociación exacta por estructura y controles del extractor.
- `best_mlp.pt`, `best_linear.pt`, `training_config.yaml`, `training_history.csv`, `baseline_metrics_train_val.csv`: cabezas, selección validation y curvas.
- `predictions.csv`, `metrics.json`, `metrics_by_split.csv`, `polymorph_metrics.csv`, `pairwise_comparisons.csv`, `E1_report.md`: resultados energéticos y ranking.
- `figures/*.png`: curvas, paridad DFT, errores, energías relativas, selección del mínimo y errores por familia.

`prepare_dataset.py` y `extract_embeddings.py` rechazan salidas existentes para no sobrescribir corridas anteriores. `evaluate.py` también se niega a reemplazar predicciones o métricas existentes.
