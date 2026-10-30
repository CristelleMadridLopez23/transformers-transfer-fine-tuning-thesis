# Diagnósticos MatGPT E1.1 y E1.2

Estos scripts estudian cabezas predictivas y pooling sobre estados ocultos
MatGPT ya extraídos. Mantienen congelados los 114 026 668 parámetros del
checkpoint E0. No ejecutan E2 ni descongelan el Transformer.

## Reglas de ejecución

- Ejecuta los comandos desde la raíz del repositorio.
- Usa el entorno donde se ejecutó E1 y están disponibles PyTorch, NumPy, pandas,
  SciPy, scikit-learn, joblib, PyYAML y Matplotlib.
- Los scripts no reescriben datos o reportes E1. Cada fase nueva escribe en un
  directorio dedicado y se detiene si encuentra resultados previos.
- La selección se limita a train y validation. El test histórico ya fue
  inspeccionado durante E1; la evaluación final es descriptiva, no un test ciego.
- No ejecutes la evaluación final hasta cerrar y guardar la selección en un CSV.
- Las figuras se guardan como PNG conforme a las normas del repositorio.

## Fase 0 — reproducción de E1

Reproduce las predicciones de la MLP histórica usando el checkpoint guardado,
sin entrenar de nuevo. Si falla, detén las fases posteriores e investiga el
hash, IDs, normalización o asociación entre predicciones y estructuras.

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/reproduce_e1.py
```

Se compara cada `structure_id` con el CSV histórico, con tolerancia por defecto
de 1e-5 eV/átomo. Resultado: `Alternative2/reports/matgpt_e1_diagnostics/e1_reproduction.json`.

## Fase 1 — E1.1

Entrena Ridge con cuatro valores de alpha, MLP8 y MLP32 con cuatro valores de
lambda pairwise, tres semillas y muestreo que mantiene juntas las composiciones.
Se informan train y validation; no se calculan métricas test.

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/train_heads.py --experiment e1_1 --device cpu --out-dir Alternative2/reports/matgpt_e1_1_v3
```

La MLP se detiene con el objetivo normalizado de validation correspondiente a
su lambda. Las escalas absoluta y pairwise se calculan solo con train. La
regresión Ridge estandariza características con estadísticas ajustadas solo en
train. Las métricas incluyen MAE absoluta/relativa, ranking, estado fundamental,
empates, Spearman por composición cuando está definido, rangos y MAE por familia.

Resume los compromisos sin reducirlos a una puntuación artificial:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/validation_pareto.py --experiment-dir Alternative2/reports/matgpt_e1_1_v3
```

Genera las figuras E1.1 en español:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/plot_results.py --experiment-dir Alternative2/reports/matgpt_e1_1_v3
```

Revisa `validation_report.md`, `metrics.csv`, `metrics_by_configuration.csv`,
`predictions_train_val.csv`, `mae_by_family.csv`, las curvas y los PNG. Decide
cuál candidato, si alguno, merece compararse en E1.2. No consultes test para
esa decisión. Si Ridge es el candidato, registra también su alpha.

## Fase 2 — extracción y geometría E1.2

Extrae mean pooling sobre todos y solo los tokens SLICES; excluye `<start>`,
`<end>` y `<pad>`. La inferencia individual evita padding artificial. Se
verifican hash E0, logits, dimensiones, valores finitos, repetibilidad y
sensibilidad a padding final.

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/extract_masked_mean.py --device cpu
```

Calcula similitudes coseno, distancias euclidianas, comparaciones entre
composiciones con muestreo reproducible, asociación con diferencias DFT,
longitud SLICES y colisiones de secuencia:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/embedding_diagnostics.py
```

Genera PNG de las geometrías de embeddings:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/plot_results.py --embedding-distances Alternative2/reports/matgpt_e1_2/embedding_distances.csv
```

## Fase 3 — comparación controlada de pooling

Primero ejecuta los dos controles `MLP32 + pérdida absoluta`, cambiando solo el
pooling. Conserva directorios de resultados diferentes:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/train_heads.py --experiment e1_2 --pooling last_token --head mlp32 --pairwise-lambda 0 --device cpu --out-dir Alternative2/reports/matgpt_e1_2_last_mlp32_abs_v1
```

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/train_heads.py --experiment e1_2 --pooling masked_mean --head mlp32 --pairwise-lambda 0 --dataset Alternative2/reports/matgpt_e1_v2/dataset.csv --embeddings Alternative2/data/embeddings/matgpt_e1_2_masked_mean_v1/matgpt_uncond_masked_mean.npz --metadata Alternative2/data/embeddings/matgpt_e1_2_masked_mean_v1/embedding_metadata.csv --audit Alternative2/data/embeddings/matgpt_e1_2_masked_mean_v1/embedding_audit.json --device cpu --out-dir Alternative2/reports/matgpt_e1_2_mean_mlp32_abs_v1
```

Resume y grafica estos dos controles:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/validation_pareto.py --experiment-dir Alternative2/reports/matgpt_e1_2_last_mlp32_abs_v1
```

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/validation_pareto.py --experiment-dir Alternative2/reports/matgpt_e1_2_mean_mlp32_abs_v1
```

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/plot_results.py --last-dir Alternative2/reports/matgpt_e1_2_last_mlp32_abs_v1 --mean-dir Alternative2/reports/matgpt_e1_2_mean_mlp32_abs_v1
```

Después de revisar validation, repite la comparación con la misma cabeza y
pérdida seleccionadas en E1.1, una ejecución para `last_token` y otra para
`masked_mean`. Para una MLP usa `--head mlp8` o `--head mlp32` y el lambda
seleccionado. Para Ridge usa `--head ridge --pairwise-lambda 0 --ridge-alpha
<alpha>`; alpha debe ser el valor seleccionado con validation. Cambia
`--out-dir` en cada ejecución. Para masked mean añade los cuatro argumentos de
caché mostrados en el comando de masked mean anterior. Luego repite
`validation_pareto.py` y `plot_results.py` para esos directorios.

Una vez fijados los nombres de configuración con las tablas de validation,
genera la comparación global E1/E1.1/E1.2. Sustituye cada valor entre
`<...>` por el nombre exacto de `configuration` que se decidió con validation:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/global_validation_comparison.py --e1-1-predictions Alternative2/reports/matgpt_e1_1_v3/predictions_train_val.csv --e1-1-configuration '<CONFIG_E1_1>' --e1-2-last-predictions Alternative2/reports/matgpt_e1_2_last_mlp32_abs_v1/predictions_train_val.csv --e1-2-last-configuration '<CONFIG_E1_2_LAST>' --e1-2-mean-predictions Alternative2/reports/matgpt_e1_2_mean_mlp32_abs_v1/predictions_train_val.csv --e1-2-mean-configuration '<CONFIG_E1_2_MEAN>' --out-dir Alternative2/reports/matgpt_e1_global_comparison_v1
```

La comparación exige exactamente los mismos IDs y etiquetas DFT de validation.
Las configuraciones E1.2 deben tener la misma cabeza y pérdida para aislar el
efecto del pooling. Como esos candidatos se seleccionan usando validation, la
comparación global en esa partición es descriptiva y puede ser optimista.

## Fase 4 — bloqueo de selección y test final

Una vez cerradas las decisiones, copia
`final_test_selection_TEMPLATE.csv` a un directorio nuevo bajo
`Alternative2/reports/` y completa una fila por configuración que se haya
decidido evaluar. `run_dir` es el directorio que contiene `checkpoints/`;
`checkpoint_file` es la ruta relativa a ese directorio. Copia literalmente
`configuration` y `seed` de `metrics.csv`; para Ridge, escribe
`not_applicable` como semilla. `pooling` acepta `last_token` o `masked_mean`.
En `selection_reason` registra brevemente qué
resultado de train/validation justificó la decisión. No incluyas modelos
elegidos mirando el test.

Solo entonces ejecuta la evaluación final:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/finalize_test.py --selection Alternative2/reports/matgpt_e1_diagnostics/final_test_selection.csv --out-dir Alternative2/reports/matgpt_e1_final_test_v1
```

El script exige justificación por candidato, verifica hashes de datos/cachés y
checkpoint, y exporta predicciones, métricas, resultados por composición y
pares. No uses esos resultados para volver a seleccionar configuraciones.

## Fase 5 — reportes integrados

Una vez terminadas validation, geometría y, si corresponde, la evaluación final,
compila los reportes a un directorio versionado nuevo:

```bash
python -u Alternative2/scripts/matgpt_e1_diagnostics/write_reports.py --e1-1 Alternative2/reports/matgpt_e1_1_v3 --e1-2-last Alternative2/reports/matgpt_e1_2_last_mlp32_abs_v1 --e1-2-mean Alternative2/reports/matgpt_e1_2_mean_mlp32_abs_v1 --embedding-summary Alternative2/reports/matgpt_e1_2/embedding_summary.json --final-test Alternative2/reports/matgpt_e1_final_test_v1 --out-dir Alternative2/reports/matgpt_e1_diagnostics_v1
```

Si la evaluación final aún no se ha ejecutado, omite `--final-test`; el reporte
lo marcará como pendiente y no inventará métricas.

## Artefactos principales

- `metrics.csv`: una fila por configuración, semilla y partición.
- `metrics_by_configuration.csv`: media y desviación entre semillas, separadas
  por partición.
- `predictions_train_val.csv`: predicción individual identificada por
  `structure_id`; no contiene predicciones test.
- `training_history.csv`: pérdida por época y MAE train/validation.
- `validation_pareto.csv` y `validation_report.md`: candidatos no dominados,
  colapsos de rango y compromisos de validation.
- `embedding_distances.csv` y `embedding_summary.json`: auditoría geométrica.
- `final_test_selection.csv`: registro explícito de selección previo a test.
- `test_predictions.csv`, `test_metrics.csv`, `test_by_composition.csv` y
  `test_pairwise.csv`: evaluación final descriptiva.

El protocolo DFT local no documenta el funcional exacto, pseudopotenciales ni
correcciones. Las etiquetas se describen como energías totales estáticas GGA
por átomo, no como energías de formación. Mantén esa limitación en cualquier
reporte científico.
