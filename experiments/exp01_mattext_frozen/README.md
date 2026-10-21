# Experimento 01 — MatText-SLICES-2M congelado

**Estado: bloqueado por dos controles científicos. Sin extracción real, entrenamiento completo, envío Slurm ni métricas finales.** Ver [`preflight.json`](../../reports/exp01_mattext_frozen/audit/preflight.json) y [`experiment_report.md`](../../reports/exp01_mattext_frozen/experiment_report.md).

## Datos y decisiones

Fuente real: `data/results(1).json`, SHA-256 `c10c8c5378beaea03a33978da772329ff31e765b6c4a6da7d0c20bc588a92012`. Contiene 480 cálculos, 80 composiciones, tres prototipos iniciales y dos tipos de cálculo por candidato. Usamos `results.E_per_at` del cálculo **estático GGA**, no r²SCAN ni `E_relax`. El total se deriva explícitamente como `E_per_at × num_atoms`, pues no existe un campo independiente de energía total.

Se verifican los metadatos INCAR (`GGA=Pe`, `NSW=0`, sin meta-GGA/híbrido/+U), convergencia, integridad de las estructuras y su relación con la relajación. CuSnS₃ hexagonal presenta una diferencia de celda y queda documentado en `energy_ambiguities.csv` y `static_relaxation_links.csv`. Se detiene la aceptación global; no se oculta la discrepancia excluyendo silenciosamente el registro.

Cuatro estáticos no disponen de SLICES aceptados en la conversión previa. Con la geometría discrepante, cinco composiciones quedan incompletas. Para no evaluar selección del mínimo con candidatos ausentes, el conjunto provisional conserva **225 filas / 75 composiciones / 75 por familia**. Familias: `dist_perovskite`, `hexagonal`, `needle`; son etiquetas del prototipo inicial y no una clasificación geométrica final.

Particiones provisionales guardadas antes de entrenar: **60 composiciones / 180 estructuras de desarrollo; 15 / 45 de test; cuatro pliegues de 15 composiciones de desarrollo; semilla 42**. Cada composición y todos sus polimorfos comparten partición/pliegue. Cualquier cambio de muestras exige regenerar y auditar las particiones. No se detectaron duplicados geométricos estrictos en las candidatas; el solapamiento con el corpus NOMAD de preentrenamiento sigue sin determinarse.

## Tokenizador y checkpoint

Checkpoint local `n0w0f/MatText-slices-2m`, revisión `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`; **no se volvió a descargar ni modificar**. SHA-256 de pesos: `6bef5652d758190e6e6eaf85ccf2a2f584b69255f1be3bb7bdd72cb1f12a74bd`.

BERT: 4 bloques, 8 cabezas, dimensión 512, FFN 3072, contexto 512, 30.522 filas de embeddings y 32.962.048 parámetros. Carga estricta, cero parámetros entrenables, salida finita y determinismo comprobados en CPU con IDs sintéticos. Esta prueba técnica no valida la correspondencia semántica del tokenizador.

Las cadenas previas proceden de SLICES 2.0.12 oficial con `graph_method="econnn"`, `relax_model=None` y estrategia 4. Ya existe exportación estrategia 3 del mismo grafo. El adaptador explícito conserva especies, índices, aristas y vectores periódicos, sin reconstruir ni relajar cristales. Conserva `slices_original` y `slices_model_input`.

Se descargó el vocabulario original de [Zenodo](https://zenodo.org/records/11484062), verificando su MD5 publicado. Los 156 IDs candidatos incluyen los tokens especiales en el orden de una configuración archivada de MatText. La documentación por defecto usa otra asignación de especiales; el checkpoint no incluye el artefacto de tokenización original. **La compatibilidad exacta no está demostrada**. Las entradas se guardan como `tokenized_candidate.jsonl`; no se genera `tokenized.jsonl` aprobado ni se permite extraer embeddings científicos.

Para desbloquear esta fase se requiere un artefacto original o evidencia pública verificable vinculada a esta revisión del checkpoint, con vocabulario/IDs, orden de especiales y comportamiento de CLS/SEP. El esquema de evidencia en `02_audit_slices.py` exige hashes y vínculo publicado; un booleano manual no basta. Si la evidencia revela una política distinta, deberá corregirse y auditarse el código antes de ejecutar.

## Protocolo implementado, pendiente de ejecución científica

- Extraer una sola vez mean y CLS con `requires_grad_(False)`, `eval()` e `inference_mode()`. Mean excluye padding y posiciones especiales mediante máscaras. Se prueba repetibilidad, forma y finitud; se guardan hashes, IDs, composiciones, familias y etiquetas por separado.
- Seleccionar un pooling **común** mediante MAE de la MLP en los cuatro pliegues de desarrollo. Ajustar los hiperparámetros de los tres predictores en esos mismos pliegues. El test queda reservado para la evaluación final.
- Ridge: normalización de X/y ajustada solo sobre entrenamiento. RF: búsqueda controlada de árboles, profundidad, mínimo de hoja y fracción de variables. MLP: dimensión del embedding → 32 → 1, GELU, dropout 0,1 y AdamW; validación interna por composición para early stopping, seguida de refit en todo el entrenamiento durante las épocas seleccionadas.
- Guardar cabezas en `artifacts/`; no se añade un optimizador al encoder ni se guardan modelos entrenados en el checkpoint original.
- Evaluar MAE/RMSE de energía (eV/átomo), MAE de diferencias dentro de composición, concordancia por pares, selección del mínimo y regret (meV/átomo). Tolerancia de cuasidegeneración: 1 meV/átomo. Pares realmente degenerados se excluyen de concordancia direccional; empates predichos cuentan 0,5. Empates de selección se resuelven por ID, sin consultar GGA.
- Baseline: familia favorecida más frecuentemente en los mínimos GGA de desarrollo. Nunca se usa la distribución r²SCAN publicada ni las etiquetas de prueba para fijar la preferencia.

## Scripts y ejecución

| Archivo en `model-pretrained/scripts/exp01_mattext_frozen/` | Función |
| --- | --- |
| `00_inventory.py` | Inventario, entorno, integridad del checkpoint y consulta Slurm. |
| `01_prepare_dataset.py` | Asociación estático/relax, filtros, referencias y particiones. |
| `02_audit_slices.py` | Adaptación sin pérdida, referencias originales, cobertura y colisiones. |
| `03_extract_embeddings.py` | Encoder congelado, máscaras, dos poolings y caché con hashes. |
| `04_train_regressors.py` | CV agrupada y tres predictores independientes. |
| `05_evaluate_models.py` | Test final y métricas energéticas/polimórficas. |
| `06_generate_reports.py` | Figuras y reporte según artefactos reales disponibles. |
| `run_experiment.py` | Preflight y orquestación con bloqueo ante fallos críticos. |
| `slurm_control.py` | Validación de recursos, envío condicionado y estado de Job IDs reales. |
| `utils.py`, `test_exp01.py` | Funciones compartidas y ocho pruebas mínimas. |

Desde la raíz y con el entorno `dmc` existente activo:

```bash
# Auditorías, pruebas y smoke real solo si los controles científicos pasan.
bash sh_files/exp01_mattext_frozen/00_preflight.sh

# Primera recuperación de referencias oficiales, si faltan en una copia nueva:
bash sh_files/exp01_mattext_frozen/00_preflight.sh --fetch-reference

# Pruebas unitarias (fixtures sintéticos explícitos; no resultados científicos).
python model-pretrained/scripts/exp01_mattext_frozen/test_exp01.py

# Auditorías individuales:
python model-pretrained/scripts/exp01_mattext_frozen/00_inventory.py
python model-pretrained/scripts/exp01_mattext_frozen/01_prepare_dataset.py
python model-pretrained/scripts/exp01_mattext_frozen/02_audit_slices.py

# Regenerar el reporte usando solo los resultados existentes:
python model-pretrained/scripts/exp01_mattext_frozen/run_experiment.py --phase report

# Envío: rechazado mientras el preflight falle o sus hashes hayan cambiado.
bash sh_files/exp01_mattext_frozen/submit_experiment.sh
bash sh_files/exp01_mattext_frozen/check_status.sh
```

Configurable en [`config.yaml`](config.yaml): rutas, semilla, particiones, tokenizador, poolings, batch, dispositivo, modelos, búsqueda, tolerancia y salidas. `EXP01_CONFIG` y `EXP01_PYTHON` permiten seleccionar configuración/intérprete en los wrappers. No se instalan dependencias durante trabajos.

Entorno observado: Python 3.10.21, PyTorch 2.8.0+cu128, Transformers 5.17.0, scikit-learn 1.7.2, pymatgen 2025.10.7 y SLICES 2.0.12. Las versiones completas, revisión Git y hashes se registran en `audit/inventory.json` y `experiment_manifest.json`.

## Slurm y artefactos reales

Asociación consultada: cuenta `cbartel`, QOS `normal`; particiones `msigpu` y `msismall` activas y abiertas a las cuentas autorizadas. La solicitud GPU de 4 h está dentro del límite de 24 h de `msigpu`. Falta la validación de admisión `sbatch --test-only`, que solo se ejecutará después de superar los controles científicos.

Extracción: 1 GPU, 8 CPU, 32 GB, 4 h, `mattext_e01_extract`. Cabezas: 8 CPU, 32 GB, 4 h en `msismall`, dependencia `afterok:<ID real de extracción>`. La activación Conda usa el entorno existente. Se bloquean envíos duplicados y configuraciones modificadas tras la auditoría.

**Job IDs: ninguno. Estado: no enviado por preflight fallido.** Logs: `reports/exp01_mattext_frozen/logs/`; pruebas y API oficial: `reports/exp01_mattext_frozen/audit/*.log`.

Se generaron únicamente tres figuras reales de auditoría, a 240 dpi: distribución energética del subconjunto provisional, longitudes candidatas y cobertura candidata. Las figuras 04–15, modelos, embeddings y métricas finales se producirán únicamente tras una ejecución científica válida. No se crearon resultados vacíos ni ficticios.
