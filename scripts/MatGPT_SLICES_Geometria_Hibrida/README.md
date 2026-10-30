# Experimento 2: MatGPT congelado + SLICES + geometría de celda

## Estado de la ejecución real (28 de octubre de 2026)

Se recuperaron las **210 geometrías reales** mediante `source_index`, `config_id` y
`source_key` del EXTXYZ original
`Alternative2/01_data/abs3/only_hex_new/configurations_only_hex_new.extxyz`.
El enlace y los 30 entrenamientos apareados terminaron. El resultado es
**EXPLORATORY_PASS**: los intervalos apareados de mejora incluyen cero y el caché
MatGPT reconstruido coincide numéricamente con referencias H3-C, aunque su
SHA-256 binario difiere del archivo histórico retirado del árbol activo.

Artefactos canónicos:

- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl`
- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv`
- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/geometry_provenance_audit.md`
- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/geometry_scientific_report.md`
- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/results/`
- `Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/final_manifest.json`

El ejecutor actual está en
`sh_files/MatGPT_SLICES_Geometria_Hibrida/ejecutar_experimento.sh`.
Valida el preflight antes de entrenar y registra tiempo/RSS de cada fase. El
CSV de entrenamiento registra las 30 corridas individuales. No volver a
ejecutarlo sobre la salida canónica existente: el script rechaza sobrescritura.

## Propósito

Medir cuánto **mejora la predicción del residuo energético intrafórmula** al agregar información de **celda cristalográfica** a la representación SLICES del MatGPT preentrenado. No es SLICES-PLUS, ni fine-tuning, ni un MLIP. Utiliza el **mismo extractor MatGPT congelado** y la regresión energética híbrida Ridge–MLP8 auditada.

**El ZIP `MatGPT_H3C_Final(1).zip` se utilizó como referencia de las dependencias/rutas y las especificaciones de H3-C; no contenía las 210 estructuras cristalográficas ni todos los artefactos necesarios.** Las geometrías se localizaron después en el EXTXYZ original y se enlazaron de forma exacta. El resto de esta sección describe el procedimiento de entrada reutilizable; no se generan estructuras ficticias ni se infieren celdas desde SLICES.

## Arquitectura bloqueada y cambio único

| Componente | SLICES solo | SLICES + celda |
|---|---|---|
| MatGPT congelado | Original, 12 bloques, 768 | Idéntico |
| Ridge media de fórmula | L03_MEAN; histórico, inalterado | Idéntico (cargado del mismo checkpoint) |
| Embedding estructural | FINAL_END: 768 | FINAL_END: 768 |
| Geometría adicional | Ninguna | 7 dimensiones estandarizadas solo con train |
| MLP residual | 768→8→1, 6,161 parámetros | 775→8→1, 6,217 parámetros |
| GELU / Dropout | GELU / 0.1 | Igual |
| Función de pérdida | SmoothL1 de residuos centrados, escala train | Igual |
| AdamW / LR / decay | AdamW / .001 / .0001 | Igual |
| 5 outer folds × 3 seeds | Sí | Sí |
| Checkpoint | Mejor SmoothL1 en inner validation | Igual |
| Ridge entrenamiento | Se cargan 5 Ridge históricos | Se cargan los mismos Ridge |

**El baseline SLICES-only se reentrena en paralelo** con exactamente la misma inicialización de parámetros y orden de minibatches, para una comparación apareada. La condición con geometría empieza con los 768 pesos SLICES y las capas posteriores idénticos al control, y los 56 pesos geométricos adicionales inicializados en cero. De esa forma, ambas cabezas comienzan con predicciones iguales (salvo redondeo numérico), aunque posteriormente se optimizan independientemente.

### Descriptores de geometría de celda (7)

1. `a_reduced_shape`, `b_reduced_shape`, `c_reduced_shape`: longitudes de celda primitiva reducida por Niggli, cada una dividida por la raíz cúbica del volumen de esa celda; captan **forma adimensional** y son menos sensibles a elecciones de supercelda.
2. `alpha_reduced_deg`, `beta_reduced_deg`, `gamma_reduced_deg`: ángulos entre vectores de celda reducida, en grados.
3. `volume_per_atom_ang3`: volumen de la estructura original dividido por N (Å³/átomo); capta **escala volumétrica física**.

La geometría de celda NO representa posiciones atómicas internas, ángulos B–S–B ni fuerzas. Si la estructura se relajó por DFT, el estudio presupone geometrías DFT disponibles al inferir: **no evalúa predecir energías desde SLICES sin estructura inicial**.

## Instalación

Desde la raíz del repositorio de tesis:

```bash
unzip MatGPT_SLICES_Geometria_Hibrida.zip -d Alternative2/scripts/MatGPT/
source .venv/bin/activate
python -m pip install pymatgen
export ROOT="$PWD"
export EXP="$ROOT/Alternative2/scripts/MatGPT/MatGPT_SLICES_Geometria_Hibrida"
```

Se necesitan también `torch`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, `tabulate` (este último para informe Markdown). Usa las versiones del entorno histórico cuando sea posible.

### Requisito previo: estructura cristalina vinculada a cada `structure_id`

**Opción A: 210 archivos de estructura (CIF/POSCAR/etc.)**.

Crear `structures_manifest.csv` con encabezado:

```csv
structure_id,path,geometry_source
```

Cada fila debe corresponder a uno de los 210 `structure_id` del dataset de desarrollo; `path` apunta a un archivo real (ruta absoluta o relativa al manifiesto), y `geometry_source` debe indicar procedencia homogénea (`DFT_relaxed`, `initial`, etc.). No colocar etiquetas de energía.

```bash
python "$EXP/exportar_geometria.py" \
  --dataset "$ROOT/Alternative2/reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv" \
  --manifest-cif "$ROOT/structures_manifest.csv" \
  --output "$ROOT/Alternative2/data/geometry_cell_development.csv"
```

**Opción B: JSONL estructural**: un JSON por línea con `structure_id`, `geometry_source` y `structure` tal como lo devuelve `pymatgen.Structure.as_dict()`; por ejemplo, se pueden convertir entradas reales `ComputedStructureEntry` de Alexandria a ese formato, preservando las correspondencias exactas de ID. No asumir que `prototype_id`, `mat_id` o `entry_id` corresponde al `structure_id` sin verificarlo.

```bash
python "$EXP/exportar_geometria.py" \
  --dataset "$ROOT/Alternative2/reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv" \
  --structures-jsonl "$ROOT/data/structures_development.jsonl" \
  --output "$ROOT/Alternative2/data/geometry_cell_development.csv"
```

Si se repite este protocolo con otro conjunto, solo se permite la fusión tras enlazar geometrías físicas exactas. Para esta corrida, la correspondencia ya quedó verificada en `geometry_provenance_audit.md`.

### Ejecutar

```bash
GEO="$ROOT/Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv"
STAGING="$ROOT/Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo"
OUT="$ROOT/Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/results"
python "$EXP/preflight.py" --repo-root "$STAGING" --geometry-csv "$GEO"

python -m unittest discover -s "$EXP/tests" -p 'test_*.py' -v

bash "$ROOT/sh_files/MatGPT_SLICES_Geometria_Hibrida/ejecutar_experimento.sh" \
  "$ROOT" "$GEO" "$STAGING" "$OUT"
```

Estas rutas describen la corrida ya completada. El ejecutor solo admite una
salida vacía en una reproducción independiente; no sobrescribe el resultado
canónico. Los resultados están en:

`Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/results/`

El entrenamiento rechaza un directorio de salida que ya contenga archivos. No sobrescribe H3-C ni sus checkpoints históricos.

**Si el entrenamiento ya finalizó y solo deseas regenerar el análisis:**

```bash
OUT="$ROOT/Alternative2/reports/MatGPT/MatGPT_SLICES_Geometria/results"
python "$EXP/evaluar.py" --output "$OUT"
python "$EXP/figuras.py" --output "$OUT" \
  --geometry-csv "$GEO"
```

## Figuras con nombres descriptivos

- `paridad_energia_por_atomo_slices_frente_a_geometria.png`
- `paridad_residuos_energeticos_intraformula.png`
- `paridad_diferencias_energeticas_entre_polimorfos.png`
- `comparacion_mae_residuos_ground_state_y_ranking.png`
- `error_diferencias_energeticas_por_magnitud_gap_dft.png`
- `exactitud_ordenamiento_por_magnitud_gap_dft.png`
- `diferencias_pareadas_con_intervalos_confianza.png`
- `distribucion_descriptores_geometricos_de_celda.png`

Cada figura se genera desde predicciones OOF verdaderas del repositorio. **Ningún resultado físico se incluye precalculado en el ZIP.**

## Qué evaluar para decidir si hay aporte geométrico

La métrica primaria sugerida es la **diferencia de MAE de ΔE firmado**, con bootstrap apareado por composición; secundaria: MAE residual intrafórmula, ranking para |ΔE DFT| >5 meV/átomo, ground state elegible y gaps de 5–25 meV/átomo. Mantener MAE absoluto como guardrail, pero recordar que Ridge (componente medio) no cambia, por lo que el aporte de geometría se mide principalmente en la MLP residual.

El aumento de 56 parámetros es pequeño, pero **no es cero**. El beneficio de la fusión no constituye por sí solo una demostración de causalidad física, ya que el cambio combina características nuevas con mayor dimensión de entrada.

Resultados sujetos a las limitaciones conocidas: 70 fórmulas de desarrollo previamente exploradas, no test ciego, y referencias `gga-static` cuya homogeneidad XC/U/correcciones todavía no está certificada. No emplear el test histórico para seleccionar geometría.

## Tests

Las pruebas unitarias verifican 6,161/6,217 parámetros, igualdad inicial de ambos modelos, gradientes en pesos geométricos, permutación de polimorfos, centrado residual, cancelación de Ridge en ΔE y consistencia de métricas. Se probó también el pipeline de métricas/gráficas **con datos sintéticos únicamente para verificar el funcionamiento del código**. Eso no es un resultado científico.
