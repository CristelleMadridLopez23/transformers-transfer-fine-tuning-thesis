# only_hex_new: extxyz → CSV → SLICES

Fuente: `data/only_hex_new/configurations_only_hex_new.extxyz`. Se conservan todas
las configuraciones, las tres familias, las etiquetas de origen y la asignación
histórica por fórmula de `data/only_hex_new/formula_split/`. El nombre del origen
describe el preprocesamiento aplicado a hexagonales; no significa que el archivo
contenga únicamente estructuras hexagonales.

El CSV conserva posiciones cartesianas, celda, números atómicos, periodicidad,
arrays, metadatos y resultados del calculador mediante columnas JSON reversibles.
Las energías de cada frame de relajación permanecen asociadas a ese frame. Las
referencias estáticas se identifican mediante `data_origin=static`; no se asignan
sus energías a geometrías de la trayectoria. No se recalcula DFT ni se cambia el
experimento congelado anterior.

La conversión usa SLICES oficial 2.0.12, `graph_method="econnn"`,
`relax_model=None`, validación estricta y estrategia 4 como el notebook previo.
`slices_raw` conserva la salida original de la API. `slices_original`/`SLICES`
contienen la forma canónica de estrategia 4 sin prefijo SLICES-PLUS.
`slices_model_input`/`SLICES_mattext` serializan exactamente el mismo grafo en
estrategia 3. Esta sintaxis **no demuestra compatibilidad del tokenizador MatText**.
No se reconstruyen ni relajan estructuras y no se utiliza GPU.

Salidas en `data/processed/only_hex_new_csv_slices/`:

- `configurations.csv`: todas las configuraciones, antes de generar grafos.
- `configurations_slices.csv`: todas las filas y su estado de conversión.
- `slices_valid.csv`: filas que pasan las comprobaciones oficiales.
- `slices_rejected.csv`: rechazos explícitos con identificador, etapa y causa.

Auditorías, progreso, colisiones topológicas y logs se guardan en
`reports/only_hex_new_csv_slices/`. No se deduplican frames con igual topología:
SLICES puede ser idéntico para geometrías y energías distintas de una trayectoria.
Los rechazos no autorizan automáticamente una selección polimórfica con familias
ausentes; cualquier conjunto supervisado posterior requiere una nueva auditoría.

Desde la raíz, con el entorno existente `dmc`:

```bash
python -B model-pretrained/scripts/only_hex_new_csv_slices/01_extxyz_to_csv.py
```

La conversión completa exige una asignación Slurm y un `csv_audit.json`
satisfactorio con hashes actuales. El envío utiliza la cuenta `cbartel` consultada
en Slurm, 4 CPU, 16 GB y 4 horas en `msismall`. No se instalan paquetes en el job.
Los valores de activación se obtienen del entorno existente antes de enviar:

```bash
mkdir -p reports/only_hex_new_csv_slices/logs
export HEX_CONDA_BASE="$(conda info --base)"
export HEX_ENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
sbatch --test-only --account=cbartel sh_files/only_hex_new_csv_slices/convert.sbatch
# Solo después de revisar auditoría y ausencia de una ejecución previa:
sbatch --parsable --account=cbartel sh_files/only_hex_new_csv_slices/convert.sbatch
```

Los Job IDs reales se documentan en `submission.json`. Consultar `squeue -j ID`
y `sacct -j ID --format=JobID,State,ExitCode,Elapsed,MaxRSS` para verificar ejecución.
No repetir el envío mientras haya un job activo ni sobrescribir salidas completas.
