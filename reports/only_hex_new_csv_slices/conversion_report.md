# Conversión only_hex_new completada

El trabajo Slurm **4310202** terminó **COMPLETED, código 0:0**, en 7 min 23 s.

Se conservaron las **8.898 configuraciones de 80 composiciones** del extxyz original.
**8.712** pasaron las validaciones oficiales SLICES; **186** quedaron registradas
con ID, geometría, energía y causa en `slices_rejected.csv`. Todos los hashes de
salida, conteos y unicidad de IDs se verificaron después de terminar el trabajo.

Archivos en `data/processed/only_hex_new_csv_slices/`:

- `configurations.csv`: datos originales convertidos a CSV (8.898 filas).
- `configurations_slices.csv`: todas las filas y su resultado SLICES (8.898).
- `slices_valid.csv`: 8.712 aceptadas; estrategias 4 y 3 equivalentes.
- `slices_rejected.csv`: 186 rechazos explícitos, sin cambiar el criterio econnn.

La conversión conserva energías, geometrías, fuerzas, tensiones, metadatos y
particiones originales por composición. El extxyz y el checkpoint no se modificaron.
La sintaxis estrategia 3 por sí sola no certifica la asignación del tokenizador
usada por el checkpoint. La auditoría posterior de todos los datos está en
`reports/exp02_only_hex_mattext_finetune/tokenizer_compatibility.md`.

Consultar `audit/slices_manifest.json`, `audit/csv_audit.json` y los logs
`logs/slices_4310202.out` / `logs/slices_4310202.err`.
