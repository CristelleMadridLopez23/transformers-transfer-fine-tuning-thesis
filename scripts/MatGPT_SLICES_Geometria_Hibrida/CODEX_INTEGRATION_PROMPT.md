# Instrucción para Codex: enlace de estructuras reales ABS₃

Identifica en el repositorio la fuente original de estructuras DFT correspondientes a los 210 `structure_id` de `Alternative2/reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv` con `split=train|val`.

1. Leer AGENTS.md, instrucciones y manifests del proyecto.
2. Localizar estructuras originales (pymatgen Structure, ComputedStructureEntry, CIF, POSCAR, Alexandria JSON). No buscar equivalencias por proximidad química o nombre si no hay ID verificable.
3. Generar `structures_development.jsonl` con exactamente un `structure_id`, `geometry_source` verdadero y `structure.as_dict()` por cada una de las 210 estructuras. Alinear IDs con el dataset, y validar `len(structure)==num_atoms` y, si es posible, la composición. No leer ni exportar energías de test.
4. Confirmar procedencia: geometrías relajadas, estáticas, reconstruidas o iniciales, sin mezclar políticas diferentes.
5. Ejecutar `exportar_geometria.py` sobre ese JSONL, revisar varianza de celda entre polimorfos y hashes.
6. Ejecutar `preflight.py`; solo con PASS proceder a `ejecutar_experimento.sh`.
7. Nunca generar estructuras de ejemplo, copiar estructuras de otros polimorfos ni rellenar ceros por geometría faltante.
8. Mantener el control `slices_only` apareado con la fusión; Ridge H3-C histórico no se reentrena y MatGPT está congelado.
9. Evaluar resultados con figuras descriptivas y declarar que la fusión añade 56 pesos al primer Linear (768→775), sin cambiar ancho 8 ni pérdidas.

Si no puedes recuperar la geometría real con correspondencia exacta, termina la auditoría indicando IDs faltantes y **no ejecutes** resultados reales.
