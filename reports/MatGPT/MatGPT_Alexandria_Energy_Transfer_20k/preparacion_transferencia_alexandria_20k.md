# Preparación corregida: Alexandria → MatGPT, 20k + 2k

## Dictamen

**Preflight de datos y compatibilidad: PASS con límites de procedencia documentados. Fine-tuning F4: iniciado en CPU local; la selección usará únicamente SmoothL1 de validation Alexandria y early stopping.** Se prepararon 20,000 estructuras de entrenamiento y 2,000 de validación; cada partición tiene una fórmula reducida única por fila y no comparten fórmulas.

## Reutilización del pipeline MatGPT

El pipeline real está en `scripts/MatGPT/MatGPT_frozen/matgpt_e1/extract_embeddings.py`: separación por espacios, tokenizador exacto `model_info_property_0.json`, tokens `<start>/<end>`, contexto 442, máscara causal oficial y extracción de `FINAL_END` desde el estado contextualizado posterior a `ln_f`. Se verificó SHA-256 del checkpoint `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414` y del vocabulario `d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee`. La dimensión de `FINAL_END` es 768.

La cadena Alexandria ya existente se adaptó desde el procedimiento validado en `data/toSLICES.ipynb`, celda 8: leer SLICES estrategia 4, omitir únicamente `space_group_num` y serializar nuevamente estrategia 4. Para el corpus se conserva el orden de átomos/aristas recuperado; no se ejecuta el canonicalizador. En todos los registros finales se compararon átomos, aristas y traslaciones periódicas antes/después del round-trip y se aplicó el chequeo básico. La comprobación topológica 3D estricta se ejecutó en la muestra requerida de 100 registros aleatorios y 20 secuencias largas; no se volvió a ejecutar ese chequeo costoso sobre cada una de las 22,000 filas. **No se ejecutó `structure2SLICES`, no se regeneró estructura y no se cambiaron tokens desconocidos a mano.**

## Procedencia DFT

El shard local `data/raw_data/alexandria_00000.json` se vincula a la colección oficial **Complete database 2025.07.02, PBE, shard 3D `alexandria_00000`** por el stem de shard y los `mat_id`; los registros elegidos cruzan `entry.energy = data.energy_total = export.energy_total` y coinciden en número de sitios. La web oficial separa la colección optimizada con PBE de PBEsol y MatPES y publica la lista de POTCAR usados ([Alexandria datasets](https://alexandria.icams.rub.de/datasets.html), [índice oficial del shard](https://alexandria.icams.rub.de/data/pbe/2025.07.02/)).

La documentación metodológica vinculada a Alexandria describe VASP, PBE, PAW 5.2, corte 520 eV, muestreo Γ centrado y la aplicación de U en óxidos/fluoruros de Co, Cr, Fe, Mn, Mo, Ni, V y W ([método publicado](https://pmc.ncbi.nlm.nih.gov/articles/PMC8641929/)). Para reducir el riesgo de mezclar la rama +U documentada, se excluyó conservadoramente cualquier candidato con uno de esos metales junto con O o F. No se inventó U por registro. Estado: `PROVENANCE_PARTIAL`: la pertenencia del shard a la colección oficial Alexandria PBE 2025.07.02 y el target crudo `entry.energy` están vinculados; faltan POTCAR e inputs VASP por `mat_id`, por lo que la homogeneidad de todos los parámetros no queda demostrada individualmente. El target no usa correcciones Materials Project, `energy_corrected`, `e_form` ni `e_above_hull`. El entrenamiento se interpreta como transferencia exploratoria de energías crudas PBE del release, con esa limitación.

## Exclusiones y división

- Export estricto inicial: 98,394 registros.
- IDs exactos compartidos con Alex‑20 excluidos: 5,397.
- SLICES originales exactas compartidas con Alex‑20: 0; además se excluyeron coincidencias exactas después de adaptar estrategia 4.
- Fórmulas de las 70 composiciones de desarrollo ABS₃ train/val y todas las ternarias reducidas 1:1:3 con S de multiplicidad 3: excluidas a nivel de fórmula. No se abrió el test histórico ni se consultó `results(1).json`.
- Químicas del grupo documentado +U O/F: excluidas conservadoramente.
- Límite de compatibilidad: `n_atoms ≤20`, porque las cadenas incluyen índices atómicos y el vocabulario original no representa índices fuera del rango validado.
- Elegibles antes de muestreo por fórmula: 85,758 filas, 82,896 fórmulas únicas.
- Final: 20,000 train y 2,000 validation; 20,000 y 2,000 fórmulas respectivamente, sin solapamiento.

El subconjunto tiene una estructura por fórmula para evitar que composiciones multiregistro dominen. Se usa una división aleatoria determinista por fórmula (semilla 42) antes de tomar filas. La muestra no usa etiquetas ABS₃ para elegir candidatos. `ID_DISJOINT` está verificado tras excluir IDs; la novedad estructural frente a Alex‑20 sigue `STRUCTURAL_NOVELTY_UNRESOLVED` porque ese CSV no aporta geometrías para comparar mediante `StructureMatcher`.

## Controles individuales

Checkpoint/hash, tokenizer/hash, vinculación con el release PBE, energía cruda de los registros seleccionados, compatibilidad del vocabulario, contexto sin truncamiento, round-trip estrategia 4, exclusiones de ID/SLICES/fórmula ABS₃, división por fórmula y smoke forward/backward: **PASS**. Parámetros POTCAR/U completos por registro: **PARTIAL**. Equivalencia cristalográfica Alex‑20: **NOT VERIFIABLE**. El entrenamiento F4 está **RUNNING**, no tiene todavía una métrica final.

El smoke de datos devolvió vector `FINAL_END` de 768 dimensiones, pérdida finita y norma de gradiente de cabeza `7.60277`; el smoke F4 separado devolvió gradiente no nulo en bloques 8–11 (`0.547` en la prueba de 8 estructuras) y mantuvo congelados embeddings y bloques 0–7. Un minibatch F4 de 8 tomó 2.08 s con 2 hilos; extrapolar linealmente da cerca de 5.7 días de pasos de entrenamiento para 100 épocas, antes de validation y preparación. La corrida real usa early stopping y permanece en curso. Los conteos, hashes, salidas y recursos están en `experiments/matgpt_alexandria_energy_transfer_20k/metrics.json`, `summary.json` y `resources.json`.

## Archivos preparados

- `experiments/matgpt_alexandria_energy_transfer_20k/alexandria_20k_train.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/alexandria_2k_validation.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/exclusion_manifest.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/dft_provenance_by_campaign.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/tokenizer_compatibility_sample.csv`

Los targets por átomo se calculan como `entry.energy / n_atoms`, en eV/átomo. El total original en eV se conserva para reconstrucción. Las curvas y métricas por época se escriben en `historial_entrenamiento.csv`; `training_trace.jsonl` aporta una traza de solo anexado con las épocas observadas, marcas de tiempo, pérdidas, estado, recursos disponibles y hashes de configuración/manifiestos. `training.log` conserva mensajes del proceso. El monitor `scripts/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/trace_run.py` es de solo lectura y no modifica ni reinicia el entrenamiento.
