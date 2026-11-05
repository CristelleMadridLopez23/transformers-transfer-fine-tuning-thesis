# MatterGPT Alexandria F4, subconjunto compatible

Preparación terminada; **no se ejecutó entrenamiento**. Se resolvió la incompatibilidad preservando el checkpoint/tokenizer MatterGPT y cada SLICES original: se excluyeron solo las filas que tienen tokens OOV o superan el contexto de 397 tokens. No hubo reemplazo de tokens ni truncamiento.

| Split original | Filas iniciales | OOV | Contexto excedido | Excluidas tras deduplicar causas | Filas listas |
|---|---:|---:|---:|---:|---:|
| Train | 20 000 | 2 749 | 9 | 2 755 | 17 245 |
| Validación | 2 000 | 282 | 1 | 282 | 1 718 |

Las tres filas train que exceden contexto también tenían OOV; una fila de validación tenía ambas causas. Las etiquetas energéticas se copiaron intactas. Los splits mantienen la asignación original; se verificó que no comparten fórmulas. Los archivos originales siguen intactos.

La configuración F4 prevista está en `config.yaml`; hashes y exclusiones están en `manifest.json` y `exclusion_manifest.csv`. El tokenizer usa condición fija `[1.0, 2.0]`, seguida por la secuencia completa; se descongelarán bloques 8–11 después de 20 épocas de warmup de la cabeza.


## Comando F4 MatterGPT (preparado; aún no ejecutado)

Ejecutar desde la raíz del repositorio, con el entorno de PyTorch 2.8 que dispone de MPS:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/private/tmp/matter-mpl-cache /Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python scripts/MatterGPT/MatterGPT_Alexandria_Energy_Transfer_20k_F4_Compatible/train_energy.py \
  --device mps --seed 42 --batch-size 8 --validation-batch-size 8 --threads 4 \
  --max-epochs 100 --patience 15 \
  --data-dir experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset --output-dir experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset
```

Este comando entrena sobre el subconjunto compatible de 17 245/1 718 filas. No cambia el checkpoint MatterGPT; usa `[1.0, 2.0]` como condición fija y selecciona por SmoothL1 de validación.
