# MatText: auditoría y fine-tuning posterior sobre only_hex_new

Estado: la compatibilidad con el código histórico oficial está verificada para
las **8.712 entradas SLICES aceptadas**. Sigue pendiente demostrar la asignación
del tokenizador usada en el checkpoint específico. El entrenamiento permanece
bloqueado; no se han generado modelos ni métricas de fine-tuning.

Fuentes principales: [paper](https://doi.org/10.1038/s43588-026-01045-2),
[suplemento](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43588-026-01045-2/MediaObjects/43588_2026_1045_MOESM1_ESM.pdf),
[código archivado v1.1.0](https://doi.org/10.5281/zenodo.21151839),
[tokenizador histórico](https://github.com/lamalab-org/xtal2txt/blob/1d9d0c6f8675efea885ef7c9c87b3195cd79aaed/src/xtal2txt/tokenizer.py).
Las descargas públicas, sus revisiones y hashes se conservan en
`reports/exp02_only_hex_mattext_finetune/audit/reference/`.

## Qué se comprobó

- El job de conversión **4310202** terminó COMPLETED, código 0:0. De 8.898 frames,
  8.712 pasaron SLICES y 186 quedaron registrados como rechazos de representación.
- Las 8.712 entradas coinciden ID por ID y máscara por máscara con `__call__` de
  la implementación oficial de 2024, ejecutada con Transformers 4.38.2.
- Se conserva el grafo completo al adaptar estrategia 4 → 3. No se omitieron
  caracteres estructurales, no aparecieron UNK y no se truncó ninguna entrada.
  Longitud máxima: 401 frente a un contexto de 512.
- Vocabulario original de 149 tokens estructurales + 7 especiales. IDs:
  UNK=149, PAD=150, CLS=151, SEP=152, MASK=153, EOS=154, BOS=155.
- El constructor con los valores por defecto documentados asigna especiales de
  otra manera. Se usa el orden de la configuración de preentrenamiento archivada,
  y una sola envoltura CLS/SEP, reproducida mediante la API histórica.
- La máscara se construye por longitud. No se confunde ID estructural 0
  (`o o o`) con padding. El config del encoder conserva `pad_token_id=0`;
  no se alteró el checkpoint ni se redimensionaron embeddings.
- Hay 2.853 entradas tokenizadas únicas y 1.119 grupos compartidos por varios
  frames. No hubo colisiones de IDs entre familias de polimorfos estáticos de una
  misma composición. Las colisiones de trayectoria se conservan y se documentan.

La compatibilidad con código publicado y la procedencia del artefacto usado en un
entrenamiento particular son comprobaciones diferentes. Se inspeccionaron las
cinco revisiones públicas del checkpoint; ninguna contiene tokenizador guardado.
Además, la configuración archivada `finetune2m/slice.yaml` apunta a un checkpoint
`30k_rt`, y no prueba la asignación del modelo 2m. No se copió esa ruta.

Para levantar el bloqueo se requiere un artefacto original publicado del
tokenizador o evidencia verificable vinculada a estos pesos que establezca el
vocabulario/IDs y la envoltura. Un booleano manual no constituye evidencia.
El esquema verificable está en `01_audit_tokenizer.checkpoint_evidence()`.

## Flujo posterior y adaptación del paper

| Componente | Método del paper / decisión local explícita |
| --- | --- |
| Modelo | BERT local MatText-SLICES-2M; carga estricta del encoder original. |
| Tokenización | Código histórico y vocabulario publicados, sin ampliación ni truncamiento. |
| Regresión | BERT pooler sobre CLS → dropout → Linear(512,1), como `BertForSequenceClassification`. |
| Parámetros | Actualizar encoder y cabeza completos; sin unfreezing gradual en este flujo. |
| Optimización | LR 2e-4; AdamW; scheduler lineal; weight decay 0; semilla 42. |
| Parada | Máximo 100 épocas; paciencia 10; umbral de mejora RMSE 0,001 eV/átomo. |
| Batch | 16, decisión local para la memoria GPU; el YAML archivado usa 64. |
| Etiqueta | Energía DFT-GGA por átomo del frame original. No se sustituye por energía de formación del benchmark del paper. |
| Particiones | Conservar las 64/8/8 composiciones originales train/val/test. Tras rechazos: 7.027/932/753 frames. |
| CV | Opción de cinco pliegues agrupados dentro de las 64 composiciones train; val/test originales quedan fuera de esos pliegues. |
| Evaluación | Test reservado; energía de todos los frames y selección polimórfica solo con conjuntos estáticos completos. |

El split local no es una réplica exacta del benchmark de cinco pliegues del paper:
se preserva el test independiente original y se agrupan composiciones para evitar
que pasos de una trayectoria o polimorfos crucen particiones. Los resultados del
experimento congelado anterior no serían una comparación controlada con este
nuevo dataset/split; cualquier comparación futura requiere las mismas entradas.

El paper advierte limitaciones para aprovechar geometría. Por eso se auditan las
colisiones y se separan precisión energética, ordenamiento estático y sensibilidad
a cambios de geometría. El corpus contiene frames fuera del equilibrio; el campo
`converged` del cálculo padre no implica fuerzas nulas en cada frame. El solapamiento
exacto con el corpus NOMAD de preentrenamiento permanece desconocido.

## Reproducir la auditoría

Desde la raíz, con `dmc` activo. Todos los paquetes históricos se aíslan del entorno
existente; los pesos no se vuelven a descargar:

```bash
bash sh_files/exp02_only_hex_mattext_finetune/prepare_runtime.sh
python -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/fetch_references.py
bash sh_files/exp02_only_hex_mattext_finetune/audit.sh
```

La auditoría devuelve código distinto de cero mientras falte evidencia crítica.
Revisar `tokenizer_compatibility.md`, `audit/tokenizer_audit.json`,
`audit/tokenizer_per_structure.csv`, `tables/tokenized_collisions.csv` y
`tables/static_composition_coverage.csv` en el directorio de reportes.
Los IDs se conservan como `data/processed/exp02_only_hex_mattext_finetune/tokenized_candidate.jsonl`;
no se crea una entrada científica aprobada hasta superar todas las comprobaciones.

Con el mismo PYTHONPATH aislado, preflight y pruebas:

```bash
export PYTHONPATH="$PWD/experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages"
python -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/04_preflight.py
```

## Fine-tuning, únicamente después de un preflight satisfactorio

```bash
bash sh_files/exp02_only_hex_mattext_finetune/submit.sh
```

El wrapper ejecuta pruebas y smoke CPU acotado, verifica los hashes de auditoría,
cuenta Slurm, duplicados y admisión con `sbatch --test-only`. Solo entonces solicita
1 GPU, 8 CPU, 32 GB y 4 h en `msigpu`, cuenta `cbartel`. Los logs se guardan en
`reports/exp02_only_hex_mattext_finetune/logs/`. No se instalan paquetes en GPU.

`02_finetune.py --fold -1` ajusta el modelo final train→val; `--fold 0` hasta
`--fold 4` ejecuta CV agrupada solo dentro de train, siempre dentro de una
asignación GPU y tras los mismos controles. El wrapper estándar envía únicamente
el ajuste final, sin lanzar cinco trabajos adicionales de forma implícita.

El mejor checkpoint supervisado y sus curvas se guardan en
`experiments/exp02_only_hex_mattext_finetune/artifacts/`. Después del ajuste final,
`03_evaluate.py` realiza una única evaluación test y compara la selección estática
con la familia favorecida más frecuentemente en composiciones completas train.
Las composiciones con candidatos estáticos ausentes se reportan explícitamente;
no se calcula un falso mínimo entre candidatos incompletos.
