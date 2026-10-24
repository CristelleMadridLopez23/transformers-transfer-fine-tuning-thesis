# Auditoría de compatibilidad entre tokenizer y pesos de MatText

**Dictamen: INCOMPATIBLE por tamaño de vocabulario; correspondencia exacta de IDs no verificable.**

El checkpoint local es `n0w0f/MatText-slices-2m` en la revisión `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`. Se leyó el header de safetensors sin cargar ni alterar los pesos. Las formas de configuración y tensor son coherentes entre sí, pero no con el tokenizer recuperado:

| Comprobación | Valor |
|---|---:|
| Checkpoint vocabulary (config) | 30522 |
| Checkpoint word-embedding rows (tensor) | 30522 |
| Recovered tokenizer.json vocabulary | 156 |
| MatText documented SLICES vocab + CLS/SEP | 151 |
| Recovered PAD ID / checkpoint PAD ID | 150 / 0 |

El tokenizer recuperado conserva en orden los 149 tokens base de `slice_vocab.txt` (sí). Esto concuerda con la documentación oficial de `SliceTokenizer`, que muestra esos tokens y añade `[CLS]` y `[SEP]` como IDs 149 y 150. Sin embargo, el `tokenizer.json` recuperado agrega siete especiales: `{'[UNK]': 149, '[PAD]': 150, '[CLS]': 151, '[SEP]': 152, '[MASK]': 153, '[EOS]': 154, '[BOS]': 155}`. Su `[PAD]` es ID 150, mientras que el config del modelo fija `pad_token_id=0`.

La revisión descargada de Hugging Face contiene 4 archivos y no incluye `tokenizer.json`, `tokenizer_config.json` ni `vocab.txt`. El config y los pesos sí declaran/contienen 30,522 filas de embedding. Por eso el tokenizador compacto puede ser un tokenizador válido para el framework MatText, pero **no se puede emparejar con seguridad con estas filas aprendidas**. Usar sus IDs con el modelo daría tensores dentro del rango, pero no demuestra que, por ejemplo, el ID de `Ga` seleccione la fila que aprendió para `Ga`; además el padding especial difiere.

## Resultado científico

La verificación previa del tokenizador recuperado **no queda validada contra estos pesos**. La forma de embedding demuestra que no es el mismo vocabulario de 156 filas. La hipótesis más probable es que este checkpoint espera un vocabulario BERT de 30,522 entradas, pero los artefactos locales no incluyen el archivo de vocabulario necesario para demostrar los IDs exactos. No debe usarse el tokenizer recuperado para extraer embeddings/finetunear este checkpoint hasta recuperar la versión exacta del tokenizer usada durante el pretraining o verificar contra un ejemplo de inferencia oficial que exponga IDs/tokens.

## Procedencia y huellas

- Checkpoint SHA-256: `6bef5652d758190e6e6eaf85ccf2a2f584b69255f1be3bb7bdd72cb1f12a74bd`
- Config SHA-256: `ddd1b4cf361b4fe3a2724563d840e5424bfe05c62aa1ff922a1cc7d7a5da40c0`
- Tokenizer recuperado SHA-256: `c56736c0f2be65e2733454315348d616bb5e9bcb40845c53b63fbaff108396e2`
- `slice_vocab.txt` SHA-256: `53e2641804ff63c8b1f48b7311a74fd64dafe858d5935e89cb3e50a5c67c60ea`
- Pesos: tensor `embeddings.word_embeddings.weight`, forma `[30522, 512]`.
- Archivos tokenizador ausentes del snapshot: `['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']`.

## Referencias

- Documentación oficial de [MatText tokenizers](https://lamalab-org.github.io/MatText/docs/tokenizers/): vocabulario `SliceTokenizer`, `[CLS]` y `[SEP]`.
- [Repositorio MatText](https://github.com/lamalab-org/MatText).
- [Checkpoint MatText-slices-2m y ejemplo AutoTokenizer](https://huggingface.co/n0w0f/MatText-slices-2m).
- Alampara et al., [MatText, arXiv:2406.17295](https://arxiv.org/abs/2406.17295).
