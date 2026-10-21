# Modelos preentrenados para ABS₃

Selección activa: **MatText-SLICES-2M y CLOUD**. MatGPT queda excluido de la comparación de predictores. Los manifiestos y verificaciones históricos de MatGPT se conservan como procedencia, sin reintroducir sus archivos ni modificar los datos del usuario.

## Artefactos disponibles — 21 de octubre de 2026

| Modelo | Ruta local | Estado |
| --- | --- | --- |
| MatText-SLICES-2M | `mattext-slices-2m/` | Pesos y configuración originales; SHA-256 verificado. Tokenizador exacto pendiente de demostrar. |
| CLOUD | `cloud/ckpt/`, `cloud/source/` | Pesos, tokenizador, código y tablas del conversor descargados. SHA-256 de pesos verificado contra el hash LFS oficial. |

Procedencia: [`mattext-manifest.json`](mattext-manifest.json) y [`cloud-manifest.json`](cloud-manifest.json). El enlace proporcionado corresponde a **CLOUD**, cuya representación es **SCOPE**; no se renombra como SLICES-BERT ni se alimenta con cadenas SLICES.

## MatText-SLICES-2M

Encoder BERT bidireccional preentrenado con masked language modeling sobre dos millones de estructuras cristalinas expresadas como SLICES. El checkpoint define **4 bloques, 8 cabezas, dimensión 512, FFN 3072, vocabulario de 30.522 filas y contexto de 512 tokens**. Contiene **32.962.048 parámetros** y no incluye una cabeza de regresión ABS₃.

Repositorio: [n0w0f/MatText-slices-2m](https://huggingface.co/n0w0f/MatText-slices-2m), revisión `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`. SHA-256 de pesos: `6bef5652d758190e6e6eaf85ccf2a2f584b69255f1be3bb7bdd72cb1f12a74bd`.

El checkpoint no publica archivos suficientes para asumir que `AutoTokenizer.from_pretrained()` funciona. El [código oficial de MatText](https://github.com/lamalab-org/MatText) y el vocabulario original de [Zenodo](https://zenodo.org/records/11484062) permiten auditar una reconstrucción candidata; aún falta demostrar la correspondencia de IDs y tokens especiales con estos pesos. No se usa un tokenizador BERT estándar ni se redimensiona la matriz.

## CLOUD

Modelo BERT preentrenado con masked language modeling sobre representación **SCOPE**, con **12 capas, 12 cabezas, dimensión 768, FFN 3072 y contexto 64** según `cloud/ckpt/config.json`. Los pesos publicados son un backbone preentrenado; requieren una cabeza supervisada y ajuste específico para predecir energías ABS₃.

Fuentes: [código oficial BattModels/CLOUD](https://github.com/BattModels/CLOUD) y [checkpoint ChangwenXu/CLOUD](https://huggingface.co/ChangwenXu/CLOUD). Revisiones fijadas:

- Pesos: `7cf997311dd19fcb3fde0585d22557656258197d`.
- Código: `9da4e7c7a5f507df2640315ad8085ad7bfaab3b5`.
- SHA-256 de `model.safetensors`: `eef300981df17e7581da3c86bb7310facbc0219d73bf44bf8798c2a2f0e8f0b9` (530.587.440 bytes).

Se incluyen tokenizador y código del conversor `cloud/source/structure_to_str.py` con las tablas originales. Se excluyeron `data/mb_cloud.json` y los datasets de ejemplo, según el manifiesto. Para una comparación futura se deberán crear entradas SCOPE de las mismas geometrías validadas y reutilizar exactamente las particiones y referencias GGA; no se ejecutó entrenamiento CLOUD.

## Experimento 01: MatText congelado

Implementación en [`scripts/exp01_mattext_frozen/`](scripts/exp01_mattext_frozen/); configuración y reproducción en [`experiments/exp01_mattext_frozen/README.md`](../experiments/exp01_mattext_frozen/README.md).

Se encontraron 480 cálculos (240 relajaciones y 240 estáticos, 80 composiciones). El conjunto provisional con tres familias completas contiene **225 estructuras y 75 composiciones**. El experimento está **bloqueado antes de extracción científica y Slurm** por una diferencia de geometría estático/relax de CuSnS₃ hexagonal y por la falta de tokenizador exacto verificable.

La adaptación de SLICES 4 a 3 conserva las unidades estructurales. Las 225 entradas candidatas no presentan omisiones ni truncamiento (máximo 352 tokens), y no hay colisiones de IDs candidatas entre polimorfos de igual composición. Esto no demuestra la asignación semántica de los IDs del checkpoint.

Las ocho pruebas técnicas pasan, incluida carga estricta del MatText original, 0 parámetros entrenables y determinismo con IDs sintéticos explícitos. No hay embeddings científicos, métricas supervisadas ni Job IDs. Ver [`tokenizer_compatibility.md`](../reports/exp01_mattext_frozen/audit/tokenizer_compatibility.md) y [`experiment_report.md`](../reports/exp01_mattext_frozen/experiment_report.md).

## Reproducción de descargas

```bash
# CLOUD solamente; revisión y SHA-256 fijados:
python model-pretrained/scripts/download_models.py --models cloud

# Selección activa del descargador: MatText y CLOUD.
# No repetir MatText si los pesos ya están presentes.
```

`new-models-validation.json` y `scripts/validate_models.py` pertenecen a la validación histórica MatText/MatGPT; no representan una ejecución del Experimento 01 ni validan CLOUD. El experimento utiliza sus propios controles, manifiestos y pruebas, conservando los registros anteriores sin alterarlos.
