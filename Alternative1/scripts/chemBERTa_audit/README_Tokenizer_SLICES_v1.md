---
title: "SLICES Tokenizer v1 | Plan ejecutable de auditoría y construcción"
date: 2026-10-21
tags: [tesis, ChemBERTa, SLICES, Alexandria, tokenizer, DAPT, reproducibilidad]
status: implementation-prepared-not-run-on-full-corpus
---

# SLICES Tokenizer v1 — Implementación y guardrails

> [!info] Contexto
> Fuente: 98,394 SLICES válidos de Alexandria (`slices_basic_valid` y `slices_valid`); 320 unidades léxicas distintas por separación con espacios. La auditoría previa reveló pérdida de símbolos, colisiones entre materiales distintos, incompatibilidad de padding y secuencias que exceden 512 tokens.

## Resultado del cruce de vocabulario (confirmado en los archivos de auditoría suministrados)

| Concepto | Valor |
|---|---:|
| Unidades SLICES distintas | 320 |
| Unidades ya existentes en `vocab` original | 24 |
| Unidades que deben agregarse | 296 |
| IDs originales accesibles (incluyendo especiales añadidos) | 593 (0–592) |
| Filas de embedding del checkpoint indicadas en `config.json` | 600 |
| IDs originales sin token accesible reservado | 7 (593–599) |
| Primer ID nuevo permitido | 600 |
| Longitud prevista del vocabulario candidato | 896 |

> [!warning] Advertencia metodológica
> Que los IDs y las filas originales queden conservados **no** significa que el conocimiento semántico de SMILES se transfiera automáticamente a SLICES. Los 296 nuevos vectores se inicializan y requieren aprendizaje (p. ej., DAPT MLM). Tampoco se debe dar por aprobado un checkpoint con `config.pad_token_id=1` y tokenizador con `[PAD]=0`.

## 1. Instalar los scripts en el proyecto

Desde la raíz del repositorio, copiar estos dos archivos dentro de `scripts/chemBERTa_audit/`:

- `audit_slices_strategy4.py`: fases A y B, auditoría de cabecera, especies, índices de enlaces, desplazamientos periódicos y duplicados preliminares.
- `build_slices_tokenizer_v1.py`: constructor WordLevel + WhitespaceSplit y controles de IDs, longitud, reconstrucción y colisiones.

Requisitos para construir: `pip install transformers tokenizers`; para la prueba del modelo: PyTorch instalado. **No se modifica el checkpoint original.**

### Paso 1 — Cerrar auditoría A/B del corpus real

```bash
python -u scripts/chemBERTa_audit/audit_slices_strategy4.py \
  --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
  --column slices --require-valid \
  --tokenizer model-pretrained/chemberta-77m-mlm \
  --context-samples 1000 --full-tokenization \
  --out reports/slices_strategy4_deep_audit
```

Comprobar:

- `summary.json` / `max_index_per_structure.csv`: índice máximo observado por cristal.
- `header_templates.csv`, `header_units.csv`: inventario de cabeceras y posición de unidades.
- `jimage_vectors.csv`: patrones de periodicidad.
- `strategy4_parse_warnings.csv`: falsaciones del parseo heurístico.
- `tokenizer_lexical_diagnostics.csv` y `tokenizer_context_sample.csv`: pérdida aislada y en contexto, sin confundir ausencia de token completo con pérdida de caracteres.
- `composition_prototype_groups.csv`: candidatos para revisión de equivalencia; no prueba estructural.

> [!caution] Limitación
> El parser de estas herramientas sigue una **heurística explícita** (tripletas terminales índice–índice–jimage y elementos inmediatamente anteriores). No certifica la gramática exacta de la versión instalada de SLICES ni la equivalencia de cristales. El paquete oficial y, si están las estructuras disponibles, `StructureMatcher`, son verificaciones independientes pendientes.

### Paso 2 — Preflight sin construir tokenizador

```bash
python -u scripts/chemBERTa_audit/build_slices_tokenizer_v1.py \
  --checkpoint model-pretrained/chemberta-77m-mlm \
  --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
  --column slices --require-valid \
  --prepare-only --out reports/slices_tokenizer_v1
```

Esta etapa extrae inventario de unidades e IDs y escribe el manifiesto. Puede ejecutarse sin librerías Hugging Face. **No genera aún un tokenizer utilizable**.

### Paso 3 — Construir candidato + auditar las 98,394 cadenas

```bash
python -u scripts/chemBERTa_audit/build_slices_tokenizer_v1.py \
  --checkpoint model-pretrained/chemberta-77m-mlm \
  --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
  --column slices --require-valid \
  --out reports/slices_tokenizer_v1
```

Produce `candidate_tokenizer/` utilizable con `AutoTokenizer.from_pretrained(...)`, además de:

- `summary.json`
- `vocab_candidate_manifest.json`
- `added_slices_tokens.csv`
- `lexical_units.csv`
- `header_patterns.csv`
- `periodicity_triplets.csv`
- `max_edge_index_distribution.csv`
- `lexical_errors.csv`
- `collision_examples.csv`
- `long_sequences.csv`

No aplica truncamiento al corpus auditado. Las secuencias superiores al límite de contexto quedan documentadas para una política explícita posterior.

### Paso 4 — Smoke de pesos (opcional, NO entrena)

```bash
python -u scripts/chemBERTa_audit/build_slices_tokenizer_v1.py \
  --checkpoint model-pretrained/chemberta-77m-mlm \
  --input data/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_all.csv \
  --column slices --require-valid --smoke-model \
  --out reports/slices_tokenizer_v1
```

La prueba carga `AutoModelForMaskedLM`, redimensiona embeddings **solo en memoria**, comprueba igualdad exacta de los pesos originales y ejecuta un forward con `position_ids` explícitos. Esto **no corrige** la discrepancia de padding para el entrenamiento posterior: sigue siendo un guardrail independiente.

## 2. Política de aprobación

- [x] Inventario léxico exploratorio previo (320 unidades).
- [x] Diseño para preservar todos los IDs heredados y las filas originales 0–599.
- [x] Código para auditar índice máximo, cabecera y periodicidad creado.
- [ ] Auditoría profunda ejecutada en los 98,394 registros y revisada.
- [ ] Validar parser frente a la implementación oficial instalada de SLICES `strategy=4`.
- [ ] Tokenizador WordLevel construido y recargado sin alteración de IDs.
- [ ] Cero fallos de reconstrucción de unidades léxicas en todo el corpus.
- [ ] Cero colisiones entre cadenas distintas después de normalizar espacios.
- [ ] Cero tokens `[UNK]` en los datos admitidos.
- [ ] Resolver incompatibilidad PAD/posiciones del checkpoint para MLM.
- [ ] Política documentada para 1,675 secuencias que exceden 512 tokens según longitud hipotética.
- [ ] Revisar duplicados cristalográficos (no basta fórmula/prototipo o `mat_id`).
- [ ] Iniciar DAPT únicamente después de despejar los bloqueos anteriores.

## 3. Referencias de implementación

- [Repositorio oficial SLICES](https://github.com/xiaohang007/SLICES)
- [Hugging Face Tokenizers — pipeline](https://huggingface.co/docs/tokenizers/pipeline)
- [RoBERTa position IDs y padding (código)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py)

> [!note] Reproducibilidad
> Los valores anteriores vienen de los reportes suministrados por el proyecto. El código se probó localmente con datos sintéticos y verificación de sintaxis; la construcción real con Transformers/Tokenizers y la auditoría completa deben ejecutarse en tu entorno de tesis.
