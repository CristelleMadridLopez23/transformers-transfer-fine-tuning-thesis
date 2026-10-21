# Predicción de energía y estabilidad polimórfica en ABS₃

Repositorio de tesis sobre representaciones cristalográficas y transferencia de aprendizaje. La comparación prevista conserva **MatText-SLICES-2M y CLOUD**. MatGPT queda fuera de la selección activa. CLOUD es el modelo del [repositorio proporcionado](https://github.com/BattModels/CLOUD): usa **SCOPE**, no SLICES.

## Estado real — 21 de octubre de 2026

- **MatText**: checkpoint local original disponible; SHA-256 verificado, 32.962.048 parámetros. Carga estricta en CPU, encoder congelado y determinismo verificados con IDs sintéticos explícitamente identificados como prueba técnica.
- **CLOUD**: descarga completada de pesos, tokenizador y código oficial; SHA-256 de los pesos verificado contra Hugging Face. Procedencia y revisiones en [`cloud-manifest.json`](model-pretrained/cloud-manifest.json). No se ha entrenado ni evaluado científicamente.
- **Experimento 01**: implementación modular y auditoría realizadas; **bloqueado antes de extracción científica y envío a Slurm**. No hay Job IDs ni métricas de regresión.

## Experimento 01: MatText como extractor congelado

Objetivo: evaluar energía DFT-GGA por átomo y selección del polimorfo de menor energía mediante Ridge, Random Forest y una MLP pequeña sobre los mismos embeddings y particiones por composición. El Transformer no se entrena. Mean pooling excluye tokens especiales y padding; CLS es alternativa seleccionable mediante validación de desarrollo.

Se encontraron **480 cálculos originales: 240 relajaciones y 240 estáticos de 80 composiciones**. Las relajaciones se conservan como referencias y no se duplican como muestras. Tras verificar etiquetas, SLICES y la disponibilidad de las tres familias, se guardaron **225 estructuras candidatas de 75 composiciones**. La admisión científica del conjunto sigue bloqueada.

Los dos controles críticos pendientes son:

1. **Geometría de CuSnS₃ hexagonal**: una celda estática difiere de la relajación final hasta 0,004696 Å por componente de celda; desplazamiento ordenado máximo de sitios de 0,001533 Å. La relación debe aclararse antes de admitir esa referencia. No se amplió la tolerancia de 0,00001 Å.
2. **Tokenizador exacto de MatText**: se recuperó el vocabulario original de Zenodo y se auditó una reconstrucción candidata. Falta evidencia publicada que vincule sus IDs, tokens especiales y envoltura CLS/SEP con este checkpoint concreto. No basta con que sus 156 IDs quepan en las 30.522 filas de embeddings.

La transformación SLICES estrategia 4 → 3 conserva las unidades estructurales de las 225 candidatas, concuerda con la exportación previa y pasa la comprobación muestreada de la API oficial. La tokenización candidata no pierde caracteres no blancos, no trunca entradas (máximo 352 tokens) y no produce colisiones de IDs entre polimorfos del mismo compuesto. Estos hechos no resuelven el segundo bloqueo.

Consultar el [README del experimento](experiments/exp01_mattext_frozen/README.md), el [reporte científico y estado](reports/exp01_mattext_frozen/experiment_report.md), el [preflight](reports/exp01_mattext_frozen/audit/preflight.json) y la [auditoría del tokenizador](reports/exp01_mattext_frozen/audit/tokenizer_compatibility.md).

## Organización

| Ruta | Contenido |
| --- | --- |
| `AGENTS.md`, `AGENT.md` | Normas canónicas y orientación humana. |
| `data/results(1).json`, `data/toSLICES.ipynb`, `data/toSLICES_output/` | Datos y procesamiento previos preservados. |
| `data/processed/exp01_mattext_frozen/` | Dataset provisional, entradas candidatas y particiones. |
| `data/embeddings/exp01_mattext_frozen/` | Destino de embeddings; todavía sin resultados científicos. |
| `model-pretrained/` | Checkpoints originales y manifiestos de procedencia. |
| `model-pretrained/scripts/exp01_mattext_frozen/` | Inventario, auditorías, extracción, predictores, evaluación, reportes y pruebas. |
| `experiments/exp01_mattext_frozen/` | Configuración, instrucciones y manifiesto. |
| `reports/exp01_mattext_frozen/` | Auditorías, CSV, logs y tres figuras reales de auditoría. |
| `sh_files/exp01_mattext_frozen/` | Shell y Slurm del experimento. |

No se reorganizaron los datos ni se modificaron las estructuras, etiquetas, notebook o checkpoint original. Los cambios y eliminaciones que ya existían en Git se preservaron.

## Reproducción

Desde la raíz, con el entorno existente `dmc` activo:

```bash
bash sh_files/exp01_mattext_frozen/00_preflight.sh
python model-pretrained/scripts/exp01_mattext_frozen/test_exp01.py
bash sh_files/exp01_mattext_frozen/check_status.sh
```

El preflight **devuelve código distinto de cero mientras persistan los bloqueos**. Es el resultado esperado de los controles científicos. Las ocho pruebas técnicas pasan; sus casos sintéticos están identificados y no se presentan como resultados del experimento.

Solo después de resolver los bloqueos y obtener un preflight satisfactorio:

```bash
bash sh_files/exp01_mattext_frozen/submit_experiment.sh
```

La configuración solicita extracción en `msigpu` (1 GPU, 8 CPU, 32 GB, 4 h) y entrenamiento/evaluación en `msismall`, con cuenta `cbartel` detectada mediante `sacctmgr`. Las particiones y la asociación se consultaron realmente. Antes de enviar se comprobarán los recursos con `sbatch --test-only`; el segundo trabajo dependerá del Job ID real del primero mediante `afterok`. **No se ha ejecutado `sbatch` para este experimento.**
