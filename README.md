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

## MatGPT — experimentos realizados y fundamento

Actualización de esta línea: 23 de octubre de 2026.

Esta línea explora si un Transformer generativo preentrenado sobre SLICES puede transferirse a la energía total estática GGA/PBE por átomo y al orden de estabilidad de polimorfos ABS₃. Se mantiene el checkpoint unconditional de MatGPT, su código local fijado al commit `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e` y el SHA-256 del checkpoint `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`. Los experimentos recientes excluyen el test histórico de 8 composiciones; la validación histórica ya fue observada en trabajos previos y no es una prueba ciega.

| Experimento | Pregunta y fundamento | Estado observado |
| --- | --- | --- |
| E0 — carga y procedencia | ¿Los pesos corresponden a la arquitectura y vocabulario de MatGPT y se cargan estrictamente? Establece identidad del checkpoint antes de cualquier inferencia científica. | PASS: carga `strict=True`; 172 tokens, contexto 442, dimensión 768, 12 capas/cabezas. |
| E1 — embeddings congelados y regresión | ¿Una representación congelada codifica señales útiles para energía ABS₃? Controla transferencia sin actualizar el Transformer. | Se extrajeron 234 embeddings de 768 dimensiones. La evaluación inicial tuvo errores de implementación que se corrigieron en la revisión v2; no se usan sus artefactos fallidos como resultados. |
| E1.1 / E1.2 — diagnóstico de cabezas | ¿Cómo cambian error absoluto y orden relativo con pooling/cabeza/pérdida? Motiva separar predicción de energía absoluta de ranking polimórfico. | Selección preliminar en las 8 composiciones de validation histórica; no convierte esa partición en evaluación independiente. |
| E1.3 — CV por composición | ¿Se sostienen los candidatos entre composiciones no vistas durante cada fold? Usa GroupKFold y deja el test histórico fuera. | PASS: 45 ejecuciones (3 configuraciones × 5 folds × 3 semillas; 210 estructuras/70 composiciones). Reporte y métricas en [`E1_3_cross_validation_report.md`](Alternative2/reports/MatGPT_frozen/matgpt_e1_3/E1_3_cross_validation_report.md). El sesgo por preselección en validation permanece explícito. |
| E2-A — fine-tuning parcial | ¿Descongelar solo el último bloque mejora la predicción frente al backbone congelado? Prueba adaptación limitada para reducir sobreajuste en el conjunto pequeño. | PASS: validación OOF de 5 folds × 3 semillas para backbone congelado y ajuste parcial, 30 ejecuciones. Reporte [`E2_report.md`](Alternative2/reports/MatGPT_finetuning/matgpt_e2/runs_endtoken_v2/cpu/E2_report.md). Se detectó y corrigió el error histórico de pooling que podía seleccionar `<start>` en vez del `<end>` real. |
| E2-T0 — diagnóstico de trayectorias previo | ¿Las secuencias SLICES de los pasos de relajación son distinguibles y coherentes con sus energías? | LIMITED: el cache previo tuvo 8,898 frames, 2,865 secuencias y colisiones energéticas; su equivalencia de grafo no estaba certificada. Se reemplaza por la auditoría ampliada E2-T1. |
| E2-Aug1 — vistas equivalentes | ¿La aleatorización de orden de sitios/aristas mejora robustez sin cambiar estructura, etiquetas ni número de muestras? Prueba regularización por presentación, conservando la pérdida E2-A. | PASS: banco de desarrollo 210 × 4 vistas = 840, equivalencia de grafo/tokens, smoke y CV 5 folds × 3 semillas. OOF: MAE absoluta macro 726.4 ± 79.8, MAE ΔE 38.75 ± 7.14 meV/átomo, GS 0.499 ± 0.135 y pairwise 0.648 ± 0.063. Frente a E2-A los IC pareados por composición incluyen cero para todas las métricas; no hay mejora consistente. Reporte/figuras [`aug1_development_only_v3`](Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/aug1_development_only_v3/). |
| E2-Aug2 — invariancia | ¿Penalizar discrepancias de energía entre dos SLICES equivalentes reduce dependencia de la serialización? Aísla consistencia de entrada con supervisión estática intacta. | PASS: smoke y CV de 5 folds × 3 semillas, 630 predicciones OOF. MAE absoluta macro 731.5 ± 82.8, MAE ΔE 37.93 ± 7.80 meV/átomo, GS 0.514 ± 0.127 y pairwise 0.636 ± 0.058. Frente a E2-A todos los IC pareados incluyen cero; reduce σ entre cuatro serializaciones (17.47 meV/átomo frente a 23.78 de E2-A) sin mejora demostrada de ranking. Comparación [`both_development_only_v2`](Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/both_development_only_v2/); invariancia [`invariance_development_only_v2`](Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/invariance_development_only_v2/). `0.1 L_inv` fue fijado a priori, no optimizado. |
| E2-T1 — auditoría de relajaciones | ¿Qué fracción de los cambios geométricos de una trayectoria permanece representable en IDs SLICES y bajo etiquetas DFT comparables? Determina si hay base defendible para usar trayectorias. | LIMITED: 12,486 frames de desarrollo, 12,486 grafos equivalentes, metadatos PBE/PAW_PBE sin +U y etiquetas estáticas cotejadas. Solo 32.3% de frames tienen IDs únicos; 1,148/1,677 grupos repetidos difieren ≥1 meV/átomo. Se permite piloto, no CV completa de T2. |
| E2-T2 — pérdida de trayectoria | ¿Las diferencias de energía dentro de un mismo cálculo añaden señal a la predicción estática? Solo es válido con evidencia T1; evalúa exclusivamente polimorfos estáticos outer. | PILOT_ONLY: una corrida fold 0/semilla 42, 35 épocas, emparejada con Aug2 en 14 composiciones (42 estructuras). MAE ΔE 38.30 frente a 39.44; GS 0.539 frente a 0.385; pairwise 0.714 frente a 0.631; MAE absoluta 731.85 frente a 720.21 meV/átomo. Los IC bootstrap pareados (2,000 remuestras) incluyen cero para todas las diferencias. T1 LIMITED no autoriza CV completa. Reporte [`E2_T2_pilot_report.md`](Alternative2/reports/MatGPT_finetuning/E2-T2/pilot_comparison_development_only_v2/E2_T2_pilot_report.md). |

Los scripts y reportes de la nueva línea se ordenan por experimento en `Alternative2/scripts/MatGPT_finetuning/` y `Alternative2/reports/MatGPT_finetuning/`. La decisión científica debe comparar métricas OOF pareadas por composición —MAE absoluta, MAE de ΔE, exactitud de estado fundamental y de pares—, dispersión entre serializaciones, folds y semillas. Los intervalos bootstrap son exploratorios. Un resultado de train loss o una disminución pequeña de MAE no bastan para recomendar E2 posterior.

### Lectura conjunta de E2-Aug y E2-T2

E2-Aug1 y E2-Aug2 completaron cada uno 15 corridas (5 folds × 3 semillas) sobre 210 estructuras/70 composiciones de desarrollo; cada candidato aporta 630 predicciones outer. Ninguna variante supera consistentemente a E2-A en ranking polimórfico: los intervalos pareados por composición para MAE de ΔE, GS y pairwise incluyen cero. Aug2 sí baja la desviación media entre cuatro serializaciones equivalentes de 23.78 a 17.47 meV/átomo; esto demuestra más invariancia a la entrada, pero no demuestra mayor exactitud. Las MAE absolutas siguen alrededor de 0.72 eV/átomo, muy lejos de un predictor cuantitativo útil para energías estáticas.

El piloto T2 es direccionalmente favorable en GS/pairwise y ΔE MAE frente a Aug2 en el único fold evaluado, pero sus IC con 14 composiciones son amplios e incluyen cero, y su error absoluto es peor. Junto con la tasa alta de colisiones de ID en T1, no hay evidencia para ejecutar T2 completo ni E2 de mayor alcance ahora. La prioridad recomendada es diagnosticar por qué SLICES/MatGPT no recupera energía absoluta y qué parte del error proviene de representación, salida/pooling o tamaño del conjunto; luego se necesitaría un holdout externo nuevo antes de cualquier afirmación independiente. El test histórico no se consultó en estos experimentos.

Nota de procedencia: durante el preflight inicial se generó un banco exploratorio que contenía 234 IDs, incluidos los 24 IDs del test, y llegó a comprobar su conversión estructural; no se entrenó ni se calcularon métricas con ese banco. Ese banco, su smoke y preflight están marcados `SUPERSEDED` y no sustentan resultados. Se regeneró el banco v2 después de excluir test antes de conversión; CV final, evaluación e invariancia usan solo 210 IDs de desarrollo. La primera auditoría T1 que comenzó sobre todos los frames también se interrumpió y no produjo un reporte válido; el T1 citado aquí es la auditoría corregida con exclusión de test antes de conversión.

### Referencias MatGPT y método

- MatGPT: [repo devalab/matgpt](https://github.com/devalab/matgpt), usando el commit histórico local indicado arriba; el modelo se basa en aprendizaje de secuencias cristalinas SLICES.
- Vasamsetti, Pal, Ehara & Priyakumar, “MatGPT: Transformer-Based Generation of Inorganic Materials with Targeted Properties”, manuscrito indicado por el repositorio como enviado a *Digital Discovery*; se cita el [código y documentación de MatGPT](https://github.com/devalab/matgpt) porque las corridas usan el commit local fijado.
- SLICES: Xiao et al., “An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning”, *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7). Fundamenta la representación periódica string-based; en nuestros experimentos la equivalencia de cada variante se verifica de forma adicional contra el grafo reconstruido.
- MatFold: Witman & Schindler, “MatFold: systematic insights into materials discovery models' performance through standardized cross-validation protocols”, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D). Fundamenta que particiones de materiales deben controlar parentesco estructural/composicional para medir generalización.
- MPtrj / CHGNet: Deng et al., “CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling”, *Nature Machine Intelligence* (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3), y [documentación MPtrj de Materials Project](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj). Se cita como referencia de datos/MLIP para situar el alcance: E2 no entrena fuerzas ni tensiones y no se presenta como potencial interatómico.
- Materiales y DFT: [Materials Project Calculation Details](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details) documenta el contexto de los cálculos del proyecto; la compatibilidad concreta se estableció con los metadatos de cada tarea fuente, no por asumir una configuración genérica del artículo.
- DFT/PBE: Perdew, Burke & Ernzerhof, *Physical Review Letters* 77, 3865 (1996), [doi:10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865); los parámetros efectivos de cada cálculo se auditan a partir de los metadatos originales.
- Aleatorización de representaciones moleculares: Arús-Pous et al. (2019), [doi:10.1186/s13321-019-0393-0](https://doi.org/10.1186/s13321-019-0393-0). Es antecedente de augmentation por serialización; no demuestra por sí solo equivalencia de SLICES cristalino, que aquí se verifica mediante grafos periódicos.
- PyTorch [SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html): función robusta usada en pérdidas absolutas y diferenciales.
- Validación y gradientes: documentación de [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), ejemplo de [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), y PyTorch [`inference_mode`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html). Estas fuentes respaldan el agrupamiento por composición, el límite de la selección previa y el cuidado al reutilizar tensores de inferencia.

Las referencias de MatFold y aleatorización motivan el diseño; no son evidencia de que estas variantes mejoren MatGPT en ABS₃. La evidencia propia debe provenir de folds agrupados por composición y de los reportes producidos aquí.

## MatGPT D1–D3 — separar química, objetivo y representación

Actualización: 23 de octubre de 2026. Reportes y figuras explicadas en español: [`Alternative2/reports/MatGPT_diagnostics/`](Alternative2/reports/MatGPT_diagnostics/); scripts y pruebas: [`Alternative2/scripts/MatGPT_diagnostics/`](Alternative2/scripts/MatGPT_diagnostics/). Se usaron los cinco folds de E1.3 y 210 estructuras/70 composiciones de desarrollo; las 24 estructuras/8 composiciones del test histórico quedaron fuera. La validation histórica ya había participado en selección de candidatas, por lo que las comparaciones son diagnósticas y no confirmatorias.

| Auditoría | Motivo | Resultado actual |
| --- | --- | --- |
| D1 — baselines químicos | Medir cuánta energía absoluta explica la fórmula antes de atribuir el error a SLICES. | PASS: Ridge de composición MAE ≈59 meV/átomo, R² OOF ≈0.990; E2-A partial_ft ≈729 meV/átomo. Ridge no ordena polimorfos: los tres reciben la misma predicción. Bootstrap Ridge−E2-A por 70 grupos: diferencia ≈−669 meV/átomo (IC95% exploratorio [−763, −575]). [`D1_report.md`](Alternative2/reports/MatGPT_diagnostics/D1/v3/D1_report.md). |
| D2 — pérdida centrada | Probar si entrenar directamente desviaciones dentro de composición modifica ranking, manteniendo MatGPT E2-A parcial. | PASS científico: 30/30 ajustes (2 pérdidas × 5 folds × 3 semillas), 1,260 predicciones OOF y 30 checkpoints. CENTER-ONLY sube GS ~5 puntos porcentuales, pero ΔE y pairwise no mejoran claramente. PARTIAL de procedencia: el hash exacto del trainer al inicio no quedó capturado; el cambio posterior fue solo bookkeeping/guarda, con protocolo científico igual. [`D2_report.md`](Alternative2/reports/MatGPT_diagnostics/D2/cv_v1/D2_report.md). |
| D3 — representación estática | Distinguir colisiones de entrada de problemas de aprendizaje: cadena SLICES, IDs, embeddings y geometría. | PASS: 70/70 composiciones tienen tres SLICES e IDs distintos; checkpoint original strict, SHA E0, 768 dimensiones y padding invariant. La distancia `<end>` casi no se asocia con |ΔE| (Spearman ρ≈0.07); masked-mean ρ≈0.28 y descriptores geométricos exploratorios, agrupados en 70 fórmulas. [`D3_report.md`](Alternative2/reports/MatGPT_diagnostics/D3/v5/D3_report.md). |

La reconciliación E2-A explica GS 0.533 frente a 0.495: el primer valor usa `argmin` en los 70 tripletes, mientras el segundo excluye 3 composiciones con gap DFT ≤1 meV/átomo, trata predicciones empatadas como no aciertos y promedia por fold antes de semillas. La diferencia usa las mismas predicciones; ver [`baseline_reconciliation.md`](Alternative2/reports/MatGPT_diagnostics/common/baseline_reconciliation.md).

Resultados D2 en las mismas 70 composiciones OOF (medias macro; errores en meV/átomo):

| Configuración | MAE absoluta | MAE de ΔE | GS | Exactitud pairwise | Lectura sencilla |
| --- | ---: | ---: | ---: | ---: | --- |
| E2-A partial_ft (baseline) | 728.9 | 37.25 | 49.5% | 66.9% | Punto de comparación. |
| D2-CENTER | 728.7 | 36.89 | 50.9% | 66.7% | Cambios prácticamente nulos. |
| D2-CENTER-ONLY | 731.7 | 38.32 | 54.5% | 68.9% | Escoge mejor el mínimo, pero no reproduce mejor las separaciones de energía; evidencia pequeña. |

El bootstrap pareado por composición encuentra para CENTER-ONLY una diferencia de GS de **+5.0 puntos** (IC95% exploratorio **[+0.5, +9.5]**, 67 grupos elegibles); el IC de pairwise incluye cero y ΔE MAE empeora en promedio ~1.1 meV/átomo, también con IC que incluye cero. CENTER no presenta una mejora clara. Por tanto, D2-CENTER-ONLY queda como hipótesis prometedora solo para selección del mínimo, no como ganador general ni recomendación para aumentar fine-tuning sin un holdout externo. La corrida completó científicamente, pero el hash exacto del trainer cargado al inicio no quedó guardado; la limitación está en `integrity_checks_cv.json`. En [el reporte D2](Alternative2/reports/MatGPT_diagnostics/D2/cv_v1/D2_report.md) se explican las métricas, los denominadores, la variación entre folds/semillas y cómo leer las tres figuras.

La evidencia D1 muestra que la composición basta para predecir bien el nivel absoluto en este conjunto, pero no ranking polimórfico. D3 no detecta colisiones exactas de SLICES en los estáticos, aunque embeddings `<end>` no ordenan diferencias energéticas de forma visible. D2 encuentra una ganancia pequeña de selección del mínimo con CENTER-ONLY, sin cambio claro en ΔE/pairwise. Esto no prueba que falte geometría: pooling, objetivo, cabeza y tamaño de datos siguen siendo alternativas. Antes de D4 o de mayor fine-tuning, hace falta confirmación con composiciones nuevas no involucradas en la preselección histórica; cualquier mejora debe evaluarse de forma pareada y sin test histórico.

Referencias metodológicas: MatGPT [repo](https://github.com/devalab/matgpt), commit local fijado; SLICES, Xiao et al. (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7); MatFold, Witman & Schindler (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D); SOAP [DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html) como descriptor publicado que no se usó; [MPtrj](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj)/CHGNet, [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3), como contexto de potenciales geométricos; y [Autograd de PyTorch](https://docs.pytorch.org/docs/stable/notes/autograd) para distinguir inferencia y entrenamiento.



## Estructura del proyecto

- `data/`: datasets fuente y procesados.
- `embeddings/`: representaciones extraídas.
- `models/`: implementaciones y checkpoints base originales.
- `scripts/`: código Python organizado por familia y experimento.
- `reports/`: informes científicos y documentación de auditoría.
- `images/<experimento>/`: todas las figuras PNG, agrupadas por experimento.
- `experiments/<experimento>/`: CSV, JSON, logs, checkpoints derivados y configuración usada.

Las carpetas históricas `Alternative2/01_data`, `Alternative2/embeddings`, `Alternative2/models`, `Alternative2/reports` y `Alternative2/scripts` son enlaces de compatibilidad. El código nuevo debe usar las rutas raíz. Los experimentos se ejecutan localmente con Python; no se conservan lanzadores shell ni configuraciones Slurm.
