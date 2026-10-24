# Diagnósticos MatGPT D1–D3

## Qué responden

- **D1 composición:** cuánto puede explicar la fórmula química de la energía absoluta, y por qué un predictor de fórmula no puede escoger entre polimorfos de igual fórmula.
- **D2 objetivo:** si sustituir pairwise por energía centrada por composición cambia el aprendizaje, manteniendo la arquitectura parcial de E2-A.
- **D3 representación:** si cadenas SLICES, tokens, embeddings originales y descriptores geométricos separan los tres polimorfos estáticos.

Los tres experimentos usan el dataset E1 de MatGPT, folds GroupKFold por composición y solo desarrollo (210 estructuras / 70 grupos). Los ocho grupos de test histórico no entran en preparación de features, escalado, entrenamiento, selección ni métricas. La validation histórica participó en selección anterior de candidatas; las nuevas comparaciones no eliminan ese sesgo. Los códigos previos E0–E2 se reutilizan sin modificar.

## Ejecución

Desde la raíz del repositorio y el entorno científico del proyecto:

```bash
python -u Alternative2/scripts/MatGPT_diagnostics/audit_previous_results.py
python -u Alternative2/scripts/MatGPT_diagnostics/d1_composition_baselines.py --output Alternative2/reports/MatGPT_diagnostics/D1/v3
python -u Alternative2/scripts/MatGPT_diagnostics/d3_static_representation.py --device cpu --output Alternative2/reports/MatGPT_diagnostics/D3/v5
python -m unittest discover -s Alternative2/scripts/MatGPT_diagnostics/tests -v
python -u Alternative2/scripts/MatGPT_diagnostics/d2_centered_training.py --phase smoke --device cpu --output Alternative2/reports/MatGPT_diagnostics/D2/smoke_v1
python -u Alternative2/scripts/MatGPT_diagnostics/d2_centered_training.py --phase pilot --device cpu --fold 0 --seed 42 --output Alternative2/reports/MatGPT_diagnostics/D2/pilot_v1
python -u Alternative2/scripts/MatGPT_diagnostics/d2_centered_training.py --phase cv --device cpu --output Alternative2/reports/MatGPT_diagnostics/D2/cv_v1
python -u Alternative2/scripts/MatGPT_diagnostics/summarize_d2.py --phase cv --input Alternative2/reports/MatGPT_diagnostics/D2/cv_v1/oof_predictions_cv.csv --output Alternative2/reports/MatGPT_diagnostics/D2/cv_v1
python -u Alternative2/scripts/MatGPT_diagnostics/audit_outputs.py --d2-cv Alternative2/reports/MatGPT_diagnostics/D2/cv_v1
python -u Alternative2/scripts/MatGPT_diagnostics/compare_diagnostics.py --d2 Alternative2/reports/MatGPT_diagnostics/D2/cv_v1 --output Alternative2/reports/MatGPT_diagnostics/comparison/v1
python -u Alternative2/scripts/MatGPT_diagnostics/plot_diagnostics.py
python -u Alternative2/scripts/MatGPT_diagnostics/write_reports.py
```

El CV de D2 son 30 entrenamientos nuevos (2 pérdidas × 5 folds × 3 semillas). El baseline D2-BASE es la predicción partial_ft E2-A ya registrada, no una corrida duplicada. El script rechaza una salida CV ya existente; elige un directorio nuevo para repetir. No ejecutar el CV completo si el smoke/piloto falla. No correr en GPU/Slurm automáticamente.

La ejecución documentada completó 30/30 ajustes en CPU y guardó 30 checkpoints pequeños (último bloque + cabeza). `audit_outputs.py` valida 1,260 predicciones OOF, 70 composiciones × tres polimorfos para cada combinación, exclusión de test y hashes. Las figuras finales son `D2/cv_v1/figures/d2_cv_error_pareado.png`, `d2_cv_ranking_folds_semillas.png` y `d2_cv_curvas_entrenamiento.png`. La lectura actual es una mejora pequeña de CENTER-ONLY en selección del mínimo, sin mejora clara de ΔE o exactitud pairwise; ver `D2_report.md` para intervalos y límites.

## Qué calcula D2

En cada batch entran grupos completos de tres polimorfos. La pérdida centrada compara cada energía con la media de su propio triplete. La escala RMS centrada y la normalización absoluta se estiman con inner-train únicamente. El checkpoint se elige por la suma de los términos incluidos en esa variante sobre inner-validation. `D2-CENTER-ONLY` no aprende offsets entre composiciones; el offset de predicción para computar MAE absoluto es la media de energías inner-train, una referencia de baseline, no una salida aprendida por la pérdida centrada.

Pairwise y centered cuadrático contienen esencialmente la misma información diferencial cuando usan todos los pares y una reducción equivalente. Con Huber, escalas y muestreo pueden optimizar de forma distinta. D2 comprueba un efecto de la función objetivo y su optimización; no presupone una señal física nueva.

## Salidas

Los resultados y figuras versionadas viven en `Alternative2/reports/MatGPT_diagnostics/`. Las tablas `oof_predictions*.csv` corresponden solo a outer-validation; no usar valores del smoke para inferencia científica. `D1_report.md`, `D3_report.md`, `baseline_reconciliation.md` y el reporte integrado explican cada magnitud en lenguaje sencillo.

## Referencias metodológicas

El código MatGPT se fija al commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e` ([repo oficial](https://github.com/devalab/matgpt)). SLICES: Xiao et al. (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7). Grouped CV: MatFold, Witman & Schindler (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D). SOAP como descriptor publicado alternativo: [documentación DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html); no se usó SOAP en D3 porque DScribe no está instalado. MPtrj y CHGNet: [documentación de MPtrj](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj), Deng et al. (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3). Uso de inferencia: [Autograd de PyTorch](https://docs.pytorch.org/docs/stable/notes/autograd.html).
