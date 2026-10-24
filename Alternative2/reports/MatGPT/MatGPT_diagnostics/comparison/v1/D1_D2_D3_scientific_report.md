# D1–D3 — qué limita la predicción de estabilidad polimórfica en ABS₃

## Resumen en una frase

La fórmula química explica muy bien el nivel de energía absoluta de estas composiciones, pero no discrimina polimorfos; las cadenas estáticas SLICES sí son distintas, aunque el pooling `<end>` preentrenado no sigue la magnitud de |ΔE|. La pérdida centrada cambia algunas métricas en el CV, pero la recomendación final depende de si esas diferencias son consistentes y de los intervalos pareados.

## Qué se hizo y qué no

Desarrollo únicamente: 210 estructuras estáticas DFT-PBE, 70 composiciones y tres polimorfos por composición; GroupKFold en cinco folds. D1 utilizó inner-train/inner-validation para Ridge. D2 comparó dos pérdidas sobre el mismo backbone parcial de E2-A; D2-BASE son las predicciones E2-A ya existentes. D3 congeló el checkpoint original unconditional. El test histórico de ocho composiciones no entró en ninguna etapa. La validation histórica sí participó en selección previa de candidatas, así que estos resultados no son una prueba independiente.

## D1: composición explica energía absoluta, no ranking

Ridge de composición obtiene MAE absoluta 59.3 meV/átomo y R² promedio por fold 0.989. El modelo E2-A partial_ft tiene MAE absoluta alrededor de 729 meV/átomo. La diferencia pareada Ridge−E2-A fue negativa y su bootstrap de 70 composiciones está en `paired_bootstrap_ci.json`. Eso demuestra que la representación actual de MatGPT/head no está calibrando bien el nivel absoluto en este benchmark; no demuestra que una feature concreta sea universal.

Los tres polimorfos de una fórmula tienen las mismas features Ridge. Por eso el predictor da idéntica energía y no resuelve el ranking, aunque prediga bien el valor absoluto. La varianza de energía observada se separa en ~99.6% entre medias de composición y ~0.4% dentro de cada fórmula. Es una descripción de este conjunto, no una explicación causal.

## D2: comparación de funciones objetivo

| Configuración | MAE absoluto, media ± DE (semillas) / (folds), meV/átomo | MAE ΔE, media ± DE (semillas) / (folds), meV/átomo | GS | Pairwise | Corridas colapsadas |
|---|---:|---:|---:|---:|---:|
| D2-BASE E2-A partial_ft | 728.9 ± 4.9 (semillas) / 728.9 ± 82.9 (folds) | 37.25 ± 1.38 / 37.25 ± 7.24 | 0.495 | 0.669 | 0/15 |
| D2-CENTER | 728.7 ± 4.1 (semillas) / 728.7 ± 82.9 (folds) | 36.88 ± 1.62 / 36.88 ± 7.01 | 0.509 | 0.667 | 0/15 |
| D2-CENTER-ONLY | 731.7 ± 2.0 (semillas) / 731.7 ± 81.4 (folds) | 38.32 ± 0.71 / 38.32 ± 6.47 | 0.545 | 0.689 | 0/15 |

Bootstrap pareado por composición (promedio de las tres semillas dentro de cada fórmula antes de remuestrear): `paired_bootstrap_ci.json`. Los intervalos son exploratorios. Si CENTER baja MAE absoluta pero no ΔE/ranking, la ganancia se limita a calibración absoluta. CENTER-ONLY puede recuperar variación relativa, pero no identifica offset entre composiciones. Pairwise y centrado codifican información diferencial relacionada; una diferencia de rendimiento puede proceder de SmoothL1, escalas, batching o convergencia, no de una señal física nueva.

## D3: distinguen las cadenas; asociación energética es otra pregunta

En 70 composiciones, las tres cadenas SLICES e IDs tokenizados son distintos. No se observó colisión exacta de cadena estática dentro de estos tripletes. Embeddings originales: checkpoint SHA-256 `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`, strict load, 768 dimensiones, padding error 1.19e-06. Las correlaciones Spearman exploratorias con |ΔE| DFT: end euclidiana 0.07, masked-mean euclidiana 0.28, geometría 0.45. Tres pares comparten cada fórmula; las correlaciones son exploratorias y no demuestran error predictivo ni causalidad.

E2-T1 ya mostró 1,148 de 1,677 grupos de IDs repetidos con rango energético ≥1 meV/átomo en frames de relajación. No se extrapola ese resultado a los estáticos: D3 no halló colisiones exactas de strings estáticos.

## Conclusión científica y recomendación

1. **Limitación absoluta:** D1 demuestra que la fórmula ofrece señal fuerte; el gran MAE de E2-A no puede explicarse diciendo simplemente que la composición es insuficiente. Revisar calibración de salida, normalización, pooling, cabeza y función objetivo.
2. **Limitación del ranking:** D1 no puede ordenar por diseño. D3 muestra inputs SLICES distintos, pero `<end>` congelado casi no se asocia con gaps DFT. Esto apunta a representación/pooling o aprendizaje supervisado, pero no los separa causalmente.
3. **D2 y siguiente paso:** CENTER no cambia de forma material las métricas frente a E2-A. CENTER-ONLY mejora GS alrededor de 5 puntos porcentuales, con IC bootstrap pareado apenas positivo; ΔE MAE y pairwise no mejoran claramente y el MAE absoluto no cambia. Es una señal modesta y exploratoria para el mínimo de triplete, no una mejora global confirmada. D2 completó científicamente, pero el SHA exacto del trainer al inicio no se capturó; los cambios durante la corrida fueron de medición de tiempo/guardas y no cambiaron el protocolo. Antes de ampliar el fine-tuning o añadir geometría, hace falta validación en composiciones nuevas que no participaron en la preselección histórica.
4. **D4:** los análisis D3 aportan una pista para diseñar —sin afirmar que vaya a funcionar— una comparación SLICES+geometría con estructuras estáticas y mismos folds/cabeza. La asociación geométrica es exploratoria. Si se evalúa, debe comparar contra CENTER-ONLY y E2-A de forma pareada y con holdout nuevo.
5. **No usar test:** ninguna cifra de este reporte proviene de los ocho grupos históricos de test. Para una afirmación independiente se necesita un holdout nuevo o nested CV completo porque validation histórica ya participó en preselección.

## Figuras

Las figuras PNG están en `D1/v3/figures`, `D2/pilot_v1/figures`, `D2/cv_v1/figures` y `D3/v5/figures`. Las figuras CV explican los puntos pareados por composición, el conteo de pliegues/semillas y la selección de épocas. Los reportes específicos explican unidades, denominadores, empates y límites interpretativos.

## Referencias verificadas

- MatGPT, [código oficial](https://github.com/devalab/matgpt), commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- Witman & Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- SOAP, [documentación DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html); alternativa geométrica publicada, no usada aquí.
- [MPtrj de Materials Project](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj) y Deng et al., CHGNet (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3), como contexto de MLIP, no comparadores.
- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html), para separar extracción en inference mode de actualización entrenable.
