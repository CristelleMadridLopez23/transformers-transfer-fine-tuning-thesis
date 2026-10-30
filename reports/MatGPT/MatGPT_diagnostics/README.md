# Resultados de auditoría MatGPT D1–D3

## Versiones de referencia

- **D1 PASS:** `D1/v3/D1_report.md`; cinco folds por composición, 210 estructuras de desarrollo, 70 fórmulas.
- **D2 PASS:** `D2/cv_v1/D2_report.md`; 30 ajustes, 1,260 predicciones OOF, 30 checkpoints ligeros e integridad en `D2/cv_v1/integrity_checks_cv.json`.
- **D3 PASS:** `D3/v5/D3_report.md`; checkpoint congelado original, 210 estructuras/70 composiciones de desarrollo, 210 pares intracomposición.
- **Síntesis:** `comparison/v1/D1_D2_D3_scientific_report.md`.
- **Procedencia y hashes:** `common/experiment_manifest.json`.

Las versiones anteriores de D1 y D3 se conservan. Los resultados canónicos son únicamente los directorios indicados arriba; las carpetas raíz y las versiones anteriores no se deben mezclar con las tablas finales. D2 completó el protocolo científico, aunque el hash exacto del trainer importado al inicio de la corrida no quedó congelado; la única diferencia posterior fue añadir bookkeeping de tiempo y una protección de salida. Esa limitación está anotada en `D2/cv_v1/integrity_checks_cv.json` y el manifiesto.

## Alcance e interpretación

Las tres auditorías usaron desarrollo únicamente (210 estructuras estáticas, 70 composiciones, tres polimorfos por composición). Las 24 estructuras y ocho composiciones del test histórico no se usaron en preparación, entrenamiento, selección ni evaluación. Los ocho grupos de validation histórica participaron en selección previa de candidatas E1.1/E1.2; por eso estos diagnósticos de desarrollo no son estimaciones confirmatorias independientes.

D1 muestra que la fórmula explica gran parte del nivel absoluto de energía, aunque no puede ordenar polimorfos de igual fórmula. D2 encuentra una mejora pequeña de CENTER-ONLY para identificar el mínimo DFT; los intervalos para MAE y pairwise incluyen cero. D3 no detecta colisiones exactas de SLICES estáticos, pero una distancia del embedding congelado `<end>` casi no se asocia con |ΔE|. Ninguno de esos resultados prueba causalidad.

## Referencias metodológicas

- MatGPT: [repositorio oficial](https://github.com/devalab/matgpt), commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- SLICES: Xiao et al., *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- Validación por grupos: Witman & Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D), y [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- SmoothL1 y gradientes: [documentación de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html) y [Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html).
- Geometría: [tutorial SOAP de DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html), citado como alternativa metodológica; no se calculó SOAP en D3.
