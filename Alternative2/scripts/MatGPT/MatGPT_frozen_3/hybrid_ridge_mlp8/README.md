# H3: Ridge para la media + MLP8 para el residuo

Este experimento combina representaciones congeladas de MatGPT. La cabeza Ridge usa el promedio de `L03_MEAN` de los tres polimorfos para estimar la energía media de una composición. La MLP8 usa `FINAL_END` de cada polimorfo y predice residuos centrados. La suma produce las energías finales en eV/átomo.

## Ejecución reproducible

Desde la raíz del proyecto y con el entorno `transformers-transfer-fine-tune-tesis`:

```bash
python -m unittest discover -s Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/tests -v
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/audit_previous_results.py
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/run_cross_validation.py --phase recombine
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/run_cross_validation.py --phase pilot
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/run_cross_validation.py --phase cv
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/evaluate_h3.py
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/generate_figures.py
python -u Alternative2/scripts/MatGPT_frozen_3/hybrid_ridge_mlp8/finalize_manifest.py
```

Las fases son secuenciales: primero H3-A/B, luego el piloto H3-C y finalmente la CV completa. Las corridas existentes con estado PASS se reutilizan durante la CV. El preflight debe estar en PASS antes de empezar. No se cargan etiquetas del test histórico.

## Protocolo fijado

- MatGPT unconditional de E0, SHA-256 y commit fijados en `config.yaml`.
- Datos de desarrollo: 210 estructuras, 70 composiciones; 5 folds agrupados; semillas 42, 123 y 2026.
- Ridge: `StandardScaler` + `Ridge(solver="lsqr")`; alpha elegido por MAE de validación interna por composición, grilla `[0.01, 0.1, 1, 10, 100, 1000]`; después se reajusta sobre el entrenamiento externo completo.
- MLP8: 768→8→1, GELU, dropout 0.1, AdamW, LR 1e-3, weight decay 1e-4, máximo 200 épocas y paciencia 20. El objetivo SmoothL1 de residuos centrados usa la escala calculada en inner-train.
- Empates y exclusión GS: tolerancia 1 meV/átomo. Bootstrap pareado: 2,000 muestras de composiciones, semilla 13072026.

## Salidas

Los artefactos de cada fase, OOF, métricas, intervalos bootstrap y figuras PNG se escriben en `Alternative2/reports/MatGPT_frozen_3/hybrid_ridge_mlp8/`. El informe principal es `H3_scientific_report.md`; `manifest.json` enumera hashes de entradas y artefactos. Ninguna figura se exporta a SVG/PDF.

## Referencias

- [MatGPT](https://github.com/devalab/matgpt), código local fijado al commit histórico indicado en la configuración.
- Xiao et al., SLICES, *Nature Communications* (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Witman & Schindler, MatFold, *Digital Discovery* (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D).
