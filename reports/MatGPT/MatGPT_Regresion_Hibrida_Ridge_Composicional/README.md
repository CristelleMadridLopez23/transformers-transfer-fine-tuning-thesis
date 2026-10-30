# Regresión híbrida con Ridge composicional

Experimento OOF de desarrollo ABS₃: el Ridge de 17 fracciones estequiométricas predice la energía media por fórmula. Se combina en paralelo con residuos MLP8 Frozen o LoRA previamente guardados y centrados por fórmula. No se entrenaron MLP8/LoRA, no se ejecutó inferencia nueva de MatGPT y el test histórico permaneció cerrado.

## Reproducción

Desde la raíz del repositorio: `bash sh_files/MatGPT_Regresion_Hibrida_Ridge_Composicional/run.sh`. El wrapper registra tiempo, CPU y RSS máximo en `resources.json` y salida completa en `run.log`.

## Archivos principales

- `oof_predictions_by_structure.csv`: cuatro condiciones, IDs físicos, pliegues y semillas.
- `composition_only_predictions.csv`: un valor por estructura, con residuo cero y sin adjudicar ranking entre empates.
- `ridge_selection.csv`, `ridge_validation_grid.csv` y `models/`: selección interna y cinco modelos por pliegue.
- `metrics_by_seed.csv`, `metrics_by_fold_seed.csv`, `composition_metrics_by_formula.csv`, `pair_predictions.csv`.
- `paired_bootstrap_by_composition.csv`: IC95% apareado por fórmula.
- `invariance_audit.csv`: prueba de igualdad de ΔE, ranking y residuo.
- `figures/`: nueve figuras científicas y diagrama; PNG y SVG.

## Interpretación

Cambiar Ridge solo cambia el término constante por composición. ΔE y ranking deben conservarse. Una reducción del error absoluto mide una mejor estimación de la media composicional, no más información estructural de MatGPT.
