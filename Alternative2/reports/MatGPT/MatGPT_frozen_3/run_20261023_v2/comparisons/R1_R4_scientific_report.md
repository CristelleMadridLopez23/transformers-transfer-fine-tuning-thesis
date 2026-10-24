# MatGPT frozen R1–R4: informe científico comparativo

## Auditoría y alcance

Checkpoint E0 SHA-256: `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`; commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`. Development: 210 estructuras/70 composiciones, 3 polimorfos por composición; cinco outer folds y 11 grupos inner-validation dentro de cada outer-train. Test histórico (24 estructuras/8 composiciones) fue excluido antes de extraer features. No se descongeló MatGPT ni se usó geometría.

Las cabezas H2 y los hiperparámetros/pooling ya fueron investigados con parte de la validation histórica. Por ello, aunque la selección nueva ocurre dentro de cada fold y no consulta outer-validation, las comparaciones son exploratorias y no sustituyen un holdout externo nuevo.

## Métricas OOF pareadas

| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | absolute | 729.08 | 729.08 | 30.63 | 37.99 | 35.25 | 56.7% (67) | 67.5% (206) | 51.41 |
| H1 | absolute | 548.47 | 548.11 | 31.85 | 39.45 | 37.07 | 58.2% (67) | 68.4% (206) | 55.24 |
| H2 | absolute | 352.53 | 351.29 | 32.01 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |
| R1-A-ABS | absolute | 530.49 | 439.85 | 301.58 | 353.34 | 386.60 | 41.8% (67) | 54.9% (206) | 675.05 |
| R1-D-COMBINED | mu_plus_delta | 507.78 | 507.32 | 30.44 | 38.07 | 35.63 | 56.7% (67) | 68.4% (206) | 56.47 |
| R2-REF8 | absolute | 352.53 | 351.29 | 32.01 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |
| R2-SMALL4 | absolute | 337.51 | 336.55 | 32.58 | 40.36 | 36.89 | 58.2% (67) | 67.0% (206) | 51.35 |
| R3-DROP30 | absolute | 367.96 | 366.57 | 30.23 | 37.51 | 34.92 | 62.7% (67) | 70.9% (206) | 50.97 |
| R3-EARLY8 | absolute | 419.37 | 417.79 | 32.06 | 39.51 | 36.98 | 59.7% (67) | 69.4% (206) | 52.90 |
| R3-WD1E2 | absolute | 352.78 | 351.57 | 32.00 | 39.28 | 36.49 | 59.7% (67) | 67.5% (206) | 50.39 |
| R4-FIXED-FINAL_END | mu_plus_delta | 507.78 | 507.32 | 30.44 | 38.07 | 35.63 | 56.7% (67) | 68.4% (206) | 56.47 |
| R4-FIXED-FINAL_MEAN | mu_plus_delta | 230.06 | 227.70 | 30.81 | 38.11 | 35.69 | 55.2% (67) | 63.1% (206) | 56.66 |
| R4-SELECTED-2COMP | mu_plus_delta | 217.54 | 213.61 | 32.88 | 41.06 | 37.81 | 50.7% (67) | 61.2% (206) | 51.24 |
| R4-SELECTED-ABSOLUTE | absolute | 153.27 | 125.55 | 87.92 | 99.49 | 101.08 | 35.8% (67) | 53.4% (206) | 160.53 |

D1 Ridge composition-only obtuvo MAE absoluta 59.29 meV/átomo en los mismos IDs, pero da energía idéntica a los tres candidatos de cada fórmula; ranking indeterminado. No se incorporó a R1 ni a las cabezas.

## Selecciones internas

R1 alpha por fold/target: `R1_linear_probes/selected_alpha_counts.csv`. Para R4, resumen de capas/poolings elegidos por outer fold: `R4_layers_pooling/selected_features_by_fold.csv`. La evaluación solo puntúa los candidatos seleccionados en inner-validation y los controles finales fijos; la tabla de heatmap contiene únicamente puntuaciones de inner-validation.

## Incertidumbre y comparación pareada

Bootstrap pareado por 70 composiciones, 2,000 remuestras, semillas promediadas por estructura. Las tablas `paired_bootstrap.json` comparan candidatos con H2; las entradas no aplicables a objetivos μ-only o δ-only se dejan NA. Los intervalos no son prueba de superioridad.

## Sobreajuste y selección de época

Por corrida se guardan loss train por minibatch, validación interna completa por triplete, escalas train-only, gradientes, época seleccionada y segundos de entrenamiento. La regla de parada y selección común es el objetivo H2 de SmoothL1 composicional + residual, normalizado con estadísticas de inner-train. El CSV `capacity_regularization_audit.csv` permite revisar las brechas en la época elegida; la curva no usa outer folds para detener.

## Interpretación científica

**Acceso al nivel absoluto.** H0 registra MAE 729.1 meV/átomo; H2 baja a 352.5. R4-SELECTED-ABSOLUTE obtiene 153.3 meV/átomo y su diferencia pareada candidato−H2 es -199.3 [-262.3, -132.4] meV/átomo. La selección interna eligió L03_MEAN para energía absoluta en los cinco folds. Su ranking baja a GS 35.8% y pairwise 53.4%, frente a 59.7% y 67.5% en H2.

**Nivel medio y residuo.** R1-B (μ-only) obtiene MAE de media 507.3 meV/átomo y no discrimina polimorfos. R1-C, que estima solo δ, obtiene MAE ΔE anclada 38.1; H2 obtiene 39.3. En H0, MSE del error de media = 694466, frente a MSE residual = 1789 meV²/átomo²: el error absoluto de H0 viene principalmente del offset medio, mientras que sus diferencias centradas son mucho menores. La identidad MSE se verifica por composición; los MAE no se suman.

**Capacidad y regularización.** R2-small4 da MAE absoluto 337.5 (H2 352.5) con la mitad de parámetros de cabeza; GS/pairwise son 58.2%/67.0%. R3-dropout 0.30 llega a GS 62.7% y pairwise 70.9%, con MAE absoluto 368.0. Las diferencias de R2/R3 y sus IC pareados están en `paired_bootstrap.json`; intervalos que cruzan cero no respaldan una mejora estable.

**Referencia externa.** D1 composición-only obtiene MAE 59.3 meV/átomo en los mismos 210 IDs, pero asigna una sola energía a los tres polimorfos, así que su ranking es indeterminado. D1 no forma parte de las cabezas.

R1 cuantifica accesibilidad lineal de E, μ y δ por separado; R2/R3 prueban capacidad y regularización con un backbone idéntico; R4 determina si una lectura END/MEAN de capas previas cambia los probes. No se selecciona un ganador global por promedios outer. R4-B (MLP sobre capas elegidas) queda opcional y no ejecutado.

Limitaciones: solo 70 composiciones, selección exploratoria histórica, energía DFT estática total por átomo, posible descalibración del nivel absoluto, estrategias Ridge de dimensión alta, y diferencia de objetivo/dominio frente a NequIX. Durante Ridge se imprimieron advertencias numéricas del solver; los chequeos explícitos confirmaron predicciones finitas para todas las selecciones y folds, pero las sondas lineales deben leerse con cautela. No hay fuerzas, tensiones ni phonons en este experimento.

## Figuras

Se generaron 19 PNG. El manifiesto de figuras registra que no se exportó SVG/PDF/EPS/JPG/HTML/diapositivas. Consulte `comparisons/figures/` y la interpretación en `R1_report.md`–`R4_report.md`.

## Referencias

- MatGPT (código histórico): https://github.com/devalab/matgpt, commit local fijado.
- SLICES: Xiao et al. (2023), https://doi.org/10.1038/s41467-023-42870-7.
- MatFold: Witman & Schindler (2025), https://doi.org/10.1039/D4DD00250D.
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.
- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html.
- PyTorch hooks/autograd: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html and https://docs.pytorch.org/docs/stable/notes/autograd.html.
- PBE: Perdew, Burke & Ernzerhof (1996), https://doi.org/10.1103/PhysRevLett.77.3865.
- Métricas H0–H2 de contexto: `Alternative2/reports/MatGPT_frozen_2/v1/H0_H1_H2_report.md`.
