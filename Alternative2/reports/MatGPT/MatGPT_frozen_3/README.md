# MatGPT_frozen_3 — R1–R4

Corrida activa: `run_20261023_v2`. El [informe científico R1–R4](run_20261023_v2/comparisons/R1_R4_scientific_report.md) reúne métodos, resultados, límites e intervalos pareados. Consulte también la [auditoría de referencia](run_20261023_v2/baseline_audit/baseline_audit.md), el [audit de todos los outputs](run_20261023_v2/manifests/final_audit.json), y los reportes individuales de [R1](run_20261023_v2/R1_linear_probes/R1_report.md), [R2](run_20261023_v2/R2_head_capacity/R2_report.md), [R3](run_20261023_v2/R3_regularization/R3_report.md) y [R4](run_20261023_v2/R4_layers_pooling/R4_report.md).

## Lectura rápida

- Se completaron 75 entrenamientos de cabezas (5 configuraciones × 5 folds × 3 semillas). El MatGPT original quedó congelado; solo se usaron 210 estructuras y 70 composiciones de development.
- R4-SELECTED-ABSOLUTE bajó el MAE absoluto a 153.3 meV/átomo, pero su GS/pairwise fue 35.8%/53.4%. R4 mejora el nivel de energía y empeora la selección de polimorfo.
- H0 tiene 729.1 meV/átomo de MAE absoluto, principalmente por error en el nivel medio composicional: MSE de media 694,466 frente a MSE residual 1,789 meV²/átomo².
- R1-C (solo diferencias) obtuvo 38.1 meV/átomo de MAE ΔE DFT-GS-anclada, similar a H2 (39.3). Aún hay error apreciable en el ranking y no se alcanzan metas de alta precisión.
- D1 Ridge composición-only obtuvo 59.3 meV/átomo de MAE absoluto, pero asigna el mismo valor a todos los polimorfos de una fórmula y no puede ordenarlos.

## Cómo leer las figuras principales

- `absolute_energy_parity_with_D1_ridge.png`: comprueba nivel absoluto. Cada punto es una estructura; la línea discontinua es el acuerdo perfecto y la rosa es el ajuste OLS. D1 es solo referencia de energía, no de ranking.
- `composition_mu_parity.png`: cada punto es una fórmula completa, promediada sobre sus tres polimorfos. Evalúa si una cabeza recupera el nivel energético medio.
- `energy_decomposition.png`: separa MSE de error medio y residuo centrado. La suma de ambos reconstruye el MSE total; los MAE no son aditivos.
- `parity_dft_gs_anchored.png` y `parity_delta_own_min_legacy.png`: muestran por separado las dos definiciones de ΔE. Valores negativos en la predicción DFT-GS-anclada son inversiones del orden, no errores que se deban recortar.
- `gs_confusion_matrices.png` y `pairwise_confusion_matrices.png`: muestran qué familias se intercambian. Empates predichos son abstenciones; los empates DFT se excluyen del accuracy primario.
- `fold_seed_stability.png`, `paired_composition_differences.png` y `R3_regularization_gap.png`: muestran variación entre grupos/folds/semillas y la diferencia train frente a validación interna.

Los 3 grupos con brecha DFT ≤1 meV/átomo se excluyen de GS (N=67); el accuracy pairwise incluye 206 de los 210 pares físicos. El test histórico no se leyó. La validation histórica participó en selección previa, por lo que los intervalos y las comparaciones son exploratorios. Las figuras exportadas son PNG exclusivamente.
