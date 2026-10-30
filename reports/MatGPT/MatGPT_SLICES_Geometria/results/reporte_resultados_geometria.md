# Resultados: SLICES frente a SLICES + geometría de celda

**Diseño:** MatGPT frozen, Ridge histórico sin cambios, MLP8 con 8 neuronas; 70 composiciones, 5 folds, 3 semillas.

**Advertencia:** geometría puede proceder de estructuras relajadas DFT; no equivale a predicción sin geometría conocida.

## Promedios por condición

| condition                 |     seed |   n_structures |   n_compositions |   n_pairs |   n_pairs_eligible |   mae_energy_total_ev |   mae_energy_per_atom_mev |   mae_residual_intraformula_mev |   mae_delta_pairs_mev |   mae_delta_zero_mev |   mae_delta_anchored_mev |   ground_state_eligible_accuracy |   ground_state_eligible_n |   ground_state_argmin_raw_accuracy |   pairwise_accuracy_gt5 |   energy_slope |   energy_intercept |   energy_correlation |   energy_std_ratio |   residual_slope |   residual_intercept |   residual_correlation |   residual_std_ratio |   delta_slope |   delta_intercept |   delta_correlation |   delta_std_ratio |
|:--------------------------|---------:|---------------:|-----------------:|----------:|-------------------:|----------------------:|--------------------------:|--------------------------------:|----------------------:|---------------------:|-------------------------:|---------------------------------:|--------------------------:|-----------------------------------:|------------------------:|---------------:|-------------------:|---------------------:|-------------------:|-----------------:|---------------------:|-----------------------:|---------------------:|--------------:|------------------:|--------------------:|------------------:|
| slices_only               | 730.3333 |       210.0000 |          70.0000 |  210.0000 |           197.0000 |                3.6633 |                  183.1637 |                         33.4856 |               58.5964 |              60.4704 |                  40.9980 |                           0.5473 |                   67.0000 |                             0.5667 |                  0.6633 |         0.9546 |            -0.2322 |               0.9505 |             1.0043 |           0.2914 |              -0.0000 |                 0.4620 |               0.6309 |        0.2598 |          -12.7997 |              0.4254 |            0.6114 |
| slices_plus_cell_geometry | 730.3333 |       210.0000 |          70.0000 |  210.0000 |           197.0000 |                3.6477 |                  182.3852 |                         32.6033 |               56.2729 |              60.4704 |                  39.7597 |                           0.5423 |                   67.0000 |                             0.5476 |                  0.6870 |         0.9549 |            -0.2305 |               0.9507 |             1.0045 |           0.3778 |              -0.0000 |                 0.5410 |               0.6972 |        0.3513 |          -10.7290 |              0.5124 |            0.6838 |

## Diferencias apareadas (fusión − SLICES; IC95% por composición)

| metric                         | comparison                              |   difference_mean |   ci95_lower |   ci95_upper | unit    |   formulas |   seeds |
|:-------------------------------|:----------------------------------------|------------------:|-------------:|-------------:|:--------|-----------:|--------:|
| absolute_mae_mev_atom          | slices_plus_cell_geometry - slices_only |           -0.7785 |      -1.8885 |       0.0296 | formula |         70 |       3 |
| composition_mean_mae_mev_atom  | slices_plus_cell_geometry - slices_only |            0.0000 |      -0.0000 |       0.0000 | formula |         70 |       3 |
| residual_mae_mev_atom          | slices_plus_cell_geometry - slices_only |           -0.8822 |      -2.3198 |       0.5584 | formula |         70 |       3 |
| pair_mae_mev_atom              | slices_plus_cell_geometry - slices_only |           -2.3235 |      -5.1051 |       0.1687 | formula |         70 |       3 |
| gs_correct                     | slices_plus_cell_geometry - slices_only |           -0.0048 |      -0.0571 |       0.0476 | formula |         70 |       3 |
| gs_argmin_raw_correct          | slices_plus_cell_geometry - slices_only |           -0.0190 |      -0.0714 |       0.0333 | formula |         70 |       3 |
| pairwise_accuracy_gt5          | slices_plus_cell_geometry - slices_only |            0.0237 |      -0.0068 |       0.0541 | formula |         70 |       3 |
| ground_state_eligible_accuracy | slices_plus_cell_geometry - slices_only |           -0.0050 |      -0.0628 |       0.0490 | formula |         70 |       3 |

## Interpretación

Los errores de ΔE y ranking corresponden únicamente a la rama residual MLP8: la salida Ridge común se cancela.
Una mejora en MAE relativo, GS y ranking debería verificarse especialmente para gaps de 5–25 meV/átomo.
Los resultados son exploratorios: estas 70 composiciones se usaron previamente para diseñar el estudio. El test histórico sigue cerrado.
No atribuir a geometría una mejora si el control SLICES-only replicado cambia sustancialmente frente al H3-C histórico; documentar la diferencia.
