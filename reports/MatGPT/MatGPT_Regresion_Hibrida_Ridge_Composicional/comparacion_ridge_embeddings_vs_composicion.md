# Comparación Ridge de embeddings frente a Ridge composicional

**Diseño:** 70 composiciones, 210 estructuras físicas, tres polimorfos, cinco pliegues y tres semillas residuales. Las métricas de semilla se promedian para el resumen; el bootstrap promedia primero semillas por composición y remuestrea 70 fórmulas. El test histórico permanece cerrado.

## Resultados medios de energía absoluta y polimorfos

| configuración                     |   mae_energy_per_atom_mev_atom |   mae_composition_mean_mev_atom |   mae_residual_intraformula_mev_atom |   mae_delta_signed_pairs_mev_atom |   ground_state_argmin_accuracy |   pairwise_accuracy_gt5_predicted_gap_gt1 |
|:----------------------------------|-------------------------------:|--------------------------------:|-------------------------------------:|----------------------------------:|-------------------------------:|------------------------------------------:|
| Ridge de embeddings + MLP8 Frozen |                        183.296 |                         180.177 |                               33.754 |                            58.844 |                          0.543 |                                     0.668 |
| Ridge composicional + MLP8 Frozen |                         52.019 |                          40.830 |                               33.754 |                            58.844 |                          0.543 |                                     0.668 |
| Ridge de embeddings + MLP8 LoRA   |                        182.776 |                         180.177 |                               33.539 |                            58.278 |                          0.529 |                                     0.658 |
| Ridge composicional + MLP8 LoRA   |                         52.071 |                          40.830 |                               33.539 |                            58.278 |                          0.529 |                                     0.658 |

## Diferencia apareada del Ridge composicional menos el Ridge de embeddings

- **Residuo Frozen:** MAE absoluta cambia -131.28 meV/átomo (IC95% -174.65, -94.52); MAE de media composicional cambia -139.35 meV/átomo (IC95% -183.89, -101.55).
- **Residuo LoRA:** MAE absoluta cambia -130.70 meV/átomo (IC95% -175.65, -93.84); MAE de media composicional cambia -139.35 meV/átomo (IC95% -183.07, -101.27).

| comparison                                                       | metric                                   | units     |   difference_new_minus_old |   ci95_lower |   ci95_upper |   n_compositions |   seeds_averaged_before_resampling |   bootstrap_repetitions |   bootstrap_seed |
|:-----------------------------------------------------------------|:-----------------------------------------|:----------|---------------------------:|-------------:|-------------:|-----------------:|-----------------------------------:|------------------------:|-----------------:|
| Ridge composicional frente a Ridge de embeddings; residuo Frozen | mae_absolute_ev_atom_mev                 | meV/átomo |                  -131.2763 |    -174.6495 |     -94.5238 |               70 |                                  3 |                   10000 |         20261030 |
| Ridge composicional frente a Ridge de embeddings; residuo LoRA   | mae_absolute_ev_atom_mev                 | meV/átomo |                  -130.7044 |    -175.6518 |     -93.8403 |               70 |                                  3 |                   10000 |         20261030 |
| Ridge composicional frente a Ridge de embeddings; residuo Frozen | composition_mean_absolute_error_mev_atom | meV/átomo |                  -139.3473 |    -183.8887 |    -101.5541 |               70 |                                  3 |                   10000 |         20261030 |
| Ridge composicional frente a Ridge de embeddings; residuo LoRA   | composition_mean_absolute_error_mev_atom | meV/átomo |                  -139.3473 |    -183.0677 |    -101.2720 |               70 |                                  3 |                   10000 |         20261030 |

Los intervalos se agrupan por composición; no se cuentan las semillas como materiales independientes. Una diferencia en energía absoluta no implica una mejora en ranking.

## Error de media y residuo

| model_id   |   mean_mse_offset |   mean_mse_residual |   mean_cross |   mean_mse_total |   max_closure |
|:-----------|------------------:|--------------------:|-------------:|-----------------:|--------------:|
| M1         |        0.0649328  |          0.00192499 | -3.90154e-12 |       0.0668578  |   4.14884e-16 |
| M2         |        0.00262362 |          0.00192499 |  6.94304e-11 |       0.00454861 |   1.01416e-16 |
| M3         |        0.0649328  |          0.00196061 | -1.09371e-10 |       0.0668934  |   4.92136e-16 |
| M4         |        0.00262362 |          0.00196061 |  2.19372e-11 |       0.00458423 |   1.06381e-16 |

La identidad exacta es MSE(total) = MSE(media) + MSE(residuo) + término cruzado. Las MAE de los componentes son descriptivas y **no son aditivas**.

## Control solo composicional

MAE E/N 52.03 meV/átomo; MAE de energía total 1.041 eV; MAE de pares firmados 60.47 meV/átomo. El predictor asigna la misma energía a los tres polimorfos, por lo que no se le adjudica estado fundamental.

## Selección y ajuste de Ridge

|   fold |   selected_alpha |   inner_validation_mae_ev_atom |   ridge_features |   inner_train_compositions |   inner_validation_compositions |   outer_train_compositions |   outer_test_compositions | scaler_fit_role                                           | solver   |   tolerance | intercept   | pipeline_path                                                                                                     | historical_control_alpha_match   |   max_abs_difference_existing_composition_control_ev_atom |
|-------:|-----------------:|-------------------------------:|-----------------:|---------------------------:|--------------------------------:|---------------------------:|--------------------------:|:----------------------------------------------------------|:---------|------------:|:------------|:------------------------------------------------------------------------------------------------------------------|:---------------------------------|----------------------------------------------------------:|
|      0 |             0.01 |                       0.052172 |               17 |                         45 |                              11 |                         56 |                        14 | inner_train_for_selection; outer_train_for_saved_pipeline | lsqr     |       1e-08 | True        | Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_0.joblib | True                             |                                                5.8343e-07 |
|      1 |             0.1  |                       0.045533 |               17 |                         45 |                              11 |                         56 |                        14 | inner_train_for_selection; outer_train_for_saved_pipeline | lsqr     |       1e-08 | True        | Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_1.joblib | True                             |                                                5.8343e-07 |
|      2 |             0.1  |                       0.040517 |               17 |                         45 |                              11 |                         56 |                        14 | inner_train_for_selection; outer_train_for_saved_pipeline | lsqr     |       1e-08 | True        | Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_2.joblib | True                             |                                                5.8343e-07 |
|      3 |             0.1  |                       0.043902 |               17 |                         45 |                              11 |                         56 |                        14 | inner_train_for_selection; outer_train_for_saved_pipeline | lsqr     |       1e-08 | True        | Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_3.joblib | True                             |                                                5.8343e-07 |
|      4 |             0.1  |                       0.036105 |               17 |                         45 |                              11 |                         56 |                        14 | inner_train_for_selection; outer_train_for_saved_pipeline | lsqr     |       1e-08 | True        | Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_4.joblib | True                             |                                                5.8343e-07 |

El único ajuste nuevo fue Ridge de 17 fracciones estequiométricas, con selección interna 45/11 y refit outer-train de 56 fórmulas. La rama residual y el predictor Frozen/LoRA se reutilizaron de predicciones ya guardadas. No hubo inferencia nueva de MatGPT ni entrenamiento MLP8/LoRA.

En esta tabla, **Frozen** identifica residuos MLP8 del MatGPT pretrained congelado; **LoRA** identifica residuos ya guardados del modelo adaptado mediante Low-Rank Adaptation. Ambas ramas usan la cabeza MLP8 existente; no se reajustaron para esta comparación.

**Límites:** resultados de las 70 composiciones ABS₃ de desarrollo previamente analizadas, no validación ciega externa. Las conclusiones se limitan a la descomposición del nivel medio y los residuos recuperados.
