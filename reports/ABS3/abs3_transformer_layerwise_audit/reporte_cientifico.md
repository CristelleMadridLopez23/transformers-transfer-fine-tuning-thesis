# Auditoría de representaciones por capa: MatGPT vs MatterGPT

## Diseño y controles

Cohorte de 70 composiciones/210 estructuras; cinco pliegues históricos agrupados por fórmula (45 train, 11 validation, 14 outer). Los extractores originales se mantuvieron congelados. MatGPT usó SLICES y MatterGPT el conditioning `[1.0, 2.0]`. Los targets por par son ΔE = Eⱼ−Eᵢ en meV/átomo. Scalers se ajustaron solo con inner-train; Ridge M2 quedó fijo para reconstrucción absoluta. No se consultó el test independiente.

La capa se eligió por fold y probe usando exclusivamente el mínimo de MAE ΔE de inner-validation; desempates por pairwise accuracy, mayor regularización y menor profundidad. Después se ajustó sobre las 45 fórmulas inner-train y se evaluó una sola vez en outer. FINAL y BEST_LAYER se comparan como desarrollo OOF; no son evaluación externa.

## Compatibilidad de extracción

FINAL MatGPT reproduce la caché histórica con error absoluto máximo 0 y coseno mínimo 1.000000000000. MatterGPT: máximo 0, coseno mínimo 1.000000000000. Ambos cumplen tolerancias; hashes de checkpoints y tokenización constan en manifiestos.

## Capas elegidas por validación interna

| Modelo | Probe | Capas seleccionadas en F0–F4 |
|---|---|---|
| MAT | Ridge | L7_BLOCK_6, FINAL_POST_LN_F, L9_BLOCK_8, L2_BLOCK_1, L11_BLOCK_10 |
| MAT | KRR | L10_BLOCK_9, FINAL_POST_LN_F, L9_BLOCK_8, L2_BLOCK_1, FINAL_POST_LN_F |
| MATTER | Ridge | L5_BLOCK_4, L11_BLOCK_10, L7_BLOCK_6, L1_BLOCK_0, L10_BLOCK_9 |
| MATTER | KRR | L8_BLOCK_7, L3_BLOCK_2, L9_BLOCK_8, L2_BLOCK_1, L2_BLOCK_1 |

## Comparación FINAL vs BEST_LAYER OOF

| Modelo | Probe | Capa | ΔE MAE A | Energía MAE | Pairwise | Ground state | m OLS |
|---|---|---|---:|---:|---:|---:|---:|
| MAT | Ridge | FINAL | 54.38 meV/átomo | 50.26 meV/átomo | 135/197 (68.5%) | 39/67 (58.2%) | 0.344 |
| MAT | Ridge | BEST_LAYER | 59.39 meV/átomo | 51.54 meV/átomo | 131/197 (66.5%) | 32/67 (47.8%) | 0.326 |
| MAT | KRR | FINAL | 53.28 meV/átomo | 49.57 meV/átomo | 138/197 (70.1%) | 37/67 (55.2%) | 0.239 |
| MAT | KRR | BEST_LAYER | 51.06 meV/átomo | 48.91 meV/átomo | 138/197 (70.1%) | 35/67 (52.2%) | 0.326 |
| MATTER | Ridge | FINAL | 54.11 meV/átomo | 50.61 meV/átomo | 134/197 (68.0%) | 33/67 (49.3%) | 0.333 |
| MATTER | Ridge | BEST_LAYER | 59.55 meV/átomo | 52.91 meV/átomo | 125/197 (63.5%) | 33/67 (49.3%) | 0.337 |
| MATTER | KRR | FINAL | 51.69 meV/átomo | 49.53 meV/átomo | 138/197 (70.1%) | 38/67 (56.7%) | 0.260 |
| MATTER | KRR | BEST_LAYER | 53.32 meV/átomo | 49.94 meV/átomo | 131/197 (66.5%) | 33/67 (49.3%) | 0.299 |

ΔE MAE sigue estimando A: primero media de los tres errores absolutos por fórmula y después macro-promedio de 70 fórmulas. Pairwise elegible usa |ΔE_DFT|>5 meV/átomo; predicciones |ΔE|≤1 son no resueltas e incorrectas. Ground state requiere gap DFT >1 meV/átomo y predicción sin empate >1. Full-triplet requiere todos los gaps DFT >1 meV/átomo.

## Bootstrap pareado

| Contraste | Métrica | Diferencia | IC95% |
|---|---|---:|---:|
| MAT BEST_LAYER − FINAL Ridge | deltaE_mae_estimand_A_mev_atom | 5.012 | [1.077, 9.296] |
| MAT BEST_LAYER − FINAL Ridge | pairwise_accuracy | -0.020 | [-0.072, 0.030] |
| MAT BEST_LAYER − FINAL Ridge | groundstate_accuracy | -0.104 | [-0.191, -0.029] |
| MAT BEST_LAYER − FINAL Ridge | full_triplet_accuracy | -0.045 | [-0.143, 0.048] |
| MAT BEST_LAYER − FINAL KRR | deltaE_mae_estimand_A_mev_atom | -2.221 | [-5.476, 0.333] |
| MAT BEST_LAYER − FINAL KRR | pairwise_accuracy | 0.000 | [-0.041, 0.042] |
| MAT BEST_LAYER − FINAL KRR | groundstate_accuracy | -0.030 | [-0.134, 0.074] |
| MAT BEST_LAYER − FINAL KRR | full_triplet_accuracy | 0.030 | [-0.075, 0.134] |
| MATTER BEST_LAYER − FINAL Ridge | deltaE_mae_estimand_A_mev_atom | 5.442 | [1.881, 9.035] |
| MATTER BEST_LAYER − FINAL Ridge | pairwise_accuracy | -0.046 | [-0.103, 0.010] |
| MATTER BEST_LAYER − FINAL Ridge | groundstate_accuracy | 0.000 | [-0.078, 0.087] |
| MATTER BEST_LAYER − FINAL Ridge | full_triplet_accuracy | -0.076 | [-0.182, 0.030] |
| MATTER BEST_LAYER − FINAL KRR | deltaE_mae_estimand_A_mev_atom | 1.623 | [-2.296, 5.461] |
| MATTER BEST_LAYER − FINAL KRR | pairwise_accuracy | -0.036 | [-0.105, 0.032] |
| MATTER BEST_LAYER − FINAL KRR | groundstate_accuracy | -0.075 | [-0.185, 0.031] |
| MATTER BEST_LAYER − FINAL KRR | full_triplet_accuracy | -0.136 | [-0.258, -0.015] |
| MAT GEO_FINAL − GEO GEO_KRR | energy_mae_mev_atom | -1.025 | [-3.175, 0.921] |
| MAT GEO_FINAL − GEO GEO_KRR | deltaE_mae_estimand_A_mev_atom | -3.508 | [-7.913, 0.901] |
| MAT GEO_FINAL − GEO GEO_KRR | pairwise_accuracy | -0.005 | [-0.067, 0.057] |
| MAT GEO_FINAL − GEO GEO_KRR | groundstate_accuracy | -0.060 | [-0.162, 0.043] |
| MAT GEO_FINAL − GEO GEO_KRR | full_triplet_accuracy | 0.000 | [-0.123, 0.125] |
| MAT GEO_BEST − GEO GEO_KRR | energy_mae_mev_atom | -1.282 | [-3.638, 0.989] |
| MAT GEO_BEST − GEO GEO_KRR | deltaE_mae_estimand_A_mev_atom | -2.007 | [-7.750, 3.615] |
| MAT GEO_BEST − GEO GEO_KRR | pairwise_accuracy | -0.015 | [-0.082, 0.054] |
| MAT GEO_BEST − GEO GEO_KRR | groundstate_accuracy | -0.075 | [-0.152, 0.000] |
| MAT GEO_BEST − GEO GEO_KRR | full_triplet_accuracy | -0.015 | [-0.123, 0.092] |
| MATTER GEO_FINAL − GEO GEO_KRR | energy_mae_mev_atom | -1.778 | [-4.203, 0.542] |
| MATTER GEO_FINAL − GEO GEO_KRR | deltaE_mae_estimand_A_mev_atom | -4.570 | [-9.745, 0.666] |
| MATTER GEO_FINAL − GEO GEO_KRR | pairwise_accuracy | 0.030 | [-0.041, 0.104] |
| MATTER GEO_FINAL − GEO GEO_KRR | groundstate_accuracy | 0.000 | [-0.106, 0.106] |
| MATTER GEO_FINAL − GEO GEO_KRR | full_triplet_accuracy | 0.076 | [-0.045, 0.197] |
| MATTER GEO_BEST − GEO GEO_KRR | energy_mae_mev_atom | -0.471 | [-2.845, 1.684] |
| MATTER GEO_BEST − GEO GEO_KRR | deltaE_mae_estimand_A_mev_atom | -4.504 | [-9.301, 0.150] |
| MATTER GEO_BEST − GEO GEO_KRR | pairwise_accuracy | -0.020 | [-0.092, 0.051] |
| MATTER GEO_BEST − GEO GEO_KRR | groundstate_accuracy | -0.030 | [-0.121, 0.061] |
| MATTER GEO_BEST − GEO GEO_KRR | full_triplet_accuracy | 0.030 | [-0.078, 0.141] |
| MAT BEST_LAYER − FINAL Ridge | energy_mae_mev_atom | 1.281 | [0.188, 2.397] |
| MAT BEST_LAYER − FINAL KRR | energy_mae_mev_atom | -0.657 | [-1.846, 0.327] |
| MATTER BEST_LAYER − FINAL Ridge | energy_mae_mev_atom | 2.307 | [0.778, 3.903] |
| MATTER BEST_LAYER − FINAL KRR | energy_mae_mev_atom | 0.416 | [-1.002, 1.969] |

Bootstrap por fórmula: 10 000 réplicas, semilla 20261103. Intervalos descriptivos de CV de desarrollo; no tratar pares dentro de fórmula como independientes.

## Kernel geométrico y complemento por capa

| Modelo | Condición | Capa | η | ΔE MAE OOF |
|---|---|---|---:|---:|
| GEO | GEO_KRR | GEOMETRY | 0 | 51.59 meV/átomo |
| MAT | GEO_BEST | L2_BLOCK_1;FINAL_POST_LN_F;L8_BLOCK_7 | 4.24 | 49.58 meV/átomo |
| MAT | GEO_FINAL | FINAL_POST_LN_F | 2.8 | 48.08 meV/átomo |
| MATTER | GEO_BEST | L4_BLOCK_3;FINAL_POST_LN_F;L9_BLOCK_8;L2_BLOCK_1;L10_BLOCK_9 | 4.42 | 47.08 meV/átomo |
| MATTER | GEO_FINAL | FINAL_POST_LN_F | 2.26 | 47.02 meV/átomo |

Los kernels aditivos se seleccionaron dentro de cada fold. Una mejora puntual no implica valor incremental estadísticamente concluyente; contrastes apareados están en bootstrap_paired.csv.

## Interpretación y límites

Las curvas layer-wise describen el comportamiento de probes con normalización interna. La capa con menor validación puede variar entre folds; solo el OOF de BEST_LAYER estima el procedimiento de selección. Una pendiente más cercana a 1 no sustituye MAE/RMSE ni ranking. La geometría procede de celdas finales relajadas por DFT. No se puede atribuir una diferencia MatGPT–MatterGPT exclusivamente al backbone.

El máximo pairwise OOF entre capas evaluadas fue 72.1%; meta aspiracional 90% no alcanzada. Permutaciones completadas: 100 repeticiones por combinación de modelo/probe/fold/capa.

## Artefactos y reproducibilidad

Cachés layerwise: embeddings_matgpt_layers.npz y embeddings_mattergpt_layers.npz. Manifiestos: architecture_manifest.json, checkpoint_manifest.json, layer_extraction_manifest.json, token_alignment_audit.json y embedding_cache_manifest.json. Predicciones y métricas se guardan por estructura, par, fórmula, fold, capa y probe. Recursos y duración están en resources.json. Test histórico cerrado.
