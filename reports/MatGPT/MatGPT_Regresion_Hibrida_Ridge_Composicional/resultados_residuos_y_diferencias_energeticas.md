# Resultados de residuos y diferencias energéticas

| configuración                     |   mae_residual_intraformula_mev_atom |   mae_delta_signed_pairs_mev_atom |   mae_delta_anchored_to_dft_ground_state_mev_atom |   ground_state_argmin_accuracy |   pairwise_accuracy_gt5_predicted_gap_gt1 |
|:----------------------------------|-------------------------------------:|----------------------------------:|--------------------------------------------------:|-------------------------------:|------------------------------------------:|
| Ridge de embeddings + MLP8 Frozen |                               33.754 |                            58.844 |                                            41.526 |                          0.543 |                                     0.668 |
| Ridge composicional + MLP8 Frozen |                               33.754 |                            58.844 |                                            41.526 |                          0.543 |                                     0.668 |
| Ridge de embeddings + MLP8 LoRA   |                               33.539 |                            58.278 |                                            40.387 |                          0.529 |                                     0.658 |
| Ridge composicional + MLP8 LoRA   |                               33.539 |                            58.278 |                                            40.387 |                          0.529 |                                     0.658 |

ΔE firmada = 1000·(e_B − e_A) dentro de una fórmula. El anclaje usa la misma estructura de mínimo DFT. La elegibilidad pairwise requiere |ΔE_DFT| > 5 meV/átomo; una predicción con gap ≤1 meV/átomo se cuenta como no resuelta.
