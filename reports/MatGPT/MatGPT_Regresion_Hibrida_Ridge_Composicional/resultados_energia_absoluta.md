# Resultados de energía absoluta

| configuración                     |   mae_energy_per_atom_mev_atom |   mae_composition_mean_mev_atom |   mae_residual_intraformula_mev_atom |   energy_slope_ols |   energy_dispersion_ratio |
|:----------------------------------|-------------------------------:|--------------------------------:|-------------------------------------:|-------------------:|--------------------------:|
| Ridge de embeddings + MLP8 Frozen |                        183.296 |                         180.177 |                               33.754 |              0.955 |                     1.004 |
| Ridge composicional + MLP8 Frozen |                         52.019 |                          40.830 |                               33.754 |              0.994 |                     0.998 |
| Ridge de embeddings + MLP8 LoRA   |                        182.776 |                         180.177 |                               33.539 |              0.955 |                     1.004 |
| Ridge composicional + MLP8 LoRA   |                         52.071 |                          40.830 |                               33.539 |              0.994 |                     0.998 |

**Interpretación:** el cambio de rama Ridge solo modifica la media por fórmula; la contribución residual procede de predicciones MLP8 existentes. Consulte los IC95% apareados en `paired_bootstrap_by_composition.csv`.
