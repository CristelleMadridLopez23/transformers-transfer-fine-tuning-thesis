# E2-Aug — comparación OOF por composición

Estado: PASS para la agregación de predicciones disponibles. El test histórico no se consultó. Las ocho composiciones de validation participaron en preselección histórica; estos IC no eliminan ese sesgo.

| Configuración | MAE absoluta macro (meV/átomo), media ± DE fold | MAE ΔE (meV/átomo), media ± DE fold | GS, media ± DE fold | Pairwise macro, media ± DE fold |
|---|---:|---:|---:|---:|
| E2-Aug1 | 726.441 ± 79.823 | 38.748 ± 7.141 | 0.499 ± 0.135 | 0.648 ± 0.063 |
| E2-A-partial_ft | 728.948 ± 85.807 | 37.252 ± 6.953 | 0.495 ± 0.127 | 0.669 ± 0.043 |

Bootstrap pareado: 2,000 remuestras de composiciones completas, semilla 13072026; IC porcentiles exploratorios. Consulte `paired_bootstrap_ci.json`, `metrics_per_fold_seed.csv`, `metrics_composition_fold_seed.csv` y las figuras PNG.

MAE ΔE se calcula dentro de composición, GS excluye gaps DFT ≤1 meV/átomo y considera predicciones empatadas a ≤1 como no discriminativas; pairwise omite pares DFT con diferencia ≤1 meV/átomo.
