# F4 directo en ABS₃: MatGPT y MatterGPT

## Cohorte y método

Se evaluaron 210 estructuras en 70 composiciones ABS₃ con tres polimorfos por fórmula. Se usó el dataset `data/abs3/abs3_development_f4_geometry/dataset.csv`, folds agrupados históricos, cinco pliegues y semillas 42/123/2026. El test histórico no se leyó.

Los backbones originales se congelaron en F0 y se descongelaron solo los bloques 8–11 en F4. Se compartió un warmup de MLP8 por pareja; la selección de checkpoints usó únicamente inner-validation. La geometría se escaló por fold con inner-train, y el Ridge M2 permaneció fijo.

Tres SLICES tienen marca histórica de no equivalencia bajo la validación de grafo estricta; se conservaron exactamente las entradas con las que los extractores congelados habían sido auditados. Tokenización: cero OOV y cero truncamiento.

## Resultados OOF

{
  "primary_estimand": "A; three signed intracomposition pairs; average within composition, then seeds within composition, then 70-formula macro-average",
  "MatGPT_F0": {
    "n_formulas": 70,
    "mae_delta_signed_est_a_mev_atom": 57.11100689131589,
    "mae_energy_ev_atom": 0.05089240680155941,
    "mae_residual_mev_atom": 50.89240680155941,
    "rmse_energy_global_ev_atom": 0.06698111732469549,
    "ground_state_accuracy": 0.4925373134328358,
    "ground_state_cases": 201,
    "pairwise_accuracy_eligible_including_unresolved": 0.6565143824027073,
    "pairwise_resolved_accuracy": 0.6643835616438356,
    "unresolved_fraction_eligible_pairs": 0.011844331641285956,
    "eligible_pairs": 591,
    "resolved_pairs": 584,
    "spearman_mean_by_formula": 0.361904761904762,
    "kendall_mean_by_formula": 0.31428571428571433,
    "ols_delta_slope": 0.37130090072759014,
    "ols_delta_intercept_mev_atom": -9.576211063791357,
    "ols_delta_r2": 0.2747623860130495
  },
  "MatGPT_F4": {
    "n_formulas": 70,
    "mae_delta_signed_est_a_mev_atom": 56.462812023986544,
    "mae_energy_ev_atom": 0.05074205268599483,
    "mae_residual_mev_atom": 50.742052685994835,
    "rmse_energy_global_ev_atom": 0.06686007726022042,
    "ground_state_accuracy": 0.5074626865671641,
    "ground_state_cases": 201,
    "pairwise_accuracy_eligible_including_unresolved": 0.6615905245346869,
    "pairwise_resolved_accuracy": 0.6706689536878216,
    "unresolved_fraction_eligible_pairs": 0.01353637901861252,
    "eligible_pairs": 591,
    "resolved_pairs": 583,
    "spearman_mean_by_formula": 0.38095238095238093,
    "kendall_mean_by_formula": 0.3301587301587302,
    "ols_delta_slope": 0.3713696881479581,
    "ols_delta_intercept_mev_atom": -10.459788766769808,
    "ols_delta_r2": 0.2781890973006833
  },
  "MatterGPT_F0": {
    "n_formulas": 70,
    "mae_delta_signed_est_a_mev_atom": 54.70082656037382,
    "mae_energy_ev_atom": 0.04933520515655657,
    "mae_residual_mev_atom": 49.33520515655657,
    "rmse_energy_global_ev_atom": 0.06840778541864204,
    "ground_state_accuracy": 0.5970149253731344,
    "ground_state_cases": 201,
    "pairwise_accuracy_eligible_including_unresolved": 0.6446700507614214,
    "pairwise_resolved_accuracy": 0.6767317939609236,
    "unresolved_fraction_eligible_pairs": 0.047377326565143825,
    "eligible_pairs": 591,
    "resolved_pairs": 563,
    "spearman_mean_by_formula": 0.3523809523809524,
    "kendall_mean_by_formula": 0.3238095238095239,
    "ols_delta_slope": 0.1540372553639157,
    "ols_delta_intercept_mev_atom": -12.549032389440274,
    "ols_delta_r2": 0.1635539305599445
  },
  "MatterGPT_F4": {
    "n_formulas": 70,
    "mae_delta_signed_est_a_mev_atom": 53.55304821270563,
    "mae_energy_ev_atom": 0.04920956258146699,
    "mae_residual_mev_atom": 49.20956258146699,
    "rmse_energy_global_ev_atom": 0.06786665638571444,
    "ground_state_accuracy": 0.6318407960199006,
    "ground_state_cases": 201,
    "pairwise_accuracy_eligible_including_unresolved": 0.6734348561759729,
    "pairwise_resolved_accuracy": 0.708185053380783,
    "unresolved_fraction_eligible_pairs": 0.049069373942470386,
    "eligible_pairs": 591,
    "resolved_pairs": 562,
    "spearman_mean_by_formula": 0.42142857142857143,
    "kendall_mean_by_formula": 0.39047619047619053,
    "ols_delta_slope": 0.18229004353912354,
    "ols_delta_intercept_mev_atom": -13.346692532362056,
    "ols_delta_r2": 0.1925727042701143
  }
}

## Bootstrap apareado por composición

| contrast                    |   estimate_mev_atom |   ci95_low |   ci95_high |   bootstrap_replicates |   compositions |     seed |
|:----------------------------|--------------------:|-----------:|------------:|-----------------------:|---------------:|---------:|
| MatGPT_F4 − MatGPT_F0       |           -0.648195 |   -1.202   |  -0.0941812 |                  10000 |             70 | 20261103 |
| MatterGPT_F4 − MatterGPT_F0 |           -1.14778  |   -2.41376 |   0.0921379 |                  10000 |             70 | 20261103 |
| MatterGPT_F4 − MatGPT_F4    |           -2.90976  |   -7.87428 |   2.24761   |                  10000 |             70 | 20261103 |

## Limitaciones

La evaluación OOF utiliza el conjunto de desarrollo ya explorado y no constituye una prueba ciega. MatGPT y MatterGPT difieren en arquitectura y tokenizador, por lo que la comparación entre familias no aísla causalmente el preentrenamiento.
