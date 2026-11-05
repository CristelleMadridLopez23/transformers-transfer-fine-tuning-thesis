# ABS₃: geometría y Transformers con Ridge pairwise

## Cohorte y protocolo

Dataset SHA-256: `afdec9430cde046d5a850cc95432cf90759194f882880496388dc3a65c27eb8e`. Se evaluaron 70 composiciones (210 estructuras, tres polimorfos), cinco folds agrupados y un OOF determinista por composición. Cada fold tuvo 45 fórmulas de inner-train, 11 de inner-validation y 14 outer. El test histórico no se abrió.

Ridge M2 histórico, con 17 fracciones estequiométricas, permaneció fijo; sus predicciones outer verificadas anclan E_pred. El Ridge pairwise no tiene intercepto y aprende ΔE firmada sobre z_j−z_i. α se seleccionó por ΔE MAE en validación, con exactitud pairwise como desempate; la normalización final se ajustó con inner-train más inner-validation, sin outer.

Los dos extractores permanecieron congelados. Se reutilizaron embeddings FINAL_END existentes y se incorporaron los siete descriptores de `gga-static_endpoint_cleaned` después de la extracción. Las celdas ya están relajadas por DFT, por lo que estas condiciones híbridas reciben geometría posterior a relajación.

## Resultados principales

| Condición   |   Energy MAE (meV/átomo) |   ΔE MAE (meV/átomo) |   Pairwise (%) |   Triplete exacto (%) |   Ground state (%) |
|:------------|-------------------------:|---------------------:|---------------:|----------------------:|-------------------:|
| GEO         |                  47.5488 |              49.7817 |        69.0355 |               28.7879 |            62.6866 |
| MAT         |                  50.2807 |              54.8029 |        66.4975 |               27.2727 |            50.7463 |
| MAT-G       |                  49.7662 |              53.4602 |        68.0203 |               30.303  |            52.2388 |
| MATTER      |                  51.8196 |              55.7744 |        68.5279 |               37.8788 |            55.2239 |
| MATTER-G    |                  51.4892 |              54.7013 |        69.0355 |               36.3636 |            55.2239 |

### Objetivo de ranking del 90%

El 90% es una meta descriptiva, no una regla de validez ni de selección. Pairwise se calcula en pares con |ΔE_DFT|>5 meV/átomo; predicciones con |ΔE_pred|≤1 meV/átomo se consideran no resueltas e incorrectas.

## Contrastes apareados

| contrast          | metric                   |   estimate | unit                |   ci95_low |   ci95_high |   replicates | resampling_unit   |     seed |
|:------------------|:-------------------------|-----------:|:--------------------|-----------:|------------:|-------------:|:------------------|---------:|
| MAT-G − GEO       | energy_mae_mev_atom      |   2.2174   | meV/átomo           |  -0.441754 |   4.96007   |        10000 | composition_id    | 20261103 |
| MAT-G − GEO       | deltaE_mae_mev_atom      |   3.67849  | meV/átomo           |  -2.78253  |  10.3119    |        10000 | composition_id    | 20261103 |
| MAT-G − GEO       | pairwise_accuracy_pct    |   1.01523  | puntos porcentuales |  -7.8125   |  10.2054    |        10000 | composition_id    | 20261103 |
| MAT-G − GEO       | groundstate_accuracy_pct | -10.4478   | puntos porcentuales | -25        |   3.30386   |        10000 | composition_id    | 20261103 |
| MATTER-G − GEO    | energy_mae_mev_atom      |   3.94041  | meV/átomo           |   1.21869  |   6.77536   |        10000 | composition_id    | 20261103 |
| MATTER-G − GEO    | deltaE_mae_mev_atom      |   4.91963  | meV/átomo           |  -2.02774  |  11.8487    |        10000 | composition_id    | 20261103 |
| MATTER-G − GEO    | pairwise_accuracy_pct    |   1.52284  | puntos porcentuales |  -7.92118  |  10.9453    |        10000 | composition_id    | 20261103 |
| MATTER-G − GEO    | groundstate_accuracy_pct |  -7.46269  | puntos porcentuales | -21.2121   |   7.24638   |        10000 | composition_id    | 20261103 |
| MAT-G − MAT       | energy_mae_mev_atom      |  -0.51447  | meV/átomo           |  -0.829373 |  -0.226558  |        10000 | composition_id    | 20261103 |
| MAT-G − MAT       | deltaE_mae_mev_atom      |  -1.34272  | meV/átomo           |  -1.99545  |  -0.697235  |        10000 | composition_id    | 20261103 |
| MAT-G − MAT       | pairwise_accuracy_pct    |   1.52284  | puntos porcentuales |   0        |   3.46535   |        10000 | composition_id    | 20261103 |
| MAT-G − MAT       | groundstate_accuracy_pct |   1.49254  | puntos porcentuales |   0        |   4.61538   |        10000 | composition_id    | 20261103 |
| MATTER-G − MATTER | energy_mae_mev_atom      |  -0.330373 | meV/átomo           |  -0.602345 |  -0.0705676 |        10000 | composition_id    | 20261103 |
| MATTER-G − MATTER | deltaE_mae_mev_atom      |  -1.0731   | meV/átomo           |  -1.7068   |  -0.452561  |        10000 | composition_id    | 20261103 |
| MATTER-G − MATTER | pairwise_accuracy_pct    |   0        | puntos porcentuales |  -2.04082  |   2.0202    |        10000 | composition_id    | 20261103 |
| MATTER-G − MATTER | groundstate_accuracy_pct |   0        | puntos porcentuales |  -4.41176  |   4.41341   |        10000 | composition_id    | 20261103 |
| MATTER-G − MAT    | energy_mae_mev_atom      |   1.20855  | meV/átomo           |  -0.782243 |   3.2547    |        10000 | composition_id    | 20261103 |
| MATTER-G − MAT    | deltaE_mae_mev_atom      |  -0.101581 | meV/átomo           |  -5.36049  |   5.31797   |        10000 | composition_id    | 20261103 |
| MATTER-G − MAT    | pairwise_accuracy_pct    |   2.03046  | puntos porcentuales |  -4.63978  |   8.46647   |        10000 | composition_id    | 20261103 |
| MATTER-G − MAT    | groundstate_accuracy_pct |   4.47761  | puntos porcentuales |  -5.97015  |  14.9254    |        10000 | composition_id    | 20261103 |

## Controles MLP8 históricos

{
  "MatGPT_original_geometry_M2_MLP8": {
    "energy_mae_mev_atom": 50.9711194894602,
    "deltaE_mae_signed_mev_atom": 55.59450589784513,
    "pairwise_accuracy": 0.6937394247038917,
    "source": "experiments/matgpt_f0_stoichiometric_ridge_m2_comparison/metrics.json"
  },
  "MatterGPT_original_geometry_M2_MLP8": {
    "energy_mae_mev_atom": 49.64446310912345,
    "deltaE_mae_signed_mev_atom": 53.398524303467795,
    "pairwise_accuracy": 0.6565143824027073,
    "source": "experiments/mattergpt_abs3_geometry_late_fusion/metrics.json"
  }
}

Los resultados históricos se reportan aparte: sus protocolos/cabezas no son el Ridge pairwise nuevo. Solo se comparan como referencias cuando comparten Ridge M2, targets, cohortes y estimando documentados.

## Limitaciones

La evaluación es OOF sobre el conjunto de desarrollo y no sustituye el test independiente cerrado. Ridge es lineal sobre diferencias y, algebraicamente, equivale a una puntuación lineal por estructura. Un ranking alto no garantiza calibración de magnitudes. La geometría proviene de estructuras ya relajadas por DFT. No se utilizaron semillas duplicadas para el Ridge determinista.
