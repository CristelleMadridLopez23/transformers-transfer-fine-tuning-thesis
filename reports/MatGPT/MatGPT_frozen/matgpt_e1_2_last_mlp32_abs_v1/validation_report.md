# Diagnóstico de validación E1.1/E1.2

La selección usa exclusivamente validation. Test no se consulta.
La frontera no asigna una puntuación global: conserva compromisos entre MAE absoluta, MAE relativa, exactitud de estado fundamental, ordenamiento de pares y error de rango.
Se marca como colapso un rango predicho medio menor que 1 meV/átomo.

| configuration | head_type | pooling | loss_type | pairwise_lambda | energy_mae_ev_atom_mean | relative_energy_mae_mev_atom_mean | ground_state_accuracy_mean | pairwise_ordering_accuracy_mean | mean_predicted_range_mev_atom | mean_absolute_range_error_mev_atom_mean | pareto_nondominated | predicted_range_collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_2_last_token_mlp32_lambda0 | mlp32 | last_token | absolute_huber | 0.0 | 0.3527769325889858 | 240.79724736055834 | 0.4166666666666667 | 0.6111111111111112 | 575.4216313362122 | 489.474907898712 | True | False |

No se declara ganador automático. Revise también dispersión entre semillas, MAE por familia, curvas de entrenamiento/validación, rangos por composición y predicciones colapsadas.
El conjunto test histórico ya fue inspeccionado durante E1; utilícelo solo para evaluación final descriptiva una vez bloqueadas las configuraciones.
