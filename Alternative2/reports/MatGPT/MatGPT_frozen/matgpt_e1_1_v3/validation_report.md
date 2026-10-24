# Diagnóstico de validación E1.1/E1.2

La selección usa exclusivamente validation. Test no se consulta.
La frontera no asigna una puntuación global: conserva compromisos entre MAE absoluta, MAE relativa, exactitud de estado fundamental, ordenamiento de pares y error de rango.
Se marca como colapso un rango predicho medio menor que 1 meV/átomo.

| configuration | head_type | pooling | loss_type | pairwise_lambda | energy_mae_ev_atom_mean | relative_energy_mae_mev_atom_mean | ground_state_accuracy_mean | pairwise_ordering_accuracy_mean | mean_predicted_range_mev_atom | mean_absolute_range_error_mev_atom_mean | pareto_nondominated | predicted_range_collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_1_last_token_mlp32_lambda0 | mlp32 | last_token | absolute_huber | 0.0 | 0.3527769325889858 | 240.79724736055834 | 0.4166666666666667 | 0.6111111111111112 | 575.4216313362122 | 489.474907898712 | True | False |
| e1_1_last_token_ridge_a1 | ridge | last_token | squared_error |  | 0.3852586736313527 | 327.38862224917085 | 0.625 | 0.6666666666666666 | 726.4741063117981 | 640.5273828742978 | True | False |
| e1_1_last_token_ridge_a10 | ridge | last_token | squared_error |  | 0.4162814575652516 | 290.07077926581314 | 0.625 | 0.5416666666666666 | 680.5400550365448 | 594.5933315990445 | True | False |
| e1_1_last_token_ridge_a100 | ridge | last_token | squared_error |  | 0.4823936138182931 | 247.27283842358895 | 0.625 | 0.625 | 569.9677169322968 | 484.02099349479647 | True | False |
| e1_1_last_token_mlp8_lambda1 | mlp8 | last_token | absolute_plus_pairwise_huber | 1.0 | 0.5568498206864217 | 26.73067211844206 | 0.5416666666666666 | 0.75 | 64.3039345741272 | 43.65818997691844 | True | False |
| e1_1_last_token_mlp8_lambda0.3 | mlp8 | last_token | absolute_plus_pairwise_huber | 0.3 | 0.5634023794681701 | 28.816806205128945 | 0.4166666666666667 | 0.5972222222222222 | 68.28620036443074 | 41.85510901323443 | True | False |
| e1_1_last_token_ridge_a0.1 | ridge | last_token | squared_error |  | 0.3986113955609029 | 335.903511766632 | 0.5 | 0.625 | 732.9751253128052 | 647.0284018753049 | False | False |
| e1_1_last_token_mlp8_lambda0 | mlp8 | last_token | absolute_huber | 0.0 | 0.5144004881432597 | 256.7801641579081 | 0.4583333333333333 | 0.5555555555555556 | 574.1781791051229 | 488.2314556676226 | False | False |
| e1_1_last_token_constant_train_mean | constant | last_token | constant_train_mean | 0.0 | 0.5671082530571241 | 35.68550722916677 |  | 0.0 | 0.0 | 85.94672343750025 | False | True |
| e1_1_last_token_mlp8_lambda0.1 | mlp8 | last_token | absolute_plus_pairwise_huber | 0.1 | 0.5945308418496432 | 36.237943433186764 | 0.4166666666666667 | 0.5972222222222222 | 79.19146617253621 | 45.63291383940632 | False | False |
| e1_1_last_token_mlp32_lambda0.3 | mlp32 | last_token | absolute_plus_pairwise_huber | 0.3 | 0.6102602232259793 | 33.173503099510974 | 0.3452380952380952 | 0.5833333333333334 | 72.12064663569133 | 47.23419295931995 | False | False |
| e1_1_last_token_mlp32_lambda1 | mlp32 | last_token | absolute_plus_pairwise_huber | 1.0 | 0.6167506789593845 | 30.419653600335007 | 0.42857142857142855 | 0.7083333333333334 | 71.3195006052653 | 49.55479538108318 | False | False |
| e1_1_last_token_mlp32_lambda0.1 | mlp32 | last_token | absolute_plus_pairwise_huber | 0.1 | 0.6245637282488065 | 43.15599707692798 | 0.21428571428571427 | 0.5 | 89.16570742925008 | 50.821799629872565 | False | False |

No se declara ganador automático. Revise también dispersión entre semillas, MAE por familia, curvas de entrenamiento/validación, rangos por composición y predicciones colapsadas.
El conjunto test histórico ya fue inspeccionado durante E1; utilícelo solo para evaluación final descriptiva una vez bloqueadas las configuraciones.
