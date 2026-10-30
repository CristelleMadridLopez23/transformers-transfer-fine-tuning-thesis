# D2 — ¿ayuda entrenar la energía relativa por triplete?

Estado: **PILOT**. El smoke fue una comprobación técnica, no evidencia. Se compara con E2-A partial_ft en los mismos folds y semillas; test histórico excluido.

`D2-CENTER` conserva la pérdida absoluta normalizada y añade SmoothL1 de energías centradas por composición, sustituyendo pairwise. `D2-CENTER-ONLY` usa solo la pérdida centrada. Escalas de energía y centrado proceden solo de inner-train; early stopping usa inner-validation. Mantiene MatGPT unconditional, último bloque ajustable, pooling `<end>`, MLP 768→8→1, AdamW, batch de cuatro composiciones, clipping 1.0 y paciencia 20.

La configuración YAML histórica dice 200 épocas, pero el código E2-A ejecutado fija un máximo real de 100 (smoke de 3); D2 sigue el código ejecutado, con 100 máximas. CENTER-ONLY no aprende offsets absolutos entre composiciones; su salida absoluta usa como offset la media inner-train para poder computar MAE, y no se debe interpretar como una predicción absoluta supervisada.

## Resultados pareados

```text
          configuration  seed  fold  n_compositions  n_gs_eligible  absolute_mae_mev_atom  delta_e_mae_mev_atom  ground_state_accuracy  pairwise_accuracy  pairwise_energy_mae_mev_atom  pairwise_signed_bias_mev_atom  absolute_rank_error_mean  signed_rank_error_mean  spearman_by_composition  pred_range_mev_atom  collapsed
D2-BASE E2-A partial_ft    42     0              14           13.0             713.924319             43.023121               0.384615           0.630952                     64.172013                     -12.187371                  0.714286                     0.0                 0.321429            65.036586        0.0
              D2-CENTER    42     0              14           13.0             700.449104             47.390251               0.384615           0.607143                     70.220052                     -15.643482                  0.714286                     0.0                 0.285714            71.939707        0.0
         D2-CENTER-ONLY    42     0              14           13.0             739.589950             42.121897               0.615385           0.714286                     61.574845                       2.989229                  0.523810                     0.0                 0.464286            55.255071        0.0
```

Una diferencia negativa favorece D2 en MAE; una diferencia positiva favorece D2 en GS/pairwise. Los intervalos pareados por composición están en `paired_bootstrap_pilot.json`; con un piloto de un fold y semilla solo son exploratorios.

El archivo `training_history_pilot.csv` separa objetivo promedio de minilotes inner-train y objetivo completo inner-validation. No son MAE: las curvas están normalizadas y CENTER y CENTER-ONLY contienen distintos términos. La época externa se predice después de seleccionar checkpoint con inner-validation.

La pérdida centrada y pairwise contienen información diferencial estrechamente relacionada en el límite cuadrático y de pares completos. Cualquier diferencia práctica puede venir de Huber, escalas, reducción, batching o convergencia; no implica señal física nueva.

Con solo un piloto no se declara ganador. Se requieren folds y semillas completos para concluir si el cambio es estable; después, cualquier D4 con geometría debe ser una comparación outer pareada.
