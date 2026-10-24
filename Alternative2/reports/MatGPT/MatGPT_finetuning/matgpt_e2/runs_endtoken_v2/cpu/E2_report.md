# E2-A — resultados OOF

Solo se evalúan folds externos de desarrollo; el test histórico no participa. La selección de configuraciones anteriores usó la validation histórica; estos resultados no eliminan ese sesgo. Bootstrap por composición: ver `paired_bootstrap_ci.json`; intervalos exploratorios.

## Métricas por semilla

```text
configuration  seed  absolute_mae_ev_atom  delta_e_mae_mev_atom  ground_state_accuracy  pairwise_accuracy  dft_range_mev_atom  pred_range_mev_atom  collapsed
       frozen    42              0.725900             38.770048               0.457143           0.638095           90.705635            53.919806   0.000000
       frozen   123              0.723665             37.693877               0.557143           0.673810           90.705635            55.674383   0.000000
       frozen  2026              0.731427             36.936756               0.614286           0.690476           90.705635            63.722046   0.000000
   partial_ft    42              0.727707             37.190849               0.471429           0.633333           90.705635            54.937226   0.000000
   partial_ft   123              0.724789             35.905116               0.557143           0.688095           90.705635            66.351934   0.014286
   partial_ft  2026              0.734350             38.660589               0.571429           0.685714           90.705635            67.577040   0.000000
```
