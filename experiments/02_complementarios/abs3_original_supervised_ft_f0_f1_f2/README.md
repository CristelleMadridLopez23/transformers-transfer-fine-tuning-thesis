# Fine-tuning supervisado ABS₃ F0/F1/F2

Estado: matriz completa en ejecución, CPU (3 hilos), MPS no disponible. La
corrida usa los checkpoints originales, Ridge M2, MLP8 residual y los cinco
folds agrupados por composición. La selección de checkpoints se hace solo con
inner-validation y el test independiente permanece cerrado. No se modificó el
experimento F4.

## Supervisar

Desde la raíz del repositorio:

```bash
cat experiments/02_complementarios/abs3_original_supervised_ft_f0_f1_f2/training_status.json
 tail -n 20 experiments/02_complementarios/abs3_original_supervised_ft_f0_f1_f2/training_history_full.csv
 tail -n 10 experiments/02_complementarios/abs3_original_supervised_ft_f0_f1_f2/run_records_full.csv
```

`completed_runs.csv` cuenta resultados OOF finalizados (de 180). `training_history_full.csv`
registra época, pérdida de minibatches, validación, MAE ΔE, exactitud pairwise
y gradientes. El estado final y el tiempo/RAM/CPU medidos se escribirán en
`training_status.json` y `resources_run.json`.

No iniciar una segunda copia mientras el estado sea `RUNNING`.

## Reanudar manualmente si el proceso se detiene

```bash
PYTHONUNBUFFERED=1 /Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python \
  scripts/ABS3/abs3_original_supervised_ft_f0_f1_f2/run_experiment.py --phase full --resume
```

## Archivos

- Protocolo y auditoría: `config.yaml`, `preflight_report.md`, `preflight.json`.
- Checkpoints por candidato: `continuation/` y warmups compartidos: `warmup/full/`.
- Predicciones OOF incrementales: `predictions/structures/` y `predictions/pairs/`.
- Historial y selección por fase: `training_history_full.csv`, `candidate_records_full.csv`, `run_records_full.csv`.
- Recursos: `resources.json` (preflight/piloto) y `resources_run.json` (al completar).

M2 no interviene en el ajuste ni en la selección inner-validation: la cabeza se
entrena y evalúa con residuos centrados y ΔE, donde la media composicional se
cancela. El Ridge M2 histórico se usa para reconstruir la energía absoluta OOF
una vez seleccionado el checkpoint; para cada fold se ajustó con las 56 fórmulas
outer-train, nunca con las 14 fórmulas outer-evaluation.
