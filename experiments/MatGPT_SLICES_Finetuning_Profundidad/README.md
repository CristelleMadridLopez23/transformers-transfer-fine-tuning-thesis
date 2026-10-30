# Ablación de profundidad de fine-tuning MatGPT

## Ubicación de artefactos

- Código: `scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/`
- Salidas tabulares, métricas, logs y checkpoints: este directorio.
- Figuras PNG: `images/MatGPT_SLICES_Finetuning_Profundidad/`
- Informe científico: `reports/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/reporte_cientifico.md`

## Reproducción local

Para volver a ejecutar entrenamiento y análisis, desde la raíz del repositorio:

```bash
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/train.py --phase preflight --device auto --num-threads 2
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/train.py --phase smoke --device auto --num-threads 2
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/train.py --phase all --device auto --num-threads 2
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/plots.py
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/diagrams.py
python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/write_report.py
```

Para seguir el registro de entrenamiento local: `tail -f experiments/MatGPT_SLICES_Finetuning_Profundidad/run.log`.

La corrida ya completada no necesita repetirse para consultar resultados. Para reconstruir las métricas desde checkpoints, agregar `--evaluate` a `run_local.py`. Para regenerar solo el material visual y el informe, ejecutar `python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/run_local.py`.
