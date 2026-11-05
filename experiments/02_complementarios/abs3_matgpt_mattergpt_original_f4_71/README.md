# ABS3 MatGPT/MatterGPT F4

Ejecución local. Pilot: `python scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py --stage pilot`. Completo: `python scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py --stage full`. La ejecución completa omite parejas fold-semilla ya completas; una pareja interrumpida se vuelve a ejecutar desde su warmup compartido. El log `train_history.csv` se actualiza por época. Consulte `preflight.json`, `resources.json` y `reporte_cientifico.md`.
