# Experimento C — MatGPT y LoRA para energía ABS₃

El experimento compara dos condiciones supervisadas con la misma estructura H3-C: MatGPT congelado (C0) y MatGPT con adaptadores Q/V LoRA en bloques 10–11 (C1). La rama Ridge de media composicional permanece congelada; MLP8 predice residuos centrados de energía DFT por átomo.

La validación usa cinco pliegues agrupados por fórmula, tres semillas (42, 123, 2026), 45 fórmulas de entrenamiento interno, 11 de validación interna y 14 retenidas por fold. No se usa el test histórico. Los resultados sobre las mismas 70 fórmulas de desarrollo son exploratorios.

Se aplica AdamW sin recorte de gradiente, de acuerdo con el entrenamiento histórico de MLP8 y la configuración bloqueada en `config.yaml`.

## Ejecutar

Desde la raíz del repositorio y usando el entorno `transformers-transfer-fine-tune-tesis`:

```bash
PYTHONPATH="$PWD" python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/experiment.py preflight
PYTHONPATH="$PWD" python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/tests/test_experiment.py
bash sh_files/MatGPT_SLICES_Finetuning_Energetico_LoRA/run_all.sh
```

Las 30 corridas se ejecutan secuencialmente. No se sobrescriben métricas/logs. Tras completarlas, evaluar y dibujar:

```bash
PYTHONPATH="$PWD" python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/measure_run.py --metrics reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/evaluation_resources.json --log reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/evaluation.log -- python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/evaluate.py
PYTHONPATH="$PWD" python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/measure_run.py --metrics reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/figures_resources.json --log reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/figures.log -- python model-pretrained/scripts/MatGPT_SLICES_Finetuning_Energetico_LoRA/figures.py
```

Cada proceso pesado debe conservar un nombre de log y métricas nuevo. Seguimiento de la corrida actual:

```bash
tail -f reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/runs/lora_fold0_seed123.log
```

La especificación de parámetros está en `config.yaml`; scripts, salidas, predicciones y recursos están separados por carpeta.
