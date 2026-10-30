# Experimento D: fine-tuning de MatGPT con SLICES y geometría de celda

Código en `scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/`,
resultados tabulares, métricas, checkpoints y recursos en este directorio,
figuras PNG en `images/MatGPT_SLICES_Geometria_Finetuning_Profundidad/` e
informe en `reports/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/`.

## Comandos locales

```bash
python scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/train.py --phase preflight --device cpu --num-threads 2
python -m unittest discover -s scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/tests -p 'test_*.py' -v
python scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/train.py --phase smoke --device cpu --num-threads 2
python scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/train.py --phase all --device cpu --num-threads 2
python scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/analyze.py
```

El registro vivo es `tail -f experiments/MatGPT_SLICES_Geometria_Finetuning_Profundidad/run.log`.
El análisis usa el estimando A como primario, mantiene B separado y emplea la
fórmula corregida de ΔE anclada a la misma estructura GS de DFT.
