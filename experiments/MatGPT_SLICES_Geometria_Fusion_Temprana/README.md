# MatGPT con fusión geométrica temprana

Este directorio contiene resultados, checkpoints, logs y recursos del contraste
F0/F8 entre SLICES-only, fusión temprana y fusión tardía con capacidad equiparada.

Scripts: `scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/`.
Informe: `reports/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/reporte_cientifico.md`.
Figuras PNG: `images/MatGPT_SLICES_Geometria_Fusion_Temprana/`.

Ejecución local monitorizable:

```bash
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/train.py --phase all --device cpu --num-threads 2
tail -f experiments/MatGPT_SLICES_Geometria_Fusion_Temprana/run.log
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/analyze.py --phase all
```

Solo hay tres manifiestos JSON de resultados: `metrics.json`, `summary.json` y
`resources.json`. El log permite seguir épocas; checkpoints `last.pt` admiten
reanudación y `complete.flag` evita volver a ejecutar corridas terminadas.
