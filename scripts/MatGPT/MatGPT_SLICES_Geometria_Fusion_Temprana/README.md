# Fusión geométrica temprana y tardía en MatGPT

Experimento CPU local. Conserva el contrato histórico de datos, Ridge composicional,
tokenizador, MLP8, pliegues y semillas. La geometría se evalúa únicamente en los
210 registros de desarrollo; no se abre el test histórico.

## Archivos

- `train.py`: auditoría de fuentes, smoke y entrenamiento F0/F8 con fusión temprana/tardía.
- `analyze.py`: inferencia OOF, permutación geométrica, métricas, bootstrap y figuras.
- `diagrams.py`: fuentes Mermaid y diagramas PNG.
- `config.yaml`: configuración congelada.
- Resultados/checkpoints/logs/recursos: `experiments/MatGPT_SLICES_Geometria_Fusion_Temprana/`.
- Figuras: `images/MatGPT_SLICES_Geometria_Fusion_Temprana/`.
- Informe: `reports/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/reporte_cientifico.md`.

## Ejecución local

```bash
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/train.py --phase preflight --device cpu --num-threads 2
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python -m unittest discover -s scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/tests -p 'test_*.py' -v
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/train.py --phase smoke --device cpu --num-threads 2
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/train.py --phase all --device cpu --num-threads 2
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/analyze.py --phase all
MPLCONFIGDIR=/tmp/mpl-matgpt .venv/bin/python scripts/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana/diagrams.py
```

Para monitorear: `tail -f experiments/MatGPT_SLICES_Geometria_Fusion_Temprana/run.log`.
Los pasos son reanudables por marcador/checkpoint; las corridas completas no se duplican.

## Diseño

- F0: Transformer congelado, pero el gradiente se propaga hasta la proyección geométrica.
- F8: solo los bloques 4–11 se ajustan; embeddings, bloques 0–3, `ln_f` y LM head quedan congelados.
- Temprana: `token_embedding + position_embedding + projection(geometry)` antes del decoder.
- Tardía equiparada: `FINAL_END + projection(geometry)` antes de la misma MLP8.
- La MLP8 siempre es 768→8→1; Ridge se reutiliza sin reentrenamiento.
