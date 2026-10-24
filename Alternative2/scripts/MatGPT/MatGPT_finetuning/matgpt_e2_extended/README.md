# Componentes compartidos de E2-Aug y E2-T

`generate_variants.py` construye hasta cuatro entradas para las 210 estructuras de desarrollo. La vista histórica se conserva y cada vista nueva se acepta solo si comparte el grafo periódico etiquetado (especies, multiplicidad de enlaces, traslaciones) mediante una biyección de sitios y un cambio firmado de ejes. El API local de SLICES 2.0.12 antepone opcionalmente una etiqueta de grupo espacial no presente en el vocabulario original de 172 tokens; se conserva el formato compacto que realmente usó E2-A y se registra esta limitación.

`train_aug.py` es el entrenador común de Aug1/Aug2. Cada ejecución arranca del checkpoint original y ajusta solo el último bloque de decoder y MLP8. `evaluate_aug.py` evalúa predicciones outer y remuestrea composiciones completas para comparaciones pareadas. Las salidas están versionadas bajo `reports/MatGPT_finetuning/<experimento>/development_only_v2/`.

## Reproducir

```bash
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/generate_variants.py --views 4
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug1 --phase preflight --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug1 --phase smoke --device auto --run-label smoke_integrity_v3
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug1 --phase cv --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug2 --phase preflight --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug2 --phase smoke --device auto --run-label smoke_integrity_v3
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/train_aug.py --experiment E2-Aug2 --phase cv --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/matgpt_e2_extended/evaluate_aug.py --experiment-dir Alternative2/reports/MatGPT_finetuning/E2-Aug1 --experiment-dir Alternative2/reports/MatGPT_finetuning/E2-Aug2 --out Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/development_only_v2
```

Las figuras se exportan como PNG de acuerdo con `AGENTS.md`. La evidencia incluye `config.json`, `provenance.json`, `run_manifest.csv`, historiales, predicciones por fold/semilla y checkpoints pequeños de cabeza/bloque.
