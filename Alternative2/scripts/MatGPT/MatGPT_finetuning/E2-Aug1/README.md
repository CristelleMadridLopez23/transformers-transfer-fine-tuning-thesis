# E2-Aug1 — vistas SLICES equivalentes

Evalúa si variaciones deterministas de SLICES, verificadas contra el mismo grafo periódico etiquetado, regularizan la predicción sin añadir datos ni cambiar el vocabulario. Conserva el entrenamiento parcial E2-A (último bloque Transformer + MLP8) y la pérdida absoluta más pairwise. Desarrollo: 210 estructuras / 70 composiciones; GroupKFold 5×3 semillas; test histórico excluido.

El banco de vistas de desarrollo está en `Alternative2/reports/MatGPT_finetuning/E2-Aug1/variants_development_only_v2/`. La serialización histórica es siempre la vista primaria de validación y evaluación outer; solo las entradas de entrenamiento varían por época. El test histórico se excluye antes de reconstruir estructuras o generar variantes.

```bash
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug1/run_aug.py --experiment E2-Aug1 --phase preflight --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug1/run_aug.py --experiment E2-Aug1 --phase smoke --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug1/run_aug.py --experiment E2-Aug1 --phase cv --device auto --run-label development_only_v2
```

CV PASS: 15/15 corridas, 4,023 s de entrenamiento acumulado. OOF desarrollo: MAE absoluta macro 726.4 ± 79.8 meV/átomo; MAE ΔE 38.75 ± 7.14 meV/átomo; GS 0.499 ± 0.135; pairwise 0.648 ± 0.063. Frente a E2-A, todos los IC bootstrap pareados por composición incluyen cero. Resumen conjunto y figuras corregidas en `Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/both_development_only_v2/`; no se consultó el test.

No ejecutar CV sobre una carpeta con resultados previos; `--resume` omite únicamente runs completos.
