# E2-Aug2 — consistencia entre vistas

Usa el mismo banco de grafos equivalentes y el mismo protocolo emparejado de E2-Aug1. En entrenamiento compara dos serializaciones distintas de cada misma estructura y agrega `0.1 × L_inv` a la pérdida absoluta + pairwise. `L_inv` se normaliza por la escala pairwise calculada solo con inner-train. Validación, outer-validation y evaluación usan la serialización histórica. Test histórico excluido.

```bash
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug2/run_aug.py --experiment E2-Aug2 --phase preflight --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug2/run_aug.py --experiment E2-Aug2 --phase smoke --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/E2-Aug2/run_aug.py --experiment E2-Aug2 --phase cv --device auto --run-label development_only_v2
```

CV PASS: 15/15 corridas, 6,876 s de entrenamiento acumulado. OOF desarrollo: MAE absoluta macro 731.5 ± 82.8 meV/átomo; MAE ΔE 37.93 ± 7.80 meV/átomo; GS 0.514 ± 0.127; pairwise 0.636 ± 0.058. Frente a E2-A los IC pareados incluyen cero para todas estas métricas. La dispersión entre las cuatro vistas bajó de 23.78 meV/átomo (E2-A) a 17.47 (Aug2), sin mejora de ranking demostrada. Resumen y figuras en `Alternative2/reports/MatGPT_finetuning/E2-Aug_comparison/both_development_only_v2/`; test histórico excluido.

El piloto T2 usa Aug2 fold 0/semilla 42 como referencia emparejada.
