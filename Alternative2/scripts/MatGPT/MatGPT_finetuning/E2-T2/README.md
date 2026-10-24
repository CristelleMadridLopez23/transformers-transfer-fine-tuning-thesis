# E2-T2 — fine-tuning con trayectorias

Experimento condicional. Entrena solo con frames internos de entrenamiento admitidos por E2-T1; frames repetidos del mismo ID de entrada se agrupan y no se usan como pares contradictorios. Outer-validation se evalúa una sola vez. El test histórico se excluye en todas las fases.

El script `train_t2.py` comprueba el gate de E2-T1, el smoke de Aug2 y, para CV completa, la CV completa de Aug2 antes de invocar entrenamiento. Se permite `--phase pilot` con gate `LIMITED`; la CV completa requiere `GO`. Un gate `NO-GO` o `BLOCKED` rechaza ambas fases.

```bash
python -u Alternative2/scripts/MatGPT_finetuning/E2-T2/train_t2.py --experiment E2-T2 --phase preflight --device auto --run-label development_only_v2
python -u Alternative2/scripts/MatGPT_finetuning/E2-T2/train_t2.py --experiment E2-T2 --phase smoke --device auto --run-label smoke_integrity_v3
python -u Alternative2/scripts/MatGPT_finetuning/E2-T2/train_t2.py --experiment E2-T2 --phase pilot --device cpu --fold 0 --seed 42 --run-label development_only_v2
```

La supervisión de trayectorias se balancea por composición: en cada minibatch se eligen de forma determinista hasta dos trayectorias por composición y hasta cuatro pares de secuencia distinta por trayectoria. El anchor es el último frame elegible de esa misma trayectoria. `a_traj` se deriva exclusivamente de diferencias de energía entre frames inner-train.

T1 quedó `LIMITED`, por lo que se ejecutó solo el piloto predefinido: 35 épocas, 1,296 s, fold 0/semilla 42. Frente a Aug2 pareado en 14 composiciones: MAE ΔE 38.30 vs 39.44 meV/átomo; GS 0.539 vs 0.385; pairwise 0.714 vs 0.631; MAE absoluta 731.85 vs 720.21 meV/átomo. Los IC pareados exploratorios incluyen cero en todas las diferencias; el piloto no autoriza CV completa ni una conclusión de mejora. Reporte: `Alternative2/reports/MatGPT_finetuning/E2-T2/pilot_comparison_development_only_v2/`.
