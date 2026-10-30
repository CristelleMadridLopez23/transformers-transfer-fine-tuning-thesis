# Entrenamiento energético MatGPT + LoRA

Configuración fija: Q/V en bloques 10–11, rango 4, alpha 8, dropout 0.05, AdamW; MLP8 768→8→1, 200 épocas máximas y paciencia 20. Selección por SmoothL1 residual normalizada en 11 fórmulas de validación interna. Entrenamientos secuenciales en CPU, dos hilos.

No se ajustó Ridge, no se usó geometría y no se cambió el checkpoint original.

- **frozen** — MAE E/N 183.3 ± 0.5; residuo 33.8 ± 0.7; ΔE pares 58.8 ± 1.0; GS elegible 0.532; pares elegibles 0.668.
- **lora** — MAE E/N 182.8 ± 0.3; residuo 33.5 ± 0.2; ΔE pares 58.3 ± 0.2; GS elegible 0.517; pares elegibles 0.658.

Tiempo total observado: 1.12 h. Ver logs por corrida en `runs/`.
