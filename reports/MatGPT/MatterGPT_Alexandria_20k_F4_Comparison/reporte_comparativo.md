# Comparación F4 MatGPT vs MatterGPT con transferencia Alexandria 20k

## Cohortes y alcance

MatGPT se entrenó con 20.000 fórmulas y validó con 2.000. MatterGPT solo acepta 17.245/20.000 fórmulas de train y 1.718/2.000 de validación por vocabulario y límite de contexto. La paridad Alexandria usa la intersección exacta de 1.718 IDs, con SLICES y targets cotejados por ID.

El ranking y la selección del estado fundamental se calcularon con los checkpoints F4 sobre 210 estructuras ABS₃ estáticas de desarrollo (70 tripletes), excluidas del fine-tuning Alexandria. No se leyó el test histórico. Esta es una evaluación de transferencia sobre desarrollo, no un test externo independiente.

## Hallazgos

En Alexandria compartida, el MAE fue 272.0 meV/átomo para MatGPT y 221.3 para MatterGPT.

En ABS₃ desarrollo, el MAE energético fue 381.5 meV/átomo para MatGPT y 260.9 para MatterGPT.

Exactitud pairwise: MatGPT 46.2% (91/197); MatterGPT 50.3% (99/197).

Selección exacta del ground state: MatGPT 34.3% (23/67); MatterGPT 32.8% (22/67).

## Figuras

- `curvas_entrenamiento_validacion_f4.png`
- `paridad_energia_alexandria_validation.png`
- `paridad_energia_abs3_desarrollo.png`
- `ranking_groundstate_abs3.png`

Las figuras y los CSV/JSON de predicción y métricas están vinculados desde los directorios images/ y experiments/ del proyecto.
