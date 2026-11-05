# Preflight — ABS₃ F0/F1/F2 con MatGPT y MatterGPT

## Estado

**Preflight técnico de fuentes y cachés: PASS. Entrenamiento: no iniciado.**
La solicitud prohíbe interferir con F4. Los artefactos F4 disponibles reportan
`COMPLETE` y su log más reciente tiene marca 2026-11-03 03:44 (hora local del
archivo). La lectura del listado de procesos fue denegada por macOS (`ps:
Operation not permitted`) y Computer Use no recibió permiso; por eso no pude
confirmar si existe otro proceso F4 fuera de esos artefactos. No inicié trabajo
de ajuste fino para evitar competir con un trabajo activo que no puedo observar.

## Fuentes inspeccionadas

- `scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py`:
  lectura de dataset/folds, orden de tripletes, MLP8, normalización residual,
  warmup y continuación con forward en vivo.
- `scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/extract_embeddings.py`:
  cargas estrictas, tokenizadores, extracción FINAL_END y condicionamiento.
- `scripts/MatGPT/MatGPT_F0_Stoichiometric_Ridge_M2_Comparison/run_experiment.py`:
  control M2, geometría y entrenamiento congelado MLP8.
- `scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py` y
  `scripts/MatterGPT/mattergpt_abs3_geometry_late_fusion/run_experiment.py`:
  control MatterGPT, métricas y fusión geométrica histórica.
- `scripts/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/run_experiment.py`:
  selección de alpha M2 y refit por fold.

## Controles verificados

- Dataset de desarrollo: 210 IDs, 70 fórmulas y tres polimorfos por fórmula;
  SHA-256 `afdec9430cde046d5a850cc95432cf90759194f882880496388dc3a65c27eb8e`.
- Folds históricos: SHA-256
  `a5f00a89a41e4645a68d1463f26f9eca10a09bdfd1208193a87e541ff7cd5627`; cada
  fold contiene 45/11/14 fórmulas inner-train/validation/outer.
- MatGPT original: SHA-256 esperado y observado
  `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`.
- MatterGPT original: SHA-256 esperado y observado
  `8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21`.
- Carga estricta (`strict=True`) de ambos modelos sin claves faltantes ni
  inesperadas.
- Reextracción de 210 embeddings por backbone comparada por ID con cachés
  210×768: diferencia absoluta máxima MatGPT `2.86e-6`; MatterGPT `0`.
  MatterGPT usa `[1.0, 2.0]`.
- Cinco artefactos Ridge M2 leídos desde `git HEAD`: sus hashes coinciden con
  el manifiesto histórico. No se llamó a `.fit()`.
- MPS no está disponible; la corrida F4 guardada usó CPU, 3 hilos y 9,450 s
  de pared para 30 pares modelo/fold/semilla (60 ramas: F0 y F4).
- No se abrió el test independiente.

## Implementación histórica recuperada

La MLP8 es `Linear(768 o 775, 8) → GELU → Dropout(0.1) → Linear(8,1)`.
La pérdida histórica compara residuos centrados por triplete, con SmoothL1
`beta=1` y escala igual a la desviación estándar `ddof=0` de residuos
inner-train. Geometría se estandariza por fold usando solo las 45 fórmulas
inner-train. El Ridge M2 por fold se seleccionó con 45/11 y se volvió a ajustar
en las 56 fórmulas outer-train; estas 56 no incluyen outer-evaluation.

En el protocolo histórico, la selección de la cabeza usaba SmoothL1 de
validación. Esta solicitud fija como criterio primario la MAE de ΔE firmada de
inner-validation. Para F0/F1/F2 habría que aplicar esta regla nueva e idéntica
a las doce condiciones; los resultados no serían una reproducción exacta del
criterio histórico, pero sí una comparación emparejada.

## Configuración prevista

- F0: Transformer congelado; cabeza MLP8 entrenable.
- F1: solo bloque 11 entrenable; cabeza MLP8 entrenable.
- F2: solo bloques 10–11 entrenables; cabeza MLP8 entrenable.
- Warmup común de hasta 20 épocas; continuación común de hasta 71 épocas;
  AdamW, LR de cabeza `1e-4`, patience 12 y `min_delta=0`.
- Grid interno de LR del backbone F1/F2: `1e-6`, `3e-6`, `1e-5`; MLP8 y
  geometría iguales entre ramas. L2-SP desactivado.
- Selección exclusiva por inner-validation. Outer se reserva para la evaluación
  OOF final. Test independiente cerrado.

## Piloto técnico

Se completó un piloto de una época para las doce combinaciones de modelo,
modalidad y política en fold 0/semilla 42. Se ejecutaron 28 candidatos contando
las tres tasas F1/F2. F1 actualizó 9,451,776 parámetros de MatGPT y 7,087,872
de MatterGPT; F2 actualizó 18,903,552 y 14,175,744, respectivamente. Los
gradientes fueron finitos y no nulos en bloques habilitados y MLP8; los pesos
congelados no cambiaron en el primer paso. Las pérdidas y predicciones de
validación fueron finitas. El piloto no consultó outer y no sirve como
estimación de desempeño: solo comprobó la ruta de entrenamiento de una época.

Consumo medido: 394.2 s de pared, 708.5 s de CPU, RSS máximo 3.55 GB, CPU con
3 hilos; PyTorch 2.8.0, Python 3.10.21. El piloto terminó correctamente.

## Conteo y recursos

La matriz final contiene 180 resultados (12 condiciones × 5 folds × 3 semillas).
Al seleccionar entre tres tasas de backbone, el cálculo real requiere hasta
420 continuaciones (60 F0 y 360 candidatos F1/F2), más 60 warmups compartidos.
El antecedente F4 consumió 9,450 s para 30 pares modelo/fold/semilla
(30 ramas congeladas y 30 ramas F4) en CPU. Esta referencia no predice
exactamente la duración F1/F2, pero indica que la matriz completa es
un trabajo largo de CPU. MPS no está disponible y no pude verificar procesos
activos por las restricciones del sistema operativo. El piloto técnico concluyó y la matriz completa está en ejecución; estado incremental en `training_status.json`.

## Leakage y límites

- Los polimorfos permanecen juntos por fórmula.
- El scaler geométrico y la escala residual se ajustan solo en inner-train.
- Ridge M2 se ajusta con las 56 fórmulas outer-train (45 inner-train + 11 inner-validation), excluyendo las 14 outer. El código de esta ejecución no usa μ_M2 en la pérdida de entrenamiento ni en la selección inner-validation: ambas usan scores/residuos centrados por triplete y diferencias ΔE, donde un desplazamiento común μ_c se cancela exactamente. M2 entra únicamente al reconstruir energía absoluta outer, después de seleccionar el checkpoint. Por ello, no hay etiquetas de validación filtradas a través de M2 en la selección, y se conserva el estimando composicional histórico para energía absoluta.
- El conjunto representa composiciones no usadas en el ajuste supervisado de
  cada fold; no se ha demostrado ausencia de estas composiciones del
  preentrenamiento generativo.
- F4 histórico queda como referencia secundaria de otro protocolo.
