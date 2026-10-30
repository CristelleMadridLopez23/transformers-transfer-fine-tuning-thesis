# Ablación de profundidad de fine-tuning de MatGPT en ABS₃

## Objetivo y protocolo

**HECHO VERIFICADO.** Se compararon MatGPT congelado (F0) y la actualización directa de los bloques 10–11 (F2), 8–11 (F4) y 4–11 (F8). Se conservaron los embeddings de tokens y posiciones, la normalización final, la cabeza generativa, Ridge composicional de 17 fracciones y MLP8 residual. No se usaron LoRA ni descriptores geométricos.

**HECHO VERIFICADO.** El conjunto de desarrollo contiene 210 estructuras físicas, 70 fórmulas y tres polimorfos por fórmula. Los cinco pliegues agrupan por composición: 45 fórmulas para entrenamiento interno, 11 para validación interna y 14 para evaluación outer por pliegue. Cada fórmula aparece en una sola partición outer. El test histórico ABS₃ no se leyó.

Se seleccionó la época usando exclusivamente SmoothL1 normalizada del residuo energético dentro de la validación interna. La predicción es la media Ridge por fórmula más el residuo MLP8 centrado; la energía total se deriva multiplicando E/N por el número de átomos. ΔE se calcula entre polimorfos de la misma fórmula.

## Resultados agrupados por fórmula

Las cifras agregadas promedian primero las tres semillas dentro de cada estructura y composición. Las unidades estadísticas son 70 fórmulas, no 630 predicciones tratadas como materiales independientes.

| Condición | MAE E/N (meV/átomo) | MAE residuo (meV/átomo) | MAE ΔE (meV/átomo) | Estado fundamental exacto | Pares correctos / elegibles |
|---|---:|---:|---:|---:|---:|
| F0 | 50.48 | 31.21 | 58.84 | 54.29% | 131 / 197 |
| F2 | 51.31 | 32.56 | 58.56 | 56.67% | 130 / 197 |
| F4 | 51.09 | 31.76 | 57.27 | 59.05% | 134 / 197 |
| F8 | 50.57 | 30.89 | 55.25 | 56.19% | 135 / 197 |

## Hallazgos principales

**RESULTADO EXPERIMENTAL.** La MAE de energía absoluta fue 50.48 meV/átomo en F0; F2, F4 y F8 obtuvieron 51.31, 51.09 y 50.57 meV/átomo. Ninguna profundidad mostró una reducción estadísticamente respaldada frente a F0 por el intervalo bootstrap apareado del 95%.

**RESULTADO EXPERIMENTAL.** La MAE de diferencias por pares fue 58.84 meV/átomo en F0, 58.56 en F2, 57.27 en F4 y 55.25 en F8. El contraste F8 − F0 fue -1.30 meV/átomo (IC95% -3.55 a 1.00), una reducción descriptiva compatible con una mejora dentro de este protocolo.

**LIMITACIÓN.** Aunque la MAE media de ΔE quedó por debajo del predictor ΔE=0 (60.47 meV/átomo), los intervalos bootstrap de los cuatro contrastes frente a ese baseline incluyen cero; no hay evidencia concluyente de que las predicciones superen al baseline cero.

**RESULTADO EXPERIMENTAL.** F4 obtuvo el mayor acierto exacto del estado fundamental (59.0%); F8 tuvo 135 aciertos en 197 pares elegibles frente a 131 en F0. Estas tendencias de clasificación se describen sobre los pliegues de desarrollo y no seleccionan una profundidad final.

**INFERENCIA.** En esta ablación, F8 muestra una reducción pareada de error ΔE frente a F0, mientras que la energía absoluta queda esencialmente igual. El cambio representacional en F8 es mayor; esto no prueba que el drift cause la tendencia energética.

## Cambios pareados e incertidumbre

Los intervalos se obtuvieron con 10 000 remuestreos apareados agrupados por composición, después de promediar las semillas. Un intervalo que incluye cero no respalda una diferencia clara en esa métrica.

- F2 − F0 | MAE energía absoluta (meV/átomo): diferencia 0.83 (IC95% -0.20 a 1.90).
- F4 − F0 | MAE energía absoluta (meV/átomo): diferencia 0.61 (IC95% -0.66 a 1.87).
- F8 − F0 | MAE energía absoluta (meV/átomo): diferencia 0.10 (IC95% -0.76 a 0.95).
- F4 − F2 | MAE energía absoluta (meV/átomo): diferencia -0.21 (IC95% -1.13 a 0.65).
- F8 − F4 | MAE energía absoluta (meV/átomo): diferencia -0.52 (IC95% -1.75 a 0.67).
- F2 − F0 | MAE residuo intrafórmula (meV/átomo): diferencia 1.35 (IC95% 0.06 a 2.69).
- F4 − F0 | MAE residuo intrafórmula (meV/átomo): diferencia 0.55 (IC95% -0.88 a 1.96).
- F8 − F0 | MAE residuo intrafórmula (meV/átomo): diferencia -0.32 (IC95% -1.72 a 1.02).
- F4 − F2 | MAE residuo intrafórmula (meV/átomo): diferencia -0.80 (IC95% -1.94 a 0.30).
- F8 − F4 | MAE residuo intrafórmula (meV/átomo): diferencia -0.87 (IC95% -2.40 a 0.65).
- F2 − F0 | MAE ΔE (meV/átomo): diferencia 1.69 (IC95% -0.86 a 4.22).
- F4 − F0 | MAE ΔE (meV/átomo): diferencia 0.44 (IC95% -2.17 a 3.02).
- F8 − F0 | MAE ΔE (meV/átomo): diferencia -1.30 (IC95% -3.55 a 1.00).
- F4 − F2 | MAE ΔE (meV/átomo): diferencia -1.24 (IC95% -3.26 a 0.75).
- F8 − F4 | MAE ΔE (meV/átomo): diferencia -1.74 (IC95% -4.38 a 0.95).
- F0 − ΔE=0 | MAE de diferencias (meV/átomo): diferencia -6.04 (IC95% -13.25 a 1.17).
- F2 − ΔE=0 | MAE de diferencias (meV/átomo): diferencia -4.36 (IC95% -12.85 a 3.79).
- F4 − ΔE=0 | MAE de diferencias (meV/átomo): diferencia -5.60 (IC95% -13.40 a 1.97).
- F8 − ΔE=0 | MAE de diferencias (meV/átomo): diferencia -7.35 (IC95% -15.01 a 0.14).

## Adaptación y recursos

**RESULTADO EXPERIMENTAL.** Estado de recursos: `COMPLETE`; corridas F2/F4/F8 completadas: 45 de 45; tiempo acumulado observado: 34269.5 s; CPU acumulada: 51877.3 s; RSS máximo observado: 7.59 GB.

Los diagnósticos de FINAL_END comparan el checkpoint original con F2/F4/F8 en las 168 estructuras de desarrollo outer-train del pliegue 0. El drift caracteriza el cambio de representación; por sí solo no demuestra que ese cambio cause diferencias de rendimiento.

- F2: similitud coseno media 1.0000; cambio L2 relativo 0.0079; rango efectivo espectral 38.0; distancia intra-fórmula normalizada 0.6147.
- F4: similitud coseno media 1.0000; cambio L2 relativo 0.0076; rango efectivo espectral 38.1; distancia intra-fórmula normalizada 0.6160.
- F8: similitud coseno media 0.9981; cambio L2 relativo 0.0618; rango efectivo espectral 38.2; distancia intra-fórmula normalizada 0.5798.

## Interpretación científica

**INFERENCIA.** La profundidad más útil debe juzgarse por el cambio en residuos y ΔE, además de la calibración absoluta. Ridge estima una media composicional común a los tres polimorfos y, por construcción, no puede mejorar el orden relativo entre ellos.

**LIMITACIÓN.** Las métricas outer proceden de cinco pliegues de desarrollo y tres semillas; no equivalen a un conjunto externo independiente. Los valores gga-static del conjunto conservan su procedencia histórica y esta ablación no certifica homogeneidad completa de funcionales/correcciones.

**LIMITACIÓN.** F0 se reutilizó de las 15 corridas congeladas históricas, cuya compatibilidad se basa en código, semillas, target y configuración registrados; no se repitió F0. Las comparaciones por profundidad son exploratorias y no constituyen una selección anidada nueva.

## Figuras

Las diez figuras de resultados y los diagramas de arquitectura están en `../../../images/MatGPT_SLICES_Finetuning_Profundidad/`. Incluyen paridad energética, residuos, diferencias entre pares, errores por intervalo de gap, curvas de selección interna, cambio de representación y contraste de recursos.

## Reproducción

La configuración usada está en `../../../experiments/MatGPT_SLICES_Finetuning_Profundidad/config_used.yaml`. En local, entrenar con `python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/train.py --phase all --device auto --num-threads 2`; luego ejecutar `python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/run_local.py --evaluate`. El log, estado, recursos y métricas están en `experiments/MatGPT_SLICES_Finetuning_Profundidad/`.
