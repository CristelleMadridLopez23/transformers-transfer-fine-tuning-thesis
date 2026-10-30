# Preentrenamiento frente a backbone aleatorio congelado

[HECHO VERIFICADO] Los tres controles aleatorios usan el mismo MatGPT oficial, tokenizer, 768 dimensiones, dos poolings H3-C, folds y arquitectura de regresor. Se inicializaron con semilla + 500000; sus hashes exactos están en `random_initialization_lock.json`. Ningún backbone aleatorio recibió gradientes. No se reutilizaron las métricas del antiguo E2, que tenía otra cabeza.

[RESULTADO EXPERIMENTAL] Preentrenado vs aleatorio: MAE E/N 183.3 vs 124.9 meV/átomo; MAE residual 33.8 vs 36.0; MAE pares 58.8 vs 62.2. Diferencia apareada preentrenado menos aleatorio: MAE absoluto +58.40 [+23.01, +100.21] meV/átomo; MAE pares -3.38 [-8.72, +1.89] meV/átomo (intervalos 95 %, bootstrap de fórmula).

[RESULTADO EXPERIMENTAL] Acierto de mínimo elegible 53.2 % vs 40.3 %; diferencia apareada +12.38 [+1.90, +22.86] puntos porcentuales. Orden de pares >5: 66.8 % vs 54.3 %; diferencia apareada +12.52 [+5.53, +19.50] puntos.

[INFERENCIA] En este conjunto, el preentrenamiento favorece decisiones de ranking, pero empeora el error absoluto frente al control aleatorio. La diferencia de MAE de pares tiene intervalo que incluye cero; por tanto no se demuestra mejora de precisión de ΔE. No extrapolar a otras cabezas, datos o a entrenamiento desde cero.

[LIMITACIÓN] El código histórico conserva rutas internas anteriores a una reorganización del repositorio. La reproducción de solo lectura usa rutas actuales explícitas y recarga los checkpoints originales; no modifica el código ni los pesos históricos. Los resultados de desarrollo no son una validación externa ciega.

