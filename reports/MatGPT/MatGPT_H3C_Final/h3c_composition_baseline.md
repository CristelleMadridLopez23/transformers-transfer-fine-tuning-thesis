# Control de composición solamente

[HECHO VERIFICADO] El vector contiene fracciones estequiométricas de 17 elementos (`Ba, Ca, Cd, Cu, Ge, Hf, Mg, Nb, Pb, S, Si, Sn, Sr, Ti, V, Zn, Zr`). Cada polimorfo de una fórmula recibe exactamente el mismo vector; no entran IDs, energías, topología ni geometría. Ridge usa el mismo grid alpha, StandardScaler, selección interna y refit histórico, con 17 entradas inevitables en vez de 768. La rama residual centrada es algebraicamente cero para tres entradas idénticas; no se le atribuye un entrenamiento útil ni ranking arbitrario.

[RESULTADO EXPERIMENTAL] MAE E/N: **52.0 meV/átomo**; MAE de la media de fórmula: 40.8 meV/átomo; MAE residual: 35.1; MAE de pares = baseline ΔE cero = 60.5. Los tres polimorfos siempre empatan.

[RESULTADO EXPERIMENTAL] Diferencia H3-C preentrenado menos composición: MAE absoluto +131.27 [+93.78, +179.40] meV/átomo; MAE de pares -1.63 [-7.98, +4.69] meV/átomo. La primera comparación favorece la fórmula; la segunda no demuestra una mejora clara de ΔE.

[INFERENCIA] En estas composiciones, la química explica gran parte del nivel energético absoluto. La fórmula sola no contiene identidad de polimorfo y no permite selección estructural.

[LIMITACIÓN] El código histórico conserva rutas internas anteriores a una reorganización del repositorio. La reproducción de solo lectura usa rutas actuales explícitas y recarga los checkpoints originales; no modifica el código ni los pesos históricos. Los resultados de desarrollo no son una validación externa ciega.

