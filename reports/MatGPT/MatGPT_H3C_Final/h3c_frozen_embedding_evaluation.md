# Evaluación de embeddings MatGPT congelados

[HECHO VERIFICADO] El predictor preentrenado usa `L03_MEAN` para Ridge y `FINAL_END` para MLP8; no se actualizaron parámetros de MatGPT. Las 630 predicciones son 210 estructuras físicas repetidas por tres semillas. Las comparaciones estadísticas remuestrean 70 fórmulas después de promediar semillas dentro de cada fórmula.

[RESULTADO EXPERIMENTAL] MAE de energía total derivada: 3.666 eV; MAE E/N: 183.3 meV/átomo. Error de media composicional: 180.2 meV/átomo. Error de residuo intrafórmula: 33.8 meV/átomo. Pendiente de paridad absoluta: 0.955; razón de dispersión absoluta: 1.004. Pendiente residual: 0.308; razón residual: 0.656.

[RESULTADO EXPERIMENTAL] El MAE de pares firmados es 58.8 meV/átomo; predecir ΔE=0 da 60.5. La mejora media de 1.6 meV/átomo es pequeña respecto a la dispersión entre fórmulas. Estado fundamental elegible: 53.2 %; orden de pares elegibles >5: 66.8 %. No se adjudica un mínimo al predictor de ΔE cero.

[INFERENCIA] Los embeddings preentrenados permiten recuperar señal de ordenamiento sobre este protocolo, pero la precisión de separaciones energéticas pequeñas no queda demostrada por el ranking. Una cabeza fija no mide toda la información del Transformer.

[LIMITACIÓN] El código histórico conserva rutas internas anteriores a una reorganización del repositorio. La reproducción de solo lectura usa rutas actuales explícitas y recarga los checkpoints originales; no modifica el código ni los pesos históricos. Los resultados de desarrollo no son una validación externa ciega.

