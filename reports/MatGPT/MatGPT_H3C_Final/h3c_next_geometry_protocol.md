# Protocolo para una futura comparación con geometría

[HECHO VERIFICADO] La interfaz estable `interface.py` expone `load_inputs()` (ID físico, fórmula, N, SLICES y ambos embeddings), `predict_triplets()` y `evaluate_intraformula()`. Por defecto no entrega etiquetas DFT; estas solo se solicitan en entrenamiento o evaluación autorizada.

[HIPÓTESIS] En un estudio posterior, descriptores geométricos derivados de la estructura podrán concatenarse a la rama estructural H3-C. Se conservarán los cinco pliegues, las tres semillas, los mismos targets por átomo, Ridge, anchura oculta 8, GELU, dropout .1, pérdida y regla de selección. Solo aumentará la dimensión de entrada de la primera capa y su recuento de parámetros, que deberá documentarse antes de evaluar.

[LIMITACIÓN] No se han calculado ni añadido descriptores geométricos. No hay resultado de mejora geométrica en este módulo. Cualquier afirmación futura requiere validación sobre composiciones no usadas para seleccionar el diseño.
