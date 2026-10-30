# Protocolo para una etapa posterior con geometría

Este informe no implementa geometría. Si se abre una etapa posterior, comparar en los mismos folds y semillas la entrada SLICES sola frente a SLICES más descriptores geométricos con MatGPT y la MLP8 fijados. Mantener el checkpoint y Ridge congelados; ajustar normalización geométrica solo con train interno; no introducir geometría en la selección de C1. Declarar por separado procedencia estructural y disponibilidad de geometría al predecir.

El experimento C por sí solo no demuestra que geometría sea la causa de las limitaciones de los embeddings.
