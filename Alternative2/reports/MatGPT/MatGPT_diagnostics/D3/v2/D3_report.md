# D3 — ¿SLICES y MatGPT distinguen los polimorfos estáticos?

## En palabras sencillas

Se preguntaron tres cosas distintas: ¿cambia la cadena SLICES?, ¿cambia el embedding que produce el checkpoint?, y ¿esas diferencias siguen la diferencia de energía DFT? Que dos entradas se distingan no significa que el modelo sepa cuál es más estable.

## Auditoría de datos y modelo

Se analizaron los 210 casos de desarrollo: 70 composiciones y 3 estructuras estáticas por composición. El test histórico estuvo excluido. En **70 de 70 composiciones**, los tres SLICES son diferentes; lo mismo ocurre para los tres hashes de input IDs. No hay colisión exacta de cadena/IDs entre los tres polimorfos estáticos en este subconjunto, ni composición con los tres strings iguales. El campo `topology_sha256` también se exportó, pero esta auditoría no ejecutó un algoritmo independiente de isomorfismo de grafos: por eso no declara equivalencia topológica a partir de la cadena solamente.

Los embeddings congelados son del checkpoint original unconditional, cargado con `strict=True`; SHA-256 **`4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`**. El modelo se mantuvo en eval con cero parámetros entrenables. El vector `<end>` tiene 768 dimensiones; también se calculó masked-mean excluyendo `<start>`, `<end>` y padding. Al añadir padding derecho, el máximo cambio del vector `<end>` fue 1.19e-06, prácticamente cero. No se añadió energía a los tokens.

La geometría auxiliar se calculó desde la celda y las posiciones estáticas: volumen/átomo, densidad, distribución de distancias periódicas y conteos por átomo bajo cortes de 2.5, 3.0, 3.5 y 4.0 Å. Esos descriptores no usan energía, fuerzas ni tensión. Las distancias y volúmenes no dependen del orden de sitios ni de una rotación rígida. Los cortes son descriptores exploratorios, no una definición química universal de coordinación.

## Relación entre distancia de representación y diferencia energética

En los **210 pares intracomposición** (tres pares por fórmula), las correlaciones Spearman exploratorias entre distancia y diferencia absoluta DFT fueron: `<end>` euclidiana ρ=0.068, p ingenuo=0.323; `<end>` coseno ρ=-0.048, p ingenuo=0.491; masked-mean euclidiana ρ=0.283, p ingenuo=3.21e-05; masked-mean coseno ρ=-0.278, p ingenuo=4.42e-05; descriptor geométrico L1 ρ=0.349, p ingenuo=2.07e-07.

Las correlaciones de masked-mean y geometría son positivas pero modestas; la de `<end>` es casi nula. Los p-valores son descriptivos sin corrección por multiplicidad ni ajuste por dependencia: tres pares de la misma composición no son observaciones independientes. No se interpretan como pruebas de significancia ni de desempeño predictivo. Una representación puede separar estructuras y aun así ordenar mal su energía.

## Cómo leer las figuras

- `d3_distincion_slices.png`: cuenta cuántas cadenas de SLICES diferentes hay en los tres polimorfos de cada composición. En este conjunto, las 70 tienen tres cadenas distintas.
- `d3_distancias_y_deltae.png`: cada punto es un par de polimorfos de la misma fórmula; eje x es diferencia DFT absoluta en meV/átomo. Los paneles muestran distancia de embeddings y una métrica L1 geométrica de features mixtas. La nota recuerda que 210 pares se agrupan en 70 composiciones.

## Conexión con E2-T1

E2-T1 ya auditó 12,486 frames de relajación en desarrollo: 4,033 secuencias de IDs únicas y 1,148 de 1,677 grupos repetidos con rango energético ≥1 meV/átomo. Ese resultado describe configuraciones intermedias; no se reproduce ni se extrapola a los estáticos. Aquí no se encontró colisión exacta entre cadenas estáticas de los tres polimorfos, pero aun así la similitud geométrica/energética de los frames de relajación puede crear colisiones a lo largo de las trayectorias.

## Interpretación y siguiente experimento

D3 descarta una explicación simple de “los estáticos tienen exactamente el mismo input SLICES”: los 3 polimorfos por composición tienen cadenas e IDs distintos. Sin embargo, los embeddings `<end>` preentrenados no muestran una relación apreciable con |ΔE| en este análisis, y las asociaciones masked-mean/geométricas son solo exploratorias. Esto deja abierta una limitación de pooling/objetivo/cabeza, y también la posibilidad de que geometría aporte señal adicional. **D3 por sí solo no justifica afirmar causalidad ni que D4 vaya a mejorar.** Sí ofrece fundamento razonable para diseñar D4 como comparación controlada, si D2 confirma primero que el objetivo energético no es la explicación principal.

## Referencias

- SLICES, representación periódica invertible e invariante: Xiao et al., *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7). El artículo define SLICES como codificación de composición y topología; D3 aquí prueba si las diferencias geométricas estáticas se reflejan en cadenas/embeddings del dataset.
- MatGPT: [código oficial](https://github.com/devalab/matgpt), commit local E0 `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- SOAP: [documentación DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html). SOAP se menciona como alternativa publicada de descriptor geométrico local; no fue usado aquí (DScribe no está instalado).
- MPtrj/CHGNet: [documentación del conjunto MPtrj](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj) y Deng et al. (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3). Contexto de potenciales que aprenden energía/fuerzas/tensión desde estructuras; no es comparación de rendimiento en esta auditoría.
- PyTorch: [mecánicas de Autograd e inference mode](https://docs.pytorch.org/docs/stable/notes/autograd). El modo inference evita autograd; los embeddings de D3 solo se usan para análisis, no para optimizar cabezas.
- MatFold, protocolo de partición por composición: Witman y Schindler, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D). Referencia metodológica; los folds utilizados provienen del manifiesto E1.3 del proyecto.

## Procedencia

Ver `d3_summary.json`, `static_slices_inventory.csv`, `collision_analysis.csv`, `frozen_embeddings.csv`, `embedding_pair_distances.csv` y `geometry_descriptors.csv`. Los resultados son del conjunto de desarrollo únicamente; validation histórica participó en preselecciones E1, lo que limita comparaciones confirmatorias.
