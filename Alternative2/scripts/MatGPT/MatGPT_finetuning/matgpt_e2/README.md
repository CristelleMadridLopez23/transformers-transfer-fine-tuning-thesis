# MatGPT E2-A

E2-A compara el baseline congelado y el ajuste exclusivo del último bloque del decoder MatGPT, usando los folds E1.3, agrupación por composición y semillas `[42, 123, 2026]`. Se mantienen MLP8, la pérdida Huber absoluta más pairwise (`λ=1`), escalas calculadas solo en inner-train y selección por objetivo de validation interna. El protocolo efectivo deriva de E1.3: 200 épocas, paciencia 20, 8 composiciones por lote, AdamW y weight decay 1e-4; backbone LR 1e-5 para partial-FT. No se utiliza el test histórico.

Los comandos deben ejecutarse desde la raíz del repositorio. Ejecuta preflight, smoke y piloto antes de lanzar CV. El preflight valida checkpoint, código y folds; smoke ejecuta unos pasos y audita gradientes; pilot entrena un fold/semilla; CV completa 5×3 por modelo. Para comparar de forma pareada, el baseline congelado se vuelve a entrenar con el protocolo E2.

Cada época registra segundos de pared (incluye validation interna), cada ajuste registra `training_wall_seconds`, y cada fase escribe `runtime_summary.json` con tiempo total y tiempos por ajuste. En CUDA/MPS se sincroniza el dispositivo para que la medición incluya operaciones asíncronas. Los resultados se separan por dispositivo (`pilot/cpu`, `runs/cpu`, etc.).

La corrección del pooling de `<end>` escribe la nueva corrida en carpetas `*_endtoken_v2`, separadas de resultados previos. Smoke guarda `representation_integrity.json` con posiciones de `<end>`, diversidad de embeddings/predicciones en tres SLICES, invariancia al padding derecho y normas de gradiente/cambio de pesos.

MatGPT commit E0 fijado: `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`. Referencias: [MatGPT](https://github.com/devalab/matgpt), [SLICES](https://doi.org/10.1038/s41467-023-42870-7), [MatFold](https://doi.org/10.1039/D4DD00250D), [PyTorch autograd](https://docs.pytorch.org/docs/stable/notes/autograd).
