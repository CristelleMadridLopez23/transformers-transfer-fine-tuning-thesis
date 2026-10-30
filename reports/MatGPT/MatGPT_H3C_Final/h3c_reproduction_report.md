# Reproducción de H3-C

[HECHO VERIFICADO] Se recargaron cinco pipelines Ridge y quince checkpoints MLP8. La reproducción cubre 630 predicciones: 210 estructuras físicas, 70 fórmulas, cinco pliegues y tres semillas. El test histórico no se leyó.

[RESULTADO EXPERIMENTAL] Diferencia máxima frente al CSV histórico: **7.08e-07 eV/átomo** para la predicción final, 4.77e-07 para Ridge y 9.97e-17 para residuos. Estado: **PASS**. Las discrepancias son de precisión de serialización/aritmética float32; no se forzaron métricas.

[RESULTADO EXPERIMENTAL] Media sobre tres semillas: MAE E/N 183.30 meV/átomo y MAE de pares firmados 58.84 meV/átomo, cercanos a las referencias aproximadas 183.3 y 58.8. La exactitud de mínimo por `argmin` sobre las 70 fórmulas es 54.3 %, que reproduce 54.3 %. Con la regla histórica que excluye empates DFT de ≤1 meV/átomo y abstenciones del predictor, es 53.2 % sobre 67 fórmulas elegibles por semilla. El orden de pares >5 meV/átomo es 66.8 % sobre 197 pares elegibles por semilla; el protocolo histórico de 1 meV tiene otros denominadores.

[HECHO VERIFICADO] Hashes de dataset, pliegues, cache, checkpoint y predicciones constan en `reproduction/reproduction_manifest.json`. Las filas recalculadas están en `reproduction/replayed_predictions.csv`. El checkpoint y los datos originales se conservaron separados.

[LIMITACIÓN] El código histórico conserva rutas internas anteriores a una reorganización del repositorio. La reproducción de solo lectura usa rutas actuales explícitas y recarga los checkpoints originales; no modifica el código ni los pesos históricos. Los resultados de desarrollo no son una validación externa ciega.

