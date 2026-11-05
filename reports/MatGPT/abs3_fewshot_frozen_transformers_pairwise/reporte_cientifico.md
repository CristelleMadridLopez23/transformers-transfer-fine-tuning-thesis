# ABS₃ few-shot pairwise frozen Transformers — preflight bloqueado

## Estado

**BLOCKED_PRECHECK.** No se iniciaron entrenamientos, layer mixing ni evaluación OOF. Los dos checkpoints Alexandria F4 requeridos no están disponibles en el workspace ni en los objetos Git inspeccionados.

Los checkpoints buscados eran MatGPT F4-100 (`15ba8a05…31d657f`) y MatterGPT F4-71 (`99da365e…926ef3`). Las carpetas donde residían fueron eliminadas en la limpieza solicitada anteriormente. No los sustituí por modelos distintos ni inferí sus pesos desde las métricas publicadas.

## Fuentes que sí están disponibles

- MatGPT original: hash esperado verificado.
- MatterGPT original: hash esperado verificado.
- Cachés originales MatGPT y MatterGPT disponibles.
- Dataset, manifiesto de folds y geometría histórica presentes.
- Ridge M2: los cinco hashes esperados están recuperables de objetos Git; no se reentrenaron.
- Baselines congelados conservados en las carpetas registradas en `preflight.json`.

## Alcance bloqueado

Sin los dos checkpoints F4 no es posible extraer sus representaciones ni completar la comparación de cuatro extractores solicitada. Por tanto no ejecuté Ridge pairwise, siamese ΔE, siamese + ranking, bootstrap ni layer mixing. No se generaron métricas OOF ni figuras. El test histórico no se consultó.

Para reanudar la comparación completa, deben restaurarse los dos archivos exactos con los SHA-256 indicados en `preflight.json`; después se pueden regenerar únicamente sus cachés de embeddings y continuar con la auditoría de inicialización y entrenamiento.
