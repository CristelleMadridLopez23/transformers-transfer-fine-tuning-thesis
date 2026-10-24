# E2-T1 — auditoría de trayectorias de relajación

**Estado: LIMITED.** Decisión T2: **LIMITED**. La auditoría excluye test histórico antes de reconstruir y tokenizar frames.

## Cobertura y procedencia

- Frames relax en fuente total: 13,874.
- Frames development auditados: 12,486.
- Frames históricos de test excluidos antes de conversión: 1,388.
- Trayectorias de desarrollo: 210.
- Secuencias de IDs únicas: 4033 / 12486 (ratio 0.3230).
- Equivalencia de grafo reserializada: 12486 / 12486.
- Energía raw/cache: máximo |Δ| 8.88e-16 eV/átomo en 7734 frames coincidentes.
- Metadatos PBE en relax: True; pareo static/relax: PASS.
- Static/relax difieren en ISMEAR en 153 de 210 pares; el export no expone descomposición independiente de correcciones. No se combinan energías absolutas.

## Colisiones y candidatos

- Grupos de IDs repetidos: 1677; con rango ≥1 meV/átomo: 1148.
- Rango máximo de colisión: 213.507 meV/átomo.
- MAE mínima media dentro de grupos repetidos (mediana como predictor): 2.104 meV/átomo.
- Candidatos de secuencia única en desarrollo: 4033.
- Composiciones elegibles por fold: `{"0": 45, "1": 45, "2": 45, "3": 45, "4": 45}`.
- Fracción de grupos repetidos con rango ≥1 meV/átomo: 68.5%; umbral operativo de proyecto: 50%.
La cota de colisión corresponde a un predictor determinista que recibe solo IDs; no es un error de optimización. Las diferencias de energía se calculan dentro de cada relajación.

## Límites

T1 es una auditoría de datos y representabilidad, no una evaluación de rendimiento. Fuerzas y estrés se conservan como descriptores de los frames; no se usan como pérdidas del modelo textual. Los cambios geométricos que no cambian el grafo SLICES no son observables para MatGPT.
El test histórico se excluyó antes de conversión, selección y entrenamiento; tampoco se calcularon métricas supervisadas de test.

Artefactos: `frame_tokenization.csv`, `trajectory_manifest.csv`, `collision_summary.csv`, `trajectory_statistics.csv`, `selection_candidates.csv`, `metadata_compatibility.json` y `figures/`.
