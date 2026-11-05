# Auditoría MatGPT vs MatterGPT en ABS₃

**Estado: PASS_WITH_CORRECTIONS**

MatterGPT usa el checkpoint congelado SHA-256 `8d0071fa…a72b21`, tokenizador `b6a3fb75…e668e9` y condición fija `[1.0, 2.0]`. La geometría tardía conserva los siete descriptores de `gga-static_endpoint_cleaned`, join por `structure_id`; los dos manifiestos contienen los mismos valores (diferencia máxima 0) y documentan StandardScaler ajustado en 45 fórmulas / 135 estructuras inner-train.

## Hechos verificados

Las cuatro condiciones contienen 70 composiciones, 210 estructuras, tripletes completos, 5 folds y semillas 42/123/2026; el manifiesto confirma 45/11/14 fórmulas y la predicción outer corresponde al fold asignado. Join por `structure_id + formula + polymorph + fold + seed`; targets DFT, átomo, familia y asignaciones coinciden. En cada fuente se cumple E_pred = μ_Ridge + residuo. No se entrenaron modelos, no se modificaron checkpoints ni se leyó el test histórico.

## Discrepancia 52.02 vs 183.16 meV/átomo

La fuente MatGPT histórica usa M2 (Ridge de 17 fracciones estequiométricas); el estudio geométrico MatGPT usa M1 (Ridge aplicado a embeddings `L03_MEAN`). La reconstrucción directa de los cinco M1 sobre los embeddings guardados reproduce la media del triplete hasta 4.77×10⁻⁷ eV/átomo; M2 sobre las 17 fracciones reproduce las predicciones hasta 1.78×10⁻¹⁵ eV/átomo. Las medias por fórmula son diferentes, aunque cada Ridge es constante dentro del triplete. Por ello cambia el error absoluto y se cancela exactamente en ΔE. Las cabezas MLP8 son distintas y explican diferencias residuales pequeñas entre sus MAE ΔE: el cambio centrado por triplete entre MatGPT histórico y reciente SLICES tiene RMS 14.45 meV/átomo y máximo absoluto 54.52 meV/átomo; para la condición geométrica esos valores son 17.78 y 65.14 meV/átomo. Targets, cohorte, unidades y join no explican el salto. Ambos valores 52.02 y 183.16 corresponden a A. Como discrepancia histórica separada, el resumen original MatGPT reporta 50.479 (B: ensemble de semillas); sus métricas mezclan estimandos A y B. Se publican ambos para evitar confundirlas.

**A (primario):** corrida individual; se calcula cada error por fórmula/semilla, se promedian semillas dentro de fórmula y después las 70 fórmulas. **B:** se promedian predicciones por estructura entre semillas antes de medir. ΔE primaria son los tres pares firmados i<j.

## Bootstrap apareado de MAE ΔE A

| Contraste (izq. − der.) | Diferencia | IC95 % |
|---|---:|---:|
| MatterGPT F0 geometría − MatterGPT F0 SLICES | -2.330 | [-3.803, -0.961] |
| MatGPT F0 geometría − MatGPT F0 SLICES geom-study | -2.323 | [-5.085, 0.313] |
| MatterGPT F0 geometría − MatGPT F0 geometría | -2.874 | [-8.612, 3.021] |
| MatterGPT F0 geometría − Ridge-only M2 | -7.072 | [-11.902, -2.158] |

Bootstrap percentil por composición, 10 000 réplicas, semilla 20261030. MatterGPT geometría − MatterGPT SLICES reproduce el contraste predefinido. La evaluación usa OOF de desarrollo reutilizado y no representa test independiente.

## Interpretación

Las comparaciones geométricas C1/C2 dentro de cada extractor son coherentes respecto a su Ridge y targets; el punto estimado mejora en ambos. Para MatterGPT el IC95% excluye cero; para MatGPT el IC95% lo incluye. El mejor ranking pairwise A corresponde a MatGPT con geometría (0.687), mientras MatterGPT con geometría obtiene la mejor exactitud de ground state (0.587). El contraste MatGPT histórico vs MatGPT geométrico no es una comparación homogénea de energía absoluta debido a M2/M1. El orden ΔE sí es comparable porque el Ridge común por fórmula se cancela. MatterGPT tiene mejor estimación puntual de ΔE frente a MatGPT con geometría; el IC apareado incluye cero. En OLS de pares firmados, MatGPT con geometría tiene la pendiente más cercana a 1 (0.351; R²=0.300); MatterGPT con geometría da 0.180 (R²=0.236). Ambos comprimen la dispersión de ΔE y ninguna condición ofrece calibración ideal. Las épocas de validación mínimas medianas son 3→5 en MatGPT slices→geometría y 14→15 en MatterGPT; las 15 corridas de cada condición paran antes de 200 épocas. Las pérdidas de train registradas no son siempre comparables con la validación completa; no se deduce menor sobreajuste solo de esas medianas. Los errores por polimorfo y bins 0–5, 5–10, 10–25, 25–50, 50–100 y >100 meV/átomo se exportan en CSV; las tablas contienen los conteos. Los historiales disponibles se resumen por mínimo de validación en `auditoria_historiales.csv`. Para MatGPT F0 histórico no hay curva de entrenamiento F0 en el historial exportado (el F0 fue reutilizado); no se infiere su sobreajuste.

Errores por polimorfo y gaps DFT se desglosan en `metricas_por_polimorfo.csv` y `metricas_por_gap.csv`; los historiales disponibles se resumen en `auditoria_historiales.csv`. No hay curva F0 de entrenamiento MatGPT en el historial exportado, pues ese control se reutilizó; por eso su sobreajuste no puede medirse aquí.

El ranking se recalculó con 591 pares elegibles (|ΔE_DFT|>5 meV/átomo); ground state usa 201 casos fórmula-semilla elegibles (gap DFT>1). No se infiere causalidad física ni reducción de sobreajuste a partir del error OOF.
