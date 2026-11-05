# Auditoría de suficiencia representacional ABS₃ — MatGPT vs MatterGPT

## Alcance y controles

Se reutilizaron las 70 fórmulas ABS₃, 210 estructuras estáticas, los cinco folds históricos 45/11/14 y cachés originales validadas de embeddings 210×768. MatGPT usó FINAL_END; MatterGPT usó el estado contextualizado terminal posterior a ln_f y conditioning [1.0, 2.0]. Ambos backbones permanecieron congelados. Geometría: siete descriptores de `gga-static_endpoint_cleaned`, con scaler ajustado por fold solo en inner-train. Ridge M2 histórico se mantuvo fijo para reconstrucción absoluta. No se consultó el test independiente.

MAE ΔE es estimando A: promedio de los tres errores firmados por fórmula y macro-promedio de 70 fórmulas. Energía absoluta se ancló a μ_c de Ridge M2 y scores centrados dentro de cada triplete. Ranking elegible: |ΔE_DFT|>5 meV/átomo; |ΔE_pred|≤1 se cuenta incorrecto. Estado fundamental elegible con gap al segundo mínimo >1 meV/átomo; empate predicho no es acierto. El orden completo excluye tripletes con algún par DFT ≤1 meV/átomo.

## Resultados de probes energéticos OOF

| Extractor | Probe | Energy MAE | ΔE MAE A | Pairwise | Estado fundamental | Triplete exacto | Pendiente OLS |
|---|---|---:|---:|---:|---:|---:|---:|
| GEO | Ridge | 47.63 | 50.03 | 133/197 (67.51%) | 41/67 (61.19%) | 19/66 (28.79%) | 0.305 |
| GEO | PLS_Ridge | 48.04 | 50.11 | 134/197 (68.02%) | 41/67 (61.19%) | 21/66 (31.82%) | 0.266 |
| GEO | KRR_RBF | 48.17 | 51.10 | 137/197 (69.54%) | 42/67 (62.69%) | 22/66 (33.33%) | 0.263 |
| MAT | Ridge | 50.26 | 54.38 | 135/197 (68.53%) | 39/67 (58.21%) | 23/66 (34.85%) | 0.344 |
| MAT | PLS_Ridge | 50.36 | 58.52 | 119/197 (60.41%) | 31/67 (46.27%) | 15/66 (22.73%) | 0.386 |
| MAT | KRR_RBF | 49.57 | 53.28 | 138/197 (70.05%) | 37/67 (55.22%) | 22/66 (33.33%) | 0.239 |
| MATTER | Ridge | 50.61 | 54.11 | 134/197 (68.02%) | 33/67 (49.25%) | 24/66 (36.36%) | 0.333 |
| MATTER | PLS_Ridge | 51.93 | 57.91 | 130/197 (65.99%) | 32/67 (47.76%) | 26/66 (39.39%) | 0.347 |
| MATTER | KRR_RBF | 49.53 | 51.69 | 138/197 (70.05%) | 38/67 (56.72%) | 31/66 (46.97%) | 0.260 |

Entre probes con embeddings, KRR logró ΔE MAE 53.28 para MatGPT y 51.69 para MatterGPT; sus MAE de energía absoluta fueron 49.57 y 49.53 meV/átomo. Los rankings observados siguen lejos de 90%: máximo 70.1%.

### Referencia histórica SLICES-only

El artefacto histórico `experiments/abs3_geometry_transformers_pairwise_ridge/slices_only_comparison.csv` reporta MatGPT: energía MAE 50.28, ΔE MAE 54.80 y ranking 66.50%; MatterGPT: 51.82, 55.77 y 68.53%, respectivamente. En esta auditoría, Ridge da MatGPT 50.26/54.38/68.53% y MatterGPT 50.61/54.11/68.02%. Los IDs, targets y los cinco folds OOF son los mismos. La diferencia de protocolo comprobada es que esta auditoría selecciona alpha en inner-validation y ajusta el probe final solo con las 45 fórmulas inner-train; el benchmark histórico vuelve a ajustar con inner-train + inner-validation después de seleccionar alpha. Por eso se reportan como referencias con distinto conjunto de ajuste, no como discrepancia de datos ni como comparación directa de superioridad.

Los intervalos bootstrap apareados por composición están en `bootstrap_paired.csv`. Los contrastes se interpretan como OOF de desarrollo, no como confirmación en test. La tabla de probes también incluye geometría sola como referencia bajo la misma implementación de scaler y splits.

En el kernel aditivo, frente a geometría KRR sola, el cambio de MAE ΔE fue -2.02 meV/átomo para MatGPT (IC95% [-6.50, 2.43]) y -3.90 para MatterGPT (IC95% [-8.71, 1.08]); ambos intervalos incluyen cero. En los controles con 100 permutaciones intrafórmula, el MAE ΔE medio fue 69.02 (Ridge) y 64.38 (KRR) para MatGPT, y 69.80 y 63.62 para MatterGPT. Son controles de ruptura de asociación, no estimaciones del rendimiento de otro modelo entrenable.

## Información estructural recuperable

La fracción de suma de cuadrados intracomposicional fue 0.523 en MatterGPT y 0.577 en MatGPT; el resto corresponde a variación entre medias de composición. Esto demuestra variación de embedding entre polimorfos, pero por sí solo no demuestra que la variación codifique la energía.

Las similitudes coseno y distancias normalizadas se describen en `structural_distances.csv`. La geometría recuperable por Ridge y KRR está desglosada por descriptor y fold en `geometry_probe_results.csv`; se incluye R² predictivo contra la media de train como referencia.

## Negative control y valor condicional

Se permutaron 100 veces los embeddings entre los tres polimorfos dentro de cada fórmula, sin cambiar DFT ni folds. Los resultados por fold/permutación están en `permutation_controls.csv`. La permutación destruye la asociación con la identidad del polimorfo sin borrar diferencias químicas compartidas por fórmula; no prueba ausencia absoluta de información.

El kernel aditivo `k_geo + η·k_embedding` eligió η por validación interna; configuraciones y predicciones están en `conditional_value.csv` y `conditional_value_predictions.csv`. El contraste frente a GEO se calcula por composición en `bootstrap_paired.csv`. La geometría proviene de celdas ya relajadas por DFT.

## Curvas few-shot

`learning_curves.csv` contiene 10 subconjuntos de inner-train para n=10,20,30 y el conjunto completo para n=45; la selección sigue siendo interna, y outer solo describe la curva predefinida. Los subconjuntos reutilizan la misma población outer y no constituyen ensayos independientes. No se extrapola más allá de 45 fórmulas.

## Clasificación según probes evaluados

- **Level 1 — información estructural recuperable:** PASS descriptivo en ambos extractores: hay dispersión intracomposicional, y la predicción OOF de algunos descriptores supera controles constantes. Esto no implica reconstrucción completa de estructura atómica.
- **Level 2 — información energética recuperable:** PARCIAL; Ridge/PLS/KRR recuperan señal sobre algunos controles, pero la calibración ΔE sigue comprimida y varía por fold. Las permutaciones y CIs delimitan la evidencia.
- **Level 3 — valor complementario:** INCIERTO; el kernel aditivo es exploratorio y su contraste apareado debe leerse con IC, no solo con estimaciones puntuales.
- **Level 4 — suficiencia práctica:** NO alcanzada; ninguna condición se aproxima a 90% de pairwise ranking con el protocolo OOF de desarrollo.

## Diagnóstico y límites

La evidencia observada no apunta a embeddings constantes: ambos retienen variación entre polimorfos. Aun así, la señal energética que estos probes recuperan sobre composiciones nuevas es limitada y las diferencias entre modelos/probes son inciertas. Esto es compatible con una mezcla de muestra pequeña (45 fórmulas inner-train), alta dimensionalidad, compresión energética y propiedades locales no explícitas. No identifica de forma causal cuál es el cuello de botella. No se puede concluir que falte toda información energética ni justificar fine-tuning solo a partir de probes lineales/no lineales seleccionados aquí.

Un siguiente paso razonable es probar estados contextualizados intermedios con el mismo Ridge/PLS/KRR y validación anidada; añadir descriptores locales requeriría un experimento nuevo y etiquetas/estructuras disponibles. Antes de abrir un test externo, fijar el probe y el protocolo con estos resultados de desarrollo.

## Artefactos y recursos

Los CSV conservan distancias, targets geométricos, selección PLS/KRR, curvas, controles, OOF por estructura/par y bootstrap por composición. `resources.json` registra CPU/hilo, duración y réplicas. Sin Transformers entrenados, sin MPS/CUDA y sin acceso a test.
