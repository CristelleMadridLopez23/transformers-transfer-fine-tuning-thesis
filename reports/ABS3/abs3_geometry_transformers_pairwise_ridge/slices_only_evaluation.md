# Evaluación SLICES-only: MatGPT frente a MatterGPT

## Protocolo y cohorte

Este resumen reutiliza las predicciones OOF de las condiciones `MAT` y `MATTER` del benchmark pairwise Ridge. Los extractores originales permanecen congelados; cada modelo recibe solo SLICES y su embedding contextualizado de 768 dimensiones. No se incluye geometría. La cohorte contiene 70 composiciones, 210 estructuras y tres polimorfos por composición; se usaron los cinco folds históricos agrupados por fórmula. Cada composición aparece una vez en OOF. Ridge pairwise determinista no se replicó artificialmente con semillas.

Para reconstruir energía absoluta se usó el Ridge M2 estequiométrico histórico, fijo y compartido. `ΔE` es la diferencia firmada `E_j − E_i` entre dos polimorfos de la misma fórmula. `ΔE MAE` es el estimando A: promedio de los tres errores absolutos por fórmula y luego macro-promedio sobre fórmulas. El RMSE indicado es global sobre observaciones, no el promedio de RMSE por fórmula. El bootstrap remuestrea composiciones completas (10 000 réplicas; semilla 20261103); los intervalos son descriptivos para OOF de desarrollo y no una evaluación de test externo.

## Resultados

| Métrica | MatGPT SLICES-only | MatterGPT SLICES-only |
|---|---:|---:|
| Energy MAE (meV/átomo) | 50.28 | 51.82 |
| Energy RMSE global (meV/átomo) | 66.22 | 66.80 |
| Sesgo energético (meV/átomo) | +1.63 | +1.63 |
| ΔE MAE firmado, estimando A (meV/átomo) | 54.80 | 55.77 |
| ΔE RMSE global (meV/átomo) | 72.70 | 74.26 |
| Pairwise accuracy elegible | 131/197 = 66.50% | 135/197 = 68.53% |
| Pairwise accuracy resuelto | 131/195 = 67.18% | 135/195 = 69.23% |
| Pares no resueltos | 2/197 = 1.02% | 2/197 = 1.02% |
| Tripletes con orden exacto | 18/66 = 27.27% | 25/66 = 37.88% |
| Selección del ground state | 34/67 = 50.75% | 37/67 = 55.22% |
| Pendiente OLS de ΔE; intercepto (meV/átomo) | 0.332; −7.72 | 0.317; −9.49 |
| R² descriptivo de ΔE | 0.281 | 0.250 |
| Spearman / Kendall medios por composición | 0.407 / 0.343 | 0.414 / 0.371 |

Un par es elegible para ranking si `|ΔE_DFT| > 5 meV/átomo`; las predicciones con `|ΔE_pred| ≤ 1 meV/átomo` cuentan como incorrectas. Ground state es elegible si el gap DFT al segundo polimorfo supera 1 meV/átomo. El orden completo excluye fórmulas con algún empate DFT de hasta 1 meV/átomo.

## Diferencia apareada

Convención: diferencia `MatterGPT − MatGPT`; valor positivo favorece a MatGPT en errores, y favorece a MatterGPT en exactitud.

| Contraste | Diferencia | IC95% bootstrap por composición |
|---|---:|---:|
| Energy MAE (meV/átomo) | +1.54 | [−0.48, 3.67] |
| Energy RMSE global (meV/átomo) | +0.58 | [−1.50, 2.72] |
| ΔE MAE (meV/átomo) | +0.97 | [−4.23, 6.33] |
| Pairwise accuracy (puntos porcentuales) | +2.03 | [−4.59, 8.50] |
| Triplete exacto (puntos porcentuales) | +10.61 | [−1.56, 23.19] |
| Ground state (puntos porcentuales) | +4.48 | [−5.97, 14.93] |

Para la exactitud pairwise individual, los IC95% descriptivos son 59.07–73.50% para MatGPT y 60.30–76.24% para MatterGPT. Ambos resultados quedan por debajo de la meta del 90%; ninguna condición la alcanza ni estos intervalos son compatibles con 90%. No hay evidencia bootstrap clara de una diferencia entre extractores en Energy MAE, ΔE MAE, ranking pairwise, triplete exacto o ground state: los intervalos apareados incluyen cero.

## Lectura

En este ajuste pairwise SLICES-only, MatGPT obtiene menores errores puntuales de energía absoluta y ΔE. MatterGPT obtiene exactitudes puntuales algo mayores para pares, orden completo y selección del estado fundamental. Los intervalos apareados son amplios y compatibles con ausencia de diferencia, así que las diferencias puntuales no establecen superioridad. Ambos comprimen los gaps (`pendiente OLS ≈ 0.32`, frente al ideal 1), y el ranking pairwise permanece lejos del objetivo del 90%.

Estos resultados corresponden a validación cruzada OOF sobre el conjunto de desarrollo y alpha seleccionado con inner-validation. No se consultó el test independiente. Las celdas y representaciones estructurales corresponden a la cohorte histórica; no se afirma generalización externa.

## Artefactos fuente

- `summary.json`: métricas OOF agregadas.
- `predicciones_oof_por_estructura.csv` y `predicciones_oof_por_par.csv`: predicciones usadas.
- `metricas_por_composicion.csv`: unidad de bootstrap.
- `slices_only_comparison.csv`: tabla de métricas de estas dos condiciones.
- `bootstrap_apareado.csv`: contraste `MATTER − MAT` añadido a los contrastes del benchmark.
- `resources.json`: coste del benchmark y del análisis SLICES-only.
