# MatGPT F0 con Ridge M2: comparación controlada ABS₃

**Estado: PASS**


**Artefactos reutilizados:** ejecución validada en `experiments/matgpt_f0_stoichiometric_ridge_m2_comparison`; no se repitió entrenamiento para esta carpeta.

## Hechos verificados

- Cohorte: 70 fórmulas, 210 estructuras estáticas, tripletes completos, cinco folds; inner-train/validation/outer = 45/11/14; semillas 42, 123 y 2026. Los joins se hicieron por IDs, fórmula, familia, fold y semilla.
- Ridge M2: se reutilizaron los cinco pipelines congelados de 17 fracciones, SHA verificado, sin fit; alphas [0.01, 0.1, 0.1, 0.1, 0.1]. Las medias predichas coinciden con MatterGPT M2 y son comunes a los tres polimorfos.
- MatGPT checkpoint SHA-256 `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`; entrada `FINAL_END` (210×768). La caché reconstruida conserva checkpoint/dataset y pasa auditoría numérica, pero no es bit a bit idéntica a la histórica: drift máximo de vector 7.22e-06; se informa como limitación.
- Geometría estática `gga-static_endpoint_cleaned`, siete descriptores históricos, join por `structure_id`; StandardScaler se ajustó únicamente con 45 fórmulas / 135 estructuras inner-train por fold.
- Inicialización paso cero: las 15 parejas C0/C1 son equivalentes a tolerancia 1e-7 y los siete pesos geométricos empiezan en cero. Se entrenaron 30 cabezas; ambos backbones quedaron congelados. No se abrió el test histórico.

## Energía absoluta y métrica primaria ΔE

| Condición (M2) | MAE E/N A | RMSE global E/N A | MAE residual A | MAE ΔE firmado A | MAE ΔE anclado A | GS | Pairwise | no resueltos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MatGPT M2 SLICES | 52.478 | 68.933 | 34.221 | 59.600 | 41.760 | 0.522 (201) | 0.672 (591) | 5 |
| MatGPT M2 + geometría | 50.971 | 66.878 | 32.274 | 55.595 | 39.107 | 0.567 (201) | 0.694 (591) | 8 |
| MatterGPT M2 SLICES | 50.616 | 69.158 | 32.264 | 55.729 | 38.681 | 0.562 (201) | 0.648 (591) | 20 |
| MatterGPT M2 + geometría | 49.644 | 68.109 | 31.055 | 53.399 | 37.290 | 0.587 (201) | 0.657 (591) | 29 |

Estimando A promedia corridas por fórmula/semilla y semillas dentro de fórmula, luego macro-promedia las 70 fórmulas. El RMSE E/N global se calcula como raíz de la media de los errores cuadrados en 630 observaciones (no como promedio de RMSE por fórmula). El estimando B promedia primero las tres predicciones por estructura y aparece en `metricas_recalculadas_internas.csv`. ΔE primaria incluye los tres pares firmados i<j; el filtro |ΔE_DFT|>5 se usa solo para ranking; |ΔE_pred|≤1 se clasifica como no resuelto y es incorrecto. GS excluye gaps DFT≤1 y los empates predichos no son aciertos.

## Bootstrap apareado A

| Contraste (izquierda − derecha) | ΔMAE ΔE | IC95 % |
|---|---:|---:|
| RETRAINED_M2_C1 − RETRAINED_M2_C0 | -4.005 | [-6.311, -1.750] |
| MatterGPT_F0_geometry − MatterGPT_F0 | -2.330 | [-3.799, -0.932] |
| MatterGPT_F0 − RETRAINED_M2_C0 | -3.871 | [-9.497, 1.890] |
| MatterGPT_F0_geometry − RETRAINED_M2_C1 | -2.196 | [-7.879, 3.902] |

## Errores por polimorfo y tamaño de gap DFT

| Condición | Polimorfo | MAE E/N (meV/átomo) | n estructura-semilla |
|---|---|---:|---:|
| MatGPT M2 SLICES | perovskita | 60.932 | 210 |
| MatGPT M2 SLICES | hexagonal | 51.916 | 210 |
| MatGPT M2 SLICES | tipo aguja | 44.587 | 210 |
| MatGPT M2 + geometría | perovskita | 59.984 | 210 |
| MatGPT M2 + geometría | hexagonal | 49.446 | 210 |
| MatGPT M2 + geometría | tipo aguja | 43.484 | 210 |
| MatterGPT M2 SLICES | perovskita | 58.775 | 210 |
| MatterGPT M2 SLICES | hexagonal | 52.486 | 210 |
| MatterGPT M2 SLICES | tipo aguja | 40.587 | 210 |
| MatterGPT M2 + geometría | perovskita | 57.929 | 210 |
| MatterGPT M2 + geometría | hexagonal | 50.845 | 210 |
| MatterGPT M2 + geometría | tipo aguja | 40.160 | 210 |

Los conteos por intervalo son pares fórmula-semilla; la métrica es error absoluto ΔE por par, no por estructura.

| Condición | Intervalo de |ΔE_DFT| (meV/átomo) | n pares fórmula-semilla | MAE ΔE (meV/átomo) |
|---|---|---:|---:|
| MatGPT M2 SLICES | 0–5 | 39 | 34.177 |
| MatGPT M2 SLICES | 5–10 | 51 | 34.647 |
| MatGPT M2 SLICES | 10–25 | 123 | 46.477 |
| MatGPT M2 SLICES | 25–50 | 138 | 45.063 |
| MatGPT M2 SLICES | 50–100 | 168 | 55.911 |
| MatGPT M2 SLICES | >100 | 111 | 118.193 |
| MatGPT M2 + geometría | 0–5 | 39 | 32.577 |
| MatGPT M2 + geometría | 5–10 | 51 | 36.660 |
| MatGPT M2 + geometría | 10–25 | 123 | 42.229 |
| MatGPT M2 + geometría | 25–50 | 138 | 43.896 |
| MatGPT M2 + geometría | 50–100 | 168 | 52.141 |
| MatGPT M2 + geometría | >100 | 111 | 106.963 |

## Calibración OLS

| Condición | Pendiente E/N | R² E/N | Pendiente residuo | R² residuo | Pendiente ΔE firmado | Intercepto ΔE (meV/átomo) | R² OLS ΔE | R² predictivo ΔE | RMSE ΔE global (meV/átomo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MatGPT M2 SLICES | 0.995 | 0.993 | 0.330 | 0.223 | 0.301 | -11.45 | 0.193 | 0.132 | 79.902 |
| MatGPT M2 + geometría | 0.995 | 0.993 | 0.400 | 0.308 | 0.371 | -11.87 | 0.277 | 0.246 | 74.480 |
| MatterGPT M2 SLICES | 0.994 | 0.993 | 0.181 | 0.163 | 0.152 | -11.79 | 0.127 | 0.119 | 80.484 |
| MatterGPT M2 + geometría | 0.994 | 0.993 | 0.207 | 0.217 | 0.180 | -10.94 | 0.181 | 0.178 | 77.753 |

OLS de energía usa E_DFT como x y E_pred como y; el residuo se centra dentro de cada triplete. La pendiente ΔE usa pares firmados i<j y no es la pendiente de gaps anclados al mínimo DFT.

IC percentiles no ajustados, 10 000 remuestreos por 70 fórmulas, semillas agrupadas, bootstrap seed 20261102. El contraste primario predefinido es MatGPT C1−C0; los otros tres son secundarios. Al no calcular p-valores, no se mezclan IC sin ajustar con pruebas ajustadas. Estos intervalos describen OOF de desarrollo reutilizado, no generalización externa.

## Reconstrucción post hoc M1 → M2

Sustituir solo μ_M1 por μ_M2 conservando los residuos históricos dejó ΔE invariantes dentro de 1×10⁻⁶ eV/átomo: diferencia máxima por par 4.768e-07 eV/átomo. La reconstrucción M1→Ridge+residuo cerró con error máximo 2.384e-07 eV/átomo; la tolerancia cubre redondeo de valores float32 ya serializados en los CSV históricos. Por construcción, cambian MAE E/N y energías absolutas, pero no el ranking ni ΔE dentro de esa tolerancia. `reconstruccion_posthoc_m2.csv` preserva cada predicción/residuo original y nuevo.

| Condición histórica | MAE E/N con M1 | MAE E/N post hoc M2 | MAE ΔE |
|---|---:|---:|---:|
| SLICES | 183.164 | 52.234 | 58.596 |
| SLICES + geometría | 182.385 | 51.204 | 56.273 |

## Sobreajuste, polimorfos y calibración

La mediana de época de mejor validación fue C0=5 y C1=4; la mediana de época de parada fue 25 y 24. En el mejor punto de validación, la brecha media (validación completa menos promedio train de minilotes) fue C0=0.018 y C1=0.004 SmoothL1 normalizada. La brecha es menor en C1 en estas corridas, pero no demuestra por sí sola una reducción robusta del sobreajuste. MAE por polimorfo y gap se detallan arriba; la geometría baja el MAE E/N en los tres grupos, con mayor caída observada en hexagonal (2.470 meV/átomo) y menor en perovskita (0.948). No se infiere causalidad física.

En la paridad de pares firmados se reporta OLS separado del ajuste de gaps anclados al mínimo DFT. Pendientes lejos de 1 indican compresión/calibración imperfecta. La comparación entre extractores no aísla solo preentrenamiento: MatGPT y MatterGPT difieren en arquitectura y tokenización.

## Respuestas para la tesis

1. MAE E/N MatGPT F0 M2 (reentrenado): 52.478 meV/átomo.
2. MAE E/N MatGPT F0 + geometría M2: 50.971 meV/átomo.
3. La reconstrucción post hoc cambia solo μ y conserva residuos; el cambio M1→M2 es desplazamiento absoluto por fórmula.
4. Sí: sustituir la media común M1 por M2 deja ΔE invariante; se comprobó por fila/par.
5. El reentrenamiento sí puede cambiar ΔE: MAE A C0=59.600; C1=55.595 meV/átomo.
6–7. La geometría MatGPT cambia ΔMAE=-4.005 [-6.311, -1.750]; juzgar robustez por si el IC cruza cero.
8. MatterGPT M2 + geometría tiene menor estimación puntual de MAE ΔE que MatGPT M2 + geometría (53.399 vs 55.595 meV/átomo), pero el IC95 % apareado de la diferencia cruza cero.
9. MatGPT M2 + geometría obtiene el mayor pairwise (0.694/591); MatterGPT M2 + geometría logra el mayor GS (0.587/201).
10. La mejor pendiente OLS ΔE entre las cuatro condiciones es MatGPT M2 + geometría (0.371, R²=0.277); todas comprimen los gaps frente a la pendiente ideal 1. La tabla OLS separa ΔE firmada, residuo y energía absoluta.
11. Brecha media validación−train al mejor punto: C0=0.018, C1=0.004 SmoothL1 normalizada; su reducción en este desarrollo no constituye prueba independiente de menor sobreajuste.
12. Para la tesis, usar las cuatro condiciones M2 y el estimando A; citar M1 histórico solo como referencia y separar el control post hoc del M2 reentrenado.

## Artefactos

Predicciones OOF, pares, composición, polimorfo, gap, checkpoints, bootstrap, configuración y recursos están en este directorio. Las figuras PNG están en `images/matgpt_f0_ridge_m2_matched_mattergpt/`.
