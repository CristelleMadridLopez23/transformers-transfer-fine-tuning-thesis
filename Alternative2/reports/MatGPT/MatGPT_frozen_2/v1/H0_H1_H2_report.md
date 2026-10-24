# MatGPT congelado H0/H1/H2 — resultados de desarrollo
## Alcance y limitaciones
Se usaron 210 estructuras estáticas GGA/PBE (70 composiciones, tres polimorfos) en cinco folds agrupados; 3 semillas por modelo. Las energías son totales estáticas DFT por átomo, no energías de formación. El backbone MatGPT original se mantuvo congelado. No se usaron estructuras, etiquetas ni embeddings del test histórico. Las configuraciones y pooling previos se eligieron con parte de la validation histórica; por tanto, estos intervalos son exploratorios. H1/H2 requieren conocer las tres estructuras candidatas de una composición al inferir.
## Resultados macro OOF por configuración
La tabla siguiente resume las 15 combinaciones fold×semilla por modelo (cada combinación evalúa 14 fórmulas). Para resultados principales de composición, matrices y bootstrap, primero se promediaron las energías OOF de las tres semillas por estructura y después se recalcularon las métricas por fórmula.
| Modelo | MAE absoluta (meV/átomo) | MAE ΔE anclada DFT-GS | MAE ΔE own-min legacy | GS (denom.) | Pairwise (denom.) | Error de media | Residuo centrado |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 729.89 | 39.55 | 37.01 | 52.2% (67 fórmulas; 3 semillas) | 66.3% (618 pares elegibles×semilla) | 729.43 | 32.47 |
| H1 | 550.53 | 39.63 | 37.22 | 60.2% (67 fórmulas; 3 semillas) | 69.4% (618 pares elegibles×semilla) | 549.96 | 31.96 |
| H2 | 354.51 | 39.38 | 36.67 | 59.7% (67 fórmulas; 3 semillas) | 67.6% (618 pares elegibles×semilla) | 352.92 | 32.09 |

### Métricas principales tras promediar semillas por estructura
Cada fórmula física aparece una sola vez. GS excluye fórmulas con brecha DFT ≤1 meV/átomo y empates predichos son abstenciones; pairwise excluye pares DFT empatados.
| Modelo | MAE absoluta | ΔE anclada | ΔE own-min | GS estricta | Pairwise | Error media | MAE residual centrado |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 729.08 meV/átomo | 37.99 meV/átomo | 35.25 meV/átomo | 56.7% (67/67) | 67.5% (139/206) | 729.08 meV/átomo | 30.63 meV/átomo |
| H1 | 548.47 meV/átomo | 39.45 meV/átomo | 37.07 meV/átomo | 58.2% (67/67) | 68.4% (141/206) | 548.11 meV/átomo | 31.85 meV/átomo |
| H2 | 352.53 meV/átomo | 39.28 meV/átomo | 36.49 meV/átomo | 59.7% (67/67) | 67.5% (139/206) | 351.29 meV/átomo | 32.01 meV/átomo |

## H0: identidad de error cuadrático
MSE total macro = 696255.718 meV²/átomo²; componente del error de media = 694466.306 (99.74%); componente residual centrado = 1789.411 (0.26%); discrepancia de identidad = 2.75e-10. Esta igualdad vale para MSE; los dos MAE diagnósticos no se suman para formar el MAE absoluto. Se promediaron semillas por ID antes de esta descomposición OOF.
MAE del error de media = 729.08 meV/átomo; MAE del residuo estructural centrado = 30.63 meV/átomo. Los mayores errores individuales están en `largest_structure_errors.csv`.
El diagnóstico de pares encontró 4 pares físicos DFT empatados a ±1 meV/átomo en el ensamble por composición; el listado por familias/modelos está en `near_degenerate_pairs.csv`.
## Incertidumbre y comparación pareada
Las diferencias H1−H0, H2−H0 y H2−H1 se remuestrean por composición después de promediar semillas dentro de cada composición. Un valor negativo favorece al candidato para errores; positivo favorece al candidato para GS/pairwise.
| Comparación pareada | MAE abs Δ (IC95%) | ΔE anclada Δ (IC95%) | GS Δ (IC95%) | Pairwise Δ (IC95%) |
|---|---:|---:|---:|---:|
| H1−H0 | -180.615 [-275.434, -79.718] meV/átomo | 1.466 [-0.446, 3.494] meV/átomo | 1.429 [-4.615, 8.824] puntos % | 0.238 [-3.846, 5.340] puntos % |
| H2−H0 | -376.558 [-473.126, -278.447] meV/átomo | 1.291 [-1.637, 3.953] meV/átomo | 2.857 [-3.252, 9.539] puntos % | -0.476 [-4.808, 4.808] puntos % |
| H2−H1 | -195.943 [-271.069, -125.798] meV/átomo | -0.176 [-2.215, 1.744] meV/átomo | 1.429 [-4.545, 7.812] puntos % | -0.714 [-4.831, 2.899] puntos % |

Los IC son exploratorios; 2,000 remuestras mantienen juntas las tres estructuras de cada fórmula y no tratan semillas como muestras químicas independientes. Medias y dispersión fold/semilla están en `metrics_per_fold_seed.csv` y `metrics_per_seed.csv`.
## Cómo leer las figuras
La paridad DFT-GS anclada permite predicciones relativas negativas, señal de inversión respecto al GS DFT. La gráfica own-min usa el mínimo propio de cada eje y oculta inversiones. R² OLS corresponde a una recta con intercepto libre; R² predictivo evalúa la diagonal y=x. En matrices, cada fórmula cuenta una vez tras promediar semillas; las matrices pairwise muestran empates diagnósticos, mientras que la exactitud primaria excluye DFT ties. El scorecard usa metas heredadas aspiracionales, no calibradas para MatGPT.
## Control externo D1: Ridge composition-only
La auditoría de IDs y outer folds coincide exactamente. Ridge tiene MAE absoluta macro 59.29 meV/átomo (pooled 59.29). Las tres predicciones de cada fórmula son constantes, de modo que el ranking queda indeterminado; no se le atribuye una tasa de acierto GS. Ridge no forma parte de H0/H1/H2.

## Interpretación por arquitectura
- **H0:** MAE absoluta macro 729.9 meV/átomo; error de media 729.4; MAE residual centrado 32.5; ΔE anclada 39.5.
- **H1:** MAE absoluta macro 550.5 meV/átomo; error de media 550.0; MAE residual centrado 32.0; ΔE anclada 39.6.
- **H2:** MAE absoluta macro 354.5 meV/átomo; error de media 352.9; MAE residual centrado 32.1; ΔE anclada 39.4.
- **Efecto de H1/H2:** frente a H0, H1 reduce el MAE absoluta 180.6 meV/átomo en el ensamble OOF de semillas, sobre todo por mejorar la media composicional. H2 reduce 195.9 meV/átomo adicionales respecto a H1; su diferencia de pooling afecta la cabeza de media y el cambio de residuo centrado es pequeño.
- **Ranking:** GS del ensamble con 67 fórmulas elegibles (3 DFT casi degeneradas excluidas) H0=56.7%, H1=58.2%, H2=59.7%; exactitud pairwise agrupada sobre 206 pares DFT no empatados H0=67.5%, H1=68.4%, H2=67.5%. Las diferencias pairwise y GS tienen intervalos que incluyen cero.

No se atribuye un fallo exclusivamente al embedding: la cabeza, el objetivo y el pequeño tamaño de entrenamiento también afectan el resultado. Ridge solo es una referencia externa para MAE absoluta; no forma parte de H0/H1/H2 ni proporciona ranking.
## Referencias
- MatGPT, commit histórico: https://github.com/devalab/matgpt
- SLICES: Xiao et al. (2023), https://doi.org/10.1038/s41467-023-42870-7
- MatFold: Witman & Schindler (2025), https://doi.org/10.1039/D4DD00250D
- PBE: Perdew, Burke & Ernzerhof (1996), https://doi.org/10.1103/PhysRevLett.77.3865
- PyTorch SmoothL1Loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
- scikit-learn confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- pymatgen ComputedStructureEntry: https://pymatgen.org/pymatgen.core.html
