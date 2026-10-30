# D1 — ¿Cuánto explica la composición sin ver la estructura?

## En palabras sencillas

Ridge recibe únicamente la fórmula química. Para una fórmula dada, los tres polimorfos reciben exactamente las mismas features y, por ello, la misma predicción. El modelo puede acertar la escala absoluta al comparar una composición con otra, pero no tiene cómo decidir cuál de los tres polimorfos es más estable. Ese empate es la respuesta esperada de un baseline sin estructura.

## Datos y protocolo

Se usaron 210 energías estáticas DFT-PBE por átomo, de 70 composiciones de desarrollo con tres polimorfos cada una. Se conservaron los cinco folds agrupados por composición. Dentro de cada fold, Ridge eligió α entre (0.01, 0.1, 1, 10, 100) exclusivamente con inner-validation; el scaler se ajustó con inner-train. El Random Forest pequeño usó parámetros fijos (200 árboles, profundidad máxima 4, mínimo 3 muestras por hoja). La media de train predice la media de las etiquetas inner-train. No se incluyó el identificador de composición como feature y no se consultó el test histórico.

Features de composición: fracciones por elemento, número de elementos/átomos reducidos, estadísticos de número atómico y electronegatividad de Pauling disponibles. No se usaron energías, rankings, familia polimórfica, SLICES ni geometría como entrada.

## Resultados

| Baseline | MAE absoluto (meV/átomo), media ± DE entre folds | MAE de medias por fórmula (meV/átomo) | MAE ΔE (meV/átomo), media ± DE folds | R² absoluto medio por fold | GS / pairwise |
|---|---:|---:|---:|---:|---:|
| D1-A media train | 724.3 ± 83.0 | 724.3 | 44.20 ± 4.98 | -0.070 | 0.000 / 0.000 |
| D1-B Ridge composición | 59.3 ± 8.8 | 47.0 | 44.20 ± 4.98 | 0.989 | 0.000 / 0.000 |
| D1-C RandomForest pequeño | 318.9 ± 47.0 | 316.0 | 44.20 ± 4.98 | 0.722 | 0.000 / 0.000 |

El MAE absoluto de Ridge queda cerca de **59.3 meV/átomo**. Comparado por las mismas composiciones con E2-A partial_ft (predicción promediada entre sus tres semillas), la diferencia promedio Ridge − E2-A es **-668.6 meV/átomo**; IC bootstrap pareado 95% **[-763.5, -574.8]**, 2,000 remuestras de 70 composiciones, semilla 17042026. Negativo favorece Ridge en error absoluto. Este intervalo es exploratorio, no elimina que configuraciones MatGPT se preseleccionaron con la validation histórica.

La fracción descriptiva de varianza entre medias de composición es **99.6%**; la variación dentro de composición corresponde a **0.4%**. Cada composición conserva los tres valores DFT reales. Esta descomposición no es causal: la variación entre fórmulas y la estructura están relacionadas en estos 70 grupos.

Para el baseline constante y Ridge, el rango predicho dentro de cada fórmula es cero. Por eso su ranking polimórfico es no discriminativo: el MAE de ΔE (que resulta aproximadamente el rango DFT medio dividido por tres) no debe leerse como capacidad de seleccionar un estado fundamental. Los valores GS y pairwise se conservan en cero/abstención explícita según la regla y no se interpretan como éxitos.

## Cómo leer las figuras

- `d1_paridad_composicion.png`: cada punto es una estructura OOF; DFT está en x y predicción en y. La diagonal discontinua es el ideal; rosa es OLS con intercepto libre sobre todos los puntos mostrados. En cada panel se muestran N, m, b, R² y MAE.
- `d1_ridge_vs_matgpt.png`: compara las mismas 210 estructuras. La gran diferencia de error absoluto muestra que, en este conjunto, la fórmula informa mucho sobre el nivel de energía. No prueba que las features sean universales ni transfiere a otras composiciones.
- `d1_error_por_composicion.png`: cada caja resume 70 errores macro por composición; los puntos externos son fórmulas individuales.
- `d1_varianza_entre_dentro.png`: separa descriptivamente dispersión entre medias de fórmulas y desviaciones de los polimorfos respecto a su propia media.

**Lectura científica:** en energía absoluta este benchmark da una señal química fuerte; el mal resultado absoluto de E2-A no se explica diciendo simplemente que la composición carece de información. La pérdida/head/calibración de MatGPT requieren diagnóstico. D1 confirma que el ranking necesita una fuente de señal estructural adicional, porque el baseline químico empata cada triplete por construcción.

## Archivos reproducibles

OOF, métricas, features, elección de α, regresiones OLS y provenance se guardan en este directorio. El test histórico no se usó. Los folds y validation histórica sí fueron usados para investigación previa de candidatas, así que estas auditorías de desarrollo no eliminan el sesgo histórico de selección.
