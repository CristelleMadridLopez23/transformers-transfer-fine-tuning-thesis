# Normas operativas del proyecto

- Respetar la estructura de carpetas. No reorganizar, mover ni borrar datos sin autorización.
- Organizar cada experimento mediante un identificador único.
- Organizar el proyecto en `data/`, `embeddings/`, `models/`, `reports/`, `scripts/`, `images/` y `experiments/`.
- Guardar scripts Python en `scripts/<familia>/<experimento>/`. El proyecto corre localmente: no crear ni conservar lanzadores `.sh` o `.sbatch`.
- Guardar figuras PNG en `images/<experimento>/`, informes científicos en `reports/<familia>/<experimento>/` y tablas, manifiestos, métricas, logs y checkpoints generados en `experiments/<experimento>/`.
- Mantener checkpoints originales separados de artefactos entrenados; nunca modificar estructuras ni valores DFT originales.
- No ejecutar procesos pesados ni entrenamientos GPU en nodos de login.
- Documentar versiones, semillas, configuraciones, procedencia y resultados reales.
- Evitar fuga entre entrenamiento, validación y prueba. Agrupar polimorfos por composición.
- Exigir auditorías satisfactorias antes del envío de experimentos. Detener fases dependientes ante controles críticos fallidos.
- Nunca reemplazar resultados reales con datos sintéticos sin identificarlos explícitamente. Nunca inventar métricas, archivos de datos, configuraciones usadas ni resultados.
- Cada script Python debe incluir docstring de módulo, funciones documentadas con entradas/tipos/salidas, type hints, validación de argumentos, logging estructurado, `main()` protegido y rutas configurables relativas al proyecto. Registrar errores descriptivos y comentar transformaciones científicas.
- Preservar cambios del usuario; no usar `git reset`, `git clean` ni operaciones destructivas.

## Corrección de artefactos del mismo experimento

- Mantener una sola versión canónica de los resultados por experimento. No crear una carpeta de versión nueva solo para corregir errores de código, formato, análisis o renderizado dentro del mismo experimento.
- Si durante el trabajo se generan artefactos incompletos o erróneos y luego se corrigen, actualizar la salida canónica y retirar las versiones fallidas creadas por el agente en ese mismo trabajo. Esta regla autoriza borrar esos artefactos propios fallidos, aunque la norma general prohíba borrar datos sin autorización.
- Conservar resultados previos válidos, históricos o creados por el usuario. Crear una nueva versión únicamente cuando cambie intencionalmente el protocolo, los datos, el checkpoint o la pregunta experimental, o cuando el usuario solicite conservar revisiones.

# Presentación de figuras científicas

Para cada figura nueva o revisada, sigue los ejemplos visuales de `reports/` y
estas reglas. La figura debe poder interpretarse sin depender de explicaciones
verbales ni de leer el código que la generó. Escribe en español títulos,
subtítulos, ejes, leyendas, barras de color y notas; conserva siglas científicas
cuando sean estándar y define las que puedan resultar ambiguas.

## Títulos, paneles y observaciones

- Usa un título informativo que nombre el material o conjunto, la propiedad y
  la referencia científica usada. Por ejemplo: «ABS₃: ΔE respecto al estado
  fundamental DFT». No presentes «ΔE» sin definir desde qué estado o valor se
  calcula.
- Si comparas modelos, checkpoints, estrategias o conjuntos, usa paneles con
  títulos identificables. Mantén los mismos límites, unidades, normalización y
  definición del eje en los paneles diseñados para comparación. Indica si los
  paneles usan las mismas particiones, pliegues, estructuras y observaciones.
- Declara qué representa cada punto, barra, celda o resumen. En datos de
  polimorfos, indica si un marcador corresponde a una estructura estática, a
  una fórmula, a un promedio por fórmula o a otra unidad de análisis.
- Incluye los tamaños de muestra pertinentes: número de estructuras,
  composiciones o fórmulas, polimorfos, configuraciones, pares evaluados y
  pliegues, según corresponda. Si hay varias unidades de conteo, no las mezcles:
  nómbralas y reporta cada una por separado.
- Indica la partición (entrenamiento, validación o prueba), pliegues usados,
  denominador y ponderación de cada agregado. Aclara si cada observación pesa
  igual o si se promedian primero estructuras dentro de cada composición.
- Registra las exclusiones relevantes en una nota legible: pliegues omitidos,
  registros rechazados, valores no definidos, empates, datos faltantes,
  filtros de calidad y observaciones fuera de escala. No ocultes colas ni
  puntos extremos; si se omiten, indica cuántos y por qué.

## Color, marcadores y líneas

- Mantén colores de modelo consistentes entre figuras del mismo experimento.
  Distingue familias, particiones, modelos y categorías con codificaciones
  visuales claras. La leyenda debe definir cada color, forma de marcador, línea,
  línea de referencia, subconjunto, agregado y barra de incertidumbre visibles.
- En gráficos de polimorfos, define explícitamente cada forma de marcador
  (por ejemplo, perovskita distorsionada, hexagonal o tipo aguja) y cada color.
  Indica si la forma identifica la familia estructural y qué variable determina
  el color.
- Toda dispersión codificada por color y todo mapa de calor debe tener una
  barra de color rotulada con la propiedad, su referencia y unidades; si es
  adimensional, define su significado. Usa una normalización compartida entre
  paneles que se comparan e informa saturación, recorte, escala o valores fuera
  de rango.
- Mantén leyendas, notas, etiquetas y barras de color separadas de los datos y
  entre sí. Evita que la nota tape observaciones o que la leyenda oculte una
  región relevante.

## Energías relativas y estado fundamental

- Para una energía relativa, escribe la definición exacta en el eje o en la
  nota. Por ejemplo, ΔEᵢ = Eᵢ − E_min,DFT expresa la energía de una estructura
  por encima del polimorfo de menor energía DFT de la misma composición.
  Especifica si tanto la referencia como las predicciones se anclan al mínimo
  DFT o si cada una usa su propio mínimo.
- Explica el significado de signos, ceros y empates. Si una predicción negativa
  indica inversión del orden respecto al estado fundamental DFT, dilo en la
  nota; no dejes esa interpretación implícita.
- No mezcles estructuras de composiciones distintas al formar diferencias,
  ordenar polimorfos o calcular exactitud de selección del estado fundamental.
  Declara qué composiciones son elegibles y cuántas tienen al menos dos
  polimorfos comparables.

## Paridad entre predicción y DFT

- En cada gráfico de paridad, coloca la referencia DFT en x y la predicción del
  modelo en y. Usa los mismos límites en x e y y entre paneles comparables.
- Dibuja y etiqueta la relación ideal y = x. Añade un ajuste lineal por mínimos
  cuadrados ordinarios (OLS) con
  intercepto libre, y = m x + b, calculado con todas las observaciones que se
  muestran en ese panel. No ajustes la recta con una muestra distinta de la
  representada.
- Anota para cada panel N, pendiente m (adimensional), intercepto b con sus
  unidades, R² del ajuste y MAE contra DFT con unidades. Define la población
  usada para estos estadísticos (por ejemplo, observaciones del test mostrado)
  y exporta las estadísticas de regresión a un archivo tabular o JSON.
- Si representas ΔE, especifica el estado de referencia en ambos ejes y en el
  ajuste. Para otras comparaciones de igualdad, describe la relación de
  referencia real en vez de asumir que siempre es y = x.

## Curvas de entrenamiento y agregados

- Separa promedios de minilotes de entrenamiento, calculados durante la
  optimización, de métricas de validación completa calculadas con el checkpoint
  promedio móvil exponencial (EMA) cuando corresponda. Nombra el método de
  ponderación y el conjunto al que pertenece cada curva.
- Define el contador de época local a cada etapa, marcadores de picos,
  puntos de control seleccionados, parada temprana e intervalos de incertidumbre si
  aparecen. Anota el criterio de selección del checkpoint.
- Para histogramas, indica normalización, intervalos comunes y transformaciones.
  Para diagramas de caja o violines, define mediana, cuartiles, bigotes, puntos y unidad
  estadística; indica si el resumen es por estructura o por composición.
- En matrices de confusión, define qué significan filas y columnas, umbrales de
  clase, conteos y porcentajes, y TP/TN/FP/FN cuando corresponda.

## Exportación y revisión

- Incluye una nota breve dentro o debajo de cada figura con tamaño de muestra,
  partición, pliegues, unidad de observación, denominadores, definiciones de
  referencia y límites de interpretación necesarios para leerla correctamente.
- Inspecciona visualmente los PNG renderizados antes de entregar la figura y
  corrige texto cortado, solapamientos, símbolos ilegibles, contraste pobre o
  elementos que cubran los datos.
- Preserva las figuras históricas de experimentos anteriores. Para revisiones
  científicas intencionales de resultados válidos, usa un directorio de versión
  nuevo; para corregir errores del mismo experimento, conserva una salida
  canónica según la regla anterior. Exporta solo PNG; no generes PDF ni archivos
  comprimidos.

## Tiempo y recursos
- Para todos los experimentos, guardar en `resources.json` o un manifiesto equivalente el tiempo real transcurrido hasta terminar y los recursos realmente utilizados durante la corrida: dispositivo, tiempo de CPU, número de hilos, memoria máxima y uso de GPU/MPS cuando esté disponible. No estimar ni inventar valores; registrar como no disponible lo que no pueda medirse.
- Para todos los experimentos, registrar las métricas de evaluación utilizadas junto con su definición operativa, unidades, partición, denominador, nivel de agregación, umbrales y método de selección del checkpoint/hiperparámetros. Mantener estas definiciones en `config.yaml`, `metrics.json`, el manifiesto o el reporte del experimento, y asegurar que las figuras las identifiquen claramente.
