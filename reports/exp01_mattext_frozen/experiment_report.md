# Experimento 01 — MatText congelado

## Objetivo científico

Evaluar energía GGA por átomo y selección polimórfica en composiciones no vistas, sin actualizar el encoder.

## Procedencia y auditoría de datos

480 cálculos originales ({'relax': 240, 'static': 240}); 225 filas provisionales y 75 composiciones con las tres familias. No constituyen una autorización de entrenamiento.

Total unavailable as independent source field; derived from published static E_per_at * atoms. E_relax never used as a total energy.

Initial prototype labels; not a claim of final relaxed space-group identity.

Las relajaciones no son muestras adicionales. Los filtros por grafos/candidatos incompletos se registran por ID. Asociaciones ambiguas: 1; ver energy_ambiguities.csv. No se cambió la tolerancia para admitir discrepancias.

## SLICES y tokenización

SLICES 2.0.12 oficial, econnn y relax_model=None. 225 conversiones sintácticas sin pérdida; API oficial verificada: True. Longitud máxima candidata 352; 0 colisiones candidatas de IDs entre polimorfos.

Vocabulario de Zenodo recuperado y hash verificado. No está demostrada su asignación de tokens especiales para este checkpoint. Ver audit/tokenizer_compatibility.md.

## MatText y extracción prevista

BertModel: 4 bloques, 8 cabezas, dimensión 512, FFN 3072, 30522 filas de embeddings, contexto 512. Hash del checkpoint verificado contra el manifiesto original. Código exige requires_grad=False, eval e inference_mode, cero parámetros entrenables. Mean excluye CLS/SEP y padding por máscaras de posición; CLS se conserva como alternativa.

Embeddings científicos producidos: False. No se usa el vocabulario candidato para extraer resultados científicos.

## Predictores y validación prevista

Ridge, Random Forest y MLP 512→32→1 (GELU/dropout, AdamW) sobre los mismos embeddings. Selección de pooling común por MAE de MLP en CV de desarrollo. Hiperparámetros por CV; normalización dentro de cada entrenamiento. MLP usa validación interna agrupada para early stopping y refit por el número de épocas elegido. El test no selecciona pooling, cabeza ni hiperparámetros.

Partición provisional: 60 composiciones de desarrollo / 15 de test, 4 pliegues; semilla 42. Se deberá regenerar/revalidar si cambia el conjunto admitido.

## Evaluación y estabilidad polimórfica

Se implementan MAE/RMSE (eV/átomo), diferencias entre polimorfos, concordancia por pares, selección del mínimo y regret (meV/átomo). Cuasidegeneración: 1 meV/átomo. Baseline de familia frecuente calculado solo con mínimos GGA del entrenamiento.

## Resultados cuantitativos

No hay métricas supervisadas ni trabajos enviados: los controles críticos no pasaron.

## Limitaciones e interpretación

El solapamiento con el corpus NOMAD de preentrenamiento es desconocido. Una energía absoluta precisa no demostraría sensibilidad geométrica ni ordenamiento polimórfico. Las familias corresponden al prototipo inicial y pueden cambiar durante relajación. Igual bolsa de tokens no implica igual entrada posicional.

## Estado y conclusiones

- Ambiguous static energy/structure associations; inspect energy_ambiguities.csv
- No checkpoint-specific original tokenizer artifact proves vocabulary IDs and CLS/SEP wrapping

Se detuvieron extracción, entrenamiento y envío a Slurm. No se generaron figuras de embeddings ni métricas ficticias.
