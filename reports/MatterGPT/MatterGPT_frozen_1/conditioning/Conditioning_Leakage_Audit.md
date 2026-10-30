# Auditoría de fuga por condicionamiento

- El checkpoint requiere `num_props=2`; `prop=None` no se permite.
- Protocolo principal: una sola constante `[-1.0, 2.0]`, tomada del ejemplo público; se replica para todas las muestras y no pretende ser su e_form/gap real.
- La firma del extractor recibe `ids` y `condition`; no recibe DataFrame ni etiquetas. El cache de TEST se construyó desde `test_inputs.csv`, que solo contiene IDs, fórmula, familia, split y SLICES.
- Las energías DFT de desarrollo se usan únicamente en cabezas/splits de desarrollo. Las etiquetas TEST se abrieron después de `test_lock.json` y `predictions_prelabels.csv`.
- Sensibilidad diagnóstica a `[-2,1]` se mide en representaciones y no se usa para seleccionar protocolo.
- `model.py` antepone la proyección de propiedad en posición interna 0; se excluye del pooling. El cuarto retorno `embed` ocurre antes de `self.blocks`, por lo que no se usa como hidden final. Los hooks capturan salidas de los bloques originales.
- Limitación de procedencia: el checkout local contiene los archivos públicos del checkpoint y el ejemplo de inferencia, pero no el loader de teacher forcing del repo de entrenamiento; `>` está explícitamente respaldado por el ejemplo público y `<` por el tokenizador/decodificador local. La serialización de cierre no puede verificarse contra un loader de entrenamiento local independiente; por eso se conserva como limitación y los resultados no se presentan como reproducción bit a bit del entrenamiento original.
