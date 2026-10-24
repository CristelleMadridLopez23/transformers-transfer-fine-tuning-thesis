# MatGPT H3-TEST: evaluación reservada de H3-A

Este flujo puntúa una sola vez, después de bloquear modelo, entradas, configuración y reglas, el conjunto histórico de 24 estructuras/8 composiciones. Compara H3-A (media de energía R4 por composición más residuo estructural centrado de H2) con H2. No ajusta hiperparámetros ni selecciona modelos con el test.

El backbone MatGPT permanece congelado. Se usa el checkpoint unconditional histórico, tokenizador original y folds agrupados por composición ya fijados. La reproducción de H2 en OOF y de las cinco cabezas Ridge R4 contra predicciones archivadas es requisito para continuar. Los targets de test se leen solo en `evaluate_test.py`, después de verificar el hash del dataset, el lock, los hashes de todos los scripts y los archivos de predicción pre-etiquetas.

## Secuencia de ejecución

Desde la raíz del proyecto y con el entorno MatGPT activo:

```bash
PYTHON=/Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python
$PYTHON -m unittest discover -s Alternative2/scripts/MatGPT_frozen_3/H3_test/tests -v
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/audit_final_models.py
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/prepare_test_inputs.py
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/feature_extractor.py --device cpu --batch-size 4
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/load_frozen_ensemble.py --device cpu
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/predict_test.py --device cpu
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/evaluate_test.py
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/compare_with_development.py
$PYTHON -u Alternative2/scripts/MatGPT_frozen_3/H3_test/generate_figures.py
```

Las fases previas a `evaluate_test.py` no cargan columnas de etiquetas del test. `prepare_test_inputs.py` crea un lock de una sola vez y se niega a sobrescribirlo. Si cambia un script después del lock, la evaluación detiene el flujo antes de abrir las etiquetas; en ese caso se debe auditar y generar una nueva versión del protocolo, nunca editar silenciosamente el lock.

## Artefactos

Los resultados se guardan en `Alternative2/reports/MatGPT_frozen_3/H3_test/`: procedencia y lock; modelos Ridge reconstruidos; entradas y embeddings sin etiquetas; predicciones fold×semilla y ensemble congeladas; métricas; comparaciones desarrollo/test; y figuras PNG. Las salidas que contienen etiquetas se generan únicamente tras la evaluación final.

## Métricas y alcance

Las diferencias energéticas se expresan en meV/átomo. H3-A debe conservar exactamente los residuos y el ranking H2 al añadir una constante común a cada triplete. Por ello, sus métricas de GS, pairwise y ΔE son una comprobación algebraica, no una mejora aprendida de ranking. MAE/RMSE absolutos y error de media sí pueden cambiar. El test tiene solo ocho composiciones y varias decisiones históricas se motivaron con el conjunto de desarrollo; el resultado estima generalización al holdout reservado del protocolo, pero no es un estudio completamente independiente de todo el historial de investigación.

Referencias: [MatGPT](https://github.com/devalab/matgpt), código local fijado al commit histórico `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`; SLICES, Xiao et al., [Nature Communications 14, 7027 (2023)](https://doi.org/10.1038/s41467-023-42870-7); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
