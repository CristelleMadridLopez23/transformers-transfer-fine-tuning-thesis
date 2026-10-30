# MatterGPT congelado: auditoría y transferencia

Todo el código exclusivo de MatterGPT está en esta carpeta; los resultados están en `Alternative2/reports/MatterGPT/MatterGPT_frozen_1/`. No se movió ni alteró ningún artefacto histórico de MatGPT. Las funciones comunes de energía se mantienen fuera de esta carpeta cuando pueden servir a cualquier encoder.

El protocolo condiciona siempre el modelo con `[-1.0, 2.0]`, tomado del ejemplo público. Son constantes experimentales, no propiedades de ABS₃. El target es la energía DFT estática total por átomo. Los parámetros de MatterGPT nunca se entrenan. El test histórico solo se abre después de guardar `test/test_lock.json` y `test/predictions_prelabels.csv`.

## Orden de ejecución

Desde la raíz del repositorio y con el entorno del proyecto:

```bash
python -m unittest discover -s Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/tests -v
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/audit.py --phase preflight
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/audit.py --phase extract
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/train_cv.py --models M0 M2 M4
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/recombine_h3a.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/paired_bootstrap.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/lock_test.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/evaluate_locked_test.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/supplement_test_metrics.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/plot_results.py
python -u Alternative2/scripts/MatterGPT/MatterGPT_frozen_1/write_report.py
```

Los scripts comprueban hashes y dejan evidencia bajo `reports/MatterGPT/MatterGPT_frozen_1`. Una etapa dependiente se detiene si falla la integridad de los datos, el tokenizer, el checkpoint o las predicciones bloqueadas.

## Reuso y citas

`Alternative2/scripts/energy_metrics.py` contiene solo métricas de energía y ranking independientes del encoder. Los extractores, tokenizador, condicionamiento y cabezas específicos permanecen aquí.

Fuentes: [MatterGPT (arXiv:2408.07608)](https://arxiv.org/abs/2408.07608), [checkpoint MatterGPT](https://huggingface.co/xiaohang07/MatterGPT), [SLICES](https://doi.org/10.1038/s41467-023-42870-7), [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [MatFold](https://doi.org/10.1039/D4DD00250D).
