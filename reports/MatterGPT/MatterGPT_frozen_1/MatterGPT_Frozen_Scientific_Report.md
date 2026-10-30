# MatterGPT congelado como extractor: evaluación ABS₃

## Resumen

**Estado del estudio: PARTIAL.** La extracción/carga, CV y evaluación bloqueada se completaron, pero el checkout local no incluye el loader de teacher forcing del entrenamiento original. El cierre `<` se respalda por la tokenización/decodificación pública, no por una reproducción independiente del dataset de entrenamiento. Además, los BOS/EOS IDs de `config.json` no coinciden con los IDs textuales reconstruidos. Por eso los valores describen explícitamente el protocolo de serialización `> SLICES <` y no una réplica confirmada bit a bit del pretraining.

El checkpoint se cargó con `strict=True`, SHA-256 `8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21`, 85,567,488 parámetros y cero parámetros entrenables. La condición fue fija `[-1.0, 2.0]`, copiada del ejemplo público de inferencia, para toda estructura. No describe las propiedades reales de ninguna estructura y no es MatterGPT incondicional. El cache se extrajo en CPU, con el backbone en `eval()` y `torch.inference_mode()`.

Se completaron M0, M2 y M4 en cinco folds por composición y tres semillas (45 ejecuciones de cabezas). M3-A se construyó sin reentrenar: media composicional M4 más residuo centrado M2. Se evaluaron 210 estructuras/70 composiciones OOF y después se abrió una vez el test bloqueado de 24 estructuras/8 composiciones.

## Datos, arquitectura y procedencia

- Objetivo: energía total estática GGA/PBE por átomo (eV/átomo); métricas en meV/átomo.
- Desarrollo: 210 estructuras, 70 composiciones; cada triplete completo queda dentro de un fold. El manifiesto aporta 5×14 composiciones externas.
- Test: 24 estructuras, 8 composiciones; IDs y SLICES proceden de `test_inputs.csv`, sin etiquetas, antes del lock.
- Modelo: 12 bloques, hidden 768, 2 condiciones, block size 397.
- Poolings: BLOCK03_MEAN (raw/pre-ln_f), FINAL_MEAN (post-ln_f sobre SLICES), FINAL_END (estado post-ln_f en token de cierre `<`). Se excluyó el prefijo de condición del pooling.
- M0: MLP8 absoluto en FINAL_END, Huber normalizada + término pareado; M2: cabeza MLP8 para media sobre FINAL_MEAN y MLP8 de residuo en FINAL_END; M4: StandardScaler+Ridge, capa/pooling y alpha elegidos solo en inner-validation. M3-A conserva los residuos M2 y sustituye solo la media por la media de Ridge M4.
- Cada normalizador y early stopping de MLP se ajustó dentro de outer-training. M4 selecciona entre alphas `[0.01, 0.1, 1, 10, 100, 1000]` y lecturas predefinidas. Se usó solver LSQR (`tol=1e-8`) y suma fila a fila para evitar avisos falsos de matmul en Accelerate; se verificaron predicciones finitas.

## Auditoría del tokenizador y condicionamiento

El vocabulario reconstruido de `Voc_prior` tiene 132 tokens y 0 estructuras OOV. Longitudes SLICES: mínimo 110, mediana 164, máximo 254; quedan bajo block size. La entrada usa el prefijo público `>` y añade el marcador real `<` como cierre, siguiendo el tokenizador local (que elimina `<` al decodificar). El `config.json` declara BOS/EOS IDs 130/131, que no corresponden al `SimpleTokenizer` ordenado (prefijo/cierre tienen IDs 39/38); el forward no usa esos metadatos. Se usa el mapeo textual real y esta discrepancia queda registrada.

El estado final cambió de norma 3.793 al cambiar a la condición diagnóstica fija `[-2,1]`; esta segunda condición no se usó en selección. El extractor recibe solo tokens y constantes. No accede a energía total, formación, band gap ni ΔE. MatterGPT espera obligatoriamente `num_props=2`; no se usó `prop=None`.

## Resultados OOF de desarrollo

Cada modelo se evaluó con las mismas estructuras, folds y semillas. Las métricas se calculan tras promediar las predicciones de los 15 modelos para cada estructura.

| Modelo | N estructuras | N composiciones | MAE absoluto pooled (meV/átomo) | MAE absoluto macro (meV/átomo) | RMSE absoluto (meV/átomo) | MAE ΔE anclada pooled (meV/átomo) | MAE ΔE anclada macro (meV/átomo) | GS accuracy pooled | GS accuracy macro | Pairwise accuracy pooled | Pairwise accuracy macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 | 210.000 | 70.000 | 230.456 | 230.456 | 289.894 | 38.153 | 38.153 | 0.493 | 0.493 | 0.660 | 0.664 |
| M4 | 210.000 | 70.000 | 171.223 | 171.223 | 223.197 | 113.209 | 113.209 | 0.313 | 0.313 | 0.500 | 0.502 |
| M3-A | 210.000 | 70.000 | 144.330 | 144.330 | 183.753 | 38.153 | 38.153 | 0.493 | 0.493 | 0.660 | 0.664 |

En OOF development, M3-A reduce el MAE absoluto frente a M2 en 86.1 meV/átomo, y conserva exactamente la ΔE y el ranking de M2. Los resultados de M4 muestran la separación entre energía absoluta y ordenamiento: la regularización lineal mejora el nivel pero daña el ranking.

M3-A conserva algebraicamente el ranking de M2: error máximo de diferencias entre polimorfos `8.882e-16` eV/átomo. La comparación M3-A−M2 por bootstrap pareado de 2,000 tripletes arroja los siguientes IC95% exploratorios:

{
  "gs_accuracy": {
    "ci95_percentile": [
      0.0,
      0.0
    ],
    "interpretation": "exploratory; small number of chemical groups and prior model selection",
    "mean_difference": 0.0,
    "n_valid": 2000
  },
  "mae_absolute_mev_atom": {
    "ci95_percentile": [
      -124.20496399451854,
      -49.16386528182102
    ],
    "interpretation": "exploratory; small number of chemical groups and prior model selection",
    "mean_difference": -87.31090365055981,
    "n_valid": 2000
  },
  "mae_delta_dft_gs_anchored_mev_atom": {
    "ci95_percentile": [
      -4.263256414560601e-14,
      8.535394613318155e-14
    ],
    "interpretation": "exploratory; small number of chemical groups and prior model selection",
    "mean_difference": 2.2879476091475227e-14,
    "n_valid": 2000
  },
  "pairwise_accuracy": {
    "ci95_percentile": [
      0.0,
      0.0
    ],
    "interpretation": "exploratory; small number of chemical groups and prior model selection",
    "mean_difference": 0.0,
    "n_valid": 2000
  }
}

## Test histórico bloqueado

El lock `test_lock.json` se guardó antes de leer las etiquetas. La predicción se generó para las mismas 24 IDs y se promedió entre 5 folds × 3 semillas. La configuración primaria y las reglas de empate quedaron fijadas antes de cargar energías. El resultado sigue siendo exploratorio porque el test histórico ya se usó para evaluar MatGPT H3-A.

| Modelo | N estructuras | N composiciones | MAE absoluto pooled (meV/átomo) | MAE absoluto macro (meV/átomo) | RMSE absoluto (meV/átomo) | MAE ΔE anclada pooled (meV/átomo) | MAE ΔE anclada macro (meV/átomo) | MAE ΔE own-min (meV/átomo) | MAE media composición (meV/átomo) | MAE residuo centrado (meV/átomo) | MAE pairwise firmado (meV/átomo) | Error rango medio (meV/átomo) | GS accuracy pooled | GS accuracy macro | Pairwise accuracy pooled | Pairwise accuracy macro | Pairwise balanced accuracy | Pairwise macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 | 24.000 | 8.000 | 267.430 | 267.430 | 323.627 | 21.442 | 21.442 | 21.232 | 267.399 | 17.831 | 32.776 | 29.976 | 0.875 | 0.875 | 0.833 | 0.833 | 0.829 | 0.829 |
| M4 | 24.000 | 8.000 | 102.046 | 102.046 | 120.628 | 68.985 | 68.985 | 76.671 | 79.811 | 66.994 | 115.258 | 96.634 | 0.375 | 0.375 | 0.417 | 0.417 | 0.357 | 0.294 |
| M3-A | 24.000 | 8.000 | 81.124 | 81.124 | 91.305 | 21.442 | 21.442 | 21.232 | 79.811 | 17.831 | 32.776 | 29.976 | 0.875 | 0.875 | 0.833 | 0.833 | 0.829 | 0.829 |

La descomposición MSE de M3-A en test es MSE total = 0.00834 eV²/átomo²; término de error medio = 0.00782; término residual centrado = 0.00051; discrepancia de identidad = 1.66e-17. La identidad cuadrática no implica suma de MAE.

La comparación pareada con MatGPT H3-A usa exactamente las mismas IDs y targets DFT. Referencia ya publicada en el proyecto para MatGPT: MAE absoluto 133.545 meV/átomo, MAE ΔE anclada 24.208 meV/átomo, GS 5/8 y pares correctos 17/24. IC95% bootstrap pareado MatterGPT−MatGPT, por composición:

{
  "gs_accuracy": {
    "ci95_percentile": [
      0.0,
      0.625
    ],
    "interpretation": "descriptive/exploratory; historical test has already been observed for MatGPT",
    "mean_difference_mattergpt_minus_matgpt": 0.252125,
    "replicates": 2000,
    "sampling_unit": "composition triplet",
    "seed": 13072026
  },
  "mae_absolute_mev_atom": {
    "ci95_percentile": [
      -105.87846940956213,
      2.313600301277731
    ],
    "interpretation": "descriptive/exploratory; historical test has already been observed for MatGPT",
    "mean_difference_mattergpt_minus_matgpt": -52.78275870659422,
    "replicates": 2000,
    "sampling_unit": "composition triplet",
    "seed": 13072026
  },
  "mae_delta_dft_gs_anchored_mev_atom": {
    "ci95_percentile": [
      -16.318635208173013,
      9.78122172287331
    ],
    "interpretation": "descriptive/exploratory; historical test has already been observed for MatGPT",
    "mean_difference_mattergpt_minus_matgpt": -2.9500114378643736,
    "replicates": 2000,
    "sampling_unit": "composition triplet",
    "seed": 13072026
  },
  "pairwise_accuracy": {
    "ci95_percentile": [
      -0.16666666666666663,
      0.4583333333333333
    ],
    "interpretation": "descriptive/exploratory; historical test has already been observed for MatGPT",
    "mean_difference_mattergpt_minus_matgpt": 0.12620833333333334,
    "replicates": 2000,
    "sampling_unit": "composition triplet",
    "seed": 13072026
  }
}

## Interpretación y límites

1. **Transferencia de energía absoluta:** M4 mide cuánto de la energía absoluta puede recuperarse con un regresor lineal sobre hidden states congelados. M3-A prueba si esa media mejora al combinarse con diferencias polimórficas de M2; consultar las tablas, no atribuir el cambio solo al backbone.
2. **ΔE y ranking:** M3-A reutiliza el residuo M2 centrado; por identidad, su ΔE, GS y orden de pares deben coincidir con M2. Un cambio observado sería un error de alineación o métrica.
3. **Condicionamiento:** todo resultado describe embeddings de MatterGPT bajo la condición fija de ejemplo. La sensibilidad a otra condición evidencia dependencia del contexto, pero no autoriza escoger una condición por MAE.
4. **Comparabilidad:** MatterGPT es un generador condicionado por formación/gap; MatGPT H3-A procede de un checkpoint unconditional distinto. Esta comparación descriptiva alinea el target, IDs y unidades, pero no vuelve equivalentes sus preentrenamientos.
5. **Cobertura:** no se halló OOV ni truncamiento. Se leyó SLICES de MatGPT como la entrada estructural compartida; no se regeneró ni canonicalizó.
6. **Selección:** M4 selecciona capa/lectura y alpha con validation interna. Las cifras OOF son exploratorias por decisiones previas sobre el benchmark; test histórico también está expuesto en el proyecto.

## Referencias

- Chen et al., [MatterGPT, arXiv:2408.07608](https://arxiv.org/abs/2408.07608).
- [Checkpoint y ejemplo de inferencia MatterGPT](https://huggingface.co/xiaohang07/MatterGPT), revisión local `ee83624d85e5e60e6a8f48bc72bf2663f87273cd`.
- Xiao et al., [SLICES, Nature Communications 14, 7027 (2023)](https://doi.org/10.1038/s41467-023-42870-7).
- [Ridge API de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Witman y Schindler, [MatFold, Digital Discovery 4, 625–635 (2025)](https://doi.org/10.1039/D4DD00250D).

## Artefactos

Los datos intermedios, checkpoints de cabezas, selección Ridge, predicciones OOF/test, hashes y figuras están bajo `Alternative2/reports/MatterGPT/MatterGPT_frozen_1/`. El checkpoint original de MatterGPT permanece intacto.
