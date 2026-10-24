# MatGPT E1.3 — validación cruzada agrupada por composición

E1.3 contrasta tres candidatos congelados, predefinidos a partir de E1.1/E1.2.
No busca nuevos tamaños de cabeza, pooling, λ o α, no ejecuta E2 y no usa las
24 estructuras/8 composiciones del test histórico para ajuste ni selección.

## Entradas auditadas

- Checkpoint unconditional local: `Alternative2/models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth`.
- Código local MatGPT fijado al commit `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`; no se descarga ni sustituye por `main`.
- Etiquetas/split: `Alternative2/reports/matgpt_e1_v2/dataset.csv` y `split_manifest.json`.
- Cachés last-`<end>` y masked mean; ambos deben pasar auditoría de hash y coincidir en orden, IDs, fórmula, familia, etiqueta y split.
- Cuatro resultados E1.2: `matgpt_e1_2_last_mlp32_abs_v1`, `matgpt_e1_2_mean_mlp32_abs_v1`, `matgpt_e1_2_last_mlp8_1` y `matgpt_e1_2_mean_mlp8_1`.

`audit_cv.py --phase preflight` carga el decoder oficial con `strict=True`,
comprueba SHA-256, arquitectura, número de parámetros entrenables y las dos
auditorías de pooling. El masked mean debe excluir start/end/padding; ambos
extractores deben conservar la máscara causal y el estado tras `ln_f`, antes de
`lm_head`. Los embeddings se reutilizan y se copian a tensores ordinarios antes
de entrenar cabezas, pues sus archivos se crearon en `torch.inference_mode`.

## Protocolo

El grupo se forma con la fórmula química completa canonizada. El desarrollo es
exactamente historical train ∪ validation: 210 estructuras, 70 composiciones,
cada una con los tres polimorfos físicos. Las 24 estructuras/8 fórmulas de
historical test se registran como excluidas. `GroupKFold(n_splits=5)` produce
14 grupos (42 estructuras) de outer-validation y 56 grupos (168 estructuras)
de outer-training en cada fold. Un holdout interior determinista de 11/56
grupos (20%, 33 estructuras; semilla base 13072026 + índice de fold) se reserva
dentro de outer-training. `fold_assignments.csv` se escribe antes de entrenar:
contiene una fila por estructura y fold (cinco filas por ID de desarrollo, ya
que sus roles inner cambian por fold) y una fila por estructura del test,
marcada siempre como excluida con `outer_fold=-1`.

Cada fold×semilla entrena los tres candidatos (45 cabezas):

| Identificador | Pooling | Cabeza | Pérdida | Hipótesis a evaluar |
|---|---|---|---|---|
| `matgpt_e13_mlp8_mean_pair1` | mean SLICES enmascarado | 768→8→1, GELU, dropout 0.1 | Huber absoluta + λ=1 pairwise | Selección del estado fundamental |
| `matgpt_e13_mlp8_last_pair1` | estado `<end>` | 768→8→1, GELU, dropout 0.1 | Huber absoluta + λ=1 pairwise | Energías relativas |
| `matgpt_e13_mlp32_mean_abs` | mean SLICES enmascarado | 768→32→1, GELU, dropout 0.1 | Huber absoluta, λ=0 | Energía absoluta |

La configuración efectiva de los scripts E1.2 es AdamW (`lr=0.001`,
`weight_decay=0.0001`), Huber `beta=1`, máximo 200 épocas, early stopping con
paciencia 20 y selección del menor objetivo normalizado de validation. El
sampler mezcla grupos completos, hasta 8 composiciones por minibatch. E1.2
calculaba media/desviación target y RMS de diferencias pairwise solo sobre
train; E1.3 vuelve a estimarlas solo en inner-train para evitar fuga desde
inner-validation. La escala pairwise tiene suelo de 0.001 eV/átomo; la del
target, 1e-6 eV/átomo. No se aplica clipping ni scaler de embeddings. Solo los
parámetros de la cabeza se optimizan. El outer fold no interviene en early
stopping ni en selección del checkpoint y se evalúa una sola vez al final.

## Ejecución solicitada

Desde la raíz del repositorio, en el entorno Python del proyecto. Ejecuta cada
línea en orden y confirma que la anterior termina con `PASS` antes de seguir.
Este README entrega instrucciones; el autor no ha ejecutado el preflight ni las
45 cabezas.

1. Pruebas unitarias de la canonización de fórmulas (no usan datos científicos):

```bash
python -m unittest discover -s Alternative2/scripts/matgpt_e1_3/tests -v
```

2. Preflight de checkpoint/datos/cachés y carga estricta:

```bash
python -u Alternative2/scripts/matgpt_e1_3/audit_cv.py --phase preflight
```

3. Congela GroupKFold, inner holdouts, colisiones y provenance antes de ajustar:

```bash
python -u Alternative2/scripts/matgpt_e1_3/prepare_group_folds.py
```

4. Entrena los 45 modelos pequeños en CPU; el Transformer no se vuelve a ejecutar:

```bash
python -u Alternative2/scripts/matgpt_e1_3/train_cv_heads.py --device cpu
```

5. Calcula métricas, bootstrap y figuras desde OOF de desarrollo:

```bash
python -u Alternative2/scripts/matgpt_e1_3/evaluate_cv.py
```

6. Comprueba conteos, exclusión del test, pesos de cabeza y entregables:

```bash
python -u Alternative2/scripts/matgpt_e1_3/audit_cv.py --phase results
```

Si una fase deja artefactos parciales, consérvalos. Para reiniciar, da a los
scripts `--out-dir Alternative2/reports/matgpt_e1_3_v2` y ejecuta en ese
directorio cada fase; no borres ni reemplaces la corrida anterior.

## Resultados y estadística

`evaluate_cv.py` requiere exactamente 3 candidatos × 3 semillas × 210 IDs =
1,890 predicciones OOF, una predicción por `(configuración, semilla, ID)`.
Exporta MAE/RMSE absolutas en eV/átomo, MAE de ΔE en meV/átomo, GS, pares,
rango firmado/absoluto y ratio, colapso, empates, Spearman definido, métricas
macro por composición y MAE/RMSE por familia. El DFT ground state puede tener
empate: seleccionar cualquier mínimo DFT empatado es correcto; un empate de la
predicción se marca como no discriminativo y no cuenta como acierto. Pares DFT
empatados a ≤1 meV/átomo no se incluyen en el denominador; una predicción ML
empatada no acierta un orden DFT no empatado.

Los resúmenes de fold y semilla se reportan por separado. El bootstrap pareado
usa 2,000 remuestras de 70 composiciones, con sus tres polimorfos juntos y las
semillas promediadas dentro de composición. Los intervalos son exploratorios,
no una garantía de superioridad. Se guardan comparaciones pareadas de cada
composición, grupos extremos por regla Tukey y estadísticas OLS de la figura de
paridad ΔE. Cada figura usa PNG según la norma de exportación del repositorio.

## Límites y lectura

Los candidatos se eligieron usando las mismas 8 composiciones de validation
histórica; la CV sobre esos grupos describe robustez/sensibilidad, pero no
elimina el sesgo de preselección. Para estimación independiente se necesita un
nuevo holdout externo o nested CV completo. El test histórico se observó
durante E1, no es ciego y no se consulta aquí. Un análisis geométrico E1.2
incluyó todos los 234 embeddings; sus correlaciones con test son exploratorias
y no guiaron E1.3. Las etiquetas son energías DFT-GGA estáticas por átomo, no
energías de formación. No se debe atribuir automáticamente todo error a la
representación: cabeza, pérdida, protocolo y tamaño de muestra también
intervienen. E1.3 no ejecuta E2.

## Referencias metodológicas

- Código MatGPT, commit local fijado: [devalab/matgpt](https://github.com/devalab/matgpt), commit `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- Witman & Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D).
- Scikit-learn, [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) y [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- PyTorch, [inference_mode y autograd](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).

