# MatterGPT congelado como extractor para estabilidad polimórfica ABS₃

## Diseño y procedencia

Se evaluaron 210 estructuras en 70 composiciones (tres polimorfos por fórmula), con cinco folds y tres semillas. Se usó MatterGPT multipropiedad oficial local, cargado con `strict=True`, 12 bloques, 12 cabezas, hidden size 768, vocabulario 132, contexto 397 y `num_props=2`. El vector `[1.0, 2.0]` se pasó en `float32` con forma `[batch,1,2]`; es una condición fija (energía de formación y band gap según el código), no una etiqueta ABS₃.

Origen: [MatterGPT en Hugging Face](https://huggingface.co/xiaohang07/MatterGPT); se usó el checkpoint local. SHA-256 checkpoint: `8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21`. `Voc_prior`: `b6a3fb7514383c3637a52447db76197d677e7089487bf49bc995fbc256e668e9`. Código y wrapper: `eb32aa1721c24ac4b09f9b55028c8eb82c809aa2010ec67fd664afacba09ed8e` / `7e50a733509740c346b1478576ff5dfba2c71b99acbde2eda82f1c1de7872c81`.

## Auditorías

Compatibilidad: 210/210 (100.0%). Se conservaron byte por byte las cadenas históricas `matgpt_slices`, que coinciden con `slices_original` y `SLICES`; no se eliminaron ni alteraron tokens, índices, aristas o traslaciones. No se añadió un campo `space_group_num`: no está presente en la tabla fuente ni se serializa en la cadena SLICES usada. `strategy3_graph_equivalent=True` aparece en 207 filas (las restantes no están marcadas como equivalentes); hay hash de topología en las 210. Se preservó exactamente la representación textual histórica sin conversión cristalográfica nueva.

Cero OOV y truncamientos; máximo 256 tokens frente a contexto 397. Hay 0 estructuras sobre 20 átomos/celda. `>` inicia y `<` cierra según el uso de generación del código público. Los IDs serializados son los derivados de `Voc_prior` (>39, <38); los `bos_token_id/eos_token_id` de `config.json` (130/131) no se usan porque no coinciden con el tokenizador reconstruido. Se extrajo la posición del `<` tras `blocks → ln_f`, índice de secuencia n por el prefijo de propiedades. El `embed` de entrada pre-Transformer no se usó. La salida es 210×768, finita y determinista; el hook no alteró logits; padding y tamaño de lote no cambiaron el embedding. Condición y secuencia alteradas cambiaron la representación en controles diagnósticos.

**Limitación de serialización:** no se encontró el dataloader original del teacher-forcing del checkpoint. Por ello `> SLICES <` se justifica por el pipeline público de generación y el cierre `<`, pero la política exacta de token terminal durante el preentrenamiento no queda verificada de manera independiente.

## Entrenamiento y evaluación

Se reutilizaron los cinco Ridge estequiométricos ya ajustados (17 fracciones, scaler y alpha históricos), sin invocar `fit`. Se entrenaron 15 MLP8 (45 grupos inner-train, 11 inner-validation, 14 outer-evaluation; 42/123/2026), con entrada contextualizada fija, AdamW 1e-3, weight decay 1e-4, minibatches de ocho fórmulas, máximo 200 épocas, patience 20, GELU, dropout 0.1 y SmoothL1 beta 1. La escala de residuo se calculó solo en inner-train. Se seleccionó la menor pérdida de validación interna completa. Solo se entrenó la MLP8 (6161 parámetros); MatterGPT quedó congelado.

ΔE estructural usa ancla DFT común: `ΔE_DFT,i = E_DFT,i − E_DFT,g` y `ΔE_pred,i = E_pred,i − E_pred,g`, donde `g = argmin_j(E_DFT,j)` dentro de cada fórmula. Una predicción negativa indica inversión respecto al mínimo DFT. La métrica primaria A es MAE de diferencias firmadas `ΔE_ij = E_j − E_i` en los tres pares intrafórmula; Ridge-only predice cero en todos ellos. Empates predichos ≤1 meV/átomo quedan no resueltos. OOF cubre 210 estructuras; las semillas se promedian dentro de composición antes del bootstrap apareado de 10 000 réplicas por 70 composiciones.

## Resultados (media sobre 15 fold-semilla)
- **MatterGPT_F0:** MAE ΔE 55.729 meV/átomo; MAE E/N 0.05062 eV/átomo; RMSE E/N 0.05875 eV/átomo; MAE residual 32.264 meV/átomo; paridad ΔE OLS m=0.172, b=4.27 meV/átomo, R²=0.134; GS 56.2% (201 casos elegibles); pairwise 64.8% sobre 591 pares elegibles; no resueltos 3.4%.
- **MatGPT_F0:** MAE ΔE 58.844 meV/átomo; MAE E/N 0.05202 eV/átomo; RMSE E/N 0.06022 eV/átomo; MAE residual 33.754 meV/átomo; paridad ΔE OLS m=0.266, b=2.41 meV/átomo, R²=0.157; GS 53.2% (201 casos elegibles); pairwise 66.8% sobre 591 pares elegibles; no resueltos 1.2%.
- **Ridge_only:** MAE ΔE 60.470 meV/átomo; MAE E/N 0.05203 eV/átomo; RMSE E/N 0.06038 eV/átomo; MAE residual 35.072 meV/átomo; paridad ΔE OLS m=-0.000, b=0.00 meV/átomo, R²=no definido (predicción constante); GS 0.0% (201 casos elegibles); pairwise 0.0% sobre 591 pares elegibles; no resueltos 100.0%.

**Contrastes pareados de MAE ΔE (MatterGPT menos baseline):** frente a MatGPT F0, -3.115 meV/átomo (IC95% -8.038 a 1.992; incluye cero); frente a Ridge-only, -4.741 meV/átomo (IC95% -9.264 a -0.240; no incluye cero). MatterGPT mejora el MAE respecto a Ridge-only en este CV; la estimación puntual es menor que MatGPT F0, pero no se distingue con claridad por este bootstrap. La exactitud por pares de MatterGPT es 64.8% frente a 66.8% para MatGPT F0; la selección exacta de estado fundamental es 56.2% frente a 53.2%.

MatGPT F0 se reutilizó desde el modelo histórico M2 del Ridge híbrido, sobre las mismas filas, folds y semillas; se verificó el total de 630 predicciones y no se modificó retrospectivamente.

## Respuestas e interpretación

1. **Compatibilidad:** 210/210 estructuras tokenizables sin OOV ni truncamiento.
2. **SLICES:** cadena histórica preservada byte por byte; no se transformó la estructura. El indicador histórico `strategy3_graph_equivalent` es verdadero en 207/210 filas y falso en las otras tres.
3. **Condición:** `[1.0,2.0]` entra por `prop_nn` y afecta el embedding; se aplicó a toda la cohorte. Su región puede ser poco representada por tener energía de formación positiva.
4. **Representación:** último token `<` tras `ln_f`, dimensión 768.
5–8. **Desempeño:** MatterGPT F0 MAE ΔE 55.729; Ridge 60.470; MatGPT F0 58.844 meV/átomo. Pairwise: MatterGPT 64.8% y MatGPT 66.8%; GS: MatterGPT 56.2% y MatGPT 53.2%. Ranking y estado fundamental están detallados arriba y en JSON.
9. **Sobreajuste:** train sigue bajando mientras validación sube en épocas tardías (aprox. desde 45); early stopping limita ese sobreajuste de la cabeza.
10. **Fine-tuning parcial:** MatterGPT mejora la estimación frente a Ridge-only, pero la diferencia frente a MatGPT F0 es incierta; esto no basta para atribuir una mejora al backbone. Un ajuste parcial queda como hipótesis para un experimento separado.

## Límites

La comparación MatterGPT–MatGPT no identifica un efecto causal de backbone: cambian arquitectura, vocabulario y condiciones de preentrenamiento. La condición `[1,2]` se eligió de antemano y no se optimizó con ABS₃. El test histórico quedó cerrado; los resultados son validación cruzada OOF sobre desarrollo ya explorado, no un test ciego.

## Recursos y archivos

Tiempo de pared: 46.5 s (extracción 44.1 s; MLP8 1.8 s acumulados); CPU usuario/sistema: 65.7/28.3 s; dispositivo CPU; memoria máxima: 1111.2 MiB. Versiones: Python 3.10.21, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.2, scikit-learn 1.7.2. Se guardaron quince checkpoints MLP8 compactos sin duplicar MatterGPT.

Las siete figuras se exportan solo en PNG bajo `images/mattergpt_abs3_frozen_ridge_mlp8/`.
