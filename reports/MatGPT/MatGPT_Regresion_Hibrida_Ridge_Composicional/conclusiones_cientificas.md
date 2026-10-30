# Conclusiones científicas

## Respuesta

**RESULTADO EXPERIMENTAL:** sí, en estas 70 fórmulas de desarrollo el Ridge composicional mejora marcadamente la energía absoluta, manteniendo intacta la señal polimórfica almacenada. Frozen: MAE E/N 183.30 → 52.02 meV/átomo; LoRA: 182.78 → 52.07 meV/átomo.

Diferencia apareada de MAE absoluto, composicional menos embeddings: Frozen -131.28 meV/átomo (IC95% -174.65 a -94.52); LoRA -130.70 (IC95% -175.65 a -93.84). La media por fórmula mejora en -139.35 meV/átomo (IC95% -183.89 a -101.55).

**HECHO MATEMÁTICO Y VERIFICADO:** M1/M2 y M3/M4 tienen exactamente las mismas diferencias ΔE, residuos centrados, rankings y estados fundamentales dentro de 1×10⁻⁸ meV/átomo. La diferencia media de energía cambia; la estimación de estabilidad polimórfica no.

## Reproducción histórica

El control M1 reproduce las métricas originales principales: MAE E/N ≈183.30 meV/átomo, MAE pairwise firmado ≈58.84 meV/átomo y estado fundamental exacto 54.3%. La exactitud pairwise recalculada es 66.84%, frente al 67.2% aproximado mencionado en el pedido; el CSV histórico fuente también da 66.84% al promediar las semillas con el criterio |ΔE_DFT|>5 y predicción resuelta >1 meV/átomo. Por tanto, la diferencia 67.2% no aparece en los archivos canónicos consultados.

**HECHO VERIFICADO:** los cinco Ridge usan selección de alpha en 45 fórmulas inner-train/11 inner-validation y refit con 56 fórmulas outer-train. Alphas seleccionados: 0.01, 0.1, 0.1, 0.1 y 0.1 para los folds 0–4; coinciden con el control composicional anterior.

**LIMITACIÓN:** son predicciones OOF de folds del conjunto de desarrollo ABS₃, no un test ciego externo; el test histórico no se leyó. El experimento solo respalda la sustitución del estimador de energía media en este protocolo y no demuestra mayor información estructural aprendida por MatGPT.
