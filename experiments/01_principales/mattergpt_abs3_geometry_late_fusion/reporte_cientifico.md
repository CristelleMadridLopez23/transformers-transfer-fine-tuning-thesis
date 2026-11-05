# MatterGPT congelado + SLICES + geometría: ABS₃

## Hechos verificados

Se reutilizó la caché MatterGPT validada: 210 embeddings de dimensión 768, checkpoint SHA-256 `8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21`, tokenizador `b6a3fb7514383c3637a52447db76197d677e7089487bf49bc995fbc256e668e9`, propiedad fija `[1.0, 2.0]`; extracción histórica documentada tras `blocks → ln_f` en el último token `<`. No se ejecutó ni modificó el backbone.
La geometría histórica contiene 210 IDs, exclusivamente `gga-static_endpoint_cleaned`, SHA-256 `5be78f0e6282f0b5df4aeff8f54950e2f07750bd290a72444fb09b44aeec6eb8`. Los siete descriptores se recuperaron por `structure_id`: celda primitiva reducida de Niggli; a/b/c divididos por la raíz cúbica del volumen primitivo reducido; ángulos de esa celda; y volumen de celda original por átomo.
Tres de los 210 registros históricos no tienen hash de topología no vacío (3); la correspondencia se conserva mediante el ID estructural estable, fórmula, polimorfo, número de átomos y join uno-a-uno. El preflight original confirma la política pública `> SLICES <` y extracción tras `ln_f`, aunque el dataloader original de teacher-forcing no está disponible para verificar independientemente el token terminal durante el preentrenamiento.
Se reutilizaron los cinco Ridge de 17 fracciones sin reentrenar, cinco folds agrupados por fórmula, 45 fórmulas inner-train, 11 inner-validation, 14 outer-evaluation y semillas 42, 123, 2026. El `StandardScaler` geométrico de cada fold se ajustó con 135 estructuras pertenecientes solo a las 45 fórmulas inner-train. Test histórico cerrado.
Control de inicialización: 15 verificaciones fold-semilla sobre 3150 vectores; error absoluto máximo 0, tolerancia 1e-7. Las siete columnas geométricas comenzaron en cero y las cabezas MatterGPT de 768 dimensiones compartieron inicialización exacta.

## Resultados experimentales

Métrica primaria A: error absoluto medio de diferencias ΔE firmadas en los tres pares por fórmula; seeds promediadas dentro de fórmula para bootstrap apareado de 10 000 réplicas (70 fórmulas). C2−C1 = **-2.330 meV/átomo** (IC95% -3.803, -0.961); el IC95% excluye cero. Un valor negativo favorece la geometría.
C1 SLICES-only: MAE ΔE 55.729; MAE residual 32.264; MAE E/N 0.05062 eV/átomo; GS exacto 56.2%; exactitud pairwise elegible 64.8%.
C2 SLICES + geometría: MAE ΔE 53.399; MAE residual 31.055; MAE E/N 0.04964 eV/átomo; GS exacto 58.7%; exactitud pairwise elegible 65.7%. Denominadores explícitos en `metrics.json`.
Bootstrap C2−Ridge: -7.072 meV/átomo. Comparación histórica MatGPT F0 + geometría: MAE ΔE 56.273 meV/átomo.

## Respuestas, interpretación y límites

1–3. La geometría reduce el MAE ΔE de 55.729 a 53.399 meV/átomo y el MAE residual de 32.264 a 31.055 meV/átomo. el IC95% excluye cero en el contraste primario.
4–5. Ranking pairwise: 64.8% → 65.7%; GS exacto: 56.2% → 58.7%. Ver denominadores y no resueltos en `metrics.json`.
6. Errores por Perovskite, Hexagonal y Needle-like están tabulados en `metrics.json`; la unidad es estructura estática, media de semillas.
7. La ablación a geometría cero y la permutación dentro de triplete diagnostican dependencia predictiva bajo intervención sintética; no son prueba de causalidad física. La norma y suma absoluta de pesos geométricos por descriptor se guardan en el historial/checkpoints y `weight_diagnostics.csv`.
8. MatGPT F0 congelado se compara en mismos folds, IDs, targets y semillas; diferencias entre extractores no son efecto causal del preentrenamiento, pues también difieren arquitectura, vocabulario y condicionamiento.
9. Las curvas muestran pérdida de minibatches y validación completa. Early stopping elige el menor SmoothL1 normalizado de 11 fórmulas; no se usó el outer para selección.
10. Los resultados solo permiten decidir si conviene estudiar fine-tuning parcial. No demuestran que este sea útil; hace falta experimento predefinido separado y control de leakage.

### Definiciones

`ΔE_DFT,i = E_DFT,i − E_DFT,g`; `ΔE_pred,i = E_pred,i − E_pred,g`; g es el mínimo DFT de la misma fórmula para ambos ejes. Predicción negativa significa inversión de orden DFT. Pareja elegible si |ΔE_DFT| > 5 meV/átomo; |ΔE_pred| ≤ 1 es no resuelta y cuenta incorrecta. Fórmula elegible para estado fundamental si su gap DFT al segundo mínimo >1 meV/átomo; empate predicho ≤1 es fallo. No se mezclan composiciones.
Los intervalos se comparan por estructura de composición (70 unidades bootstrap); las tres semillas se promedian dentro de fórmula. Todas las observaciones outer OOF se reportan como validación cruzada de desarrollo, no como test ciego. Bootstrap primario sin ajuste multiplicidad; contraste C2−C1 predefinido.

## Recursos

Duración de pared 5.4 s; entrenamiento acumulado de las 15 cabezas C2 1.9 s; desglose C1/C2 y controles MatGPT en `resources.json`. CPU usuario/sistema 4.5/0.7 s; pico RSS 594.4 MiB; dispositivo CPU; versiones Python 3.10.21, PyTorch 2.8.0, NumPy 2.2.6, scikit-learn 1.7.2. Se guardaron 15 checkpoints MLP8 compactos; MatterGPT no se duplicó.

Figuras PNG 300 dpi en `images/mattergpt_abs3_geometry_late_fusion/`. Diagrama editable: `reports/MatterGPT/mattergpt_abs3_geometry_late_fusion/diagrama_arquitectura.mmd`.
