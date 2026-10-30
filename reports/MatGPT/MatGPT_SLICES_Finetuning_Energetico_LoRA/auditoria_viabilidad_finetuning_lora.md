# Auditoría de viabilidad — Experimento C

**Estado final: ejecución completa; evaluación y figuras generadas.**

Las 30 corridas (5 pliegues × 3 semillas × 2 condiciones) terminaron y produjeron 1 260 filas de predicción, equivalentes a 210 estructuras por condición/semilla. La evaluación es exploratoria sobre las 70 fórmulas de desarrollo; el test histórico permanece cerrado.

## Controles técnicos

Preflight PASS; smoke C0/C1 PASS; siete pruebas de integridad PASS. Se verificaron equivalencia inicial LoRA, selección del token `<end>`, invariancia de bloque 3/Ridge, gradientes de adaptadores y MLP8, reproducción determinista del checkpoint, normalización interna y splits agrupados por fórmula.

## Corrección de protocolo

Un primer intento se detuvo al detectar recorte de gradiente no especificado. Se eliminaron los artefactos de ese intento y se repitieron todas las corridas sin clipping; los resultados de este informe corresponden únicamente a la versión corregida.

## Resumen observado

- **frozen** — MAE E/N 183.3 ± 0.5; residuo 33.8 ± 0.7; ΔE pares 58.8 ± 1.0; GS elegible 0.532; pares elegibles 0.668.
- **lora** — MAE E/N 182.8 ± 0.3; residuo 33.5 ± 0.2; ΔE pares 58.3 ± 0.2; GS elegible 0.517; pares elegibles 0.658.

La tabla completa por fold/semilla está en `metricas_por_semilla.csv`; los IC95% apareados por fórmula están en `bootstrap_apareado.csv`.

**Procedencia:** checkpoint MatGPT SHA-256 `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`; tokenizer `d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee`; folds históricos con hash registrado en `preflight/preflight.json`. El caché reconstruido difiere en hash del original ausente y pasa la auditoría numérica guardada junto al artefacto.

Tiempo acumulado de procesos de entrenamiento: 1.12 h; RSS máximo por corrida: 2.02 GB. Para limitaciones, interpretación de bins y definiciones científicas, ver `comparacion_frozen_vs_finetuned.md`.
