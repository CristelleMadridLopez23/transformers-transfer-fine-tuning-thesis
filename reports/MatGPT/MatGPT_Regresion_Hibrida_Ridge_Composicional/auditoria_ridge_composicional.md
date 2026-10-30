# Auditoría Ridge composicional

## Fuentes y controles verificados

| Control | Resultado |
|---|---|
| Dataset train/val | 210 estructuras, 70 fórmulas |
| Asignaciones históricas | 1050 filas; 45/11/56/14 fórmulas por pliegue |
| Predicciones Frozen/LoRA existentes | 1260 filas; hash verificado |
| Replay H3-C Frozen | PASS; diferencia máxima 7e-07 eV/átomo |
| IDs, N, composición y energías | Coincidencia exacta con las etiquetas del desarrollo |
| Target del Ridge nuevo | Media de E_DFT/N de los tres polimorfos, eV/átomo |
| Datos del test histórico | No leídos ni usados |

## Características y alpha

Vector de 17 fracciones estequiométricas en el orden fijo Ba, Ca, Cd, Cu, Ge, Hf, Mg, Nb, Pb, S, Si, Sn, Sr, Ti, V, Zn y Zr. No entran IDs, etiquetas DFT, geometría, topología ni propiedades calculadas. Cada polimorfo recibe el mismo vector; las fracciones se comprueban en `composition_features.csv`.

El StandardScaler de selección se ajusta solo en las 45 fórmulas inner-train; cada alpha se puntúa sobre 11 fórmulas inner-validation. Después se crea un pipeline independiente por pliegue y se reajusta en las 56 fórmulas outer-train. Grid alpha: `[0.01, 0.1, 1, 10, 100, 1000]`, según el control composicional existente. Solver `lsqr`, tolerancia 1e-8, intercepto habilitado. Resultados en `ridge_validation_grid.csv` y `ridge_selection.csv`.

## Procedencia

Dataset SHA-256: `0719f496c96297761ede3a7e2e7ef57aa7b62400fb0004e679a66430a70c31e8`; folds: `a5f00a89a41e4645a68d1463f26f9eca10a09bdfd1208193a87e541ff7cd5627`; predicciones Frozen/LoRA: `8cc702dac3888b09f6369cea5df108c9b1220ce41c85f59dafacc18e0d13f344`; reproducción H3-C: `56df6e54e00bc282515ff940844b4274941629a2ea65ce6c772f6c9449c48fbe`; control composicional previo: `be77192438832743fd8ed2471b79a18e432af9cb04e49877fc26c7f6c0be8ed7`.
