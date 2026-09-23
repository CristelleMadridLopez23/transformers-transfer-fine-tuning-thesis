# Transferencia de Transformers moleculares a materiales inorgánicos

Para responder, en una primera etapa:

> ¿Cuánta información útil puede transferir un Transformer molecular
> preentrenado hacia materiales inorgánicos sin modificarlo?

El bracket izquierdo compara `seyonec/ChemBERTa-zinc-base-v1` y
`ibm-research/MoLFormer-XL-both-10pct` como extractores congelados. Ambos
generan vectores de 768 dimensiones; se usa el mismo pooling, partición y
regresor para los dos.

## Decisión de representación

`results_no_meta.json` contiene cristales periódicos, no moléculas discretas.
Un SMILES convencional no puede representar de forma fiel su celda, enlaces
periódicos, coordenadas ni simetría. Por eso esta etapa genera un control
deliberadamente limitado por composición:

```text
Ba1Ge1S3 -> [Ba].[Ge].[S].[S].[S]
```

Es una cadena SMILES de átomos neutros desconectados y no una conversión
estructura-cristal-a-molécula. No se infieren enlaces ni estados de oxidación.
Esto permite preguntar si el preentrenamiento molecular aporta señal química
composicional. No permite distinguir `dist_perovskite`, `hexagonal` y `needle`.
La etapa posterior deberá usar una representación periódica, por ejemplo
SLICES, para responder preguntas estructurales.

## Auditoría actual de los datos

- 480 cálculos convergidos y 480 estructuras.
- 80 fórmulas/composiciones únicas, cada una con 6 registros.
- 3 prototipos por fórmula y dos cálculos: `gga-relax` y `gga-static`.
- Todas las celdas tienen 20 sitios; la estequiometría de los sitios concuerda
  con la fórmula reportada.
- Targets tabulados: energía por átomo, energía relajada, band gap, carácter
  directo/indirecto y energía de Fermi.

El CSV generado está en
`transferability/data/SMILES/composition_smiles.csv`; su auditoría está en el
JSON adyacente.

## Notebooks

El flujo gráfico se ejecuta desde la raíz del repositorio, en este orden:

1. `transferability/preprocessing.ipynb`: audita el JSON fuente, crea una tabla
   compacta y grafica distribuciones por cálculo, prototipo y target.
2. `transferability/toSMILES.ipynb`: ejecuta la conversión, inspecciona ejemplos
   y visualiza las colisiones entre composición y polimorfos.
3. `transferability/unitCellSMILES.ipynb`: construye los controles de celda
   completa desconectada y de conectividad finita, comprueba si distinguen los
   polimorfos y visualiza una misma muestra como grafo 2D leído del SMILES y
   como celda 3D reconstruida desde CIF.
4. `transferability/evaluation/metricAnalysis.ipynb`: descarga/verifica modelos,
   extrae embeddings y compara `unit_cell_smiles` contra el baseline
   `composition_smiles` con los mismos folds y métricas.

Cada notebook llama a los scripts de `transferability/scripts/`; los notebooks
no contienen una segunda implementación del pipeline.

## Entorno mínimo para este baseline

Se recomienda Python 3.10, igual que en `requirements_tesis.txt`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r transferability/requirements.txt
```

El archivo `requirements_tesis.txt` conserva el entorno completo de la tesis;
no es necesario instalar GNNs, JAX o MLIPs para ejecutar este bracket.

## 1. Auditar el dataset

```bash
python transferability/scripts/preprocess_dataset.py
```

## 2. Preparar el control de composición

```bash
python transferability/scripts/prepare_composition_smiles.py
```

La conversión valida la razón entre fórmula y sitios y falla explícitamente si
encuentra ocupaciones parciales, desorden o una sintaxis de fórmula no soportada.

## 3. Descargar los modelos una sola vez

```bash
python transferability/scripts/download_hf_models.py
```

Los archivos quedan en `.cache/huggingface/hub`, que está ignorado por Git. El
script excluye pesos alternativos de Flax y guarda
`.cache/huggingface/download_manifest.json` con el commit exacto resuelto para
que los experimentos posteriores usen el mismo snapshot. MoLFormer requiere
`trust_remote_code=True`; por eso es especialmente importante conservar ese
commit en el manifiesto.

## Representación adicional de celda unitaria

```bash
python transferability/scripts/prepare_unit_cell_smiles.py
```

Genera `transferability/data/unit_cell_SMILES/unit_cell_smiles.csv` con dos
controles adicionales:

- `unit_cell_disconnected_smiles`: todos los sitios de la celda como especies
  desconectadas, siguiendo la convención conservadora usada para redes iónicas
  por Quirós et al.
- `unit_cell_smiles`: grafo simple inferido con `CrystalNN` y serializado con
  Open Babel. Conserva topología aproximada, pero descarta etiquetas de
  traslación, multiplicidad de aristas, geometría, órdenes de enlace y estados
  de oxidación.

Este flujo es automático y no reproduce la curación humana del trabajo de
Quirós et al.: https://doi.org/10.1186/s13321-018-0279-6

## 4. Extraer embeddings congelados

```bash
python transferability/scripts/extract_frozen_embeddings.py --device auto
```

Por defecto la extracción es offline: si falta algo en la caché, falla en vez
de descargar silenciosamente otra revisión. Se calculan embeddings sólo para
las 80 entradas únicas y luego se reasignan a los 480 registros. El pooling es
la media de la última capa, excluyendo padding y tokens especiales, idéntico en
ambos modelos. Cada archivo tiene una auditoría de longitud y tokens `[UNK]`.

## 5. Evaluar la transferencia congelada

```bash
python transferability/scripts/evaluate_frozen_transfer.py \
  --target energy_per_atom \
  --calculation gga-static
```

La evaluación usa `GroupKFold` agrupado por fórmula: ninguna composición puede
aparecer simultáneamente en entrenamiento y prueba. Esto evita que las tres
fases o los pares `relax/static` de la misma composición produzcan fuga de
datos. ChemBERTa y MoLFormer reciben exactamente los mismos folds, pooling y
regresor Ridge. Se reportan dos referencias: un predictor constante y un Ridge
sobre fracciones elementales. Esta segunda comparación es necesaria para no
atribuir al preentrenamiento una señal que una codificación composicional simple
ya puede explicar. También se calcula el error de colisión de representación:
el error optimista que queda al predecir la media conocida de cada fórmula para
sus tres polimorfos. No es un baseline predictivo, sino el límite estructural de
una entrada que sólo contiene composición.

Para band gap:

```bash
python transferability/scripts/evaluate_frozen_transfer.py \
  --target bandgap \
  --calculation gga-static \
  --output artifacts/evaluation/frozen_transfer_bandgap.json
```

## Pruebas rápidas

```bash
python -m unittest discover -s transferability/tests -v
```

## Interpretación válida

Si un embedding supera al predictor constante con fórmulas nunca vistas, hay
evidencia de señal transferible a nivel de composición. No demuestra que el
modelo entienda cristales ni periodicidad. Si no lo supera, tampoco descarta
que una representación periódica o una adaptación con SLICES pueda funcionar;
sólo limita la transferencia *zero-adaptation* desde este proxy molecular.
