# MatGPT pretraining domain audit

Auditoría del corpus original configurado para el MatGPT unconditional. El flujo exige el archivo `Training_dataset_alex20_with_encoding.csv`; los CSV de Alexandria locales no son sustitutos.

## Ejecución

Desde la raíz del proyecto:

```bash
python -m unittest discover -s Alternative2/scripts/MatGPT_frozen_3/pretraining_domain_audit/tests -v
python -u Alternative2/scripts/MatGPT_frozen_3/pretraining_domain_audit/audit_provenance.py
python -u Alternative2/scripts/MatGPT_frozen_3/pretraining_domain_audit/run_audit.py --corpus /ruta/Training_dataset_alex20_with_encoding.csv
```

El último comando genera primero un manifiesto de procedencia. Con un CSV exacto compatible produce validación de esquema y composición parseada desde el bloque de sitios SLICES (elementos iniciales; después comienzan las aristas indexadas). Si el CSV falta, termina con código 2 y un estado `BLOCKED`, sin cobertura química inferida.

La versión actual no reconstruye una exposición histórica efectiva a filas: requiere identidad y orden del CSV usados al entrenar. La semilla y `torch.randperm` por sí solas no la prueban. Solo se admiten figuras calculadas a partir del corpus confirmado y resultados OOF de desarrollo; nunca etiquetas del test histórico.
