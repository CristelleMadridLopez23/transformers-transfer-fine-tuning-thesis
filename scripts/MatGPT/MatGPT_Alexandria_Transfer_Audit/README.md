# Auditoría Alexandria–MatGPT

Auditoría de solo lectura, sin entrenamiento ni uso de etiquetas del test ABS₃. Una vista previa accidental de un registro potencialmente protegido se documenta y excluye.

- Informe: `reports/MatGPT/MatGPT_Alexandria_Transfer_Audit/auditoria_alexandria_vs_matgpt.md`
- Salidas compactas: `metrics.json`, `summary.json`, `resources.json`
- Tablas: inventario, elementos, energía, duplicados, polimorfismo y subconjuntos
- Figuras PNG: `images/MatGPT/MatGPT_Alexandria_Transfer_Audit/`

Ejecutar desde la raíz: `.venv/bin/python scripts/MatGPT/MatGPT_Alexandria_Transfer_Audit/audit.py --run`
No entrena modelos ni usa labels del test histórico; véase la nota de acceso accidental en el informe.
