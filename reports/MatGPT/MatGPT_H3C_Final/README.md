# MatGPT H3-C Final

Módulo bloqueado para estudiar energía media por composición y diferencia entre polimorfos ABS₃ con MatGPT congelado. H3-C histórico reproducido exactamente desde sus cinco Ridge y quince MLP8.

## Reproducción sin reentrenamiento

Desde la raíz del proyecto, con `.venv` activo:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 .venv/bin/python Alternative2/scripts/MatGPT/MatGPT_H3C_Final/reproduce.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s Alternative2/scripts/MatGPT/MatGPT_H3C_Final/tests -p 'test_*.py'
```

Los controles completos ya están guardados en `controls/`; sus comandos exactos y recursos por corrida figuran en los logs/JSON del módulo. El test histórico no se abre.

## Lectura

Comenzar por `h3c_architecture_audit.md`, `h3c_reproduction_report.md` y `h3c_scientific_conclusions.md`. Figuras y diagramas verificados en `figures/`; métricas por estructura, fórmula, par y bootstrap en `analysis/`. `h3c_architecture_lock.yaml` fija las decisiones arquitectónicas. `cleanup_report.md` describe el archivo recuperable de experimentos anteriores.
