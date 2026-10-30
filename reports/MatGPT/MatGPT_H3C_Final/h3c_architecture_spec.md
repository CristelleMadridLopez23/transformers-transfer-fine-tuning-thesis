# Especificación inmutable H3-C

[HECHO VERIFICADO] El contrato científico es el archivo `h3c_architecture_lock.yaml`; cualquier cambio de extractor, pooling, ancho oculto, activación, target, normalización o selección constituye **otro experimento**. El test de arquitectura falla ante cambios de dimensiones y checkpoint.

**Entradas:** SLICES estándar archivado → MatGPT incondicional congelado → `L03_MEAN` y `FINAL_END`, cada uno de 768 dimensiones. **Rama media:** promedio de tres `L03_MEAN` → StandardScaler + Ridge (768→1). **Rama residual:** tres `FINAL_END` → Linear(768,8) → GELU → Dropout(.1) → Linear(8,1) → centrado. **Suma:** energía por átomo predicha. **Postproceso:** energía total = N×e; ΔE y mínimo solo dentro de fórmula.

[HECHO VERIFICADO] Ridge y MLP8 se ajustan con objetivos distintos: media de energía DFT por fórmula y residuo respecto a la media, ambos en eV/átomo. Ridge se reajusta con 56 fórmulas después de elegir alpha; MLP8 conserva el checkpoint elegido con 11 fórmulas de validación interna y se entrenó sobre 45. No hay pérdida de pares.

[HIPÓTESIS] En futura fusión geométrica solo podrá cambiar la dimensión de la primera proyección que reciba nuevas características. El ancho 8, GELU, dropout, salida, pérdida y reglas de selección siguen bloqueados; cualquier aumento de parámetros de entrada deberá reportarse. No se implementa geometría aquí.
