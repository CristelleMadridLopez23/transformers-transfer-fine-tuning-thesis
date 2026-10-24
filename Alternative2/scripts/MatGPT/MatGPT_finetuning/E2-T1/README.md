# E2-T1 — admisibilidad de trayectorias DFT

Cruza `results(1).json` con el cache anterior, vuelve a serializar los frames faltantes, coteja cada grafo periódico con SLICES y audita parámetros PBE, colisiones de IDs, fuerzas y energía dentro de cada cálculo. No entrena ni consulta el test para decisiones supervisadas. El criterio de admisión y su resultado quedan en `Alternative2/reports/MatGPT_finetuning/E2-T1/decision.json`.

```bash
python -u Alternative2/scripts/MatGPT_finetuning/E2-T1/audit_relax_entries.py
python -u Alternative2/scripts/MatGPT_finetuning/E2-T1/audit_metadata_compatibility.py
python -u Alternative2/scripts/MatGPT_finetuning/E2-T1/plot_t1.py
python -u Alternative2/scripts/MatGPT_finetuning/E2-T1/summarize_t1.py --report-dir Alternative2/reports/MatGPT_finetuning/E2-T1
```

La auditoría usa la estructura completa de todos los 13,874 frames; puede ser intensiva en CPU. Solo `GO` habilita el entrenamiento T2 completo. `LIMITED` permite estudiar un piloto diagnóstico según el segundo prompt; `BLOCKED` detiene T2.
