# E2-T0 — auditoría de trayectorias

Estado: LIMITED. La identidad de secuencia no certifica equivalencia de grafos periódicos. Se usaron etiquetas disponibles en la fuente; no se entrenó, no se generaron etiquetas y no se usó NequIX.

```json
{
  "status": "LIMITED",
  "input": "/Users/cristellemadrid/Desktop/Transformers/transformers-transfer-fine-tuning-thesis/Alternative2/01_data/abs3/processed/only_hex_new_csv_slices/configurations_slices.csv",
  "frames_valid": 8898,
  "unique_token_sequences": 2865,
  "unique_ratio": 0.3219824679703304,
  "frames_over_442_tokens": 0,
  "collision_groups": 1127,
  "distinct_energy_collision_groups": 782,
  "topology_equivalence": "not certified by token equality; periodic graph comparison remains pending",
  "label_source_column": "energy_per_atom_ev",
  "interpretation": "SLICES idénticos con energía distinta son colisiones no distinguibles por el predictor basado solo en IDs."
}
```
