# Alexandria 20k → MatGPT

`prepare_provisional.py` is the canonical preparation/preflight command for this same experiment. It validates the exact MatGPT checkpoint/tokenizer, adapts already stored strategy‑4 SLICES, applies energy/overlap/protected-formula gates, and writes formula-disjoint 20k/2k manifests. It does not train.

```bash
.venv/bin/python scripts/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/prepare_provisional.py --run --seed 42 --train-size 20000 --validation-size 2000
```

The generated manifests, individual preflight checks, source hashes, logs, and resource measurements are in `experiments/matgpt_alexandria_energy_transfer_20k/`. The scientific report is in `reports/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/`.

The local F4 training run was interrupted after one fine-tuning epoch. Its best partial checkpoint and history are retained for provenance; the optimizer state was not saved, so this checkpoint is not an exact resume state. A transfer bundle that restarts the full protocol on an allocated MSI GPU node is in `experiments/matgpt_alexandria_energy_transfer_20k/msi_bundle/README_MSI.md`. The local `training.log` contains process messages; `training_trace.jsonl` is the append-only progress trace, including observed phases/epochs, validation losses, timing, resource status, and hashes of the manifests/configuration. Follow the trace with:

```bash
tail -f experiments/matgpt_alexandria_energy_transfer_20k/training_trace.jsonl
```

The read-only monitor is `trace_run.py`; it is attached to the current run and exits when its status becomes terminal. The tabular epoch history remains in `historial_entrenamiento.csv`; current resources/status are in `resources.json`. The run uses early stopping and can take days on CPU. Do not use `results(1).json` or the historical ABS3 test.
