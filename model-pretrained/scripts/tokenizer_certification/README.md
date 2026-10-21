# MATText tokenizer certification

Experiment: `exp02_only_hex_mattext_finetune/tokenizer_certification`.
No training, optimizer, model inference, GPU, Slurm, original input modifications or gate unlocking.

Run from the repository root with the existing `dmc` Python3.10 environment:

```bash
python -B model-pretrained/scripts/tokenizer_certification/gather.py
python -B model-pretrained/scripts/tokenizer_certification/audit.py
USE_TF=0 USE_FLAX=0 CUDA_VISIBLE_DEVICES='' PYTHONPATH=experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B model-pretrained/scripts/tokenizer_certification/probe.py
python -B model-pretrained/scripts/tokenizer_certification/report.py
python -B model-pretrained/scripts/tokenizer_certification/test_certification.py
python -B model-pretrained/scripts/tokenizer_certification/finalize.py
```

Use `gather.py --offline` for cached evidence. All audit commands accept a configurable `--output` relative to the project inside reports/. Evidence and deterministic probe JSON record pinned source revisions, hashes, versions and seed42. Native source class bodies are extracted with AST to avoid unrelated module-level network downloads, with hashed vocabulary paths supplied explicitly. Source-defined default and pretraining special configurations are compared; no arbitrary tokens or ID reassignment is introduced. Official constructors write numbered vocabulary files only under the new reports working directory.

The header inspection uses a read-only memory map and checks every BertModel tensor shape. Prior dataset counts and fingerprints are checked without converting/re-tokenizing all structures. `report.py` uses formal `decide()` rules: missing checkpoint linkage alone is PARTIALLY_VERIFIED; unresolved material contradictions are BLOCKED. CERTIFIED requires the exact original run linkage and verified special-token protocol, beyond encoding agreement. Original author output is absent; local repeatability cannot replace it.

`test_certification.py` uses synthetic malformed header bytes only for file validation; these are not scientific data. All scientific probe inputs are official examples, real project strings, or explicitly labeled diagnostic strings. Logs distinguish deliberately invalid/default-MLM diagnostics from unexpected failures. No email is sent.
