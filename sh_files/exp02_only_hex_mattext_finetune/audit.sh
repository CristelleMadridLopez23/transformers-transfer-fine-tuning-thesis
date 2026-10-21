#!/usr/bin/env bash
set -euo pipefail
MATHEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MATHEX_ROOT"
export PYTHONPATH="$MATHEX_ROOT/experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MATHEX_PYTHON="${MATHEX_PYTHON:-python}"
MATHEX_CONFIG="${MATHEX_CONFIG:-experiments/exp02_only_hex_mattext_finetune/config.json}"
mkdir -p reports/exp02_only_hex_mattext_finetune/logs
"$MATHEX_PYTHON" -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/00_probe_reference_tokenizer.py --config "$MATHEX_CONFIG"
"$MATHEX_PYTHON" -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/01_audit_tokenizer.py --config "$MATHEX_CONFIG" \
  2>&1 | tee reports/exp02_only_hex_mattext_finetune/logs/tokenizer_audit.log
