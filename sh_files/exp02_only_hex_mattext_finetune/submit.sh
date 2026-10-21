#!/usr/bin/env bash
set -euo pipefail
MATHEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MATHEX_ROOT"
export PYTHONPATH="$MATHEX_ROOT/experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MATHEX_PYTHON="${MATHEX_PYTHON:-python}"
"$MATHEX_PYTHON" -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/04_preflight.py \
  --config "${MATHEX_CONFIG:-experiments/exp02_only_hex_mattext_finetune/config.json}" --submit
