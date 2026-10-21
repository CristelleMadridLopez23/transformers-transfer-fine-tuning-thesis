#!/usr/bin/env bash
set -euo pipefail
EXP01_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$EXP01_ROOT"
EXP01_PYTHON="${EXP01_PYTHON:-python}"
"$EXP01_PYTHON" model-pretrained/scripts/exp01_mattext_frozen/slurm_control.py \
  --config "${EXP01_CONFIG:-experiments/exp01_mattext_frozen/config.yaml}" status
