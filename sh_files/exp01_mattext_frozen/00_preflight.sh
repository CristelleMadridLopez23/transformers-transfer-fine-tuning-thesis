#!/usr/bin/env bash
# Run only bounded audits and smoke tests on login; failed audits stop submission.
set -euo pipefail
EXP01_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$EXP01_ROOT"
EXP01_CONFIG="${EXP01_CONFIG:-experiments/exp01_mattext_frozen/config.yaml}"
EXP01_PYTHON="${EXP01_PYTHON:-python}"
mkdir -p reports/exp01_mattext_frozen/logs
"$EXP01_PYTHON" model-pretrained/scripts/exp01_mattext_frozen/run_experiment.py \
  --config "$EXP01_CONFIG" --phase preflight "$@" 2>&1 | tee reports/exp01_mattext_frozen/logs/preflight.log
