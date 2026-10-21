#!/usr/bin/env bash
# Install tokenizer reference dependencies in this experiment only; never modify dmc.
set -euo pipefail
MATHEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MATHEX_ROOT"
MATHEX_PYTHON="${MATHEX_PYTHON:-python}"
MATHEX_PACKAGES="$MATHEX_ROOT/experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages"
if [[ -d "$MATHEX_PACKAGES/transformers" ]]; then
  PYTHONPATH="$MATHEX_PACKAGES" "$MATHEX_PYTHON" -B -c 'from importlib.metadata import version; assert version("transformers")=="4.38.2"; assert version("tokenizers")=="0.15.2"; assert version("huggingface_hub")=="0.23.4"'
else
  "$MATHEX_PYTHON" -m pip install --disable-pip-version-check --no-compile --no-deps \
    --index-url https://pypi.org/simple --target "$MATHEX_PACKAGES" \
    transformers==4.38.2 tokenizers==0.15.2 huggingface-hub==0.23.4
fi
