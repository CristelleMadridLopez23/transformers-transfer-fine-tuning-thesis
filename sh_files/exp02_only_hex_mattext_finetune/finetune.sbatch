#!/usr/bin/env bash
#SBATCH --job-name=mattext_hex_ft
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=reports/exp02_only_hex_mattext_finetune/logs/finetune_%j.out
#SBATCH --error=reports/exp02_only_hex_mattext_finetune/logs/finetune_%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from project root}"
: "${MATHEX_CONFIG:?Use the audited submission wrapper}"
: "${MATHEX_CONDA_BASE:?Missing verified Conda base}"
: "${MATHEX_ENV_PREFIX:?Missing existing Conda environment}"
cd "$SLURM_SUBMIT_DIR"
test -f AGENTS.md || { echo 'Incorrect project directory' >&2; exit 2; }
source "$MATHEX_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$MATHEX_ENV_PREFIX"
export PYTHONPATH="$SLURM_SUBMIT_DIR/experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/02_finetune.py --config "$MATHEX_CONFIG" --fold "${SLURM_ARRAY_TASK_ID:--1}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python -B model-pretrained/scripts/exp02_only_hex_mattext_finetune/03_evaluate.py --config "$MATHEX_CONFIG"
fi
