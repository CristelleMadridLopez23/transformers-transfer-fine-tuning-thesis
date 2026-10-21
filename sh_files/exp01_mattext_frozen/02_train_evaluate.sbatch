#!/usr/bin/env bash
#SBATCH --job-name=mattext_e01_heads
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=reports/exp01_mattext_frozen/logs/train_%j.out
#SBATCH --error=reports/exp01_mattext_frozen/logs/train_%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit through submit_experiment.sh}"
: "${EXP01_CONFIG:?Missing validated experiment config}"
: "${EXP01_CONDA_BASE:?Missing verified conda base}"
: "${EXP01_ENV_PREFIX:?Missing existing conda environment}"
cd "$SLURM_SUBMIT_DIR"
test -f AGENTS.md || { echo 'Wrong project working directory' >&2; exit 2; }
source "$EXP01_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$EXP01_ENV_PREFIX"
export OMP_NUM_THREADS=2
python model-pretrained/scripts/exp01_mattext_frozen/run_experiment.py --config "$EXP01_CONFIG" --phase train-evaluate
