#!/usr/bin/env bash
#SBATCH --job-name=mattext_e01_extract
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=reports/exp01_mattext_frozen/logs/extract_%j.out
#SBATCH --error=reports/exp01_mattext_frozen/logs/extract_%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit through submit_experiment.sh}"
: "${EXP01_CONFIG:?Missing validated experiment config}"
: "${EXP01_CONDA_BASE:?Missing verified conda base}"
: "${EXP01_ENV_PREFIX:?Missing existing conda environment}"
cd "$SLURM_SUBMIT_DIR"
test -f AGENTS.md || { echo 'Wrong project working directory' >&2; exit 2; }
source "$EXP01_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$EXP01_ENV_PREFIX"
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python model-pretrained/scripts/exp01_mattext_frozen/run_experiment.py --config "$EXP01_CONFIG" --phase extract
