#!/usr/bin/env bash
#SBATCH --job-name=only_hex_new_slices
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=reports/only_hex_new_csv_slices/logs/slices_%j.out
#SBATCH --error=reports/only_hex_new_csv_slices/logs/slices_%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${HEX_CONDA_BASE:?Missing verified existing conda base}"
: "${HEX_ENV_PREFIX:?Missing existing Python environment}"
cd "$SLURM_SUBMIT_DIR"
test -f experiments/only_hex_new_csv_slices/config.json || { echo 'Incorrect project directory' >&2; exit 2; }
source "$HEX_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$HEX_ENV_PREFIX"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
python -B model-pretrained/scripts/only_hex_new_csv_slices/02_csv_to_slices.py
