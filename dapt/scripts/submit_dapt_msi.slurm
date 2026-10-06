#!/bin/bash
#SBATCH --job-name=slices_dapt
#SBATCH --partition=msigpu       # VERIFY WITH MSI BEFORE SUBMISSION
#SBATCH --gres=gpu:1             # VERIFY GPU REQUEST SYNTAX WITH MSI
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00          # VERIFY TIME LIMIT WITH MSI
#SBATCH --output=dapt/outputs/msi/logs/dapt_%j.out
#SBATCH --error=dapt/outputs/msi/logs/dapt_%j.err
# Add #SBATCH --account=<YOUR_MSI_ACCOUNT> only after confirming the required account.
# VERIFY account, partition, GPU type/request, and time limit with MSI before submission.

set -euo pipefail

# Submit this file from the project root so SLURM_SUBMIT_DIR is the project root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$PROJECT_ROOT"

source ~/.bashrc
CONDA_ENV_NAME="${CONDA_ENV_NAME:-SET_ENV_NAME}"
if [[ "$CONDA_ENV_NAME" == "SET_ENV_NAME" ]]; then
  echo "Set CONDA_ENV_NAME to the existing MSI environment before submission." >&2
  exit 2
fi
conda activate "$CONDA_ENV_NAME"

printf 'Host: '; hostname
printf 'Date: '; date
printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Python: '; python --version
nvidia-smi

python dapt/scripts/run_dapt_msi.py \
  --config dapt/configs/dapt_msi.yaml \
  --resume auto
