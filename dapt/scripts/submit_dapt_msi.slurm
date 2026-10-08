#!/bin/bash
#SBATCH --job-name=slices_dapt
#SBATCH --partition=msigpu       # VERIFY WITH MSI BEFORE SUBMISSION
#SBATCH --account=cbartel
#SBATCH --gres=gpu:1             # VERIFY GPU REQUEST SYNTAX WITH MSI
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00          # msigpu MaxTime is 1 day; the runner supports auto-resume.
#SBATCH --output=dapt/outputs/msi/logs/dapt_%j.out
#SBATCH --error=dapt/outputs/msi/logs/dapt_%j.err

# Submit this file from the project root so SLURM_SUBMIT_DIR is the project root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$PROJECT_ROOT"

# MSI's /etc/bashrc reads BASHRCSOURCED without guarding it, so source bashrc
# before enabling nounset. Keep errexit and pipefail for the training commands.
source ~/.bashrc
set -eo pipefail
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dmc}"
conda activate "$CONDA_ENV_NAME"

echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "User: $USER"
echo "Python path: $(which python)"
python --version
nvidia-smi
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

python dapt/scripts/run_dapt_msi.py \
  --config dapt/configs/dapt_msi.yaml \
  --resume auto
