#!/bin/bash
#SBATCH --job-name=vcot_pilot
#SBATCH -p mig
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

# Submit from the folder that contains pilot.py:
#   mkdir -p logs results && sbatch run_pilot.sbatch
# Optional: sbatch run_pilot.sbatch 20   (number of questions, default 10)

set -euo pipefail
N=${1:-10}

# conda's activate.d scripts reference unset variables, so relax -u while activating.
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test_env
set -u

# First run only: install dependencies if missing (no-op afterwards).
python -c "import qwen_vl_utils, datasets, accelerate" 2>/dev/null || \
    pip install -q "transformers>=4.49" accelerate datasets qwen-vl-utils pillow

# Keep the ~16 GB model download out of your home quota if needed:
# export HF_HOME=/path/to/scratch/hf_cache

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

python pilot.py --n 100

echo "Done at $(date)"
