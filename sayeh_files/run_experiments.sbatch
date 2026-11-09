#!/bin/bash
#SBATCH --job-name=vcot_exp
#SBATCH -p mig
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

# All arguments are passed to run_experiments.py, e.g.:
#   sbatch run_experiments.sbatch --dataset scienceqa --n 100
#   sbatch run_experiments.sbatch --dataset scienceqa --n 100 --reverse_choices
#   sbatch run_experiments.sbatch --dataset scienceqa --n 100 --drop_conclusion --skip_paraphrase
#   sbatch run_experiments.sbatch --dataset aokvqa --n 500

set -eo pipefail
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test_env

echo "Job $SLURM_JOB_ID on $(hostname) at $(date): $*"
python run_experiments.py "$@"
echo "Done at $(date)"
