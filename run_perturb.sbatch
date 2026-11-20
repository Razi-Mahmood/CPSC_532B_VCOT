#!/bin/bash
#SBATCH --job-name=vlmcot_perturb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_b300_sxm6_ac_2g.67gb:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Runs scripts/run_perturb.py (CoT rationale perturbations) -> results/<run_name>/
# Submit from the repo root:  sbatch run_perturb.sbatch [configs/perturb.yaml]
# ~/.bashrc exits early in non-interactive shells, so load conda directly
source /data/scratch/users/razi/miniforge3/etc/profile.d/conda.sh
conda activate vlmcot

cd "$SLURM_SUBMIT_DIR"
python scripts/run_perturb.py --config "${1:-configs/perturb.yaml}"
