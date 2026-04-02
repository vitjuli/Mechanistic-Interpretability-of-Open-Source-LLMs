#!/bin/bash
#SBATCH --job-name=qwen_activations
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                  # More memory for activation storage
#SBATCH --time=04:00:00
#SBATCH --output=logs/activations_%j.out
#SBATCH --error=logs/activations_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@cam.ac.uk

# Capture activations for circuit analysis
# Usage: sbatch slurm_scripts/02_capture_activations.slurm

echo "=========================================="
echo "Activation Capture Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

module purge
module load python/3.10
module load cuda/12.1

source venv/bin/activate

export TRANSFORMERS_CACHE=/home/${USER}/.cache/huggingface
export HF_HOME=/home/${USER}/.cache/huggingface

# Run activation capture
python scripts/03_capture_activations.py \
    --behaviour grammar_agreement \
    --split train \
    --batch_size 16

python scripts/03_capture_activations.py \
    --behaviour factual_recall \
    --split train \
    --batch_size 16

echo "Activation capture completed: $(date)"
