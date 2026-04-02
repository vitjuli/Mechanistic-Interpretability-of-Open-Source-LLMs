#!/bin/bash
#SBATCH --job-name=qwen_sae_train
#SBATCH --partition=ampere
#SBATCH --array=10-24              # Train SAEs for layers 10-24 in parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00            # SAE training can take hours
#SBATCH --output=logs/sae_train_L%a_%j.out
#SBATCH --error=logs/sae_train_L%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@cam.ac.uk

# Train SAEs in parallel using SLURM job arrays
# Each array task trains one layer
# Usage: sbatch slurm_scripts/03_train_sae_array.slurm

echo "=========================================="
echo "SAE Training - Layer ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

module purge
module load python/3.10
module load cuda/12.1

source venv/bin/activate

export TRANSFORMERS_CACHE=/home/${USER}/.cache/huggingface
export HF_HOME=/home/${USER}/.cache/huggingface

# Train SAE for this layer
LAYER=${SLURM_ARRAY_TASK_ID}

echo "Training SAE for layer ${LAYER}..."

python scripts/04_train_sae.py \
    --behaviour grammar_agreement \
    --layer ${LAYER} \
    --expansion_factor 4 \
    --max_steps 50000 \
    --batch_size 256 \
    --checkpoint_every 5000

echo "=========================================="
echo "Layer ${LAYER} completed: $(date)"
echo "=========================================="
