#!/bin/bash
#SBATCH --job-name=qwen3_baseline
#SBATCH --partition=ampere         # CSD3 GPU partition
#SBATCH --account=DAMTP-SL2-GPU    # Adjust to your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1               # Single A100 GPU for baseline
#SBATCH --mem=48G                  # Qwen3-4B needs ~15GB, extra for safety
#SBATCH --time=03:00:00            # Extended for 4 behaviors
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@cam.ac.uk

# Cambridge CSD3 SLURM job for baseline measurements
# Model: Qwen3-4B-Instruct-2507
# Behaviours: grammar_agreement, factual_recall, sentiment_continuation, arithmetic
#
# Usage: sbatch slurm_scripts/01_baseline.slurm

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-4B Baseline Performance Measurement"
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Load CSD3 modules
module purge
module load rhel8/default-amp
module load python/3.10.5-gcc-11.2.0
module load cuda/12.1
module load cudnn/8.9_cuda-12.1

# Activate virtual environment
cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Set HuggingFace cache to shared storage
export HF_HOME=/rds/project/rds-YOUR_PROJECT/hf_cache  # Update with your RDS project
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME/torch

# Create cache directories
mkdir -p $HF_HOME
mkdir -p logs

# Generate prompts for all 4 behaviours (if not already done)
if [ ! -f data/prompts/arithmetic_train.jsonl ]; then
    echo "Generating prompts for all behaviours..."
    python scripts/01_generate_prompts.py
fi

# Run baseline for all behaviours
echo ""
echo "Running baseline measurements on all 4 behaviours..."
echo ""

python scripts/02_run_baseline.py --all --split train
python scripts/02_run_baseline.py --all --split test

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Results saved to: data/results/"
echo "Visualizations: data/results/figures/"
echo "=========================================="
