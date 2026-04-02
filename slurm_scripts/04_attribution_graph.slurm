#!/bin/bash
#SBATCH --job-name=qwen3_attribution
#SBATCH --partition=ampere
#SBATCH --account=DAMTP-SL2-GPU    # Adjust to your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/attribution_%j.out
#SBATCH --error=logs/attribution_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@cam.ac.uk

# Build attribution graphs from SAE features
# Model: Qwen3-4B-Instruct-2507
# Behaviours: All 4 (grammar_agreement, factual_recall, sentiment_continuation, arithmetic)
#
# Usage: sbatch slurm_scripts/04_attribution_graph.slurm

set -e

echo "=========================================="
echo "Attribution Graph Construction"
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load CSD3 modules
module purge
module load rhel8/default-amp
module load python/3.10.5-gcc-11.2.0
module load cuda/12.1
module load cudnn/8.9_cuda-12.1

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

export HF_HOME=/rds/project/rds-YOUR_PROJECT/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Build attribution graphs for all behaviours
echo "Building attribution graphs for all behaviours..."

python scripts/06_build_attribution_graph.py \
    --all \
    --split train \
    --n_prompts 30

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Output: data/results/attribution_graphs/"
echo "=========================================="
