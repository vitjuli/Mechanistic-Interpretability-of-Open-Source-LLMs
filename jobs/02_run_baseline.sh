#!/bin/bash
#SBATCH -J baseline
#SBATCH -A CHANGEME-SL2-CPU        # CHANGE THIS to your CPU account
#SBATCH -p cclake                   # CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00             # Longer time for CPU
#SBATCH --mem=64G                   # More memory for CPU inference
#SBATCH -o logs/02_baseline_%j.out
#SBATCH -e logs/02_baseline_%j.err

# Load modules (no CUDA needed)
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# Activate environment
source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

# Set environment
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# Force CPU
export CUDA_VISIBLE_DEVICES=""

# Run
echo "=== Starting baseline evaluation (CPU) ==="
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
date

python scripts/02_run_baseline.py

echo "=== Completed ==="
date
