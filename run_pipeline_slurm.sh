#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cclake
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# SLURM job script for running full pipeline on CSD3

# Load environment
module purge
module load python/3.8  # Adjust to your Python version
source /path/to/your/venv/bin/activate  # UPDATE THIS PATH!

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/your/cache  # UPDATE THIS PATH!

# Run pipeline
echo "=========================================="
echo "SLURM JOB: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd /path/to/project  # UPDATE THIS PATH!

# Run the pipeline script
bash run_pipeline.sh

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
