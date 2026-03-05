#!/bin/bash
#SBATCH -J transcoder
#SBATCH -A CHANGEME-SL2-CPU        # CHANGE THIS to your CPU account
#SBATCH -p cclake                   # CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00             # Longer time for CPU
#SBATCH --mem=128G
#SBATCH -o logs/04_transcoder_%j.out
#SBATCH -e logs/04_transcoder_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export CUDA_VISIBLE_DEVICES=""

echo "=== Starting transcoder feature extraction (CPU) ==="
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
date

python scripts/04_extract_transcoder_features.py

echo "=== Completed ==="
date
