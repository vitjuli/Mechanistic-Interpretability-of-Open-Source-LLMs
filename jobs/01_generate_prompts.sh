#!/bin/bash
#SBATCH -J gen_prompts
#SBATCH -A CHANGEME-SL2-CPU        # CHANGE THIS to your account
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -o logs/01_prompts_%j.out
#SBATCH -e logs/01_prompts_%j.err

# Load modules
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# Activate environment
source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

# Run
echo "=== Starting prompt generation ==="
echo "Hostname: $(hostname)"
date
python scripts/01_generate_prompts.py
echo "=== Completed ==="
date
