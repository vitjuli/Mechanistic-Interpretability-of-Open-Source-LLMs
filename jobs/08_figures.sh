#!/bin/bash
#SBATCH -J figures
#SBATCH -A CHANGEME-SL2-CPU        # CHANGE THIS to your CPU account
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -o logs/08_figures_%j.out
#SBATCH -e logs/08_figures_%j.err

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source ~/rds/hpc-work/thesis/project/venv/bin/activate
cd ~/rds/hpc-work/thesis/project

echo "=== Generating figures ==="
echo "Hostname: $(hostname)"
date

python scripts/08_generate_figures.py

echo "=== Completed ==="
date
