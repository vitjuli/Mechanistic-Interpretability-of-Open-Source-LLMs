#!/bin/bash
#SBATCH -J neighbor
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH -o logs/neighbor-%j.out
#SBATCH -e logs/neighbor-%j.out
set -uo pipefail
module purge; module load rhel8/default-amp 2>/dev/null || true
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate
export PYTHONPATH="scripts:src/transcoder:src:${PYTHONPATH:-}"
mkdir -p logs; date
python -u scripts/particle_neighbor_matrix.py
echo "=== done $(date) ==="
