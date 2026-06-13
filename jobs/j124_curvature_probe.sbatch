#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j124-curvature
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/j124-%j.out
#SBATCH --error=logs/j124-%j.out
#
# 124 — Curvature probe on B1_alpha_beta dump. Tests if the 16 out-of-tolerance
# cells in (L19-25, usage, c=4-16) zone are explained by 2nd-order curvature.
# κ split by class — opposite signs = S-shape signature of nonlinear ignition.
# Expected runtime: ~15-20 min real on A100.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/B1_alpha_beta/curvature

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "==== 124: curvature probe on B1_alpha_beta ===="
date

python -u scripts/124_curvature_probe.py

echo ""
echo "==== Done: $(date) ===="
