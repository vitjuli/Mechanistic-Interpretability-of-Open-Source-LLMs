#!/bin/bash
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --job-name=j116-flip-law
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/j116-%j.out
#SBATCH --error=logs/j116-%j.out
#
# 116 — Flip-law retro: GO/NO-GO gate. CPU only, reads from 119's field_dump.
# Computes predicted surface F̂ over full corpus × all 36 layers ×
# {w_res, usage, shuffled×3, random×3} × c ∈ {0.5...32}, then retro-joins
# with measured CSVs from 86 (steering_efficiency.csv) and 89
# (calculus_points.csv). Split reconstructed bit-for-bit per 86 convention.
# Prints empirical validity radius c* and verdict:
#   GO     → single-u predictor form fixed, finalise headline analysis
#   REVISE → try U_k subspace / 2nd-order correction before B1 battery
# Expected runtime: ~10-20 min CPU.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/flip_law_retro

echo "========================================================"
echo "  116 — flip-law retro (GO/NO-GO gate)"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
date

# Use retro-compatible dump for apples-to-apples vs 86/89
python -u scripts/116_flip_law_retro.py \
    --dump_dir data/analysis/runD_v2/field_dump \
    --measured_86 data/analysis/runD_v2/usage_direction/steering_efficiency.csv \
    --measured_89 data/analysis/runD_v2/intervention_calculus/calculus_points.csv

echo ""
echo "==== Done: $(date) ===="
