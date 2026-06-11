#!/bin/bash
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --qos=cclake
#SBATCH --job-name=j118-null-calib
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j118-%j.out
#SBATCH --error=logs/j118-%j.out
#
# 118 — Null calibration. CPU only, reads from 119's field_dump.
# Computes three null distributions:
#   (1) isotropic Gaussian (analytic √(2/πd) and sampled, should match)
#   (2) covariance-matched (inflated by anisotropy)
#   (3) within-span (concept-axis sub-space)
# Plus d_eff(ℓ) via Gram trick (no 2560×2560 covariance built),
# family-bootstrap → r_w, r_u → disattenuation bound.
# Expected runtime: ~20-40 min CPU.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/null_calibration

echo "========================================================"
echo "  118 — null calibration + d_eff + disattenuation"
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

# Use retro-compatible dump (v2_train only) so all numbers match 86/89 split
python -u scripts/118_null_calibration.py \
    --dump_dir data/analysis/runD_v2/field_dump

echo ""
echo "==== Done: $(date) ===="
