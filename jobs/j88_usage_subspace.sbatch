#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j88-usub
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/j88-%j.out
#SBATCH --error=logs/j88-%j.out
#
# 88 — Usage SUBSPACE: gram spectrum, principal angles vs concept INLP map,
# DUAL erasure (recentered, all positions) with random-frame + construction-contrast nulls,
# saliency, carrier-capture (decoder columns of 163k transcoder features).
# ~3 h on A100-80GB.  Mirror image of 87 (which erased what WE read; here we erase what
# the MODEL uses).  Lesson from 87 baked in: three-tier verdict, final-tap sanity required.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/usage_subspace

echo "========================================================"
echo "  88 — usage subspace (gram + angles + DUAL erasure + carrier)"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
nvidia-smi || true
date

# ── smoke (2-4 min, no carrier, no cumulative) ─────────────────────────────
echo ""
echo "==== SMOKE (layers 20 21 35, k=5, 5 targets) ===="
date
python -u scripts/88_usage_subspace.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 20 21 35 \
    --k_grid 5 \
    --n_random 1 \
    --max_targets 5 \
    --cum_windows \
    --no_carrier \
    --out_dir data/analysis/runD_v2/usage_subspace/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run (all 36 layers, k=1/5/13, cumulative 4 and 8, carrier 10-25) ──
echo ""
echo "==== FULL RUN (all layers, k=1/5/13, cumulative windows, carrier) ===="
date
python -u scripts/88_usage_subspace.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/usage_subspace

echo ""
echo "==== Done: $(date) ===="
