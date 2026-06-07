#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j89-calc
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/j89-%j.out
#SBATCH --error=logs/j89-%j.out
#
# 89 — Intervention calculus (predicted vs measured Delta margin: R^2/slope, validity radius)
# + minimal flipping perturbation (Newton on locally-linear margin, real fwd+bwd per step).
# Decomposes delta* into s_concept (share in INLP concept map) and s_usage (share in U_u).
# Lever-cone PR = empirical non-identifiability dimension. ~1 h.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/intervention_calculus

echo "========================================================"
echo "  89 — intervention calculus + minimal flip"
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

# ── smoke (2-3 min) ───────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L21 L35, 5 calc + 3 flip targets, c=1 4) ===="
date
python -u scripts/89_intervention_calculus.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 21 35 \
    --calc_targets 5 \
    --flip_targets 3 \
    --c_grid 1 4 \
    --n_random 1 \
    --out_dir data/analysis/runD_v2/intervention_calculus/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run (default taps 8 16 21 24 30 35, c=0.5 1 4 16, 30+20 targets) ──
echo ""
echo "==== FULL RUN (taps 8 16 21 24 30 35, c=0.5 1 4 16) ===="
date
python -u scripts/89_intervention_calculus.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/intervention_calculus

echo ""
echo "==== Done: $(date) ===="
