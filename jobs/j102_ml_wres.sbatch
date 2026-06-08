#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j102-ml
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j102-%j.out
#SBATCH --error=logs/j102-%j.out
#
# 102 — Multi-layer w_res steering: does composing the push across many layers help?
#   (A) Single-layer profile — does w_res steering work at SOME layer alone?
#   (B) Composed across all layers — does maintaining a w_res signal along the
#       stack reveal the readable axis as a causal lever?
# Multi-layer nulls: random-shuffled-label w_res per layer, and norm-matched random
# directions per layer. Only valid windows (null intact ≈ 0) are reported.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/ml_wres

echo "========================================================"
echo "  102 — multi-layer w_res steering (single + composed)"
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

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
nvidia-smi || true
date

# ── smoke (~5 min) ────────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (c_single=8, c_comp=1, n=8) ===="
date
python -u scripts/102_multilayer_wres_steering.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --c_single 8 --c_comp 1 --n_single 8 --n_comp 8 --n_shuffle 1 \
    --out_dir data/analysis/runD_v2/ml_wres_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~45-60 min) ─────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (single c=2,8 + composed c=0.25,0.5,1,2; n=32) ===="
date
python -u scripts/102_multilayer_wres_steering.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --c_single 2 8 --c_comp 0.25 0.5 1 2 --n_comp 32 \
    --out_dir data/analysis/runD_v2/ml_wres

echo ""
echo "==== Done: $(date) ===="
