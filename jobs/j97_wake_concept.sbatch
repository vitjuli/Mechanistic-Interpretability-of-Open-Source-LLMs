#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j97-wake
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j97-%j.out
#SBATCH --error=logs/j97-%j.out
#
# 97 — Wake-concept by competitor suppression. Define competitor
#   d_error = mean(failed-β residual) - mean(succ-β residual) at layer L
# This is the direction along which failure differs from success — and we know
# concept (w_res) does NOT distinguish them (j92: AUC_wres f|s ≈ 0.40), so
# d_error ⊥ w_res, meaning suppressing d_error removes the competitor signal
# WITHOUT touching the concept.
#
# Project out d_error at L14/17/20 (mid-stack) + L30 (late readout-adjacent control),
# rerun downstream, measure: did failed-β prompts now flip to β?
#
# Built-in tautology guards (printed per layer):
#   * cos(d_err, w_res) — must be small (else we're killing the concept, not the competitor)
#   * concept PRESERVED check — w_res projection before/after suppression must agree
#   * L30 control — if recovery only happens at L30+, it's a readout tautology, not waking

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/wake_concept

echo "========================================================"
echo "  97 — wake concept via competitor suppression"
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

# ── smoke (~3 min) ────────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L17 only) ===="
date
python -u scripts/97_wake_concept_suppression.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 17 --n_random 2 \
    --out_dir data/analysis/runD_v2/wake_concept_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~20-30 min) ─────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (L14/17/20 + L30 control) ===="
date
python -u scripts/97_wake_concept_suppression.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/wake_concept

echo ""
echo "==== Done: $(date) ===="
