#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j98-heal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j98-%j.out
#SBATCH --error=logs/j98-%j.out
#
# 98 — Consolidated healing contest in FORCED format (j90-style 2-shot template).
# Replaces 96+97 with three critical fixes:
#   (1) Forced template lifts intact-rate to ~1.0 → intact-flip is now informative
#   (2) Fine α-sweep 0.25/0.5/1.0/1.5/2.0 × RMS — small enough to keep the null inert
#   (3) Valid-window calibration: report only at α where null intact-flip ≈ 0
#
# Two parts in one pass: (A) inject concept/surface/random at L17/20/24,
# (B) suppress d_error (project out failed-vs-succ direction) at same layers.
#
# Built-in guards:
#   * cos(d_surf, w_res) and cos(d_err, w_res) printed per layer (must be small)
#   * concept-preserved check after suppression (w_res projection ratio)
#   * valid-window flag — if null intact-flip > 0 even at α=0.25, mark UNINFORMATIVE

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/healing_forced

echo "========================================================"
echo "  98 — healing contest in forced regime"
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

# ── smoke (~4-5 min) ──────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L20, alphas 0.5 1.0, 2 randoms) ===="
date
python -u scripts/98_healing_contest_forced.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 20 --alphas 0.5 1.0 --n_random 2 \
    --out_dir data/analysis/runD_v2/healing_forced_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~25-35 min) ─────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (L17/20/24, alphas 0.25-2.0, full nulls) ===="
date
python -u scripts/98_healing_contest_forced.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/healing_forced

echo ""
echo "==== Done: $(date) ===="
