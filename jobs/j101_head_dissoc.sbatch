#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j101-hd
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/j101-%j.out
#SBATCH --error=logs/j101-%j.out
#
# 101 — Head-level dissociation: top-K answer-writers (via o_proj DLA of γbar)
# vs top-K axis-writers (via o_proj projection onto w_res), in FORCED regime.
# Ablate each set on mid-layers [12-28]; report intact-flip vs random-head null.
# Key claim sought: answer-writers break behaviour, axis-writers do not.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/head_dissoc

echo "========================================================"
echo "  101 — head dissociation (answer-writers vs axis-writers)"
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
echo "==== SMOKE (topk=6, n_eval=8, n_random=1) ===="
date
python -u scripts/101_head_dissociation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --topk 6 --n_eval 8 --n_random 1 \
    --out_dir data/analysis/runD_v2/head_dissoc_smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full (~30-45 min) ─────────────────────────────────────────────────────
echo ""
echo "==== FULL RUN (topk=10, mid [12-28], n=24) ===="
date
python -u scripts/101_head_dissociation.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --topk 10 --mid_lo 12 --mid_hi 28 --n_eval 24 \
    --out_dir data/analysis/runD_v2/head_dissoc

echo ""
echo "==== Done: $(date) ===="
