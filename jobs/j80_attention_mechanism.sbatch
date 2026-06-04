#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j80-attn
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/j80-%j.out
#SBATCH --error=logs/j80-%j.out
#
# 80 — Attention CIE mechanism test.
# Answers (A) do attention heads carry the concept (vs random-head null),
#         (B) is w_res subspace causal under CIE patching (vs random-direction null)
#
# Strategy: first a 5-prompt smoke test to verify Qwen3 module names / W_O shape /
# hook compatibility, then the full run on heldout β-target prompts.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/mechanism

echo "========================================================"
echo "  80 — attention CIE mechanism"
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
echo ""
date

# ── (1) SMOKE TEST — small, fast, verifies architecture compatibility ───────
echo ""
echo "==== SMOKE TEST (5 targets, 3 nulls) ===="
date
python -u scripts/80_attention_cie_mechanism.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 14 18 21 24 \
    --top_k_heads 5 \
    --max_targets 5 \
    --n_random_head 3 \
    --n_random_dir 3 \
    --out_dir data/analysis/runD_v2/mechanism/smoke

SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "SMOKE TEST FAILED (exit $SMOKE_RC). Aborting full run."
    exit 1
fi
echo "==== smoke OK ===="
echo ""

# ── (2) FULL RUN ────────────────────────────────────────────────────────────
echo "==== FULL RUN ===="
date
python -u scripts/80_attention_cie_mechanism.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 14 18 21 24 \
    --top_k_heads 10 \
    --n_random_head 20 \
    --n_random_dir 20 \
    --out_dir data/analysis/runD_v2/mechanism

echo ""
echo "==== Done: $(date) ===="
