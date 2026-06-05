#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j82-scan
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/j82-%j.out
#SBATCH --error=logs/j82-%j.out
#
# 82 — Attention writer scan (all 36 layers × 32 heads = 1152 heads).
# Both metrics: |d_along_wres| and intrinsic_auc (axis-invariant).
# Top-50 intervention with intact-flip behavioural metric.
# Cumulative ablation curve through k=50.
#
# Two-phase: smoke test (2 layers, 5 targets) then full run.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/writer_scan

echo "========================================================"
echo "  82 — attention writer scan"
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

# ── (1) smoke test (2 layers, 5 targets) ────────────────────────────────────
echo ""
echo "==== SMOKE TEST (layers 20 21, 5 targets, 3 nulls) ===="
date
python -u scripts/82_attention_writer_scan.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 20 21 \
    --top_k_heads 5 \
    --max_targets 5 \
    --n_random_head 3 \
    --out_dir data/analysis/runD_v2/writer_scan/smoke

SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "SMOKE TEST FAILED (exit $SMOKE_RC). Aborting full run."
    exit 1
fi
echo "==== smoke OK ===="

# ── (2) full run (all 36 layers, top-50 heads, n_random=30) ──────────────────
echo ""
echo "==== FULL RUN (all 36 layers, top-50, n_random=30) ===="
date
python -u scripts/82_attention_writer_scan.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --top_k_heads 50 \
    --n_random_head 30 \
    --out_dir data/analysis/runD_v2/writer_scan

echo ""
echo "==== Done: $(date) ===="
