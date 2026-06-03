#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j76-recall
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j76-%j.out
#SBATCH --error=logs/j76-%j.out
#
# 76 — Pre-check baseline: does Qwen3-4B know decay modes of named isotopes
# by recall (no physics hints)?  Validates D0 recall-end of the gradient
# before investing GPU in the full battery.
#
# ~110 prompts, one forward each. Fast: <30 min.
#
# Decisive metric: accuracy on the HEURISTIC-BREAKING subset (heavy β-emitters
# and medium-mass α-emitters).  If ≥0.75 → real recall, D0 valid → build gradient.
# If ≈0.5 with high heuristic-agreement → not recall, redesign.

set -uo pipefail
mkdir -p logs data/prompts data/results

echo "========================================================"
echo "  76 — recall baseline pre-check"
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

python -u scripts/76_recall_baseline_precheck.py \
    --out_dir data/prompts

echo ""
echo "==== Done: $(date) ===="
