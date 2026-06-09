#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j112-quad
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j112-%j.out
#SBATCH --error=logs/j112-%j.out
#
# 112 — Quadrant analysis of feature population. For each active feature, classify by
#   high/low activation × high/low |attribution|. Four quadrants:
#     Q1 high-act × high-attr  — used carriers (write the answer)
#     Q2 high-act × low-attr   — quiet readers (fire but don't write the answer)
#     Q3 low-act × high-attr   — sparse strong levers (rare but causal)
#     Q4 low-act × low-attr    — background
# Forced regime. Needs transcoders. ~7-10 min real.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/quadrants

echo "========================================================"
echo "  112 — feature quadrants (used vs quiet, etc)"
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
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

python -u scripts/112_feature_attribution_quadrants.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/quadrants

echo ""
echo "==== Done: $(date) ===="
