#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j94-dump
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/j94-%j.out
#SBATCH --error=logs/j94-%j.out
#
# 94 — Dump REAL per-prompt predictions using single-prompt no-padding eval
# (canonical setup confirmed by j93). Replaces the broken `baseline_logit_diff`
# CSV column as the source of truth for error-structure analysis. Writes
# `real_predictions.csv` with prompt_idx, prediction, real margin, AND all
# metadata fields (relation_type, concept_route, level_label, framing, etc.)
# for honest local re-analysis.
#
# Real runtime estimate: ~5 min (model load ~3 min + 538 forward passes ~30 sec).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/real_predictions

echo "========================================================"
echo "  94 — dump REAL per-prompt predictions (no padding)"
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

echo ""
echo "==== Single no-pad dump of 538 real predictions ===="
date
python -u scripts/94_dump_real_predictions.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/real_predictions

echo ""
echo "==== Done: $(date) ===="
