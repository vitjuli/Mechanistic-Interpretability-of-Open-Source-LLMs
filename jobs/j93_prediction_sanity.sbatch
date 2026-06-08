#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j93-pred
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j93-%j.out
#SBATCH --error=logs/j93-%j.out
#
# 93 — Prediction sanity. Three-way comparison of α/β predictions on the same 538 prompts:
#   (1) single_nopad     — one prompt per forward, no padding (canonical reference)
#   (2) leftpad_naive    — left-padded batches, default position_ids (= j92 setup)
#   (3) leftpad_posfix   — left-padded batches with explicit corrected position_ids
# Plus optional comparison vs per_prompt_cluster_effect_matrix.csv baseline.
#
# Diagnoses whether RoPE position_id corruption under left-padding is responsible
# for the j92 vs CSV discrepancy in failed-β counts.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/prediction_sanity

echo "========================================================"
echo "  93 — prediction sanity (single vs leftpad vs leftpad_posfix)"
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

# CSV with per-prompt baseline (from cluster joint ablation analyses).
# Script will auto-skip CSV comparison if file is absent.
CSV="data/analysis/runD_v2/cluster_joint_ablation_subgroup/per_prompt_cluster_effect_matrix.csv"

echo ""
echo "==== Three-way prediction sanity run ===="
date
python -u scripts/93_prediction_sanity.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --csv "$CSV" \
    --out_dir data/analysis/runD_v2/prediction_sanity

echo ""
echo "==== Done: $(date) ===="
