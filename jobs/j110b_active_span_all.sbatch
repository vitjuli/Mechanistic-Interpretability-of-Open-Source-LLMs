#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j110b-span-triad-all
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/j110b-%j.out
#SBATCH --error=logs/j110b-%j.out
#
# 110b — MERGED: active-span + per-feature triad map over ALL 36 LAYERS.
# Script 110 now emits THREE CSVs from a single capture pass:
#   - active_span_triad.csv        (aggregate per-layer span/null/angles)
#   - feature_triad_alignment.csv  (per-feature 3-axis cos + classification)
#   - triad_type_summary.csv       (per-layer 8-bucket counts)
# --layers -1 sentinel runs all 36 layers.
# Writes to active_span_all/ — separate folder from the original 7-layer
# active_span/ to preserve both runs. Replaces the deprecated j113.
# Expected runtime: ~10 min real (forward+grad over 538 prompts with 36 hooks,
# then ~3 s per layer for transcoder encode + greedy + null + per-feature cos).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/active_span_all

echo "========================================================"
echo "  110b — active-span + per-feature triad — ALL 36 LAYERS"
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

python -u scripts/110_active_span_triad.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers -1 \
    --tau 0.05 \
    --triad_threshold 0.05 \
    --out_dir data/analysis/runD_v2/active_span_all

echo ""
echo "==== Done: $(date) ===="
