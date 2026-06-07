#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j88e2-c
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j88e2-%j.out
#SBATCH --error=logs/j88e2-%j.out
#
# 88e2 — Carrier dozapusk with PROPER decoder-vs-random null.
# The first j88e used isotropic √(2 ln F / d) ≈ 0.097 baseline, which UNDER-estimates
# the null for an anisotropic over-complete transcoder dictionary (163840 in 2560-d, 64×
# redundancy). Correct null = max|cos(d_f, r)| for random unit r against the SAME
# decoder rows — fully computed inside the script now (decoder_vs_random_max_mean/p95).
#
# Extended carrier_layers: 10-25 (original hot zone) + 30, 33, 34, 35 (late usage layers).
# L34/35 are the j86 usage hot zone (AUC along u ~ 0.97) — important to check whether
# the dictionary actually carries u there or whether u ≈ γ̄ just makes it look that way.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/usage_subspace_e2

echo "========================================================"
echo "  88e2 — carrier-capture with DECODER-AWARE null"
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

# transcoder_loader.py lives at src/transcoder/
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

echo "Python: $(which python)"
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

# Carrier layers extended to include late usage region (L30, L33-35).
# Erasure-side capped at 16 + 4 = 20 layers but with --max_targets 5 the C-phase is tiny.
python -u scripts/88_usage_subspace.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 30 33 34 35 \
    --carrier_layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 30 33 34 35 \
    --k_grid 5 \
    --max_targets 5 \
    --cum_windows \
    --carrier \
    --out_dir data/analysis/runD_v2/usage_subspace_e2

echo ""
echo "==== Done: $(date) ===="
