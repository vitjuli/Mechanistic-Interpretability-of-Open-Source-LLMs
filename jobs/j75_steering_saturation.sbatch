#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j75-sat
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/j75-%j.out
#SBATCH --error=logs/j75-%j.out
#
# 75 — Steering saturation: push w_res to large magnitudes (up to 32σ),
# track BOTH flip rate AND intact rate (answer still valid) AND vs shuffled null.
# Closes the "did you push hard enough?" question with a full intact-flip curve.
#
# Self-test (already verified locally): causal toy model shows the expected
# hump-and-break pattern (flips at mid-c, collapse at large c), non-causal
# shows zero everywhere.
#
# Default config: layer 24 (strongest from j65) + final.
# Runtime estimate: ~1-2 hours.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/steering_saturation

echo "========================================================"
echo "  75 — steering saturation"
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

python -u scripts/75_steering_saturation.py \
    --concept_npz data/analysis/runD_v2/geometry_stage1/concept_directions.npz \
    --prompts     data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --steer_layers 24 \
    --include_final \
    --c_grid 0.5 1 2 4 8 16 32 \
    --max_targets 60 \
    --n_shuffle 10 \
    --n_random 10 \
    --tau_flip 0.5 \
    --intact_min 0.5 \
    --out_dir data/analysis/runD_v2/steering_saturation

echo ""
echo "==== Done: $(date) ===="
