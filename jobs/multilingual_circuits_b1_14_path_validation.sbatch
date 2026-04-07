#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_pv
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mc_b1_pv-%j.out
#SBATCH --error=logs/mc_b1_pv-%j.err
#
# multilingual_circuits_b1 — script 14: path-centric causal validation (FIX 3)
#
# Validates top-10 circuit paths from B1-sparsified circuit by ablating each
# path head (feature A) and measuring:
#   - mean_delta_act_B:          mean |Δactivation| at next feature B
#   - mean_delta_logit:          mean |Δlogit_correct - Δlogit_incorrect|
#   - propagation_consistency:   fraction of prompts where |Δact_B| > epsilon
#
# Requires: model (Qwen3-4B base), transcoders, circuits JSON from script 08.
# Each path: ~50 prompts × 2 forward passes ≈ ~5 min total for 10 paths.
#
# BEFORE SUBMITTING:
#   Verify circuits JSON exists:
#     ls -lh data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train_roleaware_k3_t05.json
#   Or use the default (plain) circuits if sparsified not available:
#     ls -lh data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
#   Then:
#     cd /rds/user/iv294/hpc-work/thesis/project
#     git pull
#     mkdir -p logs
#     sbatch jobs/multilingual_circuits_b1_14_path_validation.sbatch
#
# AFTER COMPLETION:
#   Sync outputs:
#     git add data/results/path_validation/multilingual_circuits_b1/
#     git commit -m "script 14: path validation — FIX 3 causal path metrics"

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Path Validation (Script 14, FIX 3)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

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
python --version
nvidia-smi || true
echo

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train
N_PROMPTS=96

# Circuits JSON — prefer sparsified; fall back to plain
CIRCUITS_SPARSIFIED="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}_roleaware_k3_t05.json"
CIRCUITS_PLAIN="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"

if [ -f "$CIRCUITS_SPARSIFIED" ]; then
    CIRCUITS="$CIRCUITS_SPARSIFIED"
    echo "Using sparsified circuits: $CIRCUITS"
elif [ -f "$CIRCUITS_PLAIN" ]; then
    CIRCUITS="$CIRCUITS_PLAIN"
    echo "WARNING: sparsified circuits not found, using plain: $CIRCUITS"
else
    echo "ERROR: no circuits JSON found. Run script 08 first."
    exit 1
fi

echo "Inputs:"
echo "  circuits: $CIRCUITS"
echo "  behaviour: $BEHAVIOUR / split: $SPLIT"
echo

# ══════════════════════════════════════════════════════════════
# Script 14: path-centric causal validation
# ══════════════════════════════════════════════════════════════
echo "==== Script 14: Path Validation ===="
date

python -u scripts/14_path_validation.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --circuits_json "$CIRCUITS" \
    --n_paths 10 \
    --epsilon 0.01 \
    --max_prompts 50 \
    --model_size 4b

echo ""
echo "========================================"
echo "Script 14 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Output:"
echo "  data/results/path_validation/${BEHAVIOUR}/path_validation_${BEHAVIOUR}_${SPLIT}.json"
