#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_comp
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/mc_b1_comp-%j.out
#SBATCH --error=logs/mc_b1_comp-%j.err
#
# multilingual_circuits_b1 — script 12: competition analysis
#
# For each of the 18 incorrect FR prompts: extract full logit distribution,
# identify the competing token, quantify margin, and correlate with L13/L22
# feature contributions.
#
# Requires only the model (no transcoders). 18 forward passes ≈ 2 min.
#
# Key diagnostics to check in output:
#   "Predictions = source word:  N/18"   ← expect 15+/18 (identity confusion)
#   "pearson_l22_vs_margin:  X"          ← expect positive (>0.3 confirms L22 role)
#   "competition type:  identity_confusion" ← expected
#
# BEFORE SUBMITTING:
#   Verify error_cases exist:
#     ls -lh data/results/reasoning_traces/multilingual_circuits_b1/error_cases_train.json
#   Then:
#     cd /rds/user/iv294/hpc-work/thesis/project
#     git pull
#     mkdir -p logs
#     sbatch jobs/multilingual_circuits_b1_12_competition.sbatch
#
# AFTER COMPLETION:
#   Sync outputs:
#     git add data/results/reasoning_traces/multilingual_circuits_b1/competition_analysis_train.json
#     git commit -m "script 12: competition analysis — competing token characterisation"

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 — Competition Analysis (Script 12)"
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

# Verify prerequisites
ERROR_CASES="data/results/reasoning_traces/${BEHAVIOUR}/error_cases_${SPLIT}.json"
FEAT_CSV="data/results/reasoning_traces/${BEHAVIOUR}/prompt_features_${SPLIT}.csv"

if [ ! -f "$ERROR_CASES" ]; then
    echo "ERROR: error_cases not found: $ERROR_CASES"
    echo "  Run scripts 10 + 11 (v2) first."
    exit 1
fi
if [ ! -f "$FEAT_CSV" ]; then
    echo "ERROR: prompt_features not found: $FEAT_CSV"
    exit 1
fi

echo "Inputs:"
echo "  $ERROR_CASES ($(wc -l < "$ERROR_CASES") chars)"
echo "  $FEAT_CSV ($(wc -l < "$FEAT_CSV") rows)"

# ══════════════════════════════════════════════════════════════
# Script 12: competition analysis
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Script 12: Competition Analysis ===="
date

python -u scripts/12_competition_analysis.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --top_k 10

echo ""
echo "========================================"
echo "Script 12 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Output:"
echo "  data/results/reasoning_traces/${BEHAVIOUR}/competition_analysis_${SPLIT}.json"
