#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_test_abl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=logs/phys_dt_test_abl-%j.out
#SBATCH --error=logs/phys_dt_test_abl-%j.out
#
# Re-run of Section A steps 3+4 from the combined job (28522456).
# A3 failed because script 07 looked for attribution_graph_test_n108...
# which doesn't exist — fixed here with a symlink.
# A1 (baseline) and A2 (features) already completed successfully.
# B/C/D are already done.

set -euo pipefail

BEHAVIOUR=physics_decay_type
TRAIN_N=108
TEST_N=36
GRAPH_SUFFIX=_roleaware_static
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_ABLATE=20
GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type — Test Ablation (A3+A4 rerun)"
echo "  Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python --version
nvidia-smi | head -5 || true
echo ""

# Pre-flight
TRAIN_GRAPH="${GRAPH_DIR}/attribution_graph_train_n${TRAIN_N}${GRAPH_SUFFIX}.json"
[ ! -f "$TRAIN_GRAPH" ] && echo "FATAL: train graph not found" && exit 1
[ ! -f "data/prompts/${BEHAVIOUR}_test.jsonl" ] && echo "FATAL: test prompts not found" && exit 1
echo "Pre-flight OK"
echo ""

# ── A3: Symlink train graph so script 07 can find it as the test-split graph ──
# script 07 constructs: attribution_graph_{split}_n{graph_n_prompts}{suffix}.json
# With --split test --graph_n_prompts 108, it looks for attribution_graph_test_n108...
# The symlink makes it resolve to the existing train graph.
ln -sf "$(pwd)/${TRAIN_GRAPH}" \
       "${GRAPH_DIR}/attribution_graph_test_n${TRAIN_N}${GRAPH_SUFFIX}.json"
echo "Symlinked: attribution_graph_test_n${TRAIN_N}${GRAPH_SUFFIX}.json → train graph"
echo ""

echo "==== [A3] Step 07: Ablation on test split ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split test \
    --experiment ablation \
    --n_prompts "$TEST_N" \
    --graph_n_prompts "$TRAIN_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature

echo ""
echo "==== [A4] Step 09: UI prep for test split ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split test \
    --graph_n_prompts "$TRAIN_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --community_method louvain

echo ""
echo "========================================================"
echo "  DONE: $(date)  Job: $SLURM_JOB_ID"
echo "========================================================"

# Quick summary
python3 -c "
import json, pandas as pd
from pathlib import Path

# Ablation summary
s = json.load(open('data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_test_summary.json'))
print(f'  Test ablation: {s[\"n_prompts\"]} prompts, sfr={s[\"sign_flip_rate\"]:.3f}, mean_abs={s[\"mean_abs_effect_size\"]:.3f}')

# UI run
runs = sorted(Path('data/ui_offline').glob(f'*_${BEHAVIOUR}_test_n*'))
if runs:
    print(f'  UI run: {runs[-1].name}')
" || true

echo ""
echo "After rsync, run locally:"
echo "  python scripts/15_family_circuit_analysis.py --ui_run <test-run-id>"
echo "  python scripts/prepare_decay_dashboard.py --ui_run <test-run-id>"
