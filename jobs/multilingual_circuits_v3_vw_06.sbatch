#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_v3_vw06
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mc_v3_vw06-%j.out
#SBATCH --error=logs/mc_v3_vw06-%j.err
#
# multilingual_circuits v3 — virtual-weight DAG (Phase 2)
#
# Re-runs ONLY step 06 on the existing v2 features (SLURM 25058380).
# Adds --vw_threshold 0.01 to aggregate_graphs_per_prompt_union():
#   - Computes W_enc_tgt[tgt_feats, :] @ W_dec_src[src_feats, :].T
#     for each adjacent layer pair in the graph
#   - Adds directed feature→feature edges with |weight| >= 0.01
#   - Star topology (285 edges) → DAG with inter-feature edges
#
# Steps 04, 07, 09 are NOT re-run:
#   - Features already extracted as last_5 (v2, SLURM 25058380)
#   - Interventions and UI prep use existing graph; re-run after v3 graph
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_v3_vw_06.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_v3_vw06-<JOBID>.out for:
#     "Virtual-weight edges: N added"  (expect N >> 285 for threshold=0.01)
#     "Feature nodes: 95"
#   Then optionally re-run step 09 to update the UI:
#     python scripts/09_prepare_offline_ui.py \
#         --behaviour multilingual_circuits --split train \
#         --graph_n_prompts 48 --community_method louvain
#   Then snapshot + add v3 entry to data/runs/RUN_INDEX.md

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits v3 — Virtual-Weight DAG (Step 06 only)"
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

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=multilingual_circuits
SPLIT=train
N_PROMPTS=48
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
VW_THRESHOLD=0.01

# Verify features are present (v2 run must be complete)
# Features live flat in data/results/transcoder_features/layer_N/<behaviour>_<split>_*.npy
FEAT_PROBE="data/results/transcoder_features/layer_10/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
if [ ! -f "$FEAT_PROBE" ]; then
    echo "ERROR: Feature file not found: $FEAT_PROBE"
    echo "  Run multilingual_circuits_v2_last5_02_09.sbatch first (SLURM 25058380)."
    exit 1
fi
echo "Feature probe: $FEAT_PROBE (OK)"

# ══════════════════════════════════════════════════════════════
# Step 06: Build attribution graph with virtual-weight edges
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 06: Build attribution graph (vw_threshold=${VW_THRESHOLD}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --vw_threshold "$VW_THRESHOLD"

echo ""
echo "========================================"
echo "Step 06 complete. Check graph edge count in log above."
echo "Expected: 'Virtual-weight edges: N added' where N >> 285"
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Next steps (run locally after sync):"
echo "  1. Inspect graph JSON:"
echo "       python -c \\"
echo "         \"import json; g=json.load(open('data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json')); print(len(g['edges']), 'edges')\""
echo "  2. Optionally re-run step 09 for updated UI:"
echo "       python scripts/09_prepare_offline_ui.py \\"
echo "         --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "         --graph_n_prompts ${N_PROMPTS} --community_method louvain"
echo "  3. Create v3 snapshot in data/runs/multilingual_circuits_v3_vw/"
