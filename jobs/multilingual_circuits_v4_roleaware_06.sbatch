#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_v4_ra06
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mc_v4_ra06-%j.out
#SBATCH --error=logs/mc_v4_ra06-%j.err
#
# multilingual_circuits v4 — role-aware graph (Phase 3)
#
# Re-runs ONLY step 06 on the existing v2 features (SLURM 25058380).
# New vs v3:
#   --graph_node_mode role_aware  → adds content-word position nodes
#   --vw_threshold 0.01           → adds VW inter-feature edges (same as v3)
#   --output_suffix _roleaware    → writes attribution_graph_train_n48_roleaware.json
#
# Content-word nodes are selected by:
#   Stage 1: activation existence (>=2 prompts; fallback >=1)
#   Stage 2: rank by |en_freq - fr_freq|; max 10 global, max 2 per layer
#   VW connectivity check: must connect to adjacent-layer decision node
#
# Steps 04, 07, 09 are NOT re-run (features from SLURM 25058380).
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_v4_roleaware_06.sbatch
#
# AFTER COMPLETION:
#   Check logs/mc_v4_ra06-<JOBID>.out for:
#     "Role-aware graph complete: nodes=N, edges=M"
#     "Content nodes added: X new + Y upgraded to 'both'"
#   Then run analysis script locally:
#     python scripts/a_analyze_multilingual_circuits.py \
#         --behaviour multilingual_circuits --split train \
#         --node_labels --community_summary \
#         --graph_json data/results/attribution_graphs/multilingual_circuits/attribution_graph_train_n48_roleaware.json
#   Then optionally re-run step 09 for updated UI:
#     python scripts/09_prepare_offline_ui.py \
#         --behaviour multilingual_circuits --split train \
#         --graph_n_prompts 48 --community_method louvain \
#         --graph_suffix _roleaware
#   Then snapshot + add v4 entry to data/runs/RUN_INDEX.md

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits v4 — Role-Aware Graph (Step 06 only)"
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
K_CONTENT=10
OUTPUT_SUFFIX=_roleaware

# Verify features are present (v2 run must be complete)
FEAT_PROBE="data/results/transcoder_features/layer_10/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
if [ ! -f "$FEAT_PROBE" ]; then
    echo "ERROR: Feature file not found: $FEAT_PROBE"
    echo "  Run multilingual_circuits_v2_last5_02_09.sbatch first (SLURM 25058380)."
    exit 1
fi
echo "Feature probe: $FEAT_PROBE (OK)"

# ══════════════════════════════════════════════════════════════
# Step 06: Build role-aware attribution graph
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 06: Build role-aware attribution graph ===="
echo "  --graph_node_mode role_aware"
echo "  --vw_threshold ${VW_THRESHOLD}"
echo "  --k_content ${K_CONTENT}"
echo "  --output_suffix ${OUTPUT_SUFFIX}"
date

python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --vw_threshold "$VW_THRESHOLD" \
    --graph_node_mode role_aware \
    --k_content "$K_CONTENT" \
    --output_suffix "$OUTPUT_SUFFIX"

echo ""
echo "========================================"
echo "Step 06 complete."
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Next steps (run locally after sync):"
echo "  1. Inspect graph:"
echo "       python -c \\"
echo "         \"import json; g=json.load(open('data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json')); print(len(g['nodes']), 'nodes,', len(g['edges']), 'edges')\""
echo "  2. Run Phase 3.1 + 3.2 analysis:"
echo "       python scripts/a_analyze_multilingual_circuits.py \\"
echo "         --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "         --node_labels --community_summary \\"
echo "         --graph_json data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
echo "  3. Add v4 snapshot to data/runs/RUN_INDEX.md"
