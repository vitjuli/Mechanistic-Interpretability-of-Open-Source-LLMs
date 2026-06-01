#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=decay-v2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/decay-v2-%j.out
#SBATCH --error=logs/decay-v2-%j.out
#
# physics_decay_type_probe_v2 — Full Pipeline (steps 02→04→06→07→08→09)
#
# 538 train prompts: L1/L2/L3 hierarchy + keyword-free variants
# Alpha vs beta decay classification
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   wc -l data/prompts/physics_decay_type_probe_v2_train.jsonl  # expect 538
#   sbatch jobs/decay_probe_v2_02_09.sbatch

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "physics_decay_type_probe_v2: Steps 02→09"
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

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"
GRAPH_N=108       # balanced subset for graph/ablation (matches decay_type v1 for comparability)
TOP_K=50          # feature extraction
TOP_K_ABLATION=20 # ablation coverage

# Verify prompt file
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found. Did you git pull?"
    exit 1
fi
ACTUAL=$(wc -l < "$PROMPT_FILE")
echo "Prompt file: ${ACTUAL} lines (expect 538)"

# ══════════════════════════════════════════════════════════════
# Step 02: Baseline evaluation
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 02: Baseline ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

# ══════════════════════════════════════════════════════════════
# Step 04: Extract transcoder features (all 538 prompts)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 04: Transcoder features ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k $TOP_K

# ══════════════════════════════════════════════════════════════
# Step 06: Build attribution graph (role-aware, activation-weighted)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 06: Attribution graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$GRAPH_N" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k_per_layer 3 \
    --activation_weighted

# ══════════════════════════════════════════════════════════════
# Step 07: Ablation (graph subset)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 07: Ablation ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$GRAPH_N" \
    --top_k "$TOP_K_ABLATION" \
    --layers $LAYERS \
    --graph_n_prompts "$GRAPH_N"

# ══════════════════════════════════════════════════════════════
# Step 08: Causal edges
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 08: Causal edges ===="
date
python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$GRAPH_N"

# ══════════════════════════════════════════════════════════════
# Step 09: Offline UI prep
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 09: UI prep ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --community_method louvain

# Sanity check
echo ""
echo "==== Sanity check ===="
RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${GRAPH_N} 2>/dev/null | tail -1)
if [ -n "$RUN_DIR" ]; then
    python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
else
    echo "WARNING: Could not find run directory for sanity check"
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "Job: $SLURM_JOB_ID"
date
