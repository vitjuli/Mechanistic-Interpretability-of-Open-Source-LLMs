#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_v2_last5
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mc_v2_last5-%j.out
#SBATCH --error=logs/mc_v2_last5-%j.err
#
# multilingual_circuits v2 — multi-token feature extraction (last_5)
#
# Identical to multilingual_circuits_02_09.sbatch (v1) EXCEPT step 04 uses
# --token_positions last_5 instead of the default (decision token only).
#
# Purpose: improve Claim 3 (middle-layer concentration of shared features).
# With last_5, the top_k_indices.npy has 240 rows (5 × 48 prompts) instead
# of 48. The content-word token (e.g. "rapide") at earlier positions is
# language-specific; the decision token is already semantic. This gradient
# is what Anthropic observed and what decision-token-only IoU cannot capture.
#
# Steps 06, 07, 09 are unchanged — they use the attribution graph (which is
# built from step 04 features) and interventions target the decision token.
# Only the IoU computation in a_analyze_multilingual_circuits.py changes
# (it reads position_map to group samples by prompt_idx for EN/FR splitting).
#
# Prompts are UNCHANGED from v1 (same 48 train + 16 test).
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   wc -l data/prompts/multilingual_circuits_train.jsonl   # expect 48
#   sbatch jobs/multilingual_circuits_v2_last5_02_09.sbatch
#
# AFTER COMPLETION — run analysis and snapshot on CSD3:
#   python scripts/a_analyze_multilingual_circuits.py \
#       --behaviour multilingual_circuits --split train
#   # Then sync data/analysis/multilingual_circuits/ back locally
#   # and run the snapshot script to create data/runs/multilingual_circuits_v2_last5/

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits v2 (last_5) Pipeline: Steps 02→09"
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
TOP_K=20
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

# Verify prompt file
if [ ! -f "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl" ]; then
    echo "ERROR: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl not found. Did you git pull?"
    exit 1
fi
ACTUAL=$(wc -l < "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl")
echo "Prompt file: ${ACTUAL} lines (expect 48)"
if [ "$ACTUAL" -ne 48 ]; then
    echo "ERROR: Expected 48 prompts, got ${ACTUAL}"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 02: Baseline evaluation
# (Identical to v1 — baseline does not depend on token_positions)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 04: Extract transcoder features
#
# KEY CHANGE vs v1: --token_positions last_5
# Extracts features at the last 5 token positions per prompt.
# Output: top_k_indices.npy shape (240, 50) instead of (48, 50).
# position_map.json will have 240 entries (5 per prompt).
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 04: Extract transcoder features (last_5) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50 \
    --token_positions last_5

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 06: Build attribution graph
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 06: Build attribution graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

# ══════════════════════════════════════════════════════════════
# STAGE C — Step 07: Ablation (top_k=20, per-feature, all 48 prompts)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Ablation (top_k=${TOP_K}, per_feature) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS" \
    --per_feature

# ══════════════════════════════════════════════════════════════
# STAGE C — Step 07: Patching C3 — language swap (EN → FR), per-feature
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Patching C3 — language swap (EN→FR, per_feature) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C3 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS" \
    --per_feature

# ══════════════════════════════════════════════════════════════
# STAGE D — Step 09: Prepare offline UI
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE D] Step 09: Prepare offline UI ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --community_method louvain

# Sanity check
echo ""
echo "==== Sanity check ===="
RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | tail -1)
if [ -n "$RUN_DIR" ]; then
    python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
else
    echo "WARNING: Could not find run directory for sanity check"
fi

echo ""
echo "========================================"
echo "Pipeline complete! (multilingual_circuits v2 — last_5)"
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Next steps:"
echo "  1. python scripts/a_analyze_multilingual_circuits.py \\"
echo "         --behaviour multilingual_circuits --split train"
echo "  2. Sync data/analysis/multilingual_circuits/ back locally"
echo "  3. Create v2 snapshot in data/runs/multilingual_circuits_v2_last5/"
