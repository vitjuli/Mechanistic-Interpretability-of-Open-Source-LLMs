#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ant_02_09
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ant_02_09-%j.out
#SBATCH --error=logs/ant_02_09-%j.err
#
# Runs steps 02→04→06→07→09 for antonym_operation (EN v0).
# Step 01 is NOT needed — prompts are already at:
#   data/prompts/antonym_operation_train.jsonl  (80 prompts)
#   data/prompts/antonym_operation_test.jsonl   (20 prompts)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   mkdir -p logs
#   # Verify prompts exist:
#   wc -l data/prompts/antonym_operation_train.jsonl   # expect 80
#   sbatch jobs/antonym_02_09.sbatch

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Antonym Operation Pipeline: Steps 02→09"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# --- Purge Intel/Cascade Lake modules (AVX-512 → SIGILL on AMD EPYC) ---
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# CPU instruction safety for AMD EPYC nodes (no AVX-512)
export ATEN_CPU_CAPABILITY=avx2          # PyTorch CPU kernel dispatch
export MKL_ENABLE_INSTRUCTIONS=AVX2     # Intel MKL dispatch
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"  # numpy dispatch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=antonym_operation
SPLIT=train
N_PROMPTS=80           # all 80 train prompts used for attribution graph
N_INTERVENTIONS=20     # subset used for intervention experiments (matching physics)
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

# Verify prompt file exists before doing anything GPU-expensive
if [ ! -f "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl" ]; then
    echo "ERROR: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl not found."
    echo "Prompts should already exist — did you sync the repo?"
    exit 1
fi
ACTUAL=$(wc -l < "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl")
echo "Prompt file: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl (${ACTUAL} lines)"

# ══════════════════════════════════════════════════════════════
# Step 02: Baseline evaluation
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ══════════════════════════════════════════════════════════════
# Step 04: Extract transcoder features
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 04: Extract transcoder features ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50

# ══════════════════════════════════════════════════════════════
# Step 06: Build attribution graph
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 06: Build attribution graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

# ══════════════════════════════════════════════════════════════
# Step 07: Run interventions (ablation, patching, steering)
#
# Patching note: all train prompts are forward-direction (4 template
# variants × 25 concept pairs).  create_prompt_pairs() will
# automatically use cross-template patching (same concept, different
# template_idx) and log a WARNING — this is expected behaviour.
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 07: Interventions ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment all \
    --n_prompts "$N_INTERVENTIONS" \
    --top_k 5 \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

# ══════════════════════════════════════════════════════════════
# Step 09: Prepare offline UI data
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Step 09: Prepare offline UI ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --community_method louvain

# ══════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== Sanity check ===="
date

RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | tail -1)
if [ -n "$RUN_DIR" ]; then
    python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
else
    echo "WARNING: Could not find run directory for sanity check"
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
date
