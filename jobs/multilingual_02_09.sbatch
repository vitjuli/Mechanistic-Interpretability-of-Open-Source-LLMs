#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ml_02_09
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ml_02_09-%j.out
#SBATCH --error=logs/ml_02_09-%j.err
#
# Runs steps 02→04→06→07(C1,C2,C3)→09 for multilingual_antonym.
# Reproduces Anthropic's "Multilingual Circuits" three-axis intervention study.
#
# Step 01 is NOT needed — prompts are already committed:
#   data/prompts/multilingual_antonym_train.jsonl  (80 prompts)
#   data/prompts/multilingual_antonym_test.jsonl   (24 prompts)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull                    # sync Stage A + B code
#   mkdir -p logs
#   wc -l data/prompts/multilingual_antonym_train.jsonl   # expect 80
#   sbatch jobs/multilingual_02_09.sbatch
#
# STAGE B delivers:
#   data/results/baseline/multilingual_antonym_train_baseline.csv
#   data/results/features/multilingual_antonym_train_features_*.pkl
#   data/results/graphs/multilingual_antonym_train_n80.json
#
# STAGE C delivers (three separate CSVs, one per intervention family):
#   data/results/interventions/multilingual_antonym/
#     patching_C1_operation_swap_*.csv   — antonym→synonym features
#     patching_C2_operand_swap_*.csv     — hot→small operand
#     patching_C3_language_swap_*.csv    — EN→FR routing
#     ablation_*.csv                     — all graph features (top_k=20)
#     steering_*.csv
#
# STAGE D delivers:
#   data/ui_offline/*_multilingual_antonym_train_n80/

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Antonym Pipeline: Steps 02→09"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# --- Purge Intel/Cascade Lake modules (AVX-512 → SIGILL on AMD EPYC) ---
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# CPU instruction safety for AMD EPYC nodes
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo

BEHAVIOUR=multilingual_antonym
SPLIT=train
N_PROMPTS=80           # all 80 train prompts for attribution graph
TOP_K=20               # ablation coverage fix: covers all graph features per layer
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

# Verify prompt file
if [ ! -f "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl" ]; then
    echo "ERROR: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl not found. Did you git pull?"
    exit 1
fi
ACTUAL=$(wc -l < "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl")
echo "Prompt file: ${ACTUAL} lines (expect 80)"

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 02: Baseline evaluation
# Reports accuracy + logprob_diff sliced by language and operation.
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 04: Extract transcoder features
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 04: Extract transcoder features ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 06: Build attribution graph
# Graph is built from ALL 80 train prompts (mixed language + operation).
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
# STAGE C — Step 07: Ablation + Steering (all prompts, top_k=20)
#
# top_k=20 covers all graph features per layer (avg 4.5 features/layer
# in antonym graph → top-20 ablation ensures ~100% ablation coverage).
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Ablation + Steering ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment steering \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

# ══════════════════════════════════════════════════════════════
# STAGE C — Step 07: Three patching families
#
# C1: Operation swap — antonym features patched into synonym context.
#     source=antonym prompt (concept X, EN), target=synonym prompt (concept X, EN)
#     Success: synonym margin decreases / sign_flipped on ≥30% of pairs.
#
# C2: Operand swap — hot/cold features patched into small/large context (EN only).
#     source=hot antonym (concept_idx=1), target=small antonym (concept_idx=0)
#     Success: small→large margin decreases; model shifts toward "cold".
#
# C3: Language swap — EN antonym features patched into FR context.
#     source=EN antonym (concept X), target=FR antonym (concept X)
#     Success: FR margin (grand) decreases; model shifts toward EN answer (large).
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Patching C1 — operation swap ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C1 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

echo ""
echo "==== [STAGE C] Step 07: Patching C2 — operand swap ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C2 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

echo ""
echo "==== [STAGE C] Step 07: Patching C3 — language swap ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C3 \
    --n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --graph_n_prompts "$N_PROMPTS"

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
echo "Pipeline complete!"
echo "Job: $SLURM_JOB_ID"
date
