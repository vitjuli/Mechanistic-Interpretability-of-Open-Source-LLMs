#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_02_09
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mc_02_09-%j.out
#SBATCH --error=logs/mc_02_09-%j.err
#
# Runs steps 02→04→06→07(ablation, top_k=20)→07(patching C3)→09
# for multilingual_circuits (antonym-only EN+FR, 8 concepts).
#
# Reproduces Anthropic "Multilingual Circuits" case study:
#   Claim 1: Language-specific features (low per-layer IoU at early/late layers)
#   Claim 2: Shared cross-language features (high IoU in middle layers)
#   Claim 3: Shared features concentrated in middle layers
#   Claim 4: Bridge features whose ablation degrades both EN and FR
#
# Step 01 is NOT needed — prompts are already committed:
#   data/prompts/multilingual_circuits_train.jsonl  (48 prompts: 24 EN + 24 FR)
#   data/prompts/multilingual_circuits_test.jsonl   (16 prompts:  8 EN +  8 FR)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   wc -l data/prompts/multilingual_circuits_train.jsonl   # expect 48
#   sbatch jobs/multilingual_circuits_02_09.sbatch
#
# OUTPUTS:
#   data/results/baseline/multilingual_circuits_train_baseline.csv
#   data/results/features/multilingual_circuits_train_features_*.pkl
#   data/results/graphs/multilingual_circuits_train_n48.json
#   data/results/interventions/multilingual_circuits/
#     ablation_*.csv       — per-feature: 1 row/(prompt, layer, feat) ≈ 4,600 rows
#     patching_C3_*.csv    — per-feature: 1 row/(pair, layer, feat)   ≈ 2,300 rows
#     Both CSVs have explicit feature_id column ("L{layer}_F{feat_idx}")
#   data/ui_offline/*_multilingual_circuits_train_n48/
#
# BASELINE GATE (script 02 will report; DO NOT proceed to CSD3 if gate fails):
#   EN sign_accuracy ≥ 0.90
#   FR sign_accuracy ≥ 0.75
#   overall mean_norm_logprob_diff ≥ 1.0
#
# C3 SUCCESS CRITERIA (post-hoc, script a_analyze_multilingual_circuits.py):
#   disruption_rate  = fraction of (EN→FR) pairs with effect_size < 0  (target ≥ 0.40)
#   flip_rate        = fraction where patched FR pred flips to EN answer (report only)
#   mean_effect_size ± SEM across all pairs                             (report only)

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits Pipeline: Steps 02→09"
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
# Reports accuracy + logprob_diff sliced by language.
# Baseline gate: EN sign_acc ≥ 0.90, FR sign_acc ≥ 0.75,
#                overall mean_norm_logprob_diff ≥ 1.0.
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
#
# --per_feature: one forward pass per graph feature per prompt per layer
# (Anthropic-faithful causal intervention; ~48 × 16 × ~6 feats ≈ 4,600 rows).
# top_k=20 ensures all graph features are iterated (avg ~4-6 per layer).
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
#
# source = EN antonym prompt (concept X, template T)
# target = FR antonym prompt (concept X, template T)
# --per_feature: one patch per graph feature per pair per layer
# (~24 pairs × 16 layers × ~6 feats ≈ 2,300 rows).
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
echo "Pipeline complete!"
echo "Job: $SLURM_JOB_ID"
date
