#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_02_09
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mc_b1_02_09-%j.out
#SBATCH --error=logs/mc_b1_02_09-%j.err
#
# multilingual_circuits_b1 — full pipeline: Steps 02→04→06(star)→06(role_aware)→07(ablation)→07(patching C3)→09
#
# B1 = 8 templates per (concept, language) group (T0-T7; adds T4-T7 to existing T0-T3).
# Same 8 cross-language EN+FR concepts. Stratified split: 2 test + 6 train per group.
# → 96 train (48 EN + 48 FR), 32 test (16 EN + 16 FR).
#
# Key differences from multilingual_circuits v2 (SLURM 25058380):
#   BEHAVIOUR=multilingual_circuits_b1
#   N_PROMPTS=96  (was 48)
#   Step 06b: role-aware graph added (--graph_node_mode role_aware --vw_threshold 0.01)
#   Step 07: --per_feature must be explicit (default resolves False for _b1 suffix)
#
# Parameters confirmed from code:
#   --vw_threshold 0.01  : from multilingual_circuits_v4_roleaware_06.sbatch:83
#   --k_content 10       : from multilingual_circuits_v4_roleaware_06.sbatch:84
#   --top_k 50 (step 04) : from multilingual_circuits_v2_last5_02_09.sbatch:119
#   --top_k 20 (step 07) : from multilingual_circuits_02_09.sbatch:141 comment
#   --per_feature        : required — default is (behaviour=="multilingual_circuits"), False for _b1
#   out_dir              : data/analysis/multilingual_circuits_b1/ (get_paths() line 70)
#
# Time estimate: ~16h (96 prompts ≈ 2× v2's 48; ablation per_feature dominates)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   wc -l data/prompts/multilingual_circuits_b1_train.jsonl   # expect 96
#   sbatch jobs/multilingual_circuits_b1_02_09.sbatch
#
# AFTER COMPLETION — run analysis locally after sync:
#   # 5a. Core analysis (IoU + bridge + C3 + report)
#   python scripts/a_analyze_multilingual_circuits.py \
#       --behaviour multilingual_circuits_b1 --split train
#   # Output: data/analysis/multilingual_circuits_b1/
#   #   iou_per_layer.csv, iou_per_layer_decision.csv, iou_per_layer_content.csv
#   #   bridge_features.csv, bridge_features_only.csv
#   #   c3_patching_per_feature.csv, c3_patching_stats.txt
#   #   REPORT.md
#
#   # 5b. Node labels + community summary using role-aware graph
#   python scripts/a_analyze_multilingual_circuits.py \
#       --behaviour multilingual_circuits_b1 --split train \
#       --node_labels --community_summary \
#       --graph_json data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json
#   # Output: data/analysis/multilingual_circuits_b1/
#   #   node_language_labels.csv, community_summary.json, community_summary.md
#
#   # Then snapshot + add b1 entry to data/runs/RUN_INDEX.md

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "Multilingual Circuits B1 Pipeline: Steps 02→09"
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
echo ""

BEHAVIOUR=multilingual_circuits_b1
SPLIT=train
N_PROMPTS=96
TOP_K=20
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
VW_THRESHOLD=0.01
K_CONTENT=10
OUTPUT_SUFFIX=_roleaware

# Verify prompt file
if [ ! -f "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl" ]; then
    echo "ERROR: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl not found. Did you git pull?"
    exit 1
fi
ACTUAL=$(wc -l < "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl")
echo "Prompt file: ${ACTUAL} lines (expect ${N_PROMPTS})"
if [ "$ACTUAL" -ne "$N_PROMPTS" ]; then
    echo "ERROR: Expected ${N_PROMPTS} prompts, got ${ACTUAL}"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
# STAGE A — Step 02: Baseline evaluation
# Gate: EN sign_accuracy >= 0.90, FR >= 0.65, mean_norm_diff >= 1.0
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE A] Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 04: Extract transcoder features (last_5)
#
# --context_tokens 5: extracts features at last 5 token positions per prompt.
# Output: top_k_indices.npy shape (480, 50) — 5 × 96 prompts.
# position_map.json: 480 entries with is_decision_position flag.
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 04: Extract transcoder features (last_5, top_k=50) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50 \
    --context_tokens 5

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 06a: Build decision-only star graph
#
# No --vw_threshold (star only). Used as input for Steps 07 ablation/patching.
# Output: attribution_graph_train_n96.json
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 06a: Build decision-only star graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 06b: Build role-aware graph (B2)
#
# --graph_node_mode role_aware : adds content-word position nodes
# --vw_threshold 0.01          : adds VW inter-feature edges (same as v4)
# --k_content 10               : max 10 content nodes globally (same as v4)
# --output_suffix _roleaware   : writes attribution_graph_train_n96_roleaware.json
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 06b: Build role-aware graph (vw_threshold=${VW_THRESHOLD}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --vw_threshold "$VW_THRESHOLD" \
    --k_content "$K_CONTENT" \
    --output_suffix "$OUTPUT_SUFFIX"

# ══════════════════════════════════════════════════════════════
# STAGE C — Step 07: Ablation (per-feature, top_k=20)
#
# --per_feature : one forward pass per feature (Anthropic-faithful).
#                 MUST be explicit — default for _b1 suffix resolves to False.
# --top_k 20   : ensures all graph features are iterated (avg ~4-6 per layer).
# Uses star graph (attribution_graph_train_n96.json) via --graph_n_prompts.
# Output: intervention_ablation_multilingual_circuits_b1.csv
#         (required by find_bridge_features() for Claim 4)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Ablation (top_k=${TOP_K}, per_feature) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --per_feature

# ══════════════════════════════════════════════════════════════
# STAGE C — Step 07: Patching C3 — language swap (EN→FR), per-feature
#
# C3 pairing: EN source + FR target, same (concept_index, template_idx).
# B1 has 48 EN + 48 FR train prompts → up to 48 C3 pairs (exact match always found).
# --per_feature : required for per-feature disruption_rate / bridge analysis.
# Output: intervention_patching_C3_multilingual_circuits_b1.csv
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Patching C3 (language swap, per_feature) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C3 \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --per_feature

# ══════════════════════════════════════════════════════════════
# STAGE D — Step 09: Prepare offline UI (role-aware graph)
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
echo "Pipeline complete! (multilingual_circuits_b1)"
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Key output files:"
echo "  data/results/transcoder_features/layer_*/multilingual_circuits_b1_train_*.npy"
echo "  data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96.json"
echo "  data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json"
echo "  data/results/interventions/multilingual_circuits_b1/intervention_ablation_multilingual_circuits_b1.csv"
echo "  data/results/interventions/multilingual_circuits_b1/intervention_patching_C3_multilingual_circuits_b1.csv"
echo ""
echo "Post-run analysis (run locally after sync):"
echo "  # 5a. Core analysis"
echo "  python scripts/a_analyze_multilingual_circuits.py \\"
echo "      --behaviour ${BEHAVIOUR} --split ${SPLIT}"
echo "  # Output: data/analysis/multilingual_circuits_b1/"
echo ""
echo "  # 5b. Node labels + community summary (role-aware graph)"
echo "  python scripts/a_analyze_multilingual_circuits.py \\"
echo "      --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "      --node_labels --community_summary \\"
echo "      --graph_json data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
echo "  # Output: data/analysis/multilingual_circuits_b1/ (same dir, appended)"
echo ""
echo "  # Then: snapshot + RUN_INDEX.md entry"
