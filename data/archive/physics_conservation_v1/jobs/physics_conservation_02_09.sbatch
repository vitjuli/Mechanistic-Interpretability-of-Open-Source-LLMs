#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=pc_02_09
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=logs/pc_02_09-%j.out
#SBATCH --error=logs/pc_02_09-%j.err
#
# physics_conservation — FULL PIPELINE: Steps 02→04→06a→06b→07(ablation)→07(patching)→08→10
#
# Behavior: physics_conservation
# 25 concept groups × 8 templates = 200 prompts total.
# Train split: T0-T5 (6 per concept = 150 prompts).
# Binary: ' True' (path-independent) vs ' False' (path-dependent).
# Families: conservative_named(7), conservative_abstract(7),
#           nonconservative_named(4), nonconservative_abstract(4), adversarial(3).
#
# Key differences from multilingual_circuits_b1:
#   BEHAVIOUR=physics_conservation
#   N_PROMPTS=150
#   No language split — all prompts are English
#   --context_tokens 1 (decision token only; no content-word analysis)
#   Step 07 patching: paraphrase mode (same concept, T_even→T_odd, no --patch_mode flag)
#   Step 06b: role-aware graph still valid (finds content-word nodes = physics concept tokens)
#   Analysis: use concept_family in place of language labels
#
# PREREQUISITE:
#   Run jobs/physics_conservation_pilot_02_gate.sbatch first and get PILOT GATE: PASS
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   python scripts/01_generate_prompts.py --behaviour physics_conservation
#   wc -l data/prompts/physics_conservation_train.jsonl   # expect 150
#   sbatch jobs/physics_conservation_02_09.sbatch
#
# AFTER COMPLETION:
#   Sync results, then run post-run analysis locally.
#   Key outputs:
#     data/results/attribution_graphs/physics_conservation/attribution_graph_train_n150.json
#     data/results/attribution_graphs/physics_conservation/attribution_graph_train_n150_roleaware.json
#     data/results/interventions/physics_conservation/intervention_ablation_physics_conservation.csv
#     data/results/interventions/physics_conservation/intervention_patching_physics_conservation.csv
#     data/results/reasoning_traces/physics_conservation/ (script 10)
#

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "physics_conservation Pipeline: Steps 02→10"
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
echo ""

BEHAVIOUR=physics_conservation
SPLIT=train
N_PROMPTS=150
TOP_K=20
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
VW_THRESHOLD=0.01
K_CONTENT=8
OUTPUT_SUFFIX=_roleaware

# Verify prompt file
if [ ! -f "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl" ]; then
    echo "ERROR: data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl not found."
    echo "  Run: python scripts/01_generate_prompts.py --behaviour ${BEHAVIOUR}"
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
# Gate: overall accuracy >= 0.80
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE A] Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT"

# Quick baseline gate
python3 - <<'PYEOF'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(f"data/results/baseline_physics_conservation_train.csv")))
acc = sum(1 for r in rows if str(r.get("success","")).lower() in ("true","1","yes")) / len(rows)
print(f"\nBaseline accuracy: {acc:.3f} (gate: >= 0.80)")
if acc < 0.80:
    print("GATE FAIL: accuracy below threshold. Do NOT continue pipeline.")
    sys.exit(1)
print("GATE: PASS")
PYEOF

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 04: Extract transcoder features (decision token)
#
# --context_tokens 1: extract features at the final token only (decision position).
# Simpler than last_5 since all prompts are English and the key position is the
# last ":" token where True/False is predicted.
# Output: top_k_indices.npy shape (150, 50) — 1 × 150 prompts × 50 features.
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE B] Step 04: Extract transcoder features (decision token, top_k=50) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k 50 \
    --context_tokens 1

# ══════════════════════════════════════════════════════════════
# STAGE B — Step 06a: Build decision-only star graph
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
# STAGE B — Step 06b: Build role-aware graph with VW edges
#
# --graph_node_mode role_aware : adds content-word nodes (physics concept tokens)
# --vw_threshold 0.01          : virtual-weight edges between feature nodes
# --k_content 8                : max 8 content nodes (fewer than B1's 10, since
#                                single-language = less content diversity expected)
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
# Uses default (star) graph for feature selection.
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
# STAGE C — Step 07: Patching — paraphrase mode (same concept, T_even → T_odd)
#
# No --patch_mode flag needed: physics_conservation default IS paraphrase mode.
# 75 pairs (150 prompts / 2, since each concept has 6 train templates = 3 pairs).
# Tests: do the same features fire across surface paraphrases of the same concept?
# Output: intervention_patching_physics_conservation.csv
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE C] Step 07: Patching (paraphrase mode) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --top_k "$TOP_K" \
    --layers $LAYERS \
    --per_feature

# ══════════════════════════════════════════════════════════════
# STAGE D — Step 08: Causal edges
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE D] Step 08: Causal edges ===="
date
python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --layers $LAYERS \
    --from_causal_edges

# ══════════════════════════════════════════════════════════════
# STAGE E — Step 10: Reasoning traces (layer trajectory)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE E] Step 10: Reasoning traces ===="
date
python -u scripts/10_reasoning_trace.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --layers $LAYERS

# ══════════════════════════════════════════════════════════════
# STAGE E — Step 11: Ablation supplement (remaining circuit features)
# ══════════════════════════════════════════════════════════════
echo ""
echo "==== [STAGE E] Step 11: Ablation supplement ===="
date
python -u scripts/11_ablation_supplement.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --layers $LAYERS

echo ""
echo "========================================"
echo "physics_conservation pipeline complete!"
echo "Job: $SLURM_JOB_ID"
date
echo ""
echo "Key output files:"
echo "  data/results/attribution_graphs/physics_conservation/attribution_graph_${SPLIT}_n${N_PROMPTS}.json"
echo "  data/results/attribution_graphs/physics_conservation/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
echo "  data/results/interventions/physics_conservation/intervention_ablation_physics_conservation.csv"
echo "  data/results/interventions/physics_conservation/intervention_patching_physics_conservation.csv"
echo "  data/results/reasoning_traces/physics_conservation/"
echo ""
echo "Post-run analysis steps (run locally after sync):"
echo ""
echo "  1. Concept-family feature analysis:"
echo "     python scripts/analyze_physics_conservation.py \\"
echo "         --behaviour ${BEHAVIOUR} --split ${SPLIT}"
echo ""
echo "  2. Node labels + community summary:"
echo "     python scripts/analyze_physics_conservation.py \\"
echo "         --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "         --node_labels --community_summary \\"
echo "         --graph_json data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
echo ""
echo "  3. Freeze results: snapshot to data/runs/ and add RUN_INDEX.md entry"
