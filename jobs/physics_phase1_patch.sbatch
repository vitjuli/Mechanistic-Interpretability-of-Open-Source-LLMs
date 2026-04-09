#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_p1_patch
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phys_p1_patch-%j.out
#SBATCH --error=logs/phys_p1_patch-%j.err
#
# Phase 1 patch run:
#
#   FIX 1 — physics_gauge_equivalence (B5):
#     Rewrote phase1 prompts to x_or_y question format ("same or different physics?")
#     to match B1 pattern. Completion format "...are physically the" was overwhelmingly
#     biased toward " same" (all 12 inequivalent prompts failed with diff=-10 to -14).
#     Runs full pipeline A-F.
#
#   FIX 2 — physics_decay_type (B1):
#     N=24 still produced 0 VW edges at vw_threshold=0.05.
#     Retry step 06b with --vw_threshold 0.02, then rerun step 07, 09, 08.
#     Steps 02 and 04 already passed — NOT rerun.
#
# BEFORE SUBMITTING:
#   git pull
#   wc -l data/prompts/physics_gauge_equivalence_phase1.jsonl  # expect 24
#   sbatch jobs/physics_phase1_patch.sbatch

set -uo pipefail

mkdir -p logs

echo "========================================================"
echo "  Physics Phase 1 Patch"
echo "  FIX 1: B5 (gauge_equivalence) — question format"
echo "  FIX 2: B1 (decay_type) — vw_threshold 0.02"
echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo ""

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

SPLIT=train
N_PROMPTS=24
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50
TOP_K_ABLATE=20
TOP_K_PER_LAYER=3
OUTPUT_SUFFIX=_roleaware
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=10

GATE_GAUGE=SKIP
GATE_DECAY_VW=SKIP

check_gate() {
    local BEHAVIOUR=$1
    python3 - <<PYEOF
import csv, sys
from pathlib import Path
behaviour = "${BEHAVIOUR}"
csv_path  = Path(f"data/results/baseline_{behaviour}_${SPLIT}.csv")
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found"); sys.exit(1)
rows = list(csv.DictReader(csv_path.open()))
if not rows:
    print("ERROR: empty CSV"); sys.exit(1)
sign_acc  = sum(1 for r in rows if float(r.get("logprob_diff_normalized", 0)) > 0) / len(rows)
mean_norm = sum(float(r.get("logprob_diff_normalized", 0)) for r in rows) / len(rows)
print(f"Baseline gate ({behaviour}): sign_acc={sign_acc:.3f} mean_norm={mean_norm:.3f}")
if sign_acc < 0.90 or mean_norm < 0.50:
    print(f"FAIL"); sys.exit(1)
print(f"PASS")
PYEOF
}

# ════════════════════════════════════════════════════════════════════════════
# FIX 1: physics_gauge_equivalence — full pipeline with new question-format prompts
# ════════════════════════════════════════════════════════════════════════════
BEHAVIOUR=physics_gauge_equivalence
VW_THRESHOLD=0.05
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"

echo "========================================================"
echo "  FIX 1: ${BEHAVIOUR} (full pipeline, question format)"
echo "========================================================"
date
echo ""

# Copy phase1 → train (updated question-format version)
cp data/prompts/${BEHAVIOUR}_phase1.jsonl data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl
N=$(wc -l < data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl)
echo "  Loaded ${N} prompts from phase1 (question format)"
echo ""

echo "==== [A] Step 02: Baseline ===="
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

if check_gate "$BEHAVIOUR"; then
    GATE_GAUGE=PASS
else
    GATE_GAUGE=FAIL
    echo "FAIL — skipping B-F."
fi

if [ "$GATE_GAUGE" = "PASS" ]; then

echo "==== [B] Step 04: Feature extraction ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k "$TOP_K_EXTRACT" \
    --context_tokens 5
echo ""

echo "==== [C] Step 06a: Star graph ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS
echo ""

echo "==== [C] Step 06b: Role-aware graph (vw=${VW_THRESHOLD}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --vw_threshold "$VW_THRESHOLD" \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --activation_weighted \
    --output_suffix "$OUTPUT_SUFFIX"
echo ""

echo "==== [D] Step 07: Ablation ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature
echo ""

echo "==== [E] Step 09: Offline UI prep ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX" \
    --community_method louvain
echo ""

echo "==== [F] Step 08: Causal edges ===="
date
python -c "
import json
g = json.load(open('${GRAPH_ROLEAWARE}'))
n_vw = sum(1 for e in g['edges'] if e.get('edge_type') == 'virtual_weight')
print(f'  VW edges: {n_vw}')
"
python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_json "$GRAPH_ROLEAWARE" \
    --model_size 4b \
    --agw_top_frac "$AGW_TOP_FRAC" \
    --tau_causal "$TAU_CAUSAL" \
    --n_paths "$N_PATHS" \
    --top_n_io_edges 10
echo ""

fi  # GATE_GAUGE

# ════════════════════════════════════════════════════════════════════════════
# FIX 2: physics_decay_type — retry 06b/07/09/08 with vw_threshold=0.02
# Steps 02 and 04 already passed from phase1 run — NOT rerun.
# ════════════════════════════════════════════════════════════════════════════
BEHAVIOUR=physics_decay_type
VW_THRESHOLD=0.02
OUTPUT_SUFFIX_VW02=_roleaware_vw02
GRAPH_ROLEAWARE_VW02="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX_VW02}.json"

echo "========================================================"
echo "  FIX 2: ${BEHAVIOUR} — retry 06b/07/09/08 with vw=0.02"
echo "  (02 and 04 already passed — skipping)"
echo "========================================================"
date
echo ""

echo "==== [C] Step 06b: Role-aware graph (vw=${VW_THRESHOLD}, suffix=${OUTPUT_SUFFIX_VW02}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --vw_threshold "$VW_THRESHOLD" \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --activation_weighted \
    --output_suffix "$OUTPUT_SUFFIX_VW02"
echo ""

python -c "
import json
g = json.load(open('${GRAPH_ROLEAWARE_VW02}'))
n_feat = sum(1 for n in g['nodes'] if n['type'] == 'feature')
n_vw   = sum(1 for e in g['edges'] if e.get('edge_type') == 'virtual_weight')
print(f'  Graph vw02: {n_feat} feature nodes, {n_vw} VW edges')
GATE_DECAY_VW = 'PASS' if n_vw > 0 else 'FAIL_NO_VW'
print(f'  VW gate: {GATE_DECAY_VW}')
" && GATE_DECAY_VW=CHECK_LOG

echo "==== [D] Step 07: Ablation (vw02 graph) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX_VW02" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature
echo ""

echo "==== [E] Step 09: Offline UI prep (vw02 graph) ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX_VW02" \
    --community_method louvain
echo ""

echo "==== [F] Step 08: Causal edges (vw02 graph) ===="
date
if [ ! -f "$GRAPH_ROLEAWARE_VW02" ]; then
    echo "ERROR: graph not found: $GRAPH_ROLEAWARE_VW02"
else
    python -u scripts/08_causal_edges.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --graph_json "$GRAPH_ROLEAWARE_VW02" \
        --model_size 4b \
        --agw_top_frac "$AGW_TOP_FRAC" \
        --tau_causal "$TAU_CAUSAL" \
        --n_paths "$N_PATHS" \
        --top_n_io_edges 10
fi
echo ""

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
echo "========================================================"
echo "  Phase 1 Patch Summary"
echo "  Finished: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo "  physics_gauge_equivalence gate : ${GATE_GAUGE}"
echo "  physics_decay_type vw02 gate   : see VW edge count in log above"
echo ""
echo "Key outputs:"
echo "  B5 graph  : data/results/attribution_graphs/physics_gauge_equivalence/attribution_graph_train_n24_roleaware.json"
echo "  B1 graph  : data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n24_roleaware_vw02.json"
echo "  B5 circuit: data/results/causal_edges/physics_gauge_equivalence/"
echo "  B1 circuit: data/results/causal_edges/physics_decay_type/"
echo ""
