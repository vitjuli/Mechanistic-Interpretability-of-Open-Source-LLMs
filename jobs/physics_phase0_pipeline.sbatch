#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_phase0
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phys_phase0-%j.out
#SBATCH --error=logs/phys_phase0-%j.err
#
# Phase 0 pipeline — physics_decay_type (B1) + physics_gauge_equivalence (B5)
#
# Steps run per behaviour:
#   A  Step 02   Baseline gate (sign_accuracy >= 0.90, mean_norm >= 0.5)
#   B  Step 04   Transcoder feature extraction (last_5, top_k=50)
#   C  Step 06a  Decision-only star graph (beta proxy; input to step 07)
#   C  Step 06b  Role-aware graph, activation-weighted VW (canonical FIX 1)
#   D  Step 07   Ablation per-feature (top_k=20, loads roleaware graph)
#   E  Step 09   Offline UI prep (Louvain communities)
#   F  Step 08   Causal edges + necessity + sufficiency S1/S1.5
#
# No patching (single-language behaviours, no C3 pairs).
# No steps 10-13 (behaviour-specific analysis — Phase 1 only).
#
# Go/no-go criteria checked after step 02 for each behaviour:
#   - sign_accuracy >= 0.90 on train split
#   - mean_norm_logprob_diff >= 0.50
# If a behaviour fails gate, its pipeline stages are skipped.
# Both behaviours are always attempted; exit code reflects overall result.
#
# Key parameters:
#   --context_tokens 5     : last-5 token positions (decision + 4 context)
#   --top_k 50 (step 04)   : top-50 features saved per token position
#   --top_k 20 (step 07)   : ablate top-20 features per layer
#   --top_k_per_layer 3    : max 3 attribution nodes per layer (canonical sparsified)
#   --vw_threshold 0.05    : VW edge threshold (canonical sparsified)
#   --activation_weighted  : FIX 1 activation-weighted VW edges
#   --output_suffix _roleaware : graph filename suffix
#   --n_paths 10           : path tracing budget (5 prompts only)
#   --tau_causal 0.10      : causal edge threshold
#   --agw_top_frac 0.6     : AGW prefilter fraction
#
# Prompt files (Phase 0 — 5 prompts each, concept_index=0 only):
#   data/prompts/physics_decay_type_phase0.jsonl        (B1 alpha-decay, completion format)
#   data/prompts/physics_gauge_equivalence_phase0.jsonl (B5 gauge equiv, completion format)
#
# NOTE: physics_intensive_extensive was RETIRED (B6) — Instruct model base P(' intensive')
#       ~10^-10 in question contexts; replaced by physics_gauge_equivalence (B5).
#       physics_gauge_equivalence baseline: 100%, mean_norm_diff=14.01 (2026-04-09).
#
# These are copied to _train.jsonl before the pipeline runs.
# The _phase0.jsonl originals are never modified.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   ls data/prompts/physics_decay_type_phase0.jsonl           # expect 5 lines
#   ls data/prompts/physics_gauge_equivalence_phase0.jsonl    # expect 5 lines
#   sbatch jobs/physics_phase0_pipeline.sbatch
#
# AFTER COMPLETION:
#   rsync logs/phys_phase0-JOBID.out to local
#   Check go/no-go summary at the bottom of the log
#   If GO: proceed to Phase 1 (jobs/physics_phase1_pipeline.sbatch)

set -uo pipefail

mkdir -p logs

echo "========================================================"
echo "  Physics Phase 0 Pipeline"
echo "  Behaviours: physics_decay_type  physics_gauge_equivalence"
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

# ── Shared parameters ────────────────────────────────────────────────────────
SPLIT=train
N_PROMPTS=5
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50
TOP_K_ABLATE=20
TOP_K_PER_LAYER=3
VW_THRESHOLD=0.05
OUTPUT_SUFFIX=_roleaware
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=10

# Track go/no-go per behaviour
GATE_DECAY_TYPE=SKIP
GATE_GAUGE=SKIP

# ════════════════════════════════════════════════════════════════════════════
# Helper: inline gate check for single-language physics behaviours
# ════════════════════════════════════════════════════════════════════════════
check_gate() {
    local BEHAVIOUR=$1
    python3 - <<PYEOF
import csv, sys
from pathlib import Path

behaviour = "${BEHAVIOUR}"
split     = "${SPLIT}"
csv_path  = Path(f"data/results/baseline_{behaviour}_{split}.csv")
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found")
    sys.exit(1)

rows = list(csv.DictReader(csv_path.open()))
if not rows:
    print("ERROR: empty CSV")
    sys.exit(1)

sign_acc  = sum(1 for r in rows if float(r.get("logprob_diff_normalized", 0)) > 0) / len(rows)
mean_norm = sum(float(r.get("logprob_diff_normalized", 0)) for r in rows) / len(rows)

print(f"Baseline gate ({behaviour}):")
print(f"  n_prompts     : {len(rows)}")
print(f"  sign_accuracy : {sign_acc:.3f}  (need >= 0.90)  {'PASS' if sign_acc >= 0.90 else 'FAIL'}")
print(f"  mean_norm_diff: {mean_norm:.3f}  (need >= 0.50)  {'PASS' if mean_norm >= 0.50 else 'FAIL'}")

if sign_acc < 0.90 or mean_norm < 0.50:
    print(f"\n{behaviour.upper()} GATE: FAIL — stages B-F will be skipped.")
    sys.exit(1)
print(f"\n{behaviour.upper()} GATE: PASS — proceeding.")
PYEOF
}

# ════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT: copy phase0 prompt files to _train.jsonl
# ════════════════════════════════════════════════════════════════════════════
echo "==== Pre-flight: copying phase0 prompts ===="
for BEHAVIOUR in physics_decay_type physics_gauge_equivalence; do
    SRC="data/prompts/${BEHAVIOUR}_phase0.jsonl"
    DST="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
    if [ ! -f "$SRC" ]; then
        echo "ERROR: $SRC not found. Commit phase0 JSONL files before submitting."
        exit 1
    fi
    cp "$SRC" "$DST"
    N=$(wc -l < "$DST")
    echo "  ${BEHAVIOUR}_${SPLIT}.jsonl: ${N} prompts"
    if [ "$N" -ne "$N_PROMPTS" ]; then
        echo "ERROR: Expected ${N_PROMPTS} prompts in ${DST}, got ${N}"
        exit 1
    fi
done
echo "  Pre-flight: PASS"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# BEHAVIOUR 1: physics_decay_type (B1 — alpha vs beta decay)
# ════════════════════════════════════════════════════════════════════════════
BEHAVIOUR=physics_decay_type
GRAPH_STAR="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json"
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"

echo "========================================================"
echo "  BEHAVIOUR 1: ${BEHAVIOUR}"
echo "========================================================"
date
echo ""

# ── Stage A: Baseline gate ───────────────────────────────────────────────────
echo "==== [A] Step 02: Baseline (${BEHAVIOUR}) ===="
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

if check_gate "$BEHAVIOUR"; then
    GATE_DECAY_TYPE=PASS
else
    GATE_DECAY_TYPE=FAIL
    echo "Skipping stages B-F for ${BEHAVIOUR}."
    echo ""
fi

if [ "$GATE_DECAY_TYPE" = "PASS" ]; then

# ── Stage B: Feature extraction ──────────────────────────────────────────────
echo "==== [B] Step 04: Feature extraction (${BEHAVIOUR}, top_k=${TOP_K_EXTRACT}) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k "$TOP_K_EXTRACT" \
    --context_tokens 5

echo ""

# ── Stage C: Attribution graphs ──────────────────────────────────────────────
echo "==== [C] Step 06a: Star graph (${BEHAVIOUR}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

echo ""

echo "==== [C] Step 06b: Role-aware graph (${BEHAVIOUR}, vw=${VW_THRESHOLD}, k=${TOP_K_PER_LAYER}) ===="
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

# ── Stage D: Ablation ────────────────────────────────────────────────────────
echo "==== [D] Step 07: Ablation (${BEHAVIOUR}, top_k=${TOP_K_ABLATE}) ===="
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

# ── Stage E: Offline UI prep ─────────────────────────────────────────────────
echo "==== [E] Step 09: Offline UI prep (${BEHAVIOUR}) ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX" \
    --community_method louvain

RUN_DIR_DECAY=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
echo "  UI run: ${RUN_DIR_DECAY:-NOT FOUND}"

echo ""

# ── Stage F: Causal edges ────────────────────────────────────────────────────
echo "==== [F] Step 08: Causal edges (${BEHAVIOUR}) ===="
date
if [ ! -f "$GRAPH_ROLEAWARE" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_ROLEAWARE"
    GATE_DECAY_TYPE=FAIL_08
else
    python -c "
import json
g = json.load(open('${GRAPH_ROLEAWARE}'))
n_feat = sum(1 for n in g['nodes'] if n['type'] == 'feature')
n_vw   = sum(1 for e in g['edges'] if e.get('edge_type') == 'virtual_weight')
print(f'  Graph: {n_feat} feature nodes, {n_vw} VW edges')
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
fi

echo ""

fi  # end GATE_DECAY_TYPE = PASS

# ════════════════════════════════════════════════════════════════════════════
# BEHAVIOUR 2: physics_gauge_equivalence (B5 — same vs different physics)
# ════════════════════════════════════════════════════════════════════════════
BEHAVIOUR=physics_gauge_equivalence
GRAPH_STAR="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json"
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"

echo "========================================================"
echo "  BEHAVIOUR 2: ${BEHAVIOUR}"
echo "========================================================"
date
echo ""

# ── Stage A ──────────────────────────────────────────────────────────────────
echo "==== [A] Step 02: Baseline (${BEHAVIOUR}) ===="
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

if check_gate "$BEHAVIOUR"; then
    GATE_GAUGE=PASS
else
    GATE_GAUGE=FAIL
    echo "Skipping stages B-F for ${BEHAVIOUR}."
    echo ""
fi

if [ "$GATE_GAUGE" = "PASS" ]; then

# ── Stage B ──────────────────────────────────────────────────────────────────
echo "==== [B] Step 04: Feature extraction (${BEHAVIOUR}, top_k=${TOP_K_EXTRACT}) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k "$TOP_K_EXTRACT" \
    --context_tokens 5

echo ""

# ── Stage C ──────────────────────────────────────────────────────────────────
echo "==== [C] Step 06a: Star graph (${BEHAVIOUR}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

echo ""

echo "==== [C] Step 06b: Role-aware graph (${BEHAVIOUR}, vw=${VW_THRESHOLD}, k=${TOP_K_PER_LAYER}) ===="
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

# ── Stage D ──────────────────────────────────────────────────────────────────
echo "==== [D] Step 07: Ablation (${BEHAVIOUR}, top_k=${TOP_K_ABLATE}) ===="
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

# ── Stage E ──────────────────────────────────────────────────────────────────
echo "==== [E] Step 09: Offline UI prep (${BEHAVIOUR}) ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX" \
    --community_method louvain

RUN_DIR_GAUGE=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
echo "  UI run: ${RUN_DIR_GAUGE:-NOT FOUND}"

echo ""

# ── Stage F ──────────────────────────────────────────────────────────────────
echo "==== [F] Step 08: Causal edges (${BEHAVIOUR}) ===="
date
if [ ! -f "$GRAPH_ROLEAWARE" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_ROLEAWARE"
    GATE_GAUGE=FAIL_08
else
    python -c "
import json
g = json.load(open('${GRAPH_ROLEAWARE}'))
n_feat = sum(1 for n in g['nodes'] if n['type'] == 'feature')
n_vw   = sum(1 for e in g['edges'] if e.get('edge_type') == 'virtual_weight')
print(f'  Graph: {n_feat} feature nodes, {n_vw} VW edges')
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
fi

echo ""

fi  # end GATE_GAUGE = PASS

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
echo "========================================================"
echo "  Phase 0 Summary"
echo "  Finished: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo ""
echo "  physics_decay_type       : ${GATE_DECAY_TYPE}"
echo "  physics_gauge_equivalence: ${GATE_GAUGE}"
echo ""

# Go/no-go decision
if [ "$GATE_DECAY_TYPE" = "PASS" ] && [ "$GATE_GAUGE" = "PASS" ]; then
    echo "PHASE 0 RESULT: GO — both behaviours passed gate."
    echo ""
    echo "Next steps (run locally after rsync):"
    echo "  1. Check attribution graphs for >= 4 nodes:"
    echo "     python -c \"import json; g=json.load(open('data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n5_roleaware.json')); print(sum(1 for n in g['nodes'] if n['type']==\"feature\"), 'feature nodes')\""
    echo "  2. Check ablation logit changes:"
    echo "     head data/results/interventions/physics_decay_type/intervention_ablation_physics_decay_type.csv"
    echo "  3. If GO criteria met → proceed to Phase 1 (B3, B4, B7, B8 + expand B1/B5 to 20-30 prompts)"
    echo "  4. Update community_labels.json for each new dashboard"
    EXIT_CODE=0
elif [ "$GATE_DECAY_TYPE" = "SKIP" ] && [ "$GATE_GAUGE" = "SKIP" ]; then
    echo "PHASE 0 RESULT: ERROR — no behaviours ran (pre-flight failure)."
    EXIT_CODE=1
else
    echo "PHASE 0 RESULT: PARTIAL — at least one behaviour failed gate."
    echo "  Investigate failed behaviour before starting Phase 1."
    EXIT_CODE=1
fi

echo ""
echo "Key output locations:"
echo "  Baselines   : data/results/baseline_physics_*_train.csv"
echo "  Graphs      : data/results/attribution_graphs/physics_*/attribution_graph_train_n5_roleaware.json"
echo "  Ablations   : data/results/interventions/physics_*/intervention_ablation_physics_*.csv"
echo "  Circuits    : data/results/causal_edges/physics_*/circuits_physics_*_train.json"
echo "  UI runs     : data/ui_offline/*_physics_*_train_n5/"
echo ""

exit $EXIT_CODE
