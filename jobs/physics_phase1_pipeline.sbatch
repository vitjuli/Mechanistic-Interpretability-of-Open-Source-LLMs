#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_phase1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phys_phase1-%j.out
#SBATCH --error=logs/phys_phase1-%j.err
#
# Phase 1 pipeline — physics_decay_type (B1) + physics_gauge_equivalence (B5)
#
# Extends Phase 0 to 24 prompts per behaviour (two concept poles each):
#   B1: 12 alpha (concept_index=0) + 12 beta (concept_index=1)
#   B5: 12 gauge_equivalent (concept_index=0) + 12 gauge_inequivalent (concept_index=1)
#
# Steps run per behaviour (identical to Phase 0):
#   A  Step 02   Baseline gate (sign_accuracy >= 0.90, mean_norm >= 0.5)
#   B  Step 04   Transcoder feature extraction (last_5, top_k=50)
#   C  Step 06a  Decision-only star graph
#   C  Step 06b  Role-aware graph, activation-weighted VW (canonical FIX 1)
#   D  Step 07   Ablation per-feature (top_k=20, loads roleaware graph)
#   E  Step 09   Offline UI prep (Louvain communities)
#   F  Step 08   Causal edges + necessity + sufficiency S1/S1.5
#
# Phase 1 goal: VW edges emerge (requires co-activation across >= 2 prompts).
# With N=24 and two concept poles, shared-role features should co-activate.
#
# Key parameters: same as Phase 0 (k=3/layer, vw=0.05, activation_weighted).
# N_PROMPTS updated to 24.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   ls data/prompts/physics_decay_type_phase1.jsonl          # expect 24 lines
#   ls data/prompts/physics_gauge_equivalence_phase1.jsonl   # expect 24 lines
#   sbatch jobs/physics_phase1_pipeline.sbatch
#
# AFTER COMPLETION:
#   rsync logs/phys_phase1-JOBID.out to local
#   Check go/no-go summary at the bottom of the log
#   Key metric: VW edge count > 0 in roleaware graph

set -uo pipefail

mkdir -p logs

echo "========================================================"
echo "  Physics Phase 1 Pipeline"
echo "  Behaviours: physics_decay_type  physics_gauge_equivalence"
echo "  N prompts: 24 each (two concept poles)"
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
N_PROMPTS=24
PHASE=phase1
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
# Helper: inline gate check
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
# PRE-FLIGHT: copy phase1 prompt files to _train.jsonl
# ════════════════════════════════════════════════════════════════════════════
echo "==== Pre-flight: copying phase1 prompts ===="
for BEHAVIOUR in physics_decay_type physics_gauge_equivalence; do
    SRC="data/prompts/${BEHAVIOUR}_${PHASE}.jsonl"
    DST="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
    if [ ! -f "$SRC" ]; then
        echo "ERROR: $SRC not found. Commit phase1 JSONL files before submitting."
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
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"

echo "========================================================"
echo "  BEHAVIOUR 1: ${BEHAVIOUR}"
echo "========================================================"
date
echo ""

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
if n_vw == 0:
    print('  WARNING: 0 VW edges — script 08 will fail. Consider lower --vw_threshold.')
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
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"

echo "========================================================"
echo "  BEHAVIOUR 2: ${BEHAVIOUR}"
echo "========================================================"
date
echo ""

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
if n_vw == 0:
    print('  WARNING: 0 VW edges — script 08 will fail. Consider lower --vw_threshold.')
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
echo "  Phase 1 Summary"
echo "  Finished: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo ""
echo "  physics_decay_type       : ${GATE_DECAY_TYPE}"
echo "  physics_gauge_equivalence: ${GATE_GAUGE}"
echo ""

if [ "$GATE_DECAY_TYPE" = "PASS" ] && [ "$GATE_GAUGE" = "PASS" ]; then
    echo "PHASE 1 RESULT: GO — both behaviours passed gate."
    echo ""
    echo "Key things to check after rsync:"
    echo "  1. VW edge count (should be > 0 now):"
    echo "     python -c \"import json; g=json.load(open('data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n24_roleaware.json')); print(sum(1 for e in g['edges'] if e.get('edge_type')=='virtual_weight'), 'VW edges')\""
    echo "  2. Circuit quality (causal edges / necessity):"
    echo "     cat data/results/causal_edges/physics_decay_type/circuits_physics_decay_type_train.json | python -c \"import json,sys; c=json.load(sys.stdin); print('features:', len(c.get('features',[])), 'edges:', len(c.get('causal_edges',[])))\""
    echo "  3. If circuits look clean → Phase 2 (full dataset, test split, patching)"
    EXIT_CODE=0
elif [ "$GATE_DECAY_TYPE" = "SKIP" ] && [ "$GATE_GAUGE" = "SKIP" ]; then
    echo "PHASE 1 RESULT: ERROR — no behaviours ran (pre-flight failure)."
    EXIT_CODE=1
else
    echo "PHASE 1 RESULT: PARTIAL — at least one behaviour failed gate."
    EXIT_CODE=1
fi

echo ""
echo "Key output locations:"
echo "  Baselines   : data/results/baseline_physics_*_train.csv"
echo "  Graphs      : data/results/attribution_graphs/physics_*/attribution_graph_train_n24_roleaware.json"
echo "  Ablations   : data/results/interventions/physics_*/intervention_ablation_physics_*.csv"
echo "  Circuits    : data/results/causal_edges/physics_*/circuits_physics_*_train.json"
echo "  UI runs     : data/ui_offline/*_physics_*_train_n24/"
echo ""

exit $EXIT_CODE
