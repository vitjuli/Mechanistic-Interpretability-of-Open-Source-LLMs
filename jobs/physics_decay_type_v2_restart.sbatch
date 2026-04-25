#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_v2r
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=09:00:00
#SBATCH --output=logs/phys_dt_v2r-%j.out
#SBATCH --error=logs/phys_dt_v2r-%j.out
#SBATCH --exclude=gpu-q-43
#
# physics_decay_type v2 — RESTART from step 04
# Step 02 (baseline) already done: sign_acc=0.870, WARN_PASS, job 28370718.
# Excludes gpu-q-43 which had CUDA unavailable error.
#
# Submit: sbatch jobs/physics_decay_type_v2_restart.sbatch

set -uo pipefail

mkdir -p logs

# ── PARAMETERS (must match original run) ─────────────────────────────────────
BEHAVIOUR=physics_decay_type
SPLIT=train
N_PROMPTS=108
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50
TOP_K_PER_LAYER=5
VW_THRESH_CONS=0.05
VW_THRESH_PERM=0.01
TOP_K_ABLATE=20
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=10
# ─────────────────────────────────────────────────────────────────────────────

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type v2 RESTART (steps 04 → 09)"
echo "  Baseline already done (job 28370718, sign_acc=0.870)"
echo "  N_PROMPTS=${N_PROMPTS}  TOP_K_PER_LAYER=${TOP_K_PER_LAYER}"
echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo ""

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

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

# ── CUDA sanity check — fail fast before spending time on model load ──────────
echo "---- CUDA sanity check ----"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
n = torch.cuda.device_count()
print(f'  CUDA OK: {n} device(s)')
for i in range(n):
    print(f'  [{i}] {torch.cuda.get_device_name(i)}')
x = torch.zeros(10, device='cuda')   # smoke test
del x
torch.cuda.empty_cache()
print('  Smoke test: PASS')
"
if [ $? -ne 0 ]; then
    echo "FATAL: CUDA unavailable on $(hostname) — resubmit to get a different node."
    exit 1
fi
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 04: Feature extraction
# ════════════════════════════════════════════════════════════════════════════
echo "==== [B] Step 04: Feature extraction (top_k=${TOP_K_EXTRACT}) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k "$TOP_K_EXTRACT" \
    --context_tokens 5
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 06a: Star graph (decision_only, no VW)
# ════════════════════════════════════════════════════════════════════════════
echo "==== [C] Step 06a: Star graph (decision_only) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 06b: Role-aware graph — conservative
# ════════════════════════════════════════════════════════════════════════════
echo "==== [C] Step 06b: Role-aware graph (conservative, activation-weighted, t=${VW_THRESH_CONS}, k=${TOP_K_PER_LAYER}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --vw_threshold "$VW_THRESH_CONS" \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --activation_weighted \
    --output_suffix "_roleaware"
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 06c: Role-aware graph — permissive static
# ════════════════════════════════════════════════════════════════════════════
echo "==== [C] Step 06c: Role-aware graph (permissive, static, t=${VW_THRESH_PERM}, k=${TOP_K_PER_LAYER}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --vw_threshold "$VW_THRESH_PERM" \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --no_activation_weighted \
    --output_suffix "_roleaware_static"
echo ""

# ── Select which graph to use for steps 07/08/09 ─────────────────────────────
GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
GRAPH_B="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_PROMPTS}_roleaware.json"
GRAPH_C="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_PROMPTS}_roleaware_static.json"

count_vw_edges() {
    local GRAPH_PATH=$1
    python3 -c "
import json, sys
try:
    g = json.load(open('$1'))
    vw_types = {'attribution_approx_v1', 'virtual_weight'}
    n = sum(1 for e in g.get('edges', []) if e.get('edge_type') in vw_types)
    print(n)
except:
    print(0)
" 2>/dev/null
}

VW_B=$(count_vw_edges "$GRAPH_B")
VW_C=$(count_vw_edges "$GRAPH_C")

echo "---- VW edge counts ----"
echo "  06b conservative (activation-weighted, t=${VW_THRESH_CONS}): ${VW_B} VW edges"
echo "  06c permissive   (static,              t=${VW_THRESH_PERM}): ${VW_C} VW edges"

SKIP_STEP08=""
if [ "${VW_B:-0}" -ge "${VW_C:-0}" ] && [ "${VW_B:-0}" -gt 0 ]; then
    GRAPH_SUFFIX="_roleaware"
    echo "  Selected: 06b (conservative) for steps 07/08/09"
elif [ "${VW_C:-0}" -gt 0 ]; then
    GRAPH_SUFFIX="_roleaware_static"
    echo "  Selected: 06c (permissive/static) for steps 07/08/09"
else
    GRAPH_SUFFIX="_roleaware"
    SKIP_STEP08=1
    echo "  Both variants: 0 VW edges. Step 08 will be skipped."
fi
echo ""

GRAPH_ROLEAWARE="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"

# ════════════════════════════════════════════════════════════════════════════
#  STEP 07: Ablation
# ════════════════════════════════════════════════════════════════════════════
echo "==== [D] Step 07: Ablation (graph=${GRAPH_SUFFIX}, top_k=${TOP_K_ABLATE}) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 09: Offline UI prep
# ════════════════════════════════════════════════════════════════════════════
echo "==== [E] Step 09: Offline UI prep ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --community_method louvain
RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
echo "  UI run: ${RUN_DIR:-NOT FOUND}"
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 08: Causal edges
# ════════════════════════════════════════════════════════════════════════════
if [ -z "$SKIP_STEP08" ]; then
    echo "==== [F] Step 08: Causal edges ===="
    date
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
else
    echo "==== [F] Step 08: SKIPPED (0 VW edges) ===="
    echo ""
fi

# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════════════
echo "========================================================"
echo "  RESTART SUMMARY"
echo "  Finished: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo "  VW edges (06b): ${VW_B}"
echo "  VW edges (06c): ${VW_C}"
echo "  Graph used    : attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"
echo "  Step 08 ran   : $([ -z "$SKIP_STEP08" ] && echo YES || echo NO)"
echo ""
echo "  UI run: ${RUN_DIR:-NOT FOUND}"
echo ""
echo "Top-10 features from star graph (run locally after rsync):"
echo "  python3 -c \""
echo "  import json; g=json.load(open('${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json'))"
echo "  nodes=[n for n in g['nodes'] if n.get('type')=='feature']"
echo "  top=sorted(nodes,key=lambda n:-n.get('specific_score',0))[:10]"
echo "  [print(f'  L{n[\"layer\"]}_F{n[\"feature_idx\"]}: specific={n.get(\"specific_score\",0):.4f} freq={n.get(\"frequency\",\"?\")}') for n in top]\""
