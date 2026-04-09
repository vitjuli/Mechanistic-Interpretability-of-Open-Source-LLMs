#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_sparse
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=28:00:00
#SBATCH --output=logs/mc_b1_sparse-%j.out
#SBATCH --error=logs/mc_b1_sparse-%j.err
#
# multilingual_circuits_b1 — SPARSIFIED GRAPH EXPERIMENT (Steps 02 → 13)
#
# Sparsification settings (from diag_sparsification_sweep.py, best config):
#   --top_k_per_layer 3   : keep top-3 features/layer by mean_abs_grad_attr (vs "all"=137)
#   --vw_threshold    0.05: keep only |VW weight| >= 0.05           (vs 0.01 default)
#
# Expected improvement over gradient run (SLURM 26929559):
#   communities:    7 (~6 nodes each)  vs 12 (4 giants + 8 singletons)
#   largest_pct:    23%                vs 26%
#   singletons:     1                  vs 8
#   circuit feats:  ~12                vs 13 (unchanged)
#   trajectory:     0.750              vs 0.750 (unchanged)
#   mean_abs_effect 0.995 (+10%)       vs 0.903
#
# All other parameters identical to the gradient run.
# Attribution method: gradient × activation (unchanged).
# Compares ONLY graph sparsification.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/multilingual_circuits_b1_sparsified.sbatch
#
# AFTER COMPLETION:
#   rsync outputs, then run analysis commands printed at the end of the log.

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "  multilingual_circuits_b1 — Sparsified Graph (Steps 02→13)"
echo "  node_pruning=top_3_per_layer  vw_threshold=0.05"
echo "============================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"

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

# ── Pipeline-wide parameters ─────────────────────────────────────────────────
BEHAVIOUR=multilingual_circuits_b1
SPLIT=train
N_PROMPTS=96
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50        # script 04 (unchanged)
TOP_K_FEATURES=20       # script 07 (unchanged)
AGW_TOP_FRAC=0.6        # script 08 (unchanged)
TAU_CAUSAL=0.10         # script 08 (unchanged)
N_PATHS=50              # script 08 (unchanged)
K_CONTENT=10            # script 06b (unchanged)

# ── SPARSIFICATION PARAMETERS (changed from defaults) ──────────────────────
TOP_K_PER_LAYER=3       # keep top-3 features per layer by mean_abs_grad_attr
VW_THRESHOLD=0.05       # VW edge threshold (was 0.01)
OUTPUT_SUFFIX=_roleaware_k3_t05  # distinguishes outputs from default run

# ── Derived paths ─────────────────────────────────────────────────────────────
GRAPH_STAR="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json"
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
CIRCUITS="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}${OUTPUT_SUFFIX}.json"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "==== Pre-flight checks ===="
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found. Run script 01 first."
    exit 1
fi
ACTUAL=$(wc -l < "$PROMPT_FILE")
echo "  Prompts: ${ACTUAL} (expect ${N_PROMPTS})"
[ "$ACTUAL" -ne "$N_PROMPTS" ] && echo "ERROR: count mismatch" && exit 1
echo "  Pre-flight: PASS"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE A — Step 02: Baseline gate
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE A] Step 02: Baseline ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

python3 - <<'PYEOF'
import csv, sys
from pathlib import Path
behaviour, split = "multilingual_circuits_b1", "train"
rows = list(csv.DictReader(open(f"data/results/baseline_{behaviour}_{split}.csv")))
def sacc(rs): return sum(1 for r in rs if float(r.get("logprob_diff_normalized",0))>0)/max(1,len(rs))
def mnorm(rs): return sum(float(r.get("logprob_diff_normalized",0)) for r in rs)/max(1,len(rs))
en = [r for r in rows if r.get("language","en")=="en"]
fr = [r for r in rows if r.get("language","en")=="fr"]
ea, fa, mn = sacc(en), sacc(fr), mnorm(rows)
print(f"  EN={ea:.3f}  FR={fa:.3f}  mean_norm={mn:.3f}")
if ea<0.90 or fa<0.65 or mn<1.00:
    print("GATE FAIL — aborting."); sys.exit(1)
print("GATE PASS")
PYEOF
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE B — Step 04: Transcoder feature extraction
# (identical to gradient run — activations don't change with graph params)
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE B] Step 04: Feature extraction (top_k=${TOP_K_EXTRACT}, last_5) ===="
date
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --layers $LAYERS \
    --top_k "$TOP_K_EXTRACT" \
    --context_tokens 5
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE C — Step 06a: Decision-only star graph (for script 07 ablation)
# Same as gradient run — script 07 needs this star graph.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE C] Step 06a: Star graph (beta proxy, for script 07) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE C — Step 06b: Role-aware graph — SPARSIFIED
#
# KEY CHANGE vs gradient run:
#   --top_k_per_layer 3    : prune to top-3 features per layer (vs "all" = 137 nodes)
#   --vw_threshold 0.05    : tighter VW edge threshold (vs 0.01 → removes 92% of edges)
#   --output_suffix _roleaware_k3_t05   : separate output file
#
# Attribution method: gradient × activation (unchanged from gradient run).
# Expected output: ~48 nodes, ~84 VW edges → 7 Louvain communities.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE C] Step 06b: Role-aware graph (SPARSIFIED: k=${TOP_K_PER_LAYER}/layer, t=${VW_THRESHOLD}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS \
    --graph_node_mode role_aware \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --vw_threshold "$VW_THRESHOLD" \
    --k_content "$K_CONTENT" \
    --output_suffix "$OUTPUT_SUFFIX"

# Verify output
python3 - <<PYEOF
import json
g = json.load(open("${GRAPH_ROLEAWARE}"))
feat = [n for n in g['nodes'] if n['type']=='feature']
vw   = [e for e in g['edges'] if e.get('edge_type')=='virtual_weight']
print(f"  Graph: {len(feat)} feature nodes (target ~48), {len(vw)} VW edges (target ~84)")
if len(feat) > 80:
    print(f"  WARNING: {len(feat)} nodes > 80 — check --top_k_per_layer flag is implemented in script 06")
PYEOF
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE D — Step 07: Ablation + Patching C3
# (uses star graph — unchanged, same as gradient run)
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE D] Step 07: Ablation (per_feature, top_k=${TOP_K_FEATURES}) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --layers $LAYERS \
    --top_k "$TOP_K_FEATURES" \
    --per_feature
echo ""

echo "==== [STAGE D] Step 07: Patching C3 (language swap EN→FR) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment patching \
    --patch_mode C3 \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$N_PROMPTS" \
    --layers $LAYERS \
    --top_k "$TOP_K_FEATURES" \
    --per_feature
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE E — Step 09: Offline UI prep
# Reads sparsified role-aware graph → Louvain on sparsified VW subgraph.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE E] Step 09: Offline UI prep ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --graph_suffix "$OUTPUT_SUFFIX" \
    --community_method louvain

RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
[ -z "$RUN_DIR" ] && echo "ERROR: UI run dir not found" && exit 1
RUN_ID=$(basename "$RUN_DIR")
echo "  UI run: $RUN_ID"

python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE F — Step 08: Causal edges + S1/S1.5/S2
# Reads sparsified role-aware graph.
# With fewer nodes (~48) AGW prefilter is faster; expect ~60-100 causal edges.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE F] Step 08: Causal edges + sufficiency (sparsified graph) ===="
date
[ ! -f "$GRAPH_ROLEAWARE" ] && echo "ERROR: $GRAPH_ROLEAWARE not found" && exit 1

python3 - <<PYEOF
import json
g = json.load(open("${GRAPH_ROLEAWARE}"))
feat = [n for n in g['nodes'] if n['type']=='feature']
vw   = [e for e in g['edges'] if e.get('edge_type')=='virtual_weight']
print(f"  Input graph: {len(feat)} features, {len(vw)} VW edges")
has_grad = feat and 'mean_grad_attr_conditional' in feat[0]
print(f"  Gradient attrs: {'PRESENT' if has_grad else 'MISSING — check step 06b'}")
PYEOF

python -u scripts/08_causal_edges.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_json "$GRAPH_ROLEAWARE" \
    --circuits_output "$CIRCUITS" \
    --model_size 4b \
    --agw_top_frac "$AGW_TOP_FRAC" \
    --tau_causal "$TAU_CAUSAL" \
    --n_paths "$N_PATHS" \
    --top_n_io_edges 10
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE G — Step 11: Ablation supplement
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE G] Step 11: Ablation supplement ===="
date
python -u scripts/11_ablation_supplement.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b

SUPPLEMENT_CSV="data/results/reasoning_traces/${BEHAVIOUR}/ablation_supplement_${SPLIT}.csv"
[ ! -f "$SUPPLEMENT_CSV" ] && echo "WARNING: supplement CSV not found" && SUPPLEMENT_CSV=""
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE G — Step 10: Reasoning traces
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE G] Step 10: Reasoning traces ===="
date
[ ! -f "$CIRCUITS" ] && echo "ERROR: $CIRCUITS not found (step 08 failed?)" && exit 1

STEP10_ARGS=(
    --behaviour "$BEHAVIOUR"
    --split "$SPLIT"
    --run_id "$RUN_ID"
    --circuits_json "$CIRCUITS"
    --top_k_paths 5
)
[ -n "$SUPPLEMENT_CSV" ] && STEP10_ARGS+=(--supplement_csv "$SUPPLEMENT_CSV")

python -u scripts/10_reasoning_trace.py "${STEP10_ARGS[@]}"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE H — Steps 12 + 13: Competition + L22 dose-response
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE H] Step 12: Competition / failure analysis ===="
date
python -u scripts/12_competition_analysis.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --top_k 10
echo ""

echo "==== [STAGE H] Step 13: L22 dose-response ===="
date
python -u scripts/13_l22_intervention.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --alphas 2.0 3.0
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo "  Pipeline complete: multilingual_circuits_b1 (sparsified)"
echo "  Job: $SLURM_JOB_ID   Finished: $(date)"
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  Star graph:       $GRAPH_STAR"
echo "  Roleaware graph:  $GRAPH_ROLEAWARE"
echo "  Circuit:          $CIRCUITS"
echo "  UI run:           $RUN_DIR"
echo ""
echo "Sparsification params used:"
echo "  top_k_per_layer=${TOP_K_PER_LAYER}  vw_threshold=${VW_THRESHOLD}"
echo "  output_suffix=${OUTPUT_SUFFIX}"
echo ""
echo "Post-run analysis (run locally after rsync):"
echo ""
echo "  python scripts/a_analyze_multilingual_circuits.py \\"
echo "      --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "      --node_labels --community_summary \\"
echo "      --graph_json ${GRAPH_ROLEAWARE}"
echo ""
echo "  python scripts/prepare_b1_dashboard.py \\"
echo "      --graph_suffix ${OUTPUT_SUFFIX}"
echo ""
echo "Compare vs gradient run (SLURM 26929559):"
echo "  communities:   target 7 (~6 nodes)  vs prev 12 (4 giants + 8 singletons)"
echo "  largest_pct:   target 23%           vs prev 26%"
echo "  singletons:    target 1             vs prev 8"
echo "  necessity:     run diag after — proxy suggests unchanged or improved"
