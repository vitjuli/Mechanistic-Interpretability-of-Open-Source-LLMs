#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=mc_b1_full
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=28:00:00
#SBATCH --output=logs/mc_b1_full-%j.out
#SBATCH --error=logs/mc_b1_full-%j.err
#
# multilingual_circuits_b1 — FULL PIPELINE (Steps 02 → 13)
#
# Stages:
#   A  Step 02   Baseline gate (EN ≥ 0.90, FR ≥ 0.65, mean_norm_diff ≥ 1.0)
#   B  Step 04   Transcoder feature extraction (last_5, top_k=50)
#   C  Step 06a  Attribution graph — decision-only star (beta proxy; → 07 ablation/patching)
#   C  Step 06b  Attribution graph — role-aware + VW edges (GRADIENT attribution; → 08/09/analysis)
#   D  Step 07   Ablation per-feature (top_k=20)
#   D  Step 07   Patching C3 per-feature (language swap, EN→FR)
#   E  Step 09   Offline UI prep (Louvain communities)
#   F  Step 08   Causal edges + S1/S1.5 sufficiency + S2 injection + circuit JSON
#   G  Step 10   Reasoning traces (uses circuit + supplement)
#   G  Step 11   Ablation supplement (2 features not reached by Group ablation)
#   G  Step 10   Reasoning traces v2 (re-run with supplement)
#   H  Step 12   Competition / failure analysis (incorrect prompts, quote-completion rate)
#   H  Step 13   L22 feature dose-response intervention (α=2.0,3.0)
#
# Key parameter choices:
#   --context_tokens 5  : last-5 token positions (decision + 4 context); pooled IoU reference
#   --top_k 50 (04)     : top-50 features per token saved; sufficient for all downstream top-k
#   --top_k 20 (07)     : covers all graph features (~4-6 per layer in 84-node graph)
#   --per_feature       : must be explicit — default resolves False for _b1 suffix
#   --vw_threshold 0.01 : VW edges in role-aware graph (canonical B1-v2 value)
#   --k_content 10      : max content-word nodes (B1-v2 canonical)
#   --graph_node_mode role_aware (06b): NOW uses gradient × activation attribution
#                         (α_k = a_k × ∂Δlogit/∂a_k via MLP backward hook)
#                         instead of OLS beta proxy from B1-v2
#   --agw_top_frac 0.6  : keep top 60% of VW edges as causal candidates
#   --tau_causal 0.10   : causal edge threshold (mean |Δmargin| ≥ 0.10)
#   --n_paths 50        : path tracing budget
#   --alphas 2.0 3.0    : L22 dose-response (script 13)
#
# Expected outputs:
#   data/results/transcoder_features/layer_*/multilingual_circuits_b1_train_*.npy
#   data/results/attribution_graphs/multilingual_circuits_b1/
#       attribution_graph_train_n96.json               ← star (beta proxy, for 07)
#       attribution_graph_train_n96_roleaware.json      ← role-aware (gradient, for 08/09)
#   data/results/interventions/multilingual_circuits_b1/
#       intervention_ablation_multilingual_circuits_b1.csv
#       intervention_patching_C3_multilingual_circuits_b1.csv
#   data/ui_offline/<timestamp>_multilingual_circuits_b1_train_n96/
#   data/results/causal_edges/multilingual_circuits_b1/
#       circuits_multilingual_circuits_b1_train.json
#       causal_edges_multilingual_circuits_b1_train.json
#       presentation_graph_multilingual_circuits_b1_train.json
#   data/results/reasoning_traces/multilingual_circuits_b1/
#       reasoning_traces_train.jsonl
#       ablation_supplement_train.csv
#   data/results/competition_analysis/multilingual_circuits_b1/
#   data/results/l22_intervention/multilingual_circuits_b1/
#
# Time estimate: ~22-26h (step 07 ablation dominates; gradient adds ~5 min to step 06b)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   wc -l data/prompts/multilingual_circuits_b1_train.jsonl   # expect 96
#   sbatch jobs/multilingual_circuits_b1_full_pipeline.sbatch
#
# AFTER COMPLETION:
#   Sync to local, then run analysis (see bottom of this file for commands)

set -euo pipefail

mkdir -p logs

echo "========================================================"
echo "  multilingual_circuits_b1 — Full Pipeline (Steps 02→13)"
echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# AMD EPYC: disable AVX-512 to prevent SIGILL; force AVX2 kernels
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
TOP_K_FEATURES=20       # script 07
TOP_K_EXTRACT=50        # script 04
VW_THRESHOLD=0.01
K_CONTENT=10
OUTPUT_SUFFIX=_roleaware
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=50

GRAPH_STAR="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json"
GRAPH_ROLEAWARE="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${N_PROMPTS}${OUTPUT_SUFFIX}.json"
CIRCUITS="data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo "==== Pre-flight checks ===="
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found. Run: python scripts/01_generate_prompts.py --behaviour ${BEHAVIOUR}"
    exit 1
fi
ACTUAL=$(wc -l < "$PROMPT_FILE")
echo "  Prompt file: ${ACTUAL} lines (expect ${N_PROMPTS})"
if [ "$ACTUAL" -ne "$N_PROMPTS" ]; then
    echo "ERROR: Expected ${N_PROMPTS} prompts, got ${ACTUAL}"
    exit 1
fi
echo "  Pre-flight: PASS"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE A — Step 02: Baseline evaluation
# Gate: EN sign_accuracy >= 0.90, FR >= 0.65, mean_norm_logprob_diff >= 1.0
# If gate fails, pipeline is aborted (set -e propagates exit code 1 from python).
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE A] Step 02: Baseline evaluation ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

# Inline gate check
python3 - <<'PYEOF'
import csv, sys
from pathlib import Path

behaviour = "multilingual_circuits_b1"
split     = "train"
csv_path  = Path(f"data/results/baseline_{behaviour}_{split}.csv")
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found"); sys.exit(1)

rows = list(csv.DictReader(csv_path.open()))

def sign_acc(rs):
    ok = sum(1 for r in rs if float(r.get("logprob_diff_normalized", 0)) > 0)
    return ok / len(rs) if rs else 0.0

def mean_norm(rs):
    vals = [float(r.get("logprob_diff_normalized", 0)) for r in rs]
    return sum(vals) / len(vals) if vals else 0.0

en_rows = [r for r in rows if r.get("language", "en") == "en"]
fr_rows = [r for r in rows if r.get("language", "en") == "fr"]

en_acc   = sign_acc(en_rows)
fr_acc   = sign_acc(fr_rows)
all_norm = mean_norm(rows)

print(f"Baseline gate:")
print(f"  EN sign_accuracy : {en_acc:.3f}  (need >= 0.90)  {'PASS' if en_acc >= 0.90 else 'FAIL'}")
print(f"  FR sign_accuracy : {fr_acc:.3f}  (need >= 0.65)  {'PASS' if fr_acc >= 0.65 else 'FAIL'}")
print(f"  mean_norm_diff   : {all_norm:.3f}  (need >= 1.00)  {'PASS' if all_norm >= 1.00 else 'FAIL'}")

if en_acc < 0.90 or fr_acc < 0.65 or all_norm < 1.00:
    print("\nGATE FAIL — aborting pipeline.")
    sys.exit(1)
print("\nGATE PASS — proceeding.")
PYEOF

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE B — Step 04: Extract transcoder features
#
# --context_tokens 5: captures last 5 token positions per prompt.
#   Shape: (480, 50) = 5 pos × 96 prompts, top_k=50 features.
#   position_map: 480 entries, is_decision_position=True for last token only.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE B] Step 04: Transcoder feature extraction (last_5, top_k=${TOP_K_EXTRACT}) ===="
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
# STAGE C — Step 06a: Decision-only star graph (beta proxy)
#
# Per-prompt union of top-k features ranked by activation × OLS-beta.
# Used by Step 07 ablation/patching (script 07 loads by --graph_n_prompts).
# No VW edges (star topology).
# Output: attribution_graph_train_n96.json
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE C] Step 06a: Decision-only star graph (beta proxy) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --n_prompts "$N_PROMPTS" \
    --model_size 4b \
    --layers $LAYERS

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE C — Step 06b: Role-aware graph with VW edges (GRADIENT attribution)
#
# UPGRADED from B1-v2: uses gradient × activation (α_k = a_k × ∂Δlogit/∂a_k)
# instead of OLS-beta proxy. Per-prompt, via backward hook on block.mlp.
#   Gradient path: ∂Δlogit/∂a_k ≈ (∂Δlogit/∂MLP_output) · W_dec[k,:]
#   MLP output gradient captured by register_full_backward_hook on each layer.
# Node attrs: mean_grad_attr_conditional, mean_abs_grad_attr_conditional,
#             grad_attr_sign (replacing beta/specific_score).
# Also adds content-word nodes (k_content=10) and VW edges (threshold=0.01).
# Output: attribution_graph_train_n96_roleaware.json
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE C] Step 06b: Role-aware graph (GRADIENT attribution, vw_threshold=${VW_THRESHOLD}) ===="
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

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE D — Step 07: Ablation (per-feature, star graph)
#
# --per_feature: one ablation pass per feature (not group).
#   MUST be explicit — default for _b1 suffix resolves to False.
# Reads star graph (attribution_graph_train_n96.json) via --graph_n_prompts.
# Output: intervention_ablation_multilingual_circuits_b1.csv
#         (required by find_bridge_features() in analysis script)
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

# ═════════════════════════════════════════════════════════════════════════════
# STAGE D — Step 07: Patching C3 (language swap EN→FR, per-feature)
#
# C3 pairs: same (concept_index, template_idx), EN source + FR target.
# 48 EN + 48 FR train prompts → up to 48 C3 pairs.
# Output: intervention_patching_C3_multilingual_circuits_b1.csv
#         (C3 disruption_rate, bridge feature identification)
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE D] Step 07: Patching C3 (language swap, per_feature) ===="
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
# STAGE E — Step 09: Offline UI preparation
#
# Reads role-aware graph (n96_roleaware) for graph/community data.
# Reads intervention CSVs for feature effects.
# Louvain community detection on VW-only subgraph (abs(weight), star excluded).
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE E] Step 09: Offline UI prep (Louvain communities) ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS" \
    --community_method louvain

# Capture run_id for downstream scripts
RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not find UI run directory after step 09."
    exit 1
fi
RUN_ID=$(basename "$RUN_DIR")
echo "  UI run: $RUN_ID"

# Sanity check
python -u scripts/09_sanity_check_ui_data.py --run_dir "$RUN_DIR"

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE F — Step 08: Causal edges + sufficiency
#
# Reads role-aware graph (n96_roleaware).
# Phase 1  AGW prefilter: top ${AGW_TOP_FRAC} of VW edges (|w| × freq_src × freq_tgt)
# Phase 2  Causal edge ablation loop (~84 features × 96 prompts)
# Phase 3  Edge aggregation (tau_causal=${TAU_CAUSAL}) → causal_edges JSON
# Phase 4  Path tracing (top ${N_PATHS} paths by cumulative attribution)
# Phase 5  Necessity: group ablation of all circuit features
# Phase 6  S1 sufficiency: linear combination of circuit decoders
# Phase 7  S1.5 sufficiency: layerwise re-injection
# Phase 8  S2 sufficiency: cross-prompt injection (EN→FR transfer)
# Output: circuits_multilingual_circuits_b1_train.json
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE F] Step 08: Causal edges + S1/S1.5/S2 sufficiency ===="
date
if [ ! -f "$GRAPH_ROLEAWARE" ]; then
    echo "ERROR: Role-aware graph not found: $GRAPH_ROLEAWARE"
    exit 1
fi
python -c "
import json
g = json.load(open('${GRAPH_ROLEAWARE}'))
n_feat = sum(1 for n in g['nodes'] if n['type'] == 'feature')
n_vw   = sum(1 for e in g['edges'] if e.get('edge_type') == 'virtual_weight')
print(f'  Graph: {n_feat} feature nodes, {n_vw} VW edges')
# Verify gradient attribution fields present
fn = [n for n in g['nodes'] if n['type'] == 'feature']
if fn and 'mean_grad_attr_conditional' in fn[0]:
    print('  Gradient attribution fields: PRESENT')
else:
    print('  WARNING: gradient attribution fields missing — check step 06b log')
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

# ═════════════════════════════════════════════════════════════════════════════
# STAGE G — Step 11: Ablation supplement (2 features excluded from group ablation)
#
# L23_F64429 (mean|eff|=1.330, neg_frac=77%) and L24_F76363 (mean|eff|=0.770)
# had too-high activation to be included in the group ablation via Phase 7.
# Step 11 runs targeted single-feature ablation for these.
# Must run BEFORE step 10 v2 so supplement_csv is available.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE G] Step 11: Ablation supplement ===="
date
python -u scripts/11_ablation_supplement.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b

SUPPLEMENT_CSV="data/results/reasoning_traces/${BEHAVIOUR}/ablation_supplement_${SPLIT}.csv"
echo "  Supplement CSV: $SUPPLEMENT_CSV"
if [ ! -f "$SUPPLEMENT_CSV" ]; then
    echo "WARNING: supplement CSV not found — step 10 will run without supplement"
    SUPPLEMENT_CSV=""
fi

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE G — Step 10: Reasoning traces (with supplement)
#
# Uses circuit JSON + ablation supplement to trace per-prompt feature trajectories.
# --run_id: the UI offline run directory name (captured from step 09 output).
# --top_k_paths 5: display top-5 paths per prompt.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE G] Step 10: Reasoning traces (run_id=${RUN_ID}) ===="
date
if [ ! -f "$CIRCUITS" ]; then
    echo "ERROR: Circuits JSON not found: $CIRCUITS (step 08 must have failed)"
    exit 1
fi

STEP10_ARGS=(
    --behaviour "$BEHAVIOUR"
    --split "$SPLIT"
    --run_id "$RUN_ID"
    --top_k_paths 5
)
if [ -n "$SUPPLEMENT_CSV" ]; then
    STEP10_ARGS+=(--supplement_csv "$SUPPLEMENT_CSV")
fi

python -u scripts/10_reasoning_trace.py "${STEP10_ARGS[@]}"

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE H — Step 12: Competition / failure analysis
#
# Identifies incorrect prompts; computes argmax_is_quote_rate;
# checks L13_F70603 correlation with prediction margin.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE H] Step 12: Competition / failure analysis ===="
date
python -u scripts/12_competition_analysis.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --model_size 4b \
    --top_k 10

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STAGE H — Step 13: L22 feature dose-response intervention
#
# Stimulates artifact features (L22_F108295 + L22_F32734) vs control (L22_F78043)
# at α=2.0 and α=3.0 to confirm artifact vs semantic distinction.
# Expected: artifact features HURT correct predictions at all α (monotone ↓).
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [STAGE H] Step 13: L22 dose-response intervention ===="
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
echo "========================================================"
echo "  Pipeline complete: multilingual_circuits_b1"
echo "  Job: $SLURM_JOB_ID"
echo "  Finished: $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  Attribution graphs (star + gradient roleaware):"
echo "    $GRAPH_STAR"
echo "    $GRAPH_ROLEAWARE"
echo "  Interventions:"
echo "    data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}.csv"
echo "    data/results/interventions/${BEHAVIOUR}/intervention_patching_C3_${BEHAVIOUR}.csv"
echo "  Circuit:"
echo "    $CIRCUITS"
echo "  UI:"
echo "    $RUN_DIR"
echo ""
echo "Post-run analysis (run locally after rsync):"
echo ""
echo "  # Core IoU + bridge + C3 report"
echo "  python scripts/a_analyze_multilingual_circuits.py \\"
echo "      --behaviour ${BEHAVIOUR} --split ${SPLIT}"
echo ""
echo "  # Node language labels + community summary (gradient roleaware graph)"
echo "  python scripts/a_analyze_multilingual_circuits.py \\"
echo "      --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "      --node_labels --community_summary \\"
echo "      --graph_json ${GRAPH_ROLEAWARE}"
echo ""
echo "  # Dashboard data prep"
echo "  python scripts/prepare_b1_dashboard.py"
echo "  cd dashboard_b1 && npm run dev"
