#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_v2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/phys_dt_v2-%j.out
#SBATCH --error=logs/phys_dt_v2-%j.out
#
# physics_decay_type v2 — full pipeline (108 prompts, F0-F4 families)
#
# Key changes from phase1:
#   - Single behaviour (decay_type only)
#   - N_PROMPTS=108 (54 alpha + 54 beta, four prompt families)
#   - TOP_K_PER_LAYER=5 (was 3) — more features for VW edge coverage
#   - Runs TWO role-aware graph variants:
#       06b: activation-weighted VW (canonical), threshold=0.05
#       06c: static VW (no activation weighting), threshold=0.01
#     Script selects whichever has more VW edges for steps 07/08/09
#   - Per-family baseline breakdown printed before proceeding
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   python scripts/generate_physics_decay_type_v2.py   # writes 108-prompt JSONL
#   wc -l data/prompts/physics_decay_type_train.jsonl   # expect 108
#   sbatch jobs/physics_decay_type_v2_pipeline.sbatch
#
# AFTER COMPLETION:
#   rsync logs/phys_dt_v2-JOBID.out to local
#   Check PIPELINE SUMMARY at the bottom of the log
#   Key signals: VW edge counts, F1 keyword-free accuracy, top-feature layers

set -uo pipefail

mkdir -p logs

# ── USER-EDITABLE PARAMETERS ─────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type
SPLIT=train
N_PROMPTS=108           # full v2 dataset; use 92 to exclude F4 after baseline
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50        # features extracted per layer per prompt (step 04)
TOP_K_PER_LAYER=5       # features kept per layer in role-aware graph (was 3)
VW_THRESH_CONS=0.05     # 06b: activation-weighted VW threshold
VW_THRESH_PERM=0.01     # 06c: static VW threshold (more permissive)
TOP_K_ABLATE=20         # top features passed to step 07
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=10
BASELINE_ABORT=0.80     # hard abort if overall sign_accuracy below this
# ─────────────────────────────────────────────────────────────────────────────

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type v2 Pipeline"
echo "  N_PROMPTS=${N_PROMPTS}  TOP_K_PER_LAYER=${TOP_K_PER_LAYER}"
echo "  VW conservative=${VW_THRESH_CONS}  permissive=${VW_THRESH_PERM}"
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

# ── Pre-flight: check prompt file ────────────────────────────────────────────
PROMPTS_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "FATAL: $PROMPTS_FILE not found."
    echo "       Run: python scripts/generate_physics_decay_type_v2.py"
    exit 1
fi
N_LINES=$(wc -l < "$PROMPTS_FILE")
echo "Pre-flight: $PROMPTS_FILE — ${N_LINES} prompts"
if [ "$N_LINES" -lt 100 ]; then
    echo "FATAL: Expected >= 100 prompts, got ${N_LINES}. Regenerate with v2 script."
    exit 1
fi
echo "Pre-flight: PASS"
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 02: Baseline
# ════════════════════════════════════════════════════════════════════════════
echo "==== [A] Step 02: Baseline ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

# ── Gate check ───────────────────────────────────────────────────────────────
echo ""
echo "---- Baseline gate check ----"
GATE_RESULT=$(python3 - <<PYEOF
import csv, sys
from pathlib import Path

csv_path = Path("data/results/baseline_{BEHAVIOUR}_{SPLIT}.csv".replace("{BEHAVIOUR}", "$BEHAVIOUR").replace("{SPLIT}", "$SPLIT"))
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found")
    sys.exit(1)

rows = list(csv.DictReader(csv_path.open()))
if not rows:
    print("ERROR: empty CSV")
    sys.exit(1)

sign_acc  = sum(1 for r in rows if float(r.get("logprob_diff_normalized", 0)) > 0) / len(rows)
mean_norm = sum(float(r.get("logprob_diff_normalized", 0)) for r in rows) / len(rows)

print(f"Baseline gate ($BEHAVIOUR):")
print(f"  n_prompts     : {len(rows)}")
print(f"  sign_accuracy : {sign_acc:.3f}  (abort < $BASELINE_ABORT)")
print(f"  mean_norm_diff: {mean_norm:.3f}")

# Per-family breakdown if surface_family column present
if rows and "surface_family" in rows[0]:
    by_fam = {}
    for r in rows:
        fam = r.get("surface_family", "unknown")
        by_fam.setdefault(fam, []).append(float(r.get("logprob_diff_normalized", 0)) > 0)
    print("\n  Per-family sign accuracy:")
    for fam in sorted(by_fam):
        vals = by_fam[fam]
        acc  = sum(vals) / len(vals)
        flag = "OK  " if acc >= 0.80 else "WARN"
        print(f"    [{flag}] {fam:<42} {acc:.1%}  ({int(sum(vals))}/{len(vals)})")

    # Keyword-free subset
    if "keyword_free" in rows[0]:
        kw = [r for r in rows if r.get("keyword_free", "").lower() in ("true", "1")]
        if kw:
            kw_acc = sum(1 for r in kw if float(r.get("logprob_diff_normalized", 0)) > 0) / len(kw)
            print(f"\n  Keyword-free (F1) sign accuracy: {kw_acc:.1%}  ({int(kw_acc*len(kw))}/{len(kw)})")

if sign_acc < $BASELINE_ABORT:
    print(f"\nGATE: HARD_FAIL  (sign_accuracy={sign_acc:.3f} < $BASELINE_ABORT)")
    sys.exit(2)
elif sign_acc < 0.90:
    print(f"\nGATE: WARN_PASS  (sign_accuracy={sign_acc:.3f} — below 0.90, continuing)")
else:
    print(f"\nGATE: PASS  (sign_accuracy={sign_acc:.3f})")
PYEOF
)
GATE_EXIT=$?
echo "$GATE_RESULT"

if [ $GATE_EXIT -eq 2 ]; then
    echo ""
    echo "FATAL: Baseline hard fail — pipeline aborted."
    echo "       Check per-family table above. If F4 is the cause,"
    echo "       resubmit with N_PROMPTS=92 to exclude F4."
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
#            activation-weighted VW edges, threshold=VW_THRESH_CONS
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
#  STEP 06c: Role-aware graph — permissive
#            static VW (no activation weighting), lower threshold
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

# Count VW edges in each variant.
# FIX 1 labels activation-weighted edges as "attribution_approx_v1";
# static edges remain "virtual_weight". Check both.
count_vw_edges() {
    local GRAPH_PATH=$1
    python3 -c "
import json, sys
try:
    g = json.load(open('$1'))
    vw_types = {'attribution_approx_v1', 'virtual_weight'}
    n = sum(1 for e in g.get('edges', []) if e.get('edge_type') in vw_types)
    print(n)
except Exception as e:
    print(0, file=sys.stderr)
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
    echo "  NOTE: activation-weighted VW suppressed. For future runs use --no_activation_weighted."
else
    GRAPH_SUFFIX="_roleaware"
    SKIP_STEP08=1
    echo "  Both variants: 0 VW edges."
    echo "  FINDING: No sequential VW-mediated structure at n=${N_PROMPTS}, k=${TOP_K_PER_LAYER}."
    echo "  Using 06b for ablation (step 07). Step 08 will be skipped."
    echo "  Next diagnostic: diag_concept_specificity.py + diag_layerwise_logit.py"
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
#  STEP 09: Offline UI prep  (run before 08 so UI is ready regardless)
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
#  STEP 08: Causal edges (only if VW edges exist in chosen graph)
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
    echo "  After rsync, run on login node (CPU):"
    echo "    python scripts/diag_concept_specificity.py --behaviour $BEHAVIOUR"
    echo "    python scripts/diag_layerwise_logit.py --behaviour $BEHAVIOUR"
    echo ""
fi

# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
echo "========================================================"
echo "  PIPELINE SUMMARY"
echo "  Finished: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo ""
echo "  N_PROMPTS       : ${N_PROMPTS}"
echo "  TOP_K_PER_LAYER : ${TOP_K_PER_LAYER}"
echo "  Graph used      : attribution_graph_${SPLIT}_n${N_PROMPTS}${GRAPH_SUFFIX}.json"
echo "  VW edges (06b)  : ${VW_B}"
echo "  VW edges (06c)  : ${VW_C}"
echo "  Step 08 ran     : $([ -z "$SKIP_STEP08" ] && echo YES || echo NO — 0 VW edges)"
echo ""
echo "Key output locations:"
echo "  Baseline   : data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv"
echo "  Features   : data/results/transcoder_features/layer_*/  "
echo "  Graphs     : data/results/attribution_graphs/${BEHAVIOUR}/"
echo "  Ablation   : data/results/interventions/${BEHAVIOUR}/"
echo "  Circuits   : data/results/causal_edges/${BEHAVIOUR}/  (if step 08 ran)"
echo "  UI run     : ${RUN_DIR:-data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS}/}"
echo ""
echo "First things to check after rsync:"
echo "  1. Per-family baseline accuracy (in log above, search 'Per-family')"
echo "  2. VW edge counts (printed above)"
echo "  3. Top-10 features from star graph:"
echo "     python3 -c \""
echo "     import json; g=json.load(open('${GRAPH_DIR}/attribution_graph_${SPLIT}_n${N_PROMPTS}.json'))"
echo "     nodes=[n for n in g['nodes'] if n.get('type')=='feature']"
echo "     top=sorted(nodes,key=lambda n:-n.get('specific_score',0))[:10]"
echo "     [print(f'  L{n[\"layer\"]}_F{n[\"feature_idx\"]}: specific={n.get(\"specific_score\",0):.4f} freq={n.get(\"freq\",n.get(\"frequency\",\"?\"))}') for n in top]\""
echo ""
