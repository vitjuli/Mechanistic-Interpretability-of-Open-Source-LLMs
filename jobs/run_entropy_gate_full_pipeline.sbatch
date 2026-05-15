#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=entropy-gate
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=logs/entropy-gate-%j.out
#SBATCH --error=logs/entropy-gate-%j.out
#
# physics_entropy_gate_v1 — Full Pipeline
#
# Scientific hypothesis:
#   Around L28, Qwen3-4B forms an internal thermodynamic gate-state
#   representation separating physically allowed from physically blocked
#   processes. This is not wording-driven: ARI_gate jumps at L28 while
#   ARI_wording drops (from gating_probe_v1 Family A screen, SLURM 29185755).
#
# Stages:
#   S0  Generate prompts (script 85) + token audit
#   S1  Behavioural baseline + gating metrics (script 86)
#       Gate: sign_acc ∈ [0.75, 0.95], pair_flip ≥ 0.70, AUC ≥ 0.85
#   S2  Representation decomposition (script 83, L0-L35)
#       Replicates L28 transition on full dataset
#   S3  Transcoder feature extraction (script 04, L10-L35, top_k=50)
#   S4a Star graph — decision_only (script 06)
#   S4b Role-aware graph — activation-weighted VW (script 06, t=0.05, k=8)
#   S4c Role-aware graph — static VW (script 06, t=0.01, k=8) for comparison
#   S5  Ablation per-feature on GRAPH_N subset (script 07)
#       Gate-specific: capture allow/block ΔND separately inline
#   S6  Offline UI prep (script 09)
#   S7  Causal edges (script 08) — if VW edges exist
#   S8  Minimal-pair patching: patch allow→block and block→allow at L28-L32
#
# Key parameters:
#   N_PROMPTS  = all prompts (baseline, features)
#   GRAPH_N    = 120 (subset for graph/ablation — balanced allow/block)
#   LAYERS     = L10-L35 (all available transcoders for Qwen3-4B)
#   TOP_K_PER_LAYER = 8 (larger: gate may be distributed across features)
#   VW_THRESH_CONS  = 0.05 (conservative, activation-weighted)
#   VW_THRESH_PERM  = 0.01 (permissive, static)
#
# Expected outputs:
#   data/prompts/physics_entropy_gate_v1_train.jsonl
#   data/results/physics_entropy_gate_v1/
#     baseline_results.csv, baseline_metrics.json
#     wording_breakdown.csv, pair_analysis.csv
#   data/results/gating_probe_v1/probe_by_layer.csv  (representation probe)
#   data/results/transcoder_features/layer_*/physics_entropy_gate_v1_train_*.npy
#   data/results/attribution_graphs/physics_entropy_gate_v1/
#     attribution_graph_train_n120.json               ← star
#     attribution_graph_train_n120_roleaware.json      ← activation-weighted
#     attribution_graph_train_n120_roleaware_static.json ← static VW
#   data/results/interventions/physics_entropy_gate_v1/
#     intervention_ablation_physics_entropy_gate_v1.csv
#   data/results/causal_edges/physics_entropy_gate_v1/
#     circuits_physics_entropy_gate_v1_train.json
#   data/ui_offline/<timestamp>_physics_entropy_gate_v1_train_n120/
#   data/results/physics_entropy_gate_v1/
#     gate_ablation_asymmetry.csv  ← allow vs block ΔND per feature
#     minimal_pair_patching.csv    ← layer L28-L32 patching results
#
# Time estimate:
#   S0-S2:  ~45 min (baseline + probe)
#   S3:     ~60 min (feature extraction, 26 layers × ~320 prompts)
#   S4a-c:  ~30 min (graph construction ×3)
#   S5:     ~6-8h  (ablation dominates; ~60 features × 120 prompts × 26 layers)
#   S6-S7:  ~60 min (UI + causal edges)
#   S8:     ~30 min (patching on ~40 pairs × 5 layers)
#   Total:  ~10-12h  (within 18h limit)
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   sbatch jobs/run_entropy_gate_full_pipeline.sbatch
#
# AFTER COMPLETION (run locally after rsync):
#   python scripts/86_run_entropy_gate_baseline.py  # local analysis only, no model
#   See SYNC block at the bottom of this file.

set -uo pipefail

mkdir -p logs

echo "========================================================"
echo "  physics_entropy_gate_v1 — Full Pipeline"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# AMD EPYC: disable AVX-512, force AVX2 kernels
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

# ── Pipeline-wide parameters ──────────────────────────────────────────────────
BEHAVIOUR=physics_entropy_gate_v1
SPLIT=train
PROMPT_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
GRAPH_N=120                     # subset for graph/ablation (balanced allow/block)
TOP_K_EXTRACT=50                # features extracted per layer per prompt (step 04)
TOP_K_PER_LAYER=8               # features kept per layer in role-aware graph
VW_THRESH_CONS=0.05             # activation-weighted VW threshold (conservative)
VW_THRESH_PERM=0.01             # static VW threshold (permissive)
TOP_K_ABLATE=20                 # top features for ablation
AGW_TOP_FRAC=0.6
TAU_CAUSAL=0.10
N_PATHS=10
BASELINE_ABORT=0.75

# L10-L35: full Qwen3-4B transcoder coverage (num_layers=36, verified in transcoder_config.yaml)
# Focus: L24-L35 expected to contain gate features given L28 transition
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"

GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
GRAPH_STAR="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}.json"
GRAPH_ROLEAWARE="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}_roleaware.json"
GRAPH_STATIC="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}_roleaware_static.json"
RESULTS_DIR="data/results/${BEHAVIOUR}"

echo "Parameters:"
echo "  GRAPH_N          = ${GRAPH_N}"
echo "  LAYERS           = L10-L35 (${#LAYERS} chars)"
echo "  TOP_K_PER_LAYER  = ${TOP_K_PER_LAYER}"
echo "  VW_THRESH_CONS   = ${VW_THRESH_CONS}"
echo "  VW_THRESH_PERM   = ${VW_THRESH_PERM}"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S0 — Generate prompts + token audit
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S0] Generate prompts + token audit ===="
date

python -u scripts/85_generate_entropy_gate_prompts.py

N_LINES=$(wc -l < "$PROMPT_FILE")
echo "  Prompt file: ${PROMPT_FILE} — ${N_LINES} prompts"
if [ "${N_LINES}" -lt 200 ]; then
    echo "ERROR: Expected ≥ 200 prompts, got ${N_LINES}. Check script 85."
    exit 1
fi
echo "  Pre-flight: PASS (${N_LINES} prompts)"

# Token audit inline
python3 - <<'TOKEN_AUDIT'
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
yes_ids = tok(" Yes", add_special_tokens=False).input_ids
no_ids  = tok(" No",  add_special_tokens=False).input_ids
assert len(yes_ids) == 1, f"' Yes' is multi-token: {yes_ids}"
assert len(no_ids)  == 1, f"' No' is multi-token: {no_ids}"
print(f"  Token audit PASS: ' Yes'={yes_ids[0]}  ' No'={no_ids[0]}")
TOKEN_AUDIT

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S1 — Behavioural baseline + gating metrics
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S1] Behavioural baseline + gating gate check ===="
date

python -u scripts/86_run_entropy_gate_baseline.py \
    --device cuda \
    --dtype  bfloat16

# Gate check — reads baseline_metrics.json
GATE_EXIT=0
python3 - <<PYEOF || GATE_EXIT=$?
import json, sys
from pathlib import Path

metrics_path = Path("data/results/physics_entropy_gate_v1/baseline_metrics.json")
if not metrics_path.exists():
    print(f"ERROR: {metrics_path} not found"); sys.exit(1)

m = json.load(open(metrics_path))
acc  = m["overall_sign_acc"]
flp  = m["pair_flip_rate"]
auc  = m["auc_gate_label"]
crs  = m["causal_readiness_score"]

print(f"\nGating gate check:")
print(f"  overall_sign_acc  : {acc:.4f}  ", end="")
print("WARN_CEILING" if acc > 0.95 else ("WARN_LOW" if acc < 0.75 else "PASS"))
print(f"  pair_flip_rate    : {flp:.4f}  ", end="")
print("PASS" if flp >= 0.70 else "WARN")
print(f"  auc_gate_label    : {auc:.4f}  ", end="")
print("PASS" if auc >= 0.85 else "WARN")
print(f"  CRS               : {crs:.4f}")

if acc < $BASELINE_ABORT:
    print(f"\nGATE: HARD_FAIL — sign_accuracy={acc:.4f} < {$BASELINE_ABORT}")
    sys.exit(2)
elif acc > 0.97:
    print("\nGATE: CEILING_WARN — sign_accuracy > 0.97, likely fact retrieval. Pipeline continues but interpret with caution.")
elif flp < 0.60:
    print(f"\nGATE: PAIR_FAIL — pair_flip_rate={flp:.4f} < 0.60. Model does not distinguish allow/block. Aborting.")
    sys.exit(2)
else:
    print("\nGATE: PASS — proceeding with full pipeline.")
PYEOF

if [ $GATE_EXIT -eq 2 ]; then
    echo ""; echo "FATAL: Baseline gate fail — pipeline aborted."; exit 1
fi
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S2 — Representation decomposition (L0-L35)
# Replicates the L28 gate transition found in gating_probe_v1 Family A
# on the full physics_entropy_gate_v1 dataset.
# Uses script 83 (hidden-state probe) pointed at the new prompt file.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S2] Representation decomposition (hidden-state probe, L0-L35) ===="
date

python3 - <<'PROBE_SCRIPT'
"""
Inline representation probe for physics_entropy_gate_v1.
Mirrors script 83 logic but reads physics_entropy_gate_v1_train.jsonl.
Outputs: data/results/physics_entropy_gate_v1/repr_probe_by_layer.csv
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.model_utils import ModelWrapper
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

PROMPT_PATH = Path("data/prompts/physics_entropy_gate_v1_train.jsonl")
OUT_DIR     = Path("data/results/physics_entropy_gate_v1")
MODEL_NAME  = "Qwen/Qwen3-4B"
PROBE_LAYERS = list(range(0, 36))
DEGEN_THR   = 1.0

prompts = [json.loads(l) for l in open(PROMPT_PATH)]
print(f"Loaded {len(prompts)} prompts for probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=torch.bfloat16)
mdl = mw.model
tok = mw.tokenizer
mdl.eval()

# Collect hidden states at final token for all layers
all_hs = {l: [] for l in PROBE_LAYERS}
gate_labels = []
wording_labels = []

with torch.no_grad():
    for i, row in enumerate(prompts):
        if i % 50 == 0:
            print(f"  hidden states: {i}/{len(prompts)}", end="\r", flush=True)
        inputs = tok(row["prompt"], return_tensors="pt").to(device)
        out    = mdl(**inputs, use_cache=False, output_hidden_states=True)
        for l in PROBE_LAYERS:
            hs = out.hidden_states[l][0, -1].float().cpu().numpy()
            all_hs[l].append(hs)
        gate_labels.append(1 if row["gate_label"] == "allow" else 0)
        wording_labels.append(row["wording_family"])
print()

gate_labels    = np.array(gate_labels)
wording_ints   = np.array([hash(w) % 100 for w in wording_labels])

probe_rows = []
for l in PROBE_LAYERS:
    X = np.stack(all_hs[l])
    std_val = float(X.std())
    if std_val < DEGEN_THR:
        print(f"  L{l:02d}: degenerate (std={std_val:.2f}), skipping")
        continue

    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)

    # Probe CV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(clf, Xs, gate_labels, cv=5, scoring="accuracy")
    probe_cv = float(cv_scores.mean())

    # ARI gate_label
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Xs)
    ari_gate    = float(adjusted_rand_score(gate_labels, clusters))
    ari_wording = float(adjusted_rand_score(wording_ints, clusters))

    probe_rows.append({
        "layer": l, "std": round(std_val, 3),
        "probe_cv": round(probe_cv, 4),
        "ari_gate": round(ari_gate, 4),
        "ari_wording": round(ari_wording, 4),
    })
    flag = " *** JUMP" if (ari_gate > 0.3 and l >= 25) else ""
    print(f"  L{l:02d}: probe_cv={probe_cv:.4f}  ari_gate={ari_gate:+.4f}  ari_wording={ari_wording:.4f}{flag}")

df = pd.DataFrame(probe_rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "repr_probe_by_layer.csv", index=False)

# Find first gate layer (ari_gate > 0.3)
gate_layers = df[df["ari_gate"] > 0.3]["layer"].values
first_gate  = int(gate_layers.min()) if len(gate_layers) else -1
peak_layer  = int(df.loc[df["ari_gate"].idxmax(), "layer"])
print(f"\n  First gate layer (ari_gate > 0.3): L{first_gate}")
print(f"  Peak gate layer:                    L{peak_layer} (ari_gate={df['ari_gate'].max():.4f})")
print(f"  Peak probe CV:                      L{int(df.loc[df['probe_cv'].idxmax(),'layer'])} ({df['probe_cv'].max():.4f})")
if first_gate == -1 or first_gate > 30:
    print("  WARNING: gate transition not found in L0-L35 — check prompt quality")
elif abs(first_gate - 28) > 3:
    print(f"  NOTE: gate transition at L{first_gate}, not L28 — dataset-specific shift")
else:
    print(f"  REPLICATED: L28-vicinity gate transition confirmed on full dataset")
print(f"Saved: {OUT_DIR}/repr_probe_by_layer.csv")
PROBE_SCRIPT

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S3 — Transcoder feature extraction (L10-L35, top_k=50)
#
# Extract top-50 transcoder features per layer per prompt.
# Uses all N_LINES prompts so downstream steps can choose GRAPH_N subset.
# context_tokens=5: last-5 token positions per prompt.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S3] Feature extraction (L10-L35, top_k=${TOP_K_EXTRACT}) ===="
date
N_LINES=$(wc -l < "$PROMPT_FILE")
python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --model_size 4b \
    --layers    $LAYERS \
    --top_k     "$TOP_K_EXTRACT" \
    --context_tokens 5
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S4a — Star graph (decision_only, GRAPH_N subset)
#
# Uses first GRAPH_N prompts.  The prompt file is shuffled (seed 42 in script 85),
# so the first 120 cover all wording families and both allow/block roles.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S4a] Star graph (decision_only, n=${GRAPH_N}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour  "$BEHAVIOUR" \
    --split      "$SPLIT" \
    --n_prompts  "$GRAPH_N" \
    --model_size 4b \
    --layers     $LAYERS
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S4b — Role-aware graph, activation-weighted VW (canonical)
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S4b] Role-aware graph (activation-weighted, t=${VW_THRESH_CONS}, k=${TOP_K_PER_LAYER}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour        "$BEHAVIOUR" \
    --split            "$SPLIT" \
    --n_prompts        "$GRAPH_N" \
    --model_size       4b \
    --layers           $LAYERS \
    --graph_node_mode  role_aware \
    --vw_threshold     "$VW_THRESH_CONS" \
    --top_k_per_layer  "$TOP_K_PER_LAYER" \
    --activation_weighted \
    --output_suffix    "_roleaware"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S4c — Role-aware graph, static VW (permissive)
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S4c] Role-aware graph (static, t=${VW_THRESH_PERM}, k=${TOP_K_PER_LAYER}) ===="
date
python -u scripts/06_build_attribution_graph.py \
    --behaviour        "$BEHAVIOUR" \
    --split            "$SPLIT" \
    --n_prompts        "$GRAPH_N" \
    --model_size       4b \
    --layers           $LAYERS \
    --graph_node_mode  role_aware \
    --vw_threshold     "$VW_THRESH_PERM" \
    --top_k_per_layer  "$TOP_K_PER_LAYER" \
    --no_activation_weighted \
    --output_suffix    "_roleaware_static"
echo ""

# ── Select which graph for steps S5-S7 ───────────────────────────────────────
count_vw_edges() {
    python3 -c "
import json, sys
try:
    g = json.load(open('$1'))
    vw_types = {'attribution_approx_v1', 'virtual_weight'}
    print(sum(1 for e in g.get('edges', []) if e.get('edge_type') in vw_types))
except Exception:
    print(0)
" 2>/dev/null
}

VW_B=$(count_vw_edges "$GRAPH_ROLEAWARE")
VW_C=$(count_vw_edges "$GRAPH_STATIC")
echo "---- VW edge counts ----"
echo "  S4b activation-weighted (t=${VW_THRESH_CONS}): ${VW_B} VW edges"
echo "  S4c static              (t=${VW_THRESH_PERM}): ${VW_C} VW edges"

SKIP_STEP08=""
if [ "${VW_B:-0}" -ge "${VW_C:-0}" ] && [ "${VW_B:-0}" -gt 0 ]; then
    GRAPH_SUFFIX="_roleaware"
    GRAPH_CHOSEN="$GRAPH_ROLEAWARE"
    echo "  Selected: S4b (activation-weighted) for ablation/UI/causal"
elif [ "${VW_C:-0}" -gt 0 ]; then
    GRAPH_SUFFIX="_roleaware_static"
    GRAPH_CHOSEN="$GRAPH_STATIC"
    echo "  Selected: S4c (static) — activation-weighted VW suppressed"
else
    GRAPH_SUFFIX="_roleaware"
    GRAPH_CHOSEN="$GRAPH_ROLEAWARE"
    SKIP_STEP08=1
    echo "  Both variants: 0 VW edges. Step S7 (causal edges) will be skipped."
    echo "  Action needed: check top-k features in star graph, inspect L28-L35 features."
fi
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S5 — Ablation (per-feature, top_k=20, graph subset GRAPH_N)
#
# Gate-specific output: after ablation CSV is written, compute ΔND separately
# for allow and block prompts per feature, to detect allow/block asymmetry.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S5] Ablation (per_feature, top_k=${TOP_K_ABLATE}, n=${GRAPH_N}) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --experiment     ablation \
    --n_prompts      "$GRAPH_N" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix   "$GRAPH_SUFFIX" \
    --layers         $LAYERS \
    --top_k          "$TOP_K_ABLATE" \
    --per_feature

echo ""

# ── Gate asymmetry analysis: allow vs block ΔND per feature ──────────────────
echo "---- Gate ablation asymmetry (allow vs block) ----"
ABL_CSV="data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}.csv"
python3 - <<ASYM_SCRIPT
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

abl_path = Path("${ABL_CSV}")
if not abl_path.exists():
    print("  Ablation CSV not found; skipping asymmetry analysis")
    sys.exit(0)

prompt_path = Path("data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl")
prompts_meta = {i: json.loads(l) for i, l in enumerate(open(prompt_path))}

df = pd.read_csv(abl_path)
if "prompt_idx" not in df.columns or "delta_logit_diff" not in df.columns:
    print("  Required columns missing from ablation CSV; skipping")
    sys.exit(0)

# Map prompt_idx to gate_label
df["gate_label"] = df["prompt_idx"].map(
    lambda i: prompts_meta.get(i, {}).get("gate_label", "unknown")
)

allow_df = df[df["gate_label"] == "allow"]
block_df = df[df["gate_label"] == "block"]

rows = []
for feat, g in df.groupby(["layer_idx", "feature_idx"] if "layer_idx" in df.columns
                           else ["feature_id"] if "feature_id" in df.columns
                           else ["feature"]):
    allow_g = g[g["gate_label"] == "allow"]
    block_g = g[g["gate_label"] == "block"]
    if allow_g.empty or block_g.empty:
        continue
    delta_col = "delta_logit_diff"
    nd_allow = float(allow_g[delta_col].mean())
    nd_block = float(block_g[delta_col].mean())
    asymmetry = abs(nd_allow - nd_block)
    rows.append({
        "feature": str(feat),
        "nd_allow_mean": round(nd_allow, 4),
        "nd_block_mean": round(nd_block, 4),
        "asymmetry": round(asymmetry, 4),
        "n_allow": len(allow_g),
        "n_block": len(block_g),
    })

if not rows:
    print("  No matched allow/block rows found in ablation CSV.")
    sys.exit(0)

asym_df = pd.DataFrame(rows).sort_values("asymmetry", ascending=False)
out = Path("data/results/${BEHAVIOUR}/gate_ablation_asymmetry.csv")
out.parent.mkdir(parents=True, exist_ok=True)
asym_df.to_csv(out, index=False)
print(f"  Saved: {out}")
print(f"  Top 10 asymmetric features (allow vs block ΔND):")
for _, r in asym_df.head(10).iterrows():
    print(f"    {r['feature']:<30}  allow={r['nd_allow_mean']:+.4f}  block={r['nd_block_mean']:+.4f}  asym={r['asymmetry']:.4f}")
ASYM_SCRIPT
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S6 — Offline UI prep
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S6] Offline UI prep ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix   "$GRAPH_SUFFIX" \
    --community_method louvain

RUN_DIR=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${GRAPH_N} 2>/dev/null | sort | tail -1)
echo "  UI run: ${RUN_DIR:-NOT FOUND}"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# S7 — Causal edges (only if VW edges exist)
# ═════════════════════════════════════════════════════════════════════════════
if [ -z "${SKIP_STEP08}" ]; then
    echo "==== [S7] Causal edges (graph=${GRAPH_SUFFIX}) ===="
    date
    python -u scripts/08_causal_edges.py \
        --behaviour      "$BEHAVIOUR" \
        --split          "$SPLIT" \
        --graph_json     "$GRAPH_CHOSEN" \
        --model_size     4b \
        --agw_top_frac   "$AGW_TOP_FRAC" \
        --tau_causal     "$TAU_CAUSAL" \
        --n_paths        "$N_PATHS" \
        --top_n_io_edges 10
    echo ""
else
    echo "==== [S7] Causal edges: SKIPPED (0 VW edges in both graphs) ===="
    echo "  Check: top features in star graph (layers L24-L35?)"
    echo "  Try: lower --top_k_per_layer to 5 or --vw_threshold to 0.01"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════════
# S8 — Minimal-pair patching (L28-L32 hook intervention)
#
# For each allow/block minimal pair with the same concept and wording:
#   - Patch activations from allow prompt into block prompt at layers L28-L32
#   - Record whether the block→allow flip occurs (nd flips from <0 to >0)
#   - Patch activations from block prompt into allow prompt (reverse test)
#   - Report flip rate by layer
#
# This tests the gating hypothesis: if L28 features encode allow/block,
# patching should flip the output.
# ═════════════════════════════════════════════════════════════════════════════
echo "==== [S8] Minimal-pair patching (L28-L32) ===="
date

python3 - <<'PATCH_SCRIPT'
"""
Minimal-pair activation patching for physics_entropy_gate_v1.

For each (pair_id, wording_family) pair:
  - Source: allow prompt (nd > 0)
  - Target: block prompt (nd < 0)
  - Patch: residual stream at layers 26-34 (covers L28 gate + ±3 margin)
  - Metric: does target nd flip from <0 to >0 after patching?

Also tests block→allow direction.
"""
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.model_utils import ModelWrapper

PROMPT_PATH  = Path("data/prompts/physics_entropy_gate_v1_train.jsonl")
OUT_DIR      = Path("data/results/physics_entropy_gate_v1")
MODEL_NAME   = "Qwen/Qwen3-4B"
PATCH_LAYERS = list(range(26, 35))   # L26-L34, covers L28 ± margin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mw  = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=torch.bfloat16)
mdl = mw.model
tok = mw.tokenizer
mdl.eval()

yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
no_id  = tok(" No",  add_special_tokens=False).input_ids[0]

prompts = [json.loads(l) for l in open(PROMPT_PATH)]

# Group into (pair_id, wording_family) → {allow: prompt, block: prompt}
groups = defaultdict(dict)
for p in prompts:
    key = (p["pair_id"], p["wording_family"])
    groups[key][p["pair_role"]] = p

# Only keep complete pairs
complete_pairs = [(k, v) for k, v in groups.items() if "allow" in v and "block" in v]
print(f"  Found {len(complete_pairs)} complete allow/block pairs")
if len(complete_pairs) == 0:
    print("  No pairs found — skipping patching.")
    sys.exit(0)

# Limit to 40 pairs to keep runtime bounded
import random
random.seed(42)
random.shuffle(complete_pairs)
complete_pairs = complete_pairs[:40]

def get_nd(input_ids):
    with torch.no_grad():
        out = mdl(input_ids=input_ids, use_cache=False)
        lp  = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(lp[yes_id] - lp[no_id])

def get_residuals(input_ids, layers):
    """Return dict layer→residual_at_final_token (1D tensor)."""
    residuals = {}
    hooks = []
    def make_hook(l):
        def h(module, inp, out):
            # Newer transformers may return hidden_states as a bare Tensor rather
            # than a (Tensor, ...) tuple — handle both.
            hs = out[0] if isinstance(out, tuple) else out  # (batch, seq, hidden)
            residuals[l] = hs[0, -1].detach().clone()
        return h
    for l in layers:
        hk = mdl.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(hk)
    with torch.no_grad():
        mdl(input_ids=input_ids, use_cache=False)
    for hk in hooks:
        hk.remove()
    return residuals

def patch_nd(source_ids, target_ids, patch_layers):
    """Patch residual at patch_layers from source into target forward pass."""
    src_residuals = get_residuals(source_ids, patch_layers)
    result_nd = {}

    for patch_layer in patch_layers:
        patched = {}
        def make_patch_hook(pl):
            def h(module, inp, out):
                if pl in src_residuals:
                    # Newer transformers may return hidden_states as a bare Tensor
                    # rather than a (Tensor, ...) tuple — handle both.
                    hs = out[0] if isinstance(out, tuple) else out
                    src_vec = src_residuals[pl].to(hs.dtype)
                    patched = hs.clone()
                    patched[0, -1] = src_vec
                    if isinstance(out, tuple):
                        return (patched,) + out[1:]
                    return patched
                return out
            return h
        hk = mdl.model.layers[patch_layer].register_forward_hook(make_patch_hook(patch_layer))
        with torch.no_grad():
            out = mdl(input_ids=target_ids, use_cache=False)
            lp  = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nd  = float(lp[yes_id] - lp[no_id])
        hk.remove()
        result_nd[patch_layer] = nd

    return result_nd

rows = []
for i, ((pid, wf), pair) in enumerate(complete_pairs):
    if i % 10 == 0:
        print(f"  Patching pair {i}/{len(complete_pairs)}", end="\r", flush=True)

    allow_p = pair["allow"]
    block_p = pair["block"]

    allow_ids = tok(allow_p["prompt"], return_tensors="pt").to(device).input_ids
    block_ids = tok(block_p["prompt"], return_tensors="pt").to(device).input_ids

    nd_allow_base = get_nd(allow_ids)
    nd_block_base = get_nd(block_ids)

    # Patch: allow residual → block prompt (expect block to flip to allow=Yes)
    nd_block_patched = patch_nd(allow_ids, block_ids, PATCH_LAYERS)
    # Patch: block residual → allow prompt (expect allow to flip to block=No)
    nd_allow_patched = patch_nd(block_ids, allow_ids, PATCH_LAYERS)

    for layer, nd_bp in nd_block_patched.items():
        nd_ap = nd_allow_patched.get(layer, float("nan"))
        rows.append({
            "pair_id":          pid,
            "wording_family":   wf,
            "concept":          allow_p.get("concept", ""),
            "patch_layer":      layer,
            "nd_allow_base":    round(nd_allow_base, 4),
            "nd_block_base":    round(nd_block_base, 4),
            "nd_block_patched": round(nd_bp, 4),
            "nd_allow_patched": round(nd_ap, 4),
            "block_flipped":    (nd_block_base < 0) and (nd_bp > 0),   # block→allow flip
            "allow_flipped":    (nd_allow_base > 0) and (nd_ap < 0),   # allow→block flip
        })

print()
patch_df = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
patch_df.to_csv(OUT_DIR / "minimal_pair_patching.csv", index=False)

# Summary by layer
print(f"\n  Minimal-pair patching results (allow→block patch into block prompt):")
print(f"  {'Layer':<8} {'block_flip_rate':>16} {'allow_flip_rate':>16} n_pairs")
for layer, g in patch_df.groupby("patch_layer"):
    bf = float(g["block_flipped"].mean())
    af = float(g["allow_flipped"].mean())
    print(f"  L{layer:<7} {bf:>16.4f} {af:>16.4f} {len(g)//2}")

print(f"\nSaved: {OUT_DIR}/minimal_pair_patching.csv")
PATCH_SCRIPT
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
echo "========================================================"
echo "  PIPELINE SUMMARY — physics_entropy_gate_v1"
echo "  Finished: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "  N_PROMPTS (baseline/features) : $(wc -l < ${PROMPT_FILE})"
echo "  GRAPH_N (graph/ablation)      : ${GRAPH_N}"
echo "  LAYERS                        : L10-L35"
echo "  TOP_K_PER_LAYER               : ${TOP_K_PER_LAYER}"
echo "  Graph used                    : ${GRAPH_SUFFIX}"
echo "  VW edges (activation-wtd)     : ${VW_B:-0}"
echo "  VW edges (static)             : ${VW_C:-0}"
echo "  Step S7 (causal edges) ran    : $([ -z "${SKIP_STEP08}" ] && echo YES || echo 'NO — 0 VW edges')"
echo ""
echo "Key output files:"
echo "  Baseline metrics  : data/results/${BEHAVIOUR}/baseline_metrics.json"
echo "  Repr probe        : data/results/${BEHAVIOUR}/repr_probe_by_layer.csv"
echo "  Star graph        : ${GRAPH_STAR}"
echo "  Roleaware graph   : ${GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
echo "  Ablation CSV      : data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}.csv"
echo "  Gate asymmetry    : data/results/${BEHAVIOUR}/gate_ablation_asymmetry.csv"
echo "  Minimal patching  : data/results/${BEHAVIOUR}/minimal_pair_patching.csv"
echo "  Circuits          : data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${SPLIT}.json"
echo "  UI run            : ${RUN_DIR:-data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${GRAPH_N}/}"
echo ""
echo "First things to check after rsync:"
echo "  1. repr_probe_by_layer.csv — does L28 transition replicate?"
echo "  2. baseline_metrics.json — sign_acc, pair_flip_rate, AUC, CRS"
echo "  3. gate_ablation_asymmetry.csv — which features show allow/block selectivity?"
echo "  4. minimal_pair_patching.csv — flip rate by layer (expect peak L28-L32)"
echo "  5. Top features in star graph:"
python3 -c "
import json
from pathlib import Path
g = json.load(open('${GRAPH_STAR}')) if Path('${GRAPH_STAR}').exists() else None
if g:
    nodes = [n for n in g['nodes'] if n.get('type') == 'feature']
    top   = sorted(nodes, key=lambda n: -n.get('specific_score', 0))[:10]
    for n in top:
        print(f'    L{n[\"layer\"]}_F{n[\"feature_idx\"]}: specific={n.get(\"specific_score\",0):.4f}  freq={n.get(\"freq\", n.get(\"frequency\",\"?\"))}')
" 2>/dev/null || echo "  (star graph not yet available)"
echo ""
echo "Sync to local:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/${BEHAVIOUR}/ data/results/${BEHAVIOUR}/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/${BEHAVIOUR}/ data/results/attribution_graphs/${BEHAVIOUR}/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/causal_edges/${BEHAVIOUR}/ data/results/causal_edges/${BEHAVIOUR}/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/ data/ui_offline/ --include='*${BEHAVIOUR}*' --exclude='*'"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/prompts/${BEHAVIOUR}_train.jsonl data/prompts/"
echo ""
