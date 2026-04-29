#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe_pdt
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=logs/probe_pdt-%j.out
#SBATCH --error=logs/probe_pdt-%j.out
#
# physics_decay_type_probe — latent-state probing pipeline
#
# Corpus: data/prompts/physics_decay_type_probe_train.jsonl (~470 prompts)
# Design: docs/latent_probing_corpus_phase2.md
#
# Sections:
#   A. Baseline evaluation (step 02, instruct model)
#      Checks model accuracy per level and per group.
#   B. Feature extraction (step 04, base model)
#      Saves transcoder activations for every prompt × layer.
#      This is the primary data for SE/IC convergence analysis.
#   C. Ablation interventions (step 07, base model)
#      Reuses the existing physics_decay_type train graph.
#      Ablates the same 40 circuit features on all probe prompts.
#   D. UI preparation (step 09)
#      Generates a dashboard-ready run directory for the probe corpus.
#
# After rsync to local:
#   python scripts/prepare_decay_dashboard.py --behaviour physics_decay_type_probe
#   python scripts/18_latent_state_analysis.py   # run locally for SE/IC analysis
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   # Generate probe prompts locally first, then rsync:
#   #   python scripts/generate_physics_decay_type_probe.py
#   #   rsync data/prompts/physics_decay_type_probe_train.jsonl iv294@...
#   wc -l data/prompts/physics_decay_type_probe_train.jsonl    # expect ~470
#   ls data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json
#   sbatch jobs/physics_decay_type_probe_pipeline.sbatch

# ── Parameters ────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type_probe
SPLIT=train
N_PROMPTS=470                    # update if corpus size changes
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50                 # features extracted per layer per prompt (step 04)
TOP_K_ABLATE=20                  # top features for ablation (step 07)
BASELINE_ABORT=0.75              # hard abort if overall sign_accuracy < this
# Reuse the existing physics_decay_type train graph for ablations:
GRAPH_BEHAVIOUR=physics_decay_type
GRAPH_SPLIT=train
GRAPH_N=108
GRAPH_SUFFIX=_roleaware_static
# ─────────────────────────────────────────────────────────────────────────────

cd /rds/user/iv294/hpc-work/thesis/project

section_results=()

run_section() {
    local label=$1
    local fn=$2
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  SECTION ${label}: starting at $(date)"
    echo "████████████████████████████████████████████████████████"
    ( $fn )
    local rc=$?
    if [ $rc -eq 0 ]; then
        section_results+=("PASS:${label}")
        echo "  ✓ Section ${label} PASSED at $(date)"
    else
        section_results+=("FAIL:${label}  (exit ${rc})")
        echo "  ✗ Section ${label} FAILED (exit ${rc}) at $(date) — continuing"
    fi
}

# ── Environment ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "  physics_decay_type_probe — Latent-State Probing Pipeline"
echo "  N_PROMPTS=${N_PROMPTS}  TOP_K_EXTRACT=${TOP_K_EXTRACT}"
echo "  Graph: ${GRAPH_BEHAVIOUR}/${GRAPH_SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
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

echo ""
python --version
nvidia-smi | head -12 || true

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="

PROBE_PROMPTS="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
GRAPH_PATH="data/results/attribution_graphs/${GRAPH_BEHAVIOUR}/attribution_graph_${GRAPH_SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"

if [ ! -f "$PROBE_PROMPTS" ]; then
    echo "  [FATAL] Probe prompts not found: $PROBE_PROMPTS"
    echo "          Run locally: python scripts/generate_physics_decay_type_probe.py"
    echo "          Then rsync to HPC before submitting."
    exit 1
fi

if [ ! -f "$GRAPH_PATH" ]; then
    echo "  [FATAL] Train graph not found: $GRAPH_PATH"
    echo "          Section C (ablation) requires the existing physics_decay_type train graph."
    echo "          Run the main physics_decay_type pipeline first."
    exit 1
fi

N_ACTUAL=$(wc -l < "$PROBE_PROMPTS")
echo "  Probe prompts : $N_ACTUAL (expected ~$N_PROMPTS)"
echo "  Graph         : $GRAPH_PATH — OK"

# Quick corpus sanity check
python3 -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('${PROBE_PROMPTS}')]
concepts = Counter(r.get('physics_concept','?') for r in rows)
levels   = Counter(r.get('level', 'aux') for r in rows)
is_aux   = sum(1 for r in rows if r.get('is_auxiliary'))
groups   = sorted(set(r.get('group_id','?') for r in rows))
print(f'  Concepts : {dict(concepts)}')
print(f'  Levels   : {dict(levels)}  Auxiliary: {is_aux}')
print(f'  Groups   : {len(groups)} unique ({groups[0]} ... {groups[-1]})')
# Verify all required fields present
required = ['prompt', 'correct_answer', 'incorrect_answer', 'behaviour',
            'physics_concept', 'concept_index', 'template_idx', 'surface_family',
            'level', 'group_id', 'is_auxiliary']
for f in required:
    missing = sum(1 for r in rows if f not in r)
    if missing: print(f'  [WARN] {missing} rows missing field: {f}')
print('  Sanity check OK')
"

echo "  Pre-flight done at $(date)"

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION A: Baseline evaluation (instruct model — checks accuracy per group)
# ═════════════════════════════════════════════════════════════════════════════
section_a() {
    set -e

    echo ""
    echo "---- [A1] Step 02: Baseline on probe corpus (instruct model) ----"
    date
    python -u scripts/02_run_baseline.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --no-viz

    python3 -c "
import csv, json, sys
from collections import defaultdict

rows = list(csv.DictReader(open('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')))
prompts = [json.loads(l) for l in open('${PROBE_PROMPTS}')]
prompt_map = {i: p for i, p in enumerate(prompts)}

sign_acc = sum(1 for r in rows if float(r.get('logprob_diff_normalized', 0)) > 0) / max(len(rows), 1)
print(f'  Overall sign_acc: {sign_acc:.3f} ({int(sign_acc*len(rows))}/{len(rows)})')

# Per-level accuracy
by_level = defaultdict(list)
by_group = defaultdict(list)
for i, r in enumerate(rows):
    if i < len(prompts):
        p = prompts[i]
        lv = p.get('level', 'aux')
        gid = p.get('group_id', '?')
        correct = float(r.get('logprob_diff_normalized', 0)) > 0
        by_level[lv].append(correct)
        by_group[gid].append(correct)

print('  Accuracy by level:')
for lv in sorted(by_level, key=str):
    v = by_level[lv]
    print(f'    Level {lv}: {sum(v)}/{len(v)} = {sum(v)/len(v):.1%}')

print('  Accuracy by group (any < 80% flagged):')
for gid in sorted(by_group):
    v = by_group[gid]
    acc = sum(v)/len(v)
    flag = '  *** LOW ***' if acc < 0.80 else ''
    print(f'    {gid}: {sum(v)}/{len(v)} = {acc:.1%}{flag}')

if sign_acc < $BASELINE_ABORT:
    print(f'  GATE: HARD_FAIL (sign_acc {sign_acc:.3f} < $BASELINE_ABORT)')
    sys.exit(2)
elif sign_acc < 0.90:
    print(f'  GATE: WARN_PASS (sign_acc {sign_acc:.3f} — some groups may be unreliable)')
else:
    print(f'  GATE: PASS')
"
    echo ""
    echo "  Section A output: data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv"
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION B: Feature extraction (base model — PRIMARY DATA for SE/IC analysis)
# ═════════════════════════════════════════════════════════════════════════════
section_b() {
    set -e

    echo ""
    echo "---- [B1] Step 04: Transcoder feature extraction on probe corpus ----"
    echo "  This produces the activation data used for all SE/IC convergence tests."
    date
    python -u scripts/04_extract_transcoder_features.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --model_size 4b \
        --layers $LAYERS \
        --top_k "$TOP_K_EXTRACT" \
        --context_tokens 5

    echo ""
    echo "  Verifying output..."
    # Script 04 saves to data/results/transcoder_features/layer_{n}/ (shared dirs).
    # The behaviour-specific position_map is the reliable indicator of a successful run.
    python3 -c "
from pathlib import Path
import json

# Check for position_map — saved with behaviour prefix, confirms this run completed.
posmap = Path('data/results/transcoder_features/${BEHAVIOUR}_${SPLIT}_position_map.json')
layer_dir = Path('data/results/transcoder_features/layer_10')
if not posmap.exists():
    print(f'  [ERROR] Position map not found: {posmap}')
    exit(1)

pm = json.load(open(posmap))
n_prompts = len(set(e['prompt_idx'] for e in pm)) if isinstance(pm, list) else len(pm)
print(f'  Position map: {posmap.name}  ({n_prompts} prompts)')

# Count feature files in layer dirs
n_layers = sum(1 for l in range(10, 26) if (layer_dir.parent / f'layer_{l}').exists())
print(f'  Layer directories present: {n_layers}/16')

# Check layer_10 has probe files (top_k_indices file named with behaviour)
probe_files = list(layer_dir.glob('*probe*')) if layer_dir.exists() else []
print(f'  Layer_10 probe files: {len(probe_files)}')
if not probe_files:
    print('  [NOTE] Layer files not behaviour-prefixed — features are in shared layer_N/ dirs.')
    print('         rsync: data/results/transcoder_features/ from CSD3 for local analysis.')
print('  Verification OK')
"
    echo ""
    echo "  Section B output: data/results/transcoder_features/layer_10 ... layer_25/"
    echo "  Position map   : data/results/transcoder_features/${BEHAVIOUR}_${SPLIT}_position_map.json"
    echo "  IMPORTANT: rsync data/results/transcoder_features/ to local for script 18."
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION C: Ablation (base model — reuses existing physics_decay_type graph)
# ═════════════════════════════════════════════════════════════════════════════
section_c() {
    set -e

    echo ""
    echo "---- [C] Step 07: Ablation on probe corpus (reusing train graph) ----"
    echo "  Ablating same 40 circuit features from physics_decay_type train analysis."
    echo "  Graph: ${GRAPH_PATH}"
    date

    # Script 07 constructs the graph path as:
    #   attribution_graph_{split}_n{graph_n_prompts}{graph_suffix}.json
    # under data/results/attribution_graphs/{behaviour}/.
    # For the probe behaviour, this path doesn't exist — we symlink the
    # physics_decay_type train graph into the probe behaviour's directory.
    PROBE_GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
    mkdir -p "$PROBE_GRAPH_DIR"
    SYMLINK_TARGET="${PROBE_GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
    ln -sf "$(pwd)/${GRAPH_PATH}" "$SYMLINK_TARGET"
    echo "  Symlinked train graph for probe behaviour: $SYMLINK_TARGET"

    python -u scripts/07_run_interventions.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --experiment ablation \
        --n_prompts "$N_PROMPTS" \
        --graph_n_prompts "$GRAPH_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --layers $LAYERS \
        --top_k "$TOP_K_ABLATE" \
        --per_feature

    echo ""
    echo "  Section C output:"
    echo "    data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv"

    # Quick ablation summary
    # Script 07 names the file without _{split} suffix for probe behaviour.
    python3 -c "
import pandas as pd
from pathlib import Path
# Try both naming conventions (with and without split suffix)
candidates = [
    Path('data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv'),
    Path('data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}.csv'),
]
p = next((c for c in candidates if c.exists()), None)
if p is None:
    print('  Output CSV not found (tried:', [str(c) for c in candidates], ')'); exit()
df = pd.read_csv(p)
print(f'  Ablation rows: {len(df)}')
print(f'  Unique features: {df[\"feature_id\"].nunique()}')
print(f'  Unique prompts : {df[\"prompt_idx\"].nunique()}')
sfr = df[\"sign_flipped\"].mean()
print(f'  Overall sign_flip_rate: {sfr:.4f}')
# Per-layer sfr
by_layer = df.groupby(\"layer\")[\"sign_flipped\"].mean()
print(f'  SFR by layer: {dict(by_layer.round(3))}')
" || true
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION D: UI preparation
# ═════════════════════════════════════════════════════════════════════════════
section_d() {
    set -e

    echo ""
    echo "---- [D] Step 09: UI prep for probe corpus ----"
    date
    python -u scripts/09_prepare_offline_ui.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --graph_n_prompts "$GRAPH_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --community_method louvain

    echo ""
    UI_RUN=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS} 2>/dev/null | sort | tail -1)
    echo "  UI run: ${UI_RUN:-NOT FOUND}"
    echo ""
    echo "  In the dashboard, surface_family = group_id (L1-A1, L2-BR2, etc.)"
    echo "  The FamilyBreakdown tab will show per-group accuracy and sfr — useful for SE analysis."
}

# ── Run sections ──────────────────────────────────────────────────────────────
run_section "A (baseline)"         section_a
run_section "B (feat. extraction)" section_b
run_section "C (ablation)"         section_c
run_section "D (UI prep)"          section_d

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████"
echo "  FINAL SUMMARY — Job $SLURM_JOB_ID — $(date)"
echo "████████████████████████████████████████████████████████"
n_pass=0; n_fail=0
for result in "${section_results[@]}"; do
    if [[ $result == PASS:* ]]; then echo "  ✓ ${result#PASS:}"; (( n_pass++ ))
    else echo "  ✗ ${result#FAIL:}"; (( n_fail++ )); fi
done
echo ""
echo "  ${n_pass} section(s) passed, ${n_fail} section(s) failed"
echo ""
echo "Key outputs:"
echo "  Baseline : data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv"
echo "  Features : data/results/transcoder_features/${BEHAVIOUR}/"
echo "  Ablation : data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv"
echo ""
echo "After rsync to local:"
echo "  # Dashboard:"
echo "  python scripts/prepare_decay_dashboard.py --behaviour ${BEHAVIOUR}"
echo ""
echo "  # Latent-state analysis (SE/IC/KW/CP tests):"
echo "  python scripts/18_latent_state_analysis.py \\"
echo "    --behaviour ${BEHAVIOUR} --split ${SPLIT}"
echo ""
echo "NOTE: script 18 does not exist yet — it is the next implementation step."
echo "      It loads features from Section B and runs all mechanistic tests."
echo ""
[ "$n_fail" -gt 0 ] && exit 1 || exit 0
