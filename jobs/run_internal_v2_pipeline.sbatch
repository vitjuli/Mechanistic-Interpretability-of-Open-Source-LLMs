#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=int-v2-pipe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/int-v2-pipe-%j.out
#SBATCH --error=logs/int-v2-pipe-%j.out

# ── Full mechanistic pipeline: physics_internal_candidate_selection_v2 ─────────
#
# Self-contained: includes baseline gate, feature extraction, graph,
# circuit extraction, interventions, path validation, and UI prep in one job.
# DO NOT run run_internal_v2_baseline.sbatch first — this job replaces it.
#
# Script ordering (all independently verified):
#   A: 02 baseline + gate check          → fatal if gate fails
#   B: 39 multi-candidate eval + 40 analysis  → non-fatal
#   C: 04 transcoder feature extraction   → fatal
#   D: 06 attribution graph              → fatal + file existence check
#   E: 08_causal_edges circuit extraction → fatal + file existence check
#   F: 07 interventions                  → non-fatal (WARN)
#   G: 14 path validation                → non-fatal (WARN)
#   H: 09 offline UI prep                → non-fatal (WARN + "No graph" check)
#
# Key design facts:
#   - GRAPH_N=120 < 447 (first multi-token index) → graph never sees multi-tok prompts
#   - INTERVENTION_N=100 < 447 → interventions never see multi-tok prompts
#   - Script 08 runs its OWN ablation; it is independent of script 07
#   - Script 07 auto-detects CUDA — no --device flag
#   - Script 03 (activation capture) does not exist; script 04 handles internally
#   - Script 08_generate_figures.py NOT run: hardcoded behaviour allowlist excludes v2
#   - Scripts 41/42 not yet implemented; next-step commands printed at end
#
# sbatch jobs/run_internal_v2_pipeline.sbatch

set -euo pipefail

# ── Fixed parameters ──────────────────────────────────────────────────────────
BEHAVIOUR=physics_internal_candidate_selection_v2
SPLIT=train
GRAPH_N=120
INTERVENTION_N=100
GRAPH_SUFFIX=_roleaware
TOP_K_PER_LAYER=5
VW_THRESHOLD=0.05
SINGLE_TOK_BOUNDARY=447   # first multi-token index in this JSONL (verified)

# ── Derived paths ─────────────────────────────────────────────────────────────
TRAIN_JSONL="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
BASELINE_CSV="data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv"
BASELINE_METRICS_SRC="data/results/baseline_metrics_train.json"
BASELINE_METRICS_DST="data/results/baseline_metrics_${BEHAVIOUR}_${SPLIT}.json"
GRAPH_DIR="data/results/attribution_graphs/${BEHAVIOUR}"
GRAPH_JSON="${GRAPH_DIR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json"
CIRCUITS_JSON="data/results/circuits_${BEHAVIOUR}_${SPLIT}${GRAPH_SUFFIX}.json"
INTERV_DIR="data/results/interventions/${BEHAVIOUR}"
PATH_VAL_DIR="data/results/path_validation/${BEHAVIOUR}"
PATH_VAL_JSON="${PATH_VAL_DIR}/path_validation_${BEHAVIOUR}_${SPLIT}.json"
CAND_LARGE_DIR="data/results/candidate_set_large"
CAND_BACKUP_DIR="data/results/candidate_set_large_backup_job${SLURM_JOB_ID}"
CAND_RESULT_DIR="data/results/candidate_set_large_${BEHAVIOUR}"

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  ${BEHAVIOUR} — FULL MECHANISTIC PIPELINE"
echo "  GRAPH_N=${GRAPH_N}  INTERVENTION_N=${INTERVENTION_N}  SUFFIX=${GRAPH_SUFFIX}"
echo "  Job ${SLURM_JOB_ID}  Node $(hostname)  $(date)"
echo "========================================================"
python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results data/results/figures \
    "$GRAPH_DIR" "$INTERV_DIR" "$PATH_VAL_DIR" \
    "$CAND_RESULT_DIR" data/ui_offline

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION 0: PRE-FLIGHT  $(date)"
# ══════════════════════════════════════════════════════════════════════════════

[ ! -f "$TRAIN_JSONL" ] && { echo "[FATAL] Prompt file not found: $TRAIN_JSONL"; exit 1; }
N_PROMPTS=$(wc -l < "$TRAIN_JSONL")
echo "  Prompts: $N_PROMPTS  ($TRAIN_JSONL)"

python3 -c "
import json, sys

BEHAVIOUR          = '${BEHAVIOUR}'
TRAIN_JSONL        = '${TRAIN_JSONL}'
GRAPH_N            = ${GRAPH_N}
INTERVENTION_N     = ${INTERVENTION_N}
EXPECTED_BOUNDARY  = ${SINGLE_TOK_BOUNDARY}

with open(TRAIN_JSONL) as f:
    rows = [json.loads(l) for l in f]

errors = []

# 1. Behaviour present in config (string match — no yaml import needed)
with open('configs/experiment_config.yaml') as cf:
    cfg_text = cf.read()
if BEHAVIOUR not in cfg_text:
    errors.append(f'{BEHAVIOUR!r} not found in experiment_config.yaml')
else:
    print(f'  [OK] Behaviour found in config')

# 2. All prompts have explicit_candidate_set=false
bad = [i for i, r in enumerate(rows) if r.get('explicit_candidate_set') is not False]
if bad:
    errors.append(f'explicit_candidate_set != False at indices {bad[:5]}')
else:
    print(f'  [OK] explicit_candidate_set=False for all {len(rows)} prompts')

# 3. Multi-token prompts are at the end
mt_idx = [i for i, r in enumerate(rows) if r.get('multi_token_answer', False)]
st_idx = [i for i, r in enumerate(rows) if not r.get('multi_token_answer', False)]
print(f'  Single-token: {len(st_idx)}  Multi-token: {len(mt_idx)}')
if mt_idx:
    first_mt = min(mt_idx)
    last_st  = max(st_idx) if st_idx else -1
    if first_mt != EXPECTED_BOUNDARY:
        errors.append(f'first_mt_index={first_mt} != expected {EXPECTED_BOUNDARY}')
    elif first_mt <= last_st:
        errors.append(f'Multi-token at index {first_mt} precedes single-token at {last_st}')
    else:
        print(f'  [OK] Multi-token appendix at index {first_mt} (all single-token before)')
    # 4. GRAPH_N and INTERVENTION_N within single-token range
    if GRAPH_N >= first_mt:
        errors.append(f'GRAPH_N={GRAPH_N} >= first_mt_index={first_mt}')
    else:
        print(f'  [OK] GRAPH_N={GRAPH_N} < first_mt_index={first_mt}')
    if INTERVENTION_N >= first_mt:
        errors.append(f'INTERVENTION_N={INTERVENTION_N} >= first_mt_index={first_mt}')
    else:
        print(f'  [OK] INTERVENTION_N={INTERVENTION_N} < first_mt_index={first_mt}')

if errors:
    for e in errors:
        print(f'  [FATAL] {e}')
    sys.exit(1)

print('  [OK] All pre-flight checks passed.')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Pre-flight failed"; exit 1; }

# GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || { echo "[FATAL] No GPU visible to this job"; exit 1; }
echo "  [OK] GPU visible"

# Protect existing candidate_set_large (may contain v3 outputs)
if [ -d "$CAND_LARGE_DIR" ] && [ "$(ls -A "$CAND_LARGE_DIR" 2>/dev/null)" ]; then
    cp -r "$CAND_LARGE_DIR" "$CAND_BACKUP_DIR"
    echo "  [OK] Backed up existing candidate_set_large/ → $CAND_BACKUP_DIR"
fi

echo "  Pre-flight complete  $(date)"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION A: BASELINE + GATE  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# Runs on all 487 prompts (including multi-token appendix). Gate enforced on
# single-token filters only. Script 02 uses CPU (no --device flag).

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

echo "Baseline exit code: $?  ($(date))"

# Namespace the baseline_metrics file before next run overwrites it
cp "$BASELINE_METRICS_SRC" "$BASELINE_METRICS_DST" 2>/dev/null \
    && echo "  Metrics saved → $BASELINE_METRICS_DST" || true

# Gate check — exits with code 1 if any threshold is missed
python3 -c "
import pandas as pd, sys

MULTI_TOK_FILTERS = {'antielectron', 'heavy_negative_lepton'}
FILTER_THRESHOLDS = {
    'negative_charge_atomic':     0.90,
    'positive_charge_atomic':     0.90,
    'neutral_charge_atomic':      0.90,
    'nuclear_stability_provider': 0.90,
    'em_quantum':                 0.90,
    'massless_em_mediator':       0.85,
}

bl   = pd.read_csv('${BASELINE_CSV}')
nd   = bl['logprob_diff_normalized'].replace([float('inf'), float('-inf')], float('nan'))
has_mt = 'multi_token_answer' in bl.columns
has_fp = 'filter_property'    in bl.columns
has_wf = 'wording_family'     in bl.columns
errors = []

# 1. Overall sign accuracy
sign_acc = bl['sign_correct'].mean()
print(f'Sign accuracy (all {len(bl)}): {sign_acc:.3f}  ({int(bl.sign_correct.sum())}/{len(bl)})')
if sign_acc < 0.93:
    errors.append(f'sign_accuracy={sign_acc:.3f} < 0.93')

# 2. Hard accuracy (ND > 0.5)
hard_acc = (nd > 0.5).mean()
print(f'Hard accuracy (ND>0.5):       {hard_acc:.3f}')
if hard_acc < 0.88:
    errors.append(f'hard_accuracy={hard_acc:.3f} < 0.88')

# 3. Mean ND
mean_nd = nd.dropna().mean()
print(f'Mean ND: {mean_nd:.3f}  Median: {nd.dropna().median():.3f}')
if mean_nd < 3.8:
    errors.append(f'mean_ND={mean_nd:.3f} < 3.8')

# 4. Per-filter (single-token main filters only)
if has_fp:
    st = bl[~bl['multi_token_answer'].astype(bool)] if has_mt else bl
    print()
    print('Per filter (single-token):')
    for fp, sub in st.groupby('filter_property'):
        if fp in MULTI_TOK_FILTERS:
            continue
        sv  = sub['logprob_diff_normalized'].replace([float('inf'), float('-inf')], float('nan')).dropna()
        acc = (sv > 0).mean()
        thr = FILTER_THRESHOLDS.get(fp, 0.90)
        tag = 'OK' if acc >= thr else 'FAIL'
        print(f'  {fp:38s}: {acc:.3f}  (n={len(sub)})  [{tag}]  thr={thr}')
        if acc < thr:
            errors.append(f'{fp}={acc:.3f} < {thr}')

# 5. Per wording family (single-token only)
if has_wf:
    st = bl[~bl['multi_token_answer'].astype(bool)] if has_mt else bl
    print()
    print('Per wording family (single-token):')
    for wf, sub in st.groupby('wording_family'):
        sv  = sub['logprob_diff_normalized'].replace([float('inf'), float('-inf')], float('nan')).dropna()
        acc = (sv > 0).mean()
        tag = 'OK' if acc >= 0.85 else 'FAIL'
        print(f'  {wf:30s}: {acc:.3f}  (n={len(sub)})  [{tag}]')
        if acc < 0.85:
            errors.append(f'{wf}={acc:.3f} < 0.85')

# Multi-token appendix: informational only
if has_mt and has_fp:
    mt = bl[bl['multi_token_answer'].astype(bool)]
    if len(mt):
        print(f'Multi-token appendix (n={len(mt)}): sign_acc={mt[\"sign_correct\"].mean():.3f}  [not gated]')

print()
if errors:
    print('[GATE FAIL] Stopping before feature extraction:')
    for e in errors:
        print(f'  FAIL: {e}')
    sys.exit(1)
print('[GATE PASS] All gates cleared. Proceeding.')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Baseline gate failed — job stops here"; exit 1; }

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION B: MULTI-CANDIDATE / INTERNAL EVALUATION  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# Non-fatal: failures here do not block the mechanistic pipeline.

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda \
    && RC39=0 || RC39=$?
echo "Script 39 exit code: $RC39  ($(date))"
[ $RC39 -ne 0 ] && echo "[WARN] Script 39 failed — continuing"

# Copy to namespaced directory so v3 candidate_set_large is not lost
if [ -d "$CAND_LARGE_DIR" ] && [ "$(ls -A "$CAND_LARGE_DIR" 2>/dev/null)" ]; then
    cp -r "$CAND_LARGE_DIR"/. "$CAND_RESULT_DIR/"
    echo "  Candidate outputs → $CAND_RESULT_DIR/"
fi

python -u scripts/40_analyze_candidate_state.py \
    && RC40=0 || RC40=$?
echo "Script 40 exit code: $RC40  ($(date))"
[ $RC40 -ne 0 ] && echo "[WARN] Script 40 failed — continuing"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION C: TRANSCODER FEATURE EXTRACTION  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# Runs on all 487 prompts. Script 04 handles model forward passes internally.
# Features for the multi-token appendix are extracted but never fed to the graph.

python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

echo "Feature extraction exit code: $?  ($(date))"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION D: ATTRIBUTION GRAPH  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# --n_prompts 120 reads lines 0-119 of the JSONL — all single-token.
# Output: attribution_graph_train_n120_roleaware.json

python -u scripts/06_build_attribution_graph.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_node_mode role_aware \
    --activation_weighted \
    --vw_threshold    "$VW_THRESHOLD" \
    --top_k_per_layer "$TOP_K_PER_LAYER" \
    --n_prompts       "$GRAPH_N" \
    --output_suffix   "$GRAPH_SUFFIX"

echo "Graph exit code: $?  ($(date))"

# Hard check: graph file must exist before circuit extraction
[ ! -f "$GRAPH_JSON" ] && {
    echo "[FATAL] Graph not found: $GRAPH_JSON"
    echo "Files in $GRAPH_DIR:"
    ls -lh "$GRAPH_DIR/" 2>/dev/null || echo "  (directory empty)"
    exit 1
}
echo "  [OK] Graph: $GRAPH_JSON  ($(du -sh "$GRAPH_JSON" | cut -f1))"

python3 -c "
import json
with open('${GRAPH_JSON}') as f:
    g = json.load(f)
nodes  = len(g.get('nodes', []))
edges  = len(g.get('edges', []))
params = g.get('graph', {}).get('union_params', {})
print(f'  {nodes} nodes, {edges} edges')
print(f'  edge_method: {params.get(\"edge_method\", \"unknown\")}')
print(f'  n_prompts_used: {params.get(\"n_prompts_used\", \"unknown\")}')
if nodes < 5:
    print('  [WARN] Very few graph nodes — interventions may produce sparse results')
" || true

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION E: CIRCUIT EXTRACTION  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# script 08_causal_edges.py runs its OWN ablation (Phases 1-5) to discover
# causal feature→feature edges. It does NOT read script 07 outputs.
# By default runs S1/S1.5/S2 sufficiency tests — add --skip_sufficiency if
# the job is approaching the time limit.
# Output: $CIRCUITS_JSON

python -u scripts/08_causal_edges.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_json      "$GRAPH_JSON" \
    --n_paths         50 \
    --circuits_output "$CIRCUITS_JSON"

echo "Circuit extraction exit code: $?  ($(date))"

# Hard check: circuit file must exist before path validation
[ ! -f "$CIRCUITS_JSON" ] && {
    echo "[FATAL] Circuit JSON not found: $CIRCUITS_JSON"
    echo "circuit files in data/results/:"
    ls -lh data/results/circuits_*.json 2>/dev/null || echo "  (none found)"
    exit 1
}
echo "  [OK] Circuit: $CIRCUITS_JSON  ($(du -sh "$CIRCUITS_JSON" | cut -f1))"

python3 -c "
import json
with open('${CIRCUITS_JSON}') as f:
    c = json.load(f)
feats  = c.get('features', [])
edges  = c.get('edges',    [])
paths  = c.get('paths',    [])
nec    = c.get('necessity', None)
print(f'  {len(feats)} features, {len(edges)} edges, {len(paths)} paths')
if nec is not None:
    print(f'  necessity={nec:.3f}')
if paths:
    p0 = paths[0]
    print(f'  top path: {\" → \".join(str(x) for x in p0.get(\"path\", []))}')
    print(f'  top path score: {p0.get(\"score\", \"N/A\")}')
" || echo "  [WARN] Could not summarise circuit JSON"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION F: INTERVENTIONS  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# --n_prompts 100 reads lines 0-99 — all single-token.
# --graph_n_prompts 120 tells script 07 where to find the graph (n120_roleaware).
# No --device flag: script 07 auto-detects CUDA.
# STRICT mode by default: layers with no graph features are skipped.

python -u scripts/07_run_interventions.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX" \
    --n_prompts       "$INTERVENTION_N" \
    && RC07=0 || RC07=$?
echo "Interventions exit code: $RC07  ($(date))"
[ $RC07 -ne 0 ] && echo "[WARN] Interventions completed with non-zero exit — check log for skipped layers"

echo "  Intervention files:"
ls -lh "$INTERV_DIR"/ 2>/dev/null | head -8 || echo "  (no outputs found)"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION G: PATH-LEVEL CAUSAL VALIDATION  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# Uses circuit from Section E. Runs its own ablation hooks (independent of 07).

python -u scripts/14_path_validation.py \
    --behaviour     "$BEHAVIOUR" \
    --split         "$SPLIT" \
    --circuits_json "$CIRCUITS_JSON" \
    --n_paths       10 \
    --epsilon       0.01 \
    --max_prompts   50 \
    && RC14=0 || RC14=$?
echo "Path validation exit code: $RC14  ($(date))"
[ $RC14 -ne 0 ] && echo "[WARN] Path validation failed — continuing"

if [ -f "$PATH_VAL_JSON" ]; then
    python3 -c "
import json
with open('${PATH_VAL_JSON}') as f:
    pv = json.load(f)
summary = pv.get('summary', {})
paths   = pv.get('paths',   [])
confirmed = sum(1 for p in paths if p.get('confirmed', False))
print(f'  {len(paths)} paths validated, {confirmed} confirmed')
print(f'  propagation_consistency: {summary.get(\"propagation_consistency\", \"N/A\")}')
print(f'  mean_delta_logit:        {summary.get(\"mean_delta_logit\", \"N/A\")}')
" || true
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████  SECTION H: OFFLINE UI PREPARATION  $(date)"
# ══════════════════════════════════════════════════════════════════════════════
# --graph_n_prompts must equal script 06 --n_prompts (both = GRAPH_N = 120).
# --graph_suffix must equal script 06 --output_suffix (both = _roleaware).

python -u scripts/09_prepare_offline_ui.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX" \
    > /tmp/int_v2_ui_out.txt 2>&1 \
    && RC09=0 || RC09=$?
cat /tmp/int_v2_ui_out.txt

if grep -qi "no graph found\|graph not found\|FileNotFoundError\|No such file" \
        /tmp/int_v2_ui_out.txt; then
    echo "[FATAL] UI prep could not locate graph. Expected: $GRAPH_JSON"
    exit 1
fi
[ $RC09 -ne 0 ] && echo "[WARN] UI prep exited non-zero — check output above"

UI_DIR=$(ls -td data/ui_offline/*${BEHAVIOUR}* 2>/dev/null | head -1 || echo "(not found)")
echo "  UI bundle: $UI_DIR"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================"
echo "  DONE — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID} — $(date)"
echo "========================================================"

echo ""
echo "  ── Outputs ──────────────────────────────────────────"
ls -lh "$BASELINE_CSV"        2>/dev/null || echo "  baseline CSV:      (not found)"
ls -lh "$BASELINE_METRICS_DST" 2>/dev/null || echo "  baseline metrics:  (not found)"
ls -lh "$GRAPH_JSON"          2>/dev/null || echo "  graph JSON:        (not found)"
ls -lh "$CIRCUITS_JSON"       2>/dev/null || echo "  circuit JSON:      (not found)"
ls -lh "$PATH_VAL_JSON"       2>/dev/null || echo "  path val JSON:     (not found)"
ls -lh "$INTERV_DIR"/         2>/dev/null | head -4 || echo "  interventions:     (not found)"
echo "  UI: $UI_DIR"

echo ""
echo "  ── Rsync to local Mac ───────────────────────────────"
BASE="iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
echo "  rsync -av ${BASE}/${BASELINE_CSV} data/results/"
echo "  rsync -av ${BASE}/${BASELINE_METRICS_DST} data/results/"
echo "  rsync -av ${BASE}/${CAND_RESULT_DIR}/ data/results/candidate_set_large_${BEHAVIOUR}/"
echo "  rsync -av ${BASE}/${GRAPH_DIR}/ data/results/attribution_graphs/${BEHAVIOUR}/"
echo "  rsync -av ${BASE}/${CIRCUITS_JSON} data/results/"
echo "  rsync -av ${BASE}/${INTERV_DIR}/ data/results/interventions/${BEHAVIOUR}/"
echo "  rsync -av ${BASE}/${PATH_VAL_DIR}/ data/results/path_validation/${BEHAVIOUR}/"
echo "  # For UI (find exact name from the output above, e.g.):"
echo "  rsync -av ${BASE}/${UI_DIR}/ data/ui_offline/$(basename "$UI_DIR" 2>/dev/null || echo 'INT_V2_UI_DIR')/"

echo ""
echo "  ── Post-pipeline steps (scripts 41/42 not yet written) ──"
echo "  1. Internal candidate-state diagnostics:"
echo "     python scripts/41_internal_candidate_analysis.py \\"
echo "         --behaviour ${BEHAVIOUR} --split ${SPLIT} \\"
echo "         --circuits_json ${CIRCUITS_JSON} \\"
echo "         --baseline_csv ${BASELINE_CSV} \\"
echo "         --candidates electron proton neutron photon"
echo ""
echo "  2. Explicit-vs-internal feature comparison:"
echo "     python scripts/42_explicit_vs_internal_comparison.py \\"
echo "         --internal_behaviour ${BEHAVIOUR} \\"
echo "         --explicit_behaviour physics_particle_candidate_selection_v3 \\"
echo "         --split ${SPLIT} --graph_suffix ${GRAPH_SUFFIX} --layer_report"
echo "========================================================"
