#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_all
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/phys_dt_all-%j.out
#SBATCH --error=logs/phys_dt_all-%j.out
#
# physics_decay_type — all remaining experiments in one job
#
# Sections (run sequentially, failures isolated):
#   A. Test-split pipeline  — steps 02 (instruct), 04 (base), 07 ablation, 09
#   B. Patching             — step 07 patching on train split
#   C. Steering             — step 07 steering on train split
#   D. Path validation      — script 14 on train circuit
#
# If one section fails it is recorded and the remaining sections still run.
# A final summary lists which sections passed / failed.
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   # Rsync test prompts from local:
#   #   rsync data/prompts/physics_decay_type_test.jsonl iv294@login.hpc.cam.ac.uk:...
#   wc -l data/prompts/physics_decay_type_test.jsonl      # expect 36
#   ls data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json
#   ls data/results/causal_edges/physics_decay_type/circuits_physics_decay_type_train.json
#   sbatch jobs/physics_decay_type_all_experiments.sbatch

# ── Parameters ────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type
TRAIN_SPLIT=train
TEST_SPLIT=test
TRAIN_N=108
TEST_N=36
GRAPH_SUFFIX=_roleaware_static
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50
TOP_K_ABLATE=20
STEERING_COEFF=10.0
N_PATHS=10
EPSILON=0.01
BASELINE_ABORT=0.70
CIRCUITS_JSON=data/results/causal_edges/${BEHAVIOUR}/circuits_${BEHAVIOUR}_${TRAIN_SPLIT}.json
# ─────────────────────────────────────────────────────────────────────────────

cd /rds/user/iv294/hpc-work/thesis/project

section_results=()   # accumulates "PASS:A" or "FAIL:A" entries

run_section() {
    local label=$1
    local fn=$2
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  SECTION ${label}: starting at $(date)"
    echo "████████████████████████████████████████████████████████"
    if $fn; then
        section_results+=("PASS:${label}")
        echo "  ✓ Section ${label} PASSED at $(date)"
    else
        local rc=$?
        section_results+=("FAIL:${label}  (exit ${rc})")
        echo "  ✗ Section ${label} FAILED (exit ${rc}) at $(date) — continuing"
    fi
}

# ── Environment (shared) ─────────────────────────────────────────────────────
echo "========================================================"
echo "  physics_decay_type — All Experiments"
echo "  Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================================"

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

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="

TRAIN_GRAPH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${TRAIN_SPLIT}_n${TRAIN_N}${GRAPH_SUFFIX}.json"

PREFLIGHT_OK=true

if [ ! -f "data/prompts/${BEHAVIOUR}_${TEST_SPLIT}.jsonl" ]; then
    echo "  [WARN] Test prompts not found — Section A will fail"
    echo "         Run locally: python scripts/generate_physics_decay_type_test.py"
    PREFLIGHT_OK=false
fi

if [ ! -f "$TRAIN_GRAPH" ]; then
    echo "  [FATAL] Train graph not found: $TRAIN_GRAPH"
    echo "          Sections A/B/C require this — aborting."
    exit 1
fi

if [ ! -f "$CIRCUITS_JSON" ]; then
    echo "  [WARN] Circuits JSON not found — Section D will be skipped"
fi

echo "  Train graph : $TRAIN_GRAPH — OK"
echo "  Pre-flight done at $(date)"

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION A: Test-split pipeline (02 instruct → 04 base → 07 ablation → 09)
# ═════════════════════════════════════════════════════════════════════════════
section_a() {
    set -e

    local SPLIT=$TEST_SPLIT
    local N_PROMPTS=$TEST_N

    echo ""
    echo "---- [A1] Step 02: Baseline on test split (instruct model) ----"
    date
    python -u scripts/02_run_baseline.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --no-viz

    python3 -c "
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open('data/results/baseline_${BEHAVIOUR}_${TEST_SPLIT}.csv')))
sign_acc = sum(1 for r in rows if float(r.get('logprob_diff_normalized',0)) > 0) / len(rows)
print(f'  Test baseline sign_acc: {sign_acc:.3f} ({int(sign_acc*len(rows))}/{len(rows)})')
from collections import defaultdict
by_fam = defaultdict(list)
for r in rows:
    by_fam[r.get('surface_family','?')].append(float(r.get('logprob_diff_normalized',0))>0)
for fam in sorted(by_fam):
    v = by_fam[fam]
    print(f'    {fam}: {sum(v)}/{len(v)} = {sum(v)/len(v):.1%}')
if sign_acc < $BASELINE_ABORT:
    print('  GATE: HARD_FAIL'); sys.exit(2)
elif sign_acc < 0.85:
    print('  GATE: WARN_PASS')
else:
    print('  GATE: PASS')
"

    echo ""
    echo "---- [A2] Step 04: Feature extraction on test split ----"
    date
    python -u scripts/04_extract_transcoder_features.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --model_size 4b \
        --layers $LAYERS \
        --top_k "$TOP_K_EXTRACT" \
        --context_tokens 5

    echo ""
    echo "---- [A3] Step 07: Ablation on test split ----"
    date
    python -u scripts/07_run_interventions.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --experiment ablation \
        --n_prompts "$N_PROMPTS" \
        --graph_n_prompts "$TRAIN_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --layers $LAYERS \
        --top_k "$TOP_K_ABLATE" \
        --per_feature

    echo ""
    echo "---- [A4] Step 09: UI prep for test split ----"
    date
    python -u scripts/09_prepare_offline_ui.py \
        --behaviour "$BEHAVIOUR" \
        --split "$SPLIT" \
        --graph_n_prompts "$TRAIN_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --community_method louvain

    echo ""
    echo "  Section A outputs:"
    echo "    data/results/baseline_${BEHAVIOUR}_${TEST_SPLIT}.csv"
    echo "    data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${TEST_SPLIT}.csv"
    TEST_RUN=$(ls -1d data/ui_offline/*_${BEHAVIOUR}_${TEST_SPLIT}_n${TEST_N} 2>/dev/null | sort | tail -1)
    echo "    UI run: ${TEST_RUN:-NOT FOUND}"
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION B: Patching on train split
# ═════════════════════════════════════════════════════════════════════════════
section_b() {
    set -e

    echo ""
    echo "---- [B] Step 07: Patching (train) ----"
    date
    python -u scripts/07_run_interventions.py \
        --behaviour "$BEHAVIOUR" \
        --split "$TRAIN_SPLIT" \
        --experiment patching \
        --n_prompts "$TRAIN_N" \
        --graph_n_prompts "$TRAIN_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --layers $LAYERS \
        --top_k "$TOP_K_ABLATE" \
        --per_feature

    echo ""
    echo "  Section B output:"
    echo "    data/results/interventions/${BEHAVIOUR}/intervention_patching_${BEHAVIOUR}_${TRAIN_SPLIT}.csv"
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION C: Steering on train split
# ═════════════════════════════════════════════════════════════════════════════
section_c() {
    set -e

    echo ""
    echo "---- [C] Step 07: Steering (train, coeff=${STEERING_COEFF}) ----"
    date
    python -u scripts/07_run_interventions.py \
        --behaviour "$BEHAVIOUR" \
        --split "$TRAIN_SPLIT" \
        --experiment steering \
        --n_prompts "$TRAIN_N" \
        --graph_n_prompts "$TRAIN_N" \
        --graph_suffix "$GRAPH_SUFFIX" \
        --layers $LAYERS \
        --top_k "$TOP_K_ABLATE" \
        --steering_coeff "$STEERING_COEFF" \
        --per_feature

    echo ""
    echo "  Section C output:"
    echo "    data/results/interventions/${BEHAVIOUR}/intervention_steering_${BEHAVIOUR}_${TRAIN_SPLIT}.csv"

    python3 -c "
import json
from pathlib import Path
p = Path('data/results/interventions/${BEHAVIOUR}/intervention_steering_${BEHAVIOUR}_${TRAIN_SPLIT}_summary.json')
if p.exists():
    d = json.load(open(p))
    sfr = d.get('sign_flip_rate', -1)
    print(f'  Steering sign_flip_rate: {sfr:.3f}')
    if sfr < 0.10:
        print('  NOTE: sfr < 0.10 — consider resubmitting with STEERING_COEFF=20.0')
    elif sfr > 0.50:
        print('  NOTE: sfr > 0.50 — consider resubmitting with STEERING_COEFF=5.0')
    else:
        print('  Steering coefficient looks good.')
" || true
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION D: Path validation on train circuit
# ═════════════════════════════════════════════════════════════════════════════
section_d() {
    set -e

    if [ ! -f "$CIRCUITS_JSON" ]; then
        echo "  SKIP: circuits JSON not found at $CIRCUITS_JSON"
        echo "        Run step 08 first, then re-submit this section."
        return 1
    fi

    echo ""
    echo "---- [D] Script 14: Path validation ----"
    date
    python -u scripts/14_path_validation.py \
        --behaviour "$BEHAVIOUR" \
        --split "$TRAIN_SPLIT" \
        --circuits_json "$CIRCUITS_JSON" \
        --n_paths "$N_PATHS" \
        --max_prompts "$TRAIN_N" \
        --epsilon "$EPSILON" \
        --model_size 4b

    echo ""
    python3 -c "
import json
from pathlib import Path
p = Path('data/results/path_validation/${BEHAVIOUR}/path_validation_${BEHAVIOUR}_${TRAIN_SPLIT}.json')
if not p.exists():
    print('  Output not found'); exit()
d = json.load(open(p))
s = d.get('summary', {})
print('  Path Validation Summary:')
print(f'    propagation_consistency : {s.get(\"propagation_consistency\", \"?\"):.4f}')
print(f'    mean_delta_act_B        : {s.get(\"mean_delta_act_B\", \"?\"):.4f}')
print(f'    mean_delta_logit        : {s.get(\"mean_delta_logit\", \"?\"):.4f}')
for i, pr in enumerate(d.get('paths', [])[:3]):
    print(f'    Path {i+1}: {pr.get(\"path\",[])} | consistency={pr.get(\"propagation_consistency\",\"?\"):.3f}')
" || true
}

# ── Run all sections ──────────────────────────────────────────────────────────
run_section "A (test pipeline)"  section_a
run_section "B (patching)"       section_b
run_section "C (steering)"       section_c
run_section "D (path validation)" section_d

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████"
echo "  FINAL SUMMARY — Job $SLURM_JOB_ID — $(date)"
echo "████████████████████████████████████████████████████████"
n_pass=0
n_fail=0
for result in "${section_results[@]}"; do
    if [[ $result == PASS:* ]]; then
        echo "  ✓ ${result#PASS:}"
        (( n_pass++ ))
    else
        echo "  ✗ ${result#FAIL:}"
        (( n_fail++ ))
    fi
done
echo ""
echo "  ${n_pass} section(s) passed, ${n_fail} section(s) failed"
echo ""
echo "Key outputs:"
echo "  Test baseline : data/results/baseline_${BEHAVIOUR}_${TEST_SPLIT}.csv"
echo "  Patching      : data/results/interventions/${BEHAVIOUR}/intervention_patching_${BEHAVIOUR}_${TRAIN_SPLIT}.csv"
echo "  Steering      : data/results/interventions/${BEHAVIOUR}/intervention_steering_${BEHAVIOUR}_${TRAIN_SPLIT}.csv"
echo "  Path val.     : data/results/path_validation/${BEHAVIOUR}/path_validation_${BEHAVIOUR}_${TRAIN_SPLIT}.json"
echo ""
echo "After rsync — re-run UI prep to include patching/steering in train dashboard:"
echo "  python scripts/09_prepare_offline_ui.py \\"
echo "    --behaviour ${BEHAVIOUR} --split ${TRAIN_SPLIT} \\"
echo "    --graph_n_prompts ${TRAIN_N} --graph_suffix ${GRAPH_SUFFIX}"
echo "  python scripts/prepare_decay_dashboard.py"
echo ""
[ "$n_fail" -gt 0 ] && exit 1 || exit 0
