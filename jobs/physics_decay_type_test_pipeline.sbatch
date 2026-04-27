#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=phys_dt_test
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=logs/phys_dt_test-%j.out
#SBATCH --error=logs/phys_dt_test-%j.out
#
# physics_decay_type — test split pipeline (steps 02, 04, 07, 09)
#
# Runs the test split (36 prompts) through the existing circuit.
# No new graph building or circuit extraction — uses the TRAIN graph
# to evaluate test-split behaviour.
#
# Steps:
#   02: Baseline accuracy on test split (instruct model)
#   04: Feature extraction on test split (base model)
#   07: Ablation on test split (using train graph features)
#   09: UI prep for test split
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs
#   # Generate test prompts locally, then rsync:
#   python scripts/generate_physics_decay_type_test.py
#   wc -l data/prompts/physics_decay_type_test.jsonl   # expect 36
#   # Ensure train graph exists:
#   ls data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static.json
#   sbatch jobs/physics_decay_type_test_pipeline.sbatch

set -uo pipefail

BEHAVIOUR=physics_decay_type
SPLIT=test
N_PROMPTS=36
TRAIN_GRAPH_N=108
GRAPH_SUFFIX=_roleaware_static
LAYERS="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
TOP_K_EXTRACT=50
TOP_K_ABLATE=20
BASELINE_ABORT=0.70

cd /rds/user/iv294/hpc-work/thesis/project

echo "========================================================"
echo "  physics_decay_type — Test Split Pipeline"
echo "  N_PROMPTS=${N_PROMPTS}  GRAPH from train n${TRAIN_GRAPH_N}${GRAPH_SUFFIX}"
echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"

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
nvidia-smi || true
echo ""

# Pre-flight
PROMPTS_FILE="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "FATAL: $PROMPTS_FILE not found."
    echo "       Run locally: python scripts/generate_physics_decay_type_test.py && rsync"
    exit 1
fi
N_LINES=$(wc -l < "$PROMPTS_FILE")
echo "Pre-flight: $PROMPTS_FILE — ${N_LINES} prompts"
[ "$N_LINES" -lt 30 ] && echo "FATAL: Expected >= 30 prompts" && exit 1

TRAIN_GRAPH="data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_train_n${TRAIN_GRAPH_N}${GRAPH_SUFFIX}.json"
[ ! -f "$TRAIN_GRAPH" ] && echo "FATAL: Train graph not found: $TRAIN_GRAPH" && exit 1
echo "Train graph: $TRAIN_GRAPH — OK"
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 02: Baseline on test split
# ════════════════════════════════════════════════════════════════════════════
echo "==== [A] Step 02: Baseline (test split) ===="
date
python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --no-viz

# Gate check
python3 -c "
import csv, sys
from pathlib import Path
csv_path = Path('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')
if not csv_path.exists():
    print('ERROR: baseline CSV not found'); sys.exit(1)
rows = list(csv.DictReader(csv_path.open()))
sign_acc = sum(1 for r in rows if float(r.get('logprob_diff_normalized', 0)) > 0) / len(rows)
print(f'Test baseline sign_acc: {sign_acc:.3f}  ({int(sign_acc*len(rows))}/{len(rows)})')
if 'surface_family' in rows[0]:
    from collections import defaultdict
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r.get('surface_family','?')].append(float(r.get('logprob_diff_normalized',0))>0)
    for fam in sorted(by_fam):
        vals = by_fam[fam]
        print(f'  {fam}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}')
if sign_acc < $BASELINE_ABORT:
    print(f'GATE: HARD_FAIL'); sys.exit(2)
elif sign_acc < 0.85:
    print(f'GATE: WARN_PASS')
else:
    print(f'GATE: PASS')
"
GATE_EXIT=$?
[ $GATE_EXIT -eq 2 ] && echo "FATAL: baseline hard fail" && exit 1
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 04: Feature extraction on test split
# ════════════════════════════════════════════════════════════════════════════
echo "==== [B] Step 04: Feature extraction (test, top_k=${TOP_K_EXTRACT}) ===="
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
#  STEP 07: Ablation on test split (using train graph features)
# ════════════════════════════════════════════════════════════════════════════
echo "==== [C] Step 07: Ablation (test split, train graph n${TRAIN_GRAPH_N}${GRAPH_SUFFIX}) ===="
date
python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --experiment ablation \
    --n_prompts "$N_PROMPTS" \
    --graph_n_prompts "$TRAIN_GRAPH_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --layers $LAYERS \
    --top_k "$TOP_K_ABLATE" \
    --per_feature
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  STEP 09: UI prep for test split
# ════════════════════════════════════════════════════════════════════════════
echo "==== [D] Step 09: UI prep (test split) ===="
date
python -u scripts/09_prepare_offline_ui.py \
    --behaviour "$BEHAVIOUR" \
    --split "$SPLIT" \
    --graph_n_prompts "$TRAIN_GRAPH_N" \
    --graph_suffix "$GRAPH_SUFFIX" \
    --community_method louvain
echo ""

echo "========================================================"
echo "  TEST PIPELINE COMPLETE: $(date)"
echo "  Job: $SLURM_JOB_ID"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  Baseline  : data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv"
echo "  Ablation  : data/results/interventions/${BEHAVIOUR}/intervention_ablation_${BEHAVIOUR}_${SPLIT}.csv"
echo "  UI run    : data/ui_offline/*_${BEHAVIOUR}_${SPLIT}_n${N_PROMPTS}/"
echo ""
echo "To validate:"
echo "  python scripts/validate_run.py --behaviour ${BEHAVIOUR} (update KNOWN for test split)"
