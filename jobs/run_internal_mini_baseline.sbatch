#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=int-mini-bl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/int-mini-bl-%j.out
#SBATCH --error=logs/int-mini-bl-%j.out

# ── Baseline screening for physics_internal_candidate_selection_mini ──────────
#
# This experiment uses NO explicit candidate list — model must retrieve
# the correct particle from implicit physics context.
#
# Sections:
#   A. Standard logprob baseline (script 02) — sign accuracy gate ≥ 0.80
#   B. Multi-candidate evaluation (script 39) — hard accuracy + competitor logprobs
#
# Multi-token note: positron (2 tok) and muon (2 tok) — script 39 skips these
# gracefully (18/122 prompts = 14.8%). Sign accuracy reported on all 122;
# hard accuracy on ~104 single-token prompts.
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_internal_candidate_selection_mini_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/
#
# sbatch jobs/run_internal_mini_baseline.sbatch

set -euo pipefail

BEHAVIOUR=physics_internal_candidate_selection_mini
SPLIT=train

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
echo "  physics_internal_candidate_selection_mini — BASELINE"
echo "  Split: ${SPLIT}  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results data/results/candidate_set_large

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="
TRAIN_JSONL="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "  [ERROR] Prompt file not found: $TRAIN_JSONL"
    exit 1
fi
N_PROMPTS=$(wc -l < "$TRAIN_JSONL")
echo "  Prompts: $N_PROMPTS  ($TRAIN_JSONL)"

# Spot-check: confirm explicit_candidate_set=false in all records
python3 -c "
import json
with open('$TRAIN_JSONL') as f:
    rows = [json.loads(l) for l in f]
ecs = [r.get('explicit_candidate_set') for r in rows]
assert all(v is False for v in ecs), f'Found non-False explicit_candidate_set: {set(ecs)}'
mt = sum(r.get('multi_token_answer', False) for r in rows)
print(f'  explicit_candidate_set=false: ALL {len(rows)} prompts OK')
print(f'  multi_token_answer=true: {mt}/{len(rows)} ({mt/len(rows):.1%}) — will be skipped by script 39 vocab check')
fps = {}
for r in rows:
    fps[r[\"filter_property\"]] = fps.get(r[\"filter_property\"], 0) + 1
for k,v in sorted(fps.items()):
    print(f'    {k}: {v}')
"

# ── Section A: Standard baseline ─────────────────────────────────────────────
echo ""
echo "████  SECTION A: STANDARD LOGPROB BASELINE  $(date)"

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "Baseline exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Baseline failed"; exit 1; }

python3 -c "
import pandas as pd, sys
bl = pd.read_csv('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')
acc = bl['sign_correct'].mean()
print(f'Sign accuracy: {acc:.3f} ({int(bl.sign_correct.sum())}/{len(bl)})')
if acc < 0.70:
    print('[FATAL] Accuracy < 0.70 — internal candidate retrieval failing. Stopping.')
    sys.exit(1)
elif acc < 0.80:
    print('[WARN] Accuracy 0.70-0.80 — weak but proceeding for diagnostic.')
elif acc >= 0.90:
    print('[PASS] Accuracy >= 0.90 — strong internal candidate retrieval.')
else:
    print('[PASS] Accuracy 0.80-0.90 — acceptable.')

# Per filter_property
if 'filter_property' in bl.columns:
    print()
    print('  Per filter_property:')
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        mt_flag = 'MULTI-TOK' if fp in ('antielectron','heavy_negative_lepton') else ''
        print(f'    {fp}: {(sv>0).mean():.3f} (n={len(sub)}) {mt_flag}')

# Per wording_family
if 'wording_family' in bl.columns:
    print()
    print('  Per wording_family:')
    for wf, sub in bl.groupby('wording_family'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'    {wf}: {(sv>0).mean():.3f} (n={len(sub)})')

# Mean normalized logprob diff
nd = bl['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
print(f'  Mean ND: {nd.mean():.3f}  Median: {nd.median():.3f}  Std: {nd.std():.3f}')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Baseline gate failed"; exit 1; }

# ── Section B: Multi-candidate evaluation ─────────────────────────────────────
echo ""
echo "████  SECTION B: MULTI-CANDIDATE EVALUATION  $(date)"

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Multi-candidate exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Multi-candidate evaluation failed — continuing"; }

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
ls -lh data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/ 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "========================================================"
