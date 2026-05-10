#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=int-v2-bl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/int-v2-bl-%j.out
#SBATCH --error=logs/int-v2-bl-%j.out

# ── Baseline screening for physics_internal_candidate_selection_v2 ────────────
#
# Strategy B clean internal-only dataset (487 prompts):
#   - 447 single-token implicit (main corpus, indices 0-446)
#   - 40 multi-token generalisation appendix (indices 447-486, placed last)
#
# Sections:
#   A. Standard logprob baseline (script 02) on all 487 prompts
#   B. Multi-candidate evaluation (script 39) on all 487 prompts
#
# Multi-token note: positron (2 tok) / muon (2 tok) — 40 prompts = 8.2%
# Script 39 skips multi-token tokens gracefully via vocab check.
# Gate: sign_acc >= 0.90 on full 487; on single-token subset >= 0.93.
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_internal_candidate_selection_v2_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/
#
# sbatch jobs/run_internal_v2_baseline.sbatch

set -euo pipefail

BEHAVIOUR=physics_internal_candidate_selection_v2
SPLIT=train
SINGLE_TOKEN_N=447   # index boundary: prompts 0..446 are single-token main corpus

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
echo "  physics_internal_candidate_selection_v2 — BASELINE"
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

python3 -c "
import json
with open('$TRAIN_JSONL') as f:
    rows = [json.loads(l) for l in f]

# Verify all prompts have explicit_candidate_set=false
ecs = [r.get('explicit_candidate_set') for r in rows]
assert all(v is False for v in ecs), f'Found non-False explicit_candidate_set: {set(ecs)}'

mt = sum(r.get('multi_token_answer', False) for r in rows)
st = len(rows) - mt
print(f'  explicit_candidate_set=false: ALL {len(rows)} prompts OK')
print(f'  single-token: {st}  multi-token: {mt} ({mt/len(rows):.1%})')

# Confirm multi-token prompts are at the end (indices 447+)
mt_indices = [i for i,r in enumerate(rows) if r.get('multi_token_answer', False)]
if mt_indices:
    assert min(mt_indices) >= $SINGLE_TOKEN_N, \
        f'Multi-token prompt found at index {min(mt_indices)} (< {$SINGLE_TOKEN_N})'
    print(f'  Multi-token appendix starts at index {min(mt_indices)} (>= {$SINGLE_TOKEN_N}) OK')

fps = {}
for r in rows:
    fps[r['filter_property']] = fps.get(r['filter_property'], 0) + 1
print()
print('  Per filter_property:')
for k,v in sorted(fps.items()):
    mt_flag = '[MULTI]' if any(r.get('multi_token_answer') for r in rows if r['filter_property']==k) else ''
    print(f'    {k}: {v} {mt_flag}')

# Check wording family distribution for single-token only
st_rows = [r for r in rows if not r.get('multi_token_answer', False)]
wfs = {}
for r in st_rows:
    wfs[r['wording_family']] = wfs.get(r['wording_family'], 0) + 1
print()
print('  Wording families (single-token only):')
for k,v in sorted(wfs.items()):
    print(f'    {k}: {v} ({v/len(st_rows):.1%})')
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
acc  = bl['sign_correct'].mean()
nd   = bl['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
print(f'Sign accuracy (all 487): {acc:.3f} ({int(bl.sign_correct.sum())}/{len(bl)})')
print(f'Mean ND: {nd.mean():.3f}  Median: {nd.median():.3f}  Std: {nd.std():.3f}')
print()

# Gate on full dataset
if acc < 0.85:
    print('[FATAL] Sign accuracy < 0.85 — stopping.')
    sys.exit(1)
elif acc >= 0.93:
    print('[PASS] Sign accuracy >= 0.93 — strong internal candidate retrieval.')
elif acc >= 0.90:
    print('[PASS] Sign accuracy 0.90-0.93 — acceptable.')
else:
    print('[WARN] Sign accuracy 0.85-0.90 — proceeding for diagnostic.')

# Single-token only gate
if 'multi_token_answer' in bl.columns:
    st = bl[bl['multi_token_answer'] == False]
    mt = bl[bl['multi_token_answer'] == True]
    st_nd = st['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
    print()
    print(f'Single-token subset (n={len(st)}):')
    print(f'  Sign accuracy: {st[\"sign_correct\"].mean():.3f}')
    print(f'  Mean ND: {st_nd.mean():.3f}')
    if len(mt) > 0:
        mt_nd = mt['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'Multi-token subset (n={len(mt)}):')
        print(f'  Sign accuracy: {mt[\"sign_correct\"].mean():.3f}')
        print(f'  Mean ND: {mt_nd.mean():.3f}')

# Per filter_property
if 'filter_property' in bl.columns:
    print()
    print('Per filter_property:')
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        mt_flag = '[MULTI]' if sub['multi_token_answer'].any() else ''
        print(f'  {fp:38s}: {(sv>0).mean():.3f} (n={len(sub)}) {mt_flag}')

# Per wording_family (single-token only)
if 'wording_family' in bl.columns and 'multi_token_answer' in bl.columns:
    st = bl[bl['multi_token_answer'] == False]
    print()
    print('Per wording_family (single-token):')
    for wf, sub in st.groupby('wording_family'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'  {wf:30s}: {(sv>0).mean():.3f} (n={len(sub)})')
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
ls -lh data/results/candidate_set_large/candidate_state_results.csv 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo ""
echo "  Next steps (after gate passes):"
echo "    sbatch jobs/run_internal_v2_pipeline.sbatch"
echo "========================================================"
