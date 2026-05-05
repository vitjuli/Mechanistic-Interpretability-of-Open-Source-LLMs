#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-v2-bl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cand-v2-bl-%j.out
#SBATCH --error=logs/cand-v2-bl-%j.out

# ── Baseline screening for physics_particle_candidate_selection_v2 ────────────
#
# Runs scripts 02 + 39 + 40 only (no feature extraction / graph / interventions).
# Purpose: verify sign_acc ≥ 90% before committing to the full 6-hour pipeline.
#
# Gate:
#   sign_accuracy ≥ 0.90  → PASS, proceed to full pipeline
#   sign_accuracy 0.55-0.90 → WARN, inspect per-filter breakdown
#   sign_accuracy < 0.55  → FAIL, redesign behaviour
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_particle_candidate_selection_v2_train.csv \
#     data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ \
#     data/results/candidate_set_large/
#
# sbatch jobs/run_candidate_v2_baseline.sbatch

set -euo pipefail

BEHAVIOUR=physics_particle_candidate_selection_v2
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
echo "  physics_particle_candidate_selection_v2 — BASELINE SCREENING"
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
    echo "  Run locally: python scripts/38_generate_candidate_state_v2.py"
    echo "  Then rsync to CSD3 before submitting this job."
    exit 1
fi
N_PROMPTS=$(wc -l < "$TRAIN_JSONL")
echo "  Prompts: $N_PROMPTS  ($TRAIN_JSONL)"

# Token audit
echo ""
echo "==== Token audit ===="
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
pool = [' electron',' proton',' neutron',' photon',' positron',' muon']
multi = []
for t in pool:
    n = len(tok.encode(t, add_special_tokens=False))
    status = 'OK (1-tok)' if n==1 else f'MULTI ({n} toks)'
    print(f'  {repr(t)}: {status}')
    if n > 1: multi.append(t)
if multi:
    print(f'  Note: {multi} are multi-token — will be SKIPPED by script 07 if full pipeline runs.')
" || true

# ── Standard baseline (script 02) ────────────────────────────────────────────
echo ""
echo "████  STANDARD BASELINE  $(date)"

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "Baseline exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Baseline failed"; exit 1; }

# Gate check
echo ""
echo "==== Gate check ===="
python3 -c "
import pandas as pd, sys
bl = pd.read_csv('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')
acc  = bl['sign_correct'].mean()
hard = bl['success'].astype(bool).mean() if 'success' in bl.columns else float('nan')
print(f'Sign accuracy:  {acc:.3f} ({int(bl.sign_correct.sum())}/{len(bl)})')
print(f'Hard accuracy:  {hard:.3f}')
if acc < 0.55:
    print('[FATAL] Sign accuracy < 0.55 — near chance. Redesign behaviour.')
    sys.exit(1)
elif acc >= 0.90:
    print('[PASS] Sign accuracy >= 0.90 — READY FOR FULL PIPELINE.')
else:
    print('[WARN] Sign accuracy 0.55-0.90 — check per-filter breakdown below.')
print()
if 'filter_property' in bl.columns:
    print('Per-filter sign accuracy:')
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        flag = '' if (sv>0).mean() >= 0.80 else '  ← WEAK'
        print(f'  {fp}: {(sv>0).mean():.3f} (n={len(sub)}){flag}')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Gate check failed"; exit 1; }

# ── Multi-candidate baseline (script 39) ─────────────────────────────────────
echo ""
echo "████  MULTI-CANDIDATE BASELINE  $(date)"

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Multi-candidate exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Multi-candidate baseline failed — check log"; }

# ── Candidate-state analysis (script 40) ─────────────────────────────────────
echo ""
echo "████  CANDIDATE-STATE ANALYSIS  $(date)"

python -u scripts/40_analyze_candidate_state.py

RC=$?
echo "Analysis exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Analysis failed"; }

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SCREENING DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
ls -lh data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/candidate_state_results.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/analysis_report.md 2>/dev/null || true
echo ""
echo "  If [PASS]: run full pipeline with:"
echo "    sbatch jobs/run_candidate_v2_pipeline.sbatch"
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "========================================================"
