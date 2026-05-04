#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-pipe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cand-pipe-%j.out
#SBATCH --error=logs/cand-pipe-%j.out

# ── physics_particle_candidate_selection — full pipeline ─────────────────────
# Behaviour: explicit candidate-set selection
#   "The options are electron, proton, and neutron. Which has negative charge? Answer:"
# Split:     train  (311 prompts)
# Answers:   particle names (electron/proton/neutron/photon single-token;
#             positron/muon 2-token — SKIPPED by script 07 with warning)
#
# Pipeline:
#   A. Standard baseline (script 02) → sign_correct gate
#   B. Rich multi-candidate baseline (script 39) → H1 vs H2 analysis
#   C. Feature extraction (script 04)
#   D. Attribution graph (script 06)
#   E. Interventions (script 07)  — multi-token prompts skipped gracefully
#   F. UI prep (script 09)
#   G. Candidate-state analysis (script 40)
#
# Multi-token note: positron (2-tok) and muon (2-tok) appear as correct/incorrect
# answers in ~80/311 prompts. Script 07's ablation/patching/steering loops all have
# try/except ValueError that skip these prompts. ~230 single-token prompts processed.
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ \
#     data/results/candidate_set_large/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_particle_candidate_selection/ \
#     data/results/interventions/physics_particle_candidate_selection/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection*/ \
#     data/ui_offline/
#
# sbatch jobs/run_candidate_state_pipeline.sbatch

set -euo pipefail

BEHAVIOUR=physics_particle_candidate_selection
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
echo "  physics_particle_candidate_selection pipeline"
echo "  Split: ${SPLIT}  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

mkdir -p logs data/results data/results/candidate_set_large

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="
TRAIN_JSONL="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"

# Generate prompts if not present
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "  Generating prompts..."
    python -u scripts/38_generate_candidate_state_large.py
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
    status = 'OK (1-tok)' if n==1 else f'MULTI ({n} toks — will be SKIPPED by script 07)'
    print(f'  {repr(t)}: {status}')
    if n > 1: multi.append(t)
if multi:
    print(f'  Note: {multi} prompts will be skipped in script 07 (try/except ValueError).')
    print(f'  Single-token prompts still processed normally.')
" || true

# ── Section A: Standard baseline ─────────────────────────────────────────────
echo ""
echo "████  SECTION A: STANDARD BASELINE  $(date)"

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "Baseline exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Baseline failed"; exit 1; }

# Gate check
python3 -c "
import pandas as pd, sys
bl = pd.read_csv('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')
acc = bl['sign_correct'].mean()
print(f'Sign accuracy: {acc:.3f} ({int(bl.sign_correct.sum())}/{len(bl)})')
if acc < 0.55:
    print('[FATAL] Accuracy < 0.55 — near chance. Stopping.')
    sys.exit(1)
elif acc >= 0.85:
    print('[PASS] Accuracy >= 0.85')
else:
    print('[WARN] Accuracy 0.55-0.85 — proceeding with caution.')
if 'filter_property' in bl.columns:
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'  {fp}: {(sv>0).mean():.3f} (n={len(sub)})')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Gate check failed"; exit 1; }

# ── Section B: Rich multi-candidate baseline ──────────────────────────────────
echo ""
echo "████  SECTION B: MULTI-CANDIDATE BASELINE  $(date)"

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Multi-candidate baseline exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Multi-candidate baseline failed — continuing"; }

# ── Section C: Feature extraction ────────────────────────────────────────────
echo ""
echo "████  SECTION C: FEATURE EXTRACTION  $(date)"

python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Feature extraction exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Feature extraction failed"; exit 1; }

# ── Section D: Attribution graph ─────────────────────────────────────────────
echo ""
echo "████  SECTION D: ATTRIBUTION GRAPH  $(date)"

python -u scripts/06_build_attribution_graph.py \
    --behaviour         "$BEHAVIOUR" \
    --split             "$SPLIT" \
    --activation_weighted \
    --top_k_per_layer   3

RC=$?
echo "Graph build exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Graph build failed (may be OK if few features pass threshold)"; }

# ── Section E: Interventions ─────────────────────────────────────────────────
# Note: prompts with multi-token answers (positron 2-tok, muon 2-tok) are
# SKIPPED with a warning in all experiment loops.  Single-token prompts
# (~230/311) are processed normally.
echo ""
echo "████  SECTION E: INTERVENTIONS  $(date)"

python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Interventions exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Interventions failed — check log"; }

# ── Section F: UI prep ────────────────────────────────────────────────────────
echo ""
echo "████  SECTION F: UI PREP  $(date)"

python -u scripts/09_prepare_offline_ui.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --graph_n_prompts "$N_PROMPTS"

RC=$?
echo "UI prep exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] UI prep failed — check manually"; }

# ── Section G: Candidate-state analysis ──────────────────────────────────────
echo ""
echo "████  SECTION G: CANDIDATE-STATE ANALYSIS  $(date)"

python -u scripts/40_analyze_candidate_state.py

RC=$?
echo "Analysis exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Analysis failed"; }

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
ls -lh data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/candidate_state_results.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/analysis_report.md 2>/dev/null || true
ls -lh data/results/interventions/${BEHAVIOUR}/ 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection*/ data/ui_offline/"
echo "========================================================"
