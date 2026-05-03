#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=e1sel
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/e1sel-%j.out
#SBATCH --error=logs/e1sel-%j.out

# ── physics_e1_selection — full first-pass pipeline ───────────────────────────
# Runs: baseline → feature extraction → attribution graph → interventions → UI prep
#
# Behaviour: physics_e1_selection
#   Task:    "Is this E1 transition allowed or forbidden?" (Δl=±1 rule)
#   Answers: ' allowed' (token 5420) / ' forbidden' (token 36813)
#   Train:   84 prompts  |  Test: 16 prompts
#
# Input:
#   data/prompts/physics_e1_selection_train.jsonl
#   data/prompts/physics_e1_selection_test.jsonl
#
# Key outputs:
#   data/results/baseline_physics_e1_selection_train.csv
#   data/results/interventions/physics_e1_selection/
#   data/results/physics_e1_selection/analysis_report.md
#   data/ui_offline/<timestamp>_physics_e1_selection_train_n84/
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_e1_selection_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/physics_e1_selection/ data/results/physics_e1_selection/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_e1_selection/ data/results/interventions/physics_e1_selection/
#
# sbatch jobs/run_physics_e1_selection_pipeline.sbatch

set -euo pipefail

BEHAVIOUR=physics_e1_selection
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
echo "  physics_e1_selection pipeline"
echo "  Split: ${SPLIT}  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="
TRAIN_JSONL="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
[ -f "$TRAIN_JSONL" ] && echo "  Train JSONL: $(wc -l < $TRAIN_JSONL) prompts — OK" \
  || { echo "  [FATAL] $TRAIN_JSONL not found. Run: python scripts/30_generate_physics_e1_selection_prompts.py"; exit 1; }

mkdir -p logs data/results/physics_e1_selection

# ── Section A: Baseline ───────────────────────────────────────────────────────
echo ""
echo "████  SECTION A: BASELINE  $(date)"

python -u scripts/02_run_baseline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "Baseline exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Baseline failed"; exit 1; }

# Quick gate check
python3 -c "
import pandas as pd, sys
bl = pd.read_csv('data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv')
acc = bl['sign_correct'].mean()
print(f'Baseline sign accuracy: {acc:.3f} ({bl.sign_correct.sum()}/{len(bl)})')
if acc < 0.70:
    print('[WARN] Accuracy < 0.70 — behaviour may not be reliable. Proceeding anyway.')
elif acc >= 0.85:
    print('[PASS] Accuracy >= 0.85 — proceed.')
else:
    print('[WARN] Accuracy 0.70-0.85 — acceptable but review family breakdown.')

# Check by family
if 'wording_family' in bl.columns:
    for fam, sub in bl.groupby('wording_family'):
        print(f'  {fam}: {sub.sign_correct.mean():.3f}  (n={len(sub)})')
" || true

# ── Section B: Feature extraction ────────────────────────────────────────────
echo ""
echo "████  SECTION B: FEATURE EXTRACTION  $(date)"

python -u scripts/04_extract_transcoder_features.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Feature extraction exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Feature extraction failed"; exit 1; }

# ── Section C: Attribution graph ─────────────────────────────────────────────
echo ""
echo "████  SECTION C: ATTRIBUTION GRAPH  $(date)"

python -u scripts/06_build_attribution_graph.py \
    --behaviour         "$BEHAVIOUR" \
    --split             "$SPLIT" \
    --activation_weighted \
    --top_k_per_layer   3

RC=$?
echo "Graph build exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Graph build failed (may be OK if no features pass threshold)"; }

# ── Section D: Interventions ─────────────────────────────────────────────────
echo ""
echo "████  SECTION D: INTERVENTIONS  $(date)"

python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Interventions exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Interventions failed"; exit 1; }

# ── Section E: UI prep ────────────────────────────────────────────────────────
echo ""
echo "████  SECTION E: UI PREP  $(date)"

python -u scripts/09_prep_ui_offline.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "UI prep exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] UI prep failed — check manually"; }

# ── Section F: Behaviour-specific analysis ────────────────────────────────────
echo ""
echo "████  SECTION F: BEHAVIOUR-SPECIFIC ANALYSIS  $(date)"

python -u scripts/31_analyze_physics_e1_selection.py \
    --split "$SPLIT"

RC=$?
echo "Analysis exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Analysis script failed"; }

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
ls -lh data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv 2>/dev/null || true
ls -lh data/results/physics_e1_selection/analysis_report.md 2>/dev/null || true
ls -lh data/results/interventions/${BEHAVIOUR}/ 2>/dev/null || true
echo ""
echo "  Next steps (local Mac):"
echo "    rsync data: rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv data/results/"
echo "    rsync interventions: rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "    rsync analysis: rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/${BEHAVIOUR}/ data/results/${BEHAVIOUR}/"
echo "    local analysis: python scripts/31_analyze_physics_e1_selection.py --split ${SPLIT}"
echo "========================================================"
