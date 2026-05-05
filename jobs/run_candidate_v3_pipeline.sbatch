#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-v3-pipe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cand-v3-pipe-%j.out
#SBATCH --error=logs/cand-v3-pipe-%j.out

# ── Full pipeline for physics_particle_candidate_selection_v3 (505 prompts) ───
#
# Changes vs v2 pipeline:
#   - Script 07: --n_prompts 50  (was 20; gives ~30 single-token prompts/layer)
#   - Script 06: --top_k_per_layer 5  (was 3; 80 nodes for richer graph)
#   - GRAPH_N=120  (was 100; larger graph sample for 505-prompt dataset)
#
# Multi-token: positron/muon (~30.7%) skipped gracefully by script 07.
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_physics_particle_candidate_selection_v3_train.csv data/results/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_particle_candidate_selection_v3/ data/results/interventions/physics_particle_candidate_selection_v3/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/physics_particle_candidate_selection_v3/ data/results/attribution_graphs/physics_particle_candidate_selection_v3/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection_v3*/ data/ui_offline/
#
# sbatch jobs/run_candidate_v3_pipeline.sbatch

set -euo pipefail

BEHAVIOUR=physics_particle_candidate_selection_v3
SPLIT=train
GRAPH_N=120
GRAPH_SUFFIX=_roleaware

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
echo "  physics_particle_candidate_selection_v3 — FULL PIPELINE"
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
echo "  Graph n_prompts: $GRAPH_N  suffix: $GRAPH_SUFFIX"

# ── Section A: Standard baseline ─────────────────────────────────────────────
echo ""
echo "████  SECTION A: STANDARD BASELINE  $(date)"

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
if acc < 0.55:
    print('[FATAL] Accuracy < 0.55 — stopping.')
    sys.exit(1)
elif acc >= 0.90:
    print('[PASS] Accuracy >= 0.90')
else:
    print('[WARN] Accuracy 0.55-0.90 — proceeding.')
if 'filter_property' in bl.columns:
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'  {fp}: {(sv>0).mean():.3f} (n={len(sub)})')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Gate failed"; exit 1; }

# ── Section B: Multi-candidate baseline ──────────────────────────────────────
echo ""
echo "████  SECTION B: MULTI-CANDIDATE BASELINE  $(date)"

python -u scripts/39_evaluate_candidate_state.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --device    cuda

RC=$?
echo "Multi-candidate exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Multi-candidate failed — continuing"; }

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
# top_k_per_layer=5 (80 nodes, vs 48 in v2) for richer graph with 505 prompts
echo ""
echo "████  SECTION D: ATTRIBUTION GRAPH  $(date)"

python -u scripts/06_build_attribution_graph.py \
    --behaviour         "$BEHAVIOUR" \
    --split             "$SPLIT" \
    --graph_node_mode   role_aware \
    --activation_weighted \
    --vw_threshold      0.05 \
    --top_k_per_layer   5 \
    --n_prompts         "$GRAPH_N" \
    --output_suffix     "$GRAPH_SUFFIX"

RC=$?
echo "Graph build exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Graph build failed"; }

# ── Section E: Interventions ─────────────────────────────────────────────────
# n_prompts=50: ~35 single-token prompts per layer (vs ~12 in v2)
echo ""
echo "████  SECTION E: INTERVENTIONS  $(date)"

python -u scripts/07_run_interventions.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX" \
    --n_prompts       50

RC=$?
echo "Interventions exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Interventions failed"; }

# ── Section F: UI prep ────────────────────────────────────────────────────────
echo ""
echo "████  SECTION F: UI PREP  $(date)"

python -u scripts/09_prepare_offline_ui.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX"

RC=$?
echo "UI prep exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] UI prep failed"; }

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
ls -lh data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json 2>/dev/null || true
ls -lh data/results/interventions/${BEHAVIOUR}/ 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/baseline_${BEHAVIOUR}_${SPLIT}.csv data/results/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/${BEHAVIOUR}/ data/results/attribution_graphs/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection_v3*/ data/ui_offline/"
echo "========================================================"
