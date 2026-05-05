#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-v2-pipe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cand-v2-pipe-%j.out
#SBATCH --error=logs/cand-v2-pipe-%j.out

# ── Full mechanistic pipeline for physics_particle_candidate_selection_v2 ─────
#
# Prerequisite: run_candidate_v2_baseline.sbatch passed with sign_acc=100%.
#
# Fixes vs v1 pipeline:
#   1. Script 06 uses --graph_node_mode role_aware (gradient×activation edges,
#      NOT the deprecated beta-proxy aggregate_graphs_per_prompt_union path)
#   2. Script 06 uses --output_suffix _roleaware (matches script 07 default)
#   3. Script 06 uses --n_prompts 100 (was 20 in v1 — too sparse for stable graph)
#   4. Script 06 uses --vw_threshold 0.05 (enables VW edges; None = star only)
#   5. Script 07 has NO --device flag (it auto-detects CUDA)
#   6. Script 09 uses --graph_suffix _roleaware and --graph_n_prompts 100
#
# Multi-token note: positron (2-tok) and muon (2-tok) are correct answers for
# antimatter/lepton_muon prompts (~67/224). Script 07 skips these gracefully
# via try/except ValueError. ~157 single-token prompts processed normally.
#
# After job (on local Mac):
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_particle_candidate_selection_v2/ \
#     data/results/interventions/physics_particle_candidate_selection_v2/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/physics_particle_candidate_selection_v2/ \
#     data/results/attribution_graphs/physics_particle_candidate_selection_v2/
#   rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection_v2*/ \
#     data/ui_offline/
#
# sbatch jobs/run_candidate_v2_pipeline.sbatch

set -euo pipefail

BEHAVIOUR=physics_particle_candidate_selection_v2
SPLIT=train
GRAPH_N=100          # prompts used for graph construction
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
echo "  physics_particle_candidate_selection_v2 — FULL PIPELINE"
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

# Gate check
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
    print('[WARN] Accuracy 0.55-0.90 — proceeding with caution.')
if 'filter_property' in bl.columns:
    for fp, sub in bl.groupby('filter_property'):
        sv = sub['logprob_diff_normalized'].replace([float('inf'),float('-inf')],float('nan')).dropna()
        print(f'  {fp}: {(sv>0).mean():.3f} (n={len(sub)})')
"
RC=$?
[ $RC -ne 0 ] && { echo "[FATAL] Gate check failed"; exit 1; }

# ── Section B: Multi-candidate baseline ──────────────────────────────────────
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
# FIX 1: --graph_node_mode role_aware  → gradient×activation VW edges
# FIX 2: --output_suffix _roleaware    → matches script 07 default graph_suffix
# FIX 3: --n_prompts $GRAPH_N          → was 20 (default); now 100 for stable graph
# FIX 4: --vw_threshold 0.05           → enables VW edges (None = star only)
echo ""
echo "████  SECTION D: ATTRIBUTION GRAPH  $(date)"

python -u scripts/06_build_attribution_graph.py \
    --behaviour         "$BEHAVIOUR" \
    --split             "$SPLIT" \
    --graph_node_mode   role_aware \
    --activation_weighted \
    --vw_threshold      0.05 \
    --top_k_per_layer   3 \
    --n_prompts         "$GRAPH_N" \
    --output_suffix     "$GRAPH_SUFFIX"

RC=$?
echo "Graph build exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Graph build failed (may be OK if few features pass threshold)"; }

# ── Section E: Interventions ─────────────────────────────────────────────────
# Note: positron (2-tok) and muon (2-tok) correct-answer prompts are skipped
# via try/except ValueError. ~157/224 single-token prompts processed normally.
# No --device flag: script 07 auto-detects CUDA.
echo ""
echo "████  SECTION E: INTERVENTIONS  $(date)"

python -u scripts/07_run_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT"

RC=$?
echo "Interventions exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Interventions failed — check log"; }

# ── Section F: UI prep ────────────────────────────────────────────────────────
# FIX 5: --graph_suffix matches --output_suffix from script 06
# FIX 6: --graph_n_prompts matches --n_prompts used in script 06
echo ""
echo "████  SECTION F: UI PREP  $(date)"

python -u scripts/09_prepare_offline_ui.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX"

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
ls -lh data/results/attribution_graphs/${BEHAVIOUR}/attribution_graph_${SPLIT}_n${GRAPH_N}${GRAPH_SUFFIX}.json 2>/dev/null || true
ls -lh data/results/candidate_set_large/candidate_state_results.csv 2>/dev/null || true
ls -lh data/results/candidate_set_large/analysis_report.md 2>/dev/null || true
ls -lh data/results/interventions/${BEHAVIOUR}/ 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/${BEHAVIOUR}/ data/results/attribution_graphs/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection_v2*/ data/ui_offline/"
echo "========================================================"
