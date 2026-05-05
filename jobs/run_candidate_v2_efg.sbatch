#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cand-v2-efg
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cand-v2-efg-%j.out
#SBATCH --error=logs/cand-v2-efg-%j.out

# ── Sections E + F + G for physics_particle_candidate_selection_v2 ────────────
# Picks up after sections A-D completed in job 28887249.
# Sections A (baseline), B (multi-candidate), C (features), D (graph) are done.
#
# sbatch jobs/run_candidate_v2_efg.sbatch

set -euo pipefail

BEHAVIOUR=physics_particle_candidate_selection_v2
SPLIT=train
GRAPH_N=100
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
echo "  Sections E+F+G — ${BEHAVIOUR}"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"
nvidia-smi | head -6 || true

# ── Section E: Interventions ─────────────────────────────────────────────────
echo ""
echo "████  SECTION E: INTERVENTIONS  $(date)"

python -u scripts/07_run_interventions.py \
    --behaviour       "$BEHAVIOUR" \
    --split           "$SPLIT" \
    --graph_n_prompts "$GRAPH_N" \
    --graph_suffix    "$GRAPH_SUFFIX"

RC=$?
echo "Interventions exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[WARN] Interventions failed — check log"; }

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
ls -lh data/results/interventions/${BEHAVIOUR}/ 2>/dev/null || true
echo ""
echo "  Rsync to local Mac:"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/${BEHAVIOUR}/ data/results/interventions/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/attribution_graphs/${BEHAVIOUR}/ data/results/attribution_graphs/${BEHAVIOUR}/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/candidate_set_large/ data/results/candidate_set_large/"
echo "    rsync -av iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/ui_offline/*candidate_selection_v2*/ data/ui_offline/"
echo "========================================================"
