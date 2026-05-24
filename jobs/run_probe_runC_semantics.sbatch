#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=probe-runC-sem
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-runC-sem-%j.out
#SBATCH --error=logs/probe-runC-sem-%j.out
#
# physics_decay_type_probe_v2 — full semantics pipeline (CPU only)
# Scripts: 19 (grouping) → 22 (clustering inputs) → 23 (benchmark) → 26 (semantics)
#
# Uses:
#   - ablation CSV: interventions/physics_decay_type_probe_v2/runC/
#   - graph/feature metadata: ui_offline/20260426-034658_physics_decay_type_train_n108/
#   - prompts: data/prompts/physics_decay_type_probe_v2_train.jsonl
#   - output: data/analysis/runC/{grouping,clustering,cluster_semantics}/
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull
#   mkdir -p logs

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  probe_v2 runC — Semantics Pipeline"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python: $(which python)"
python --version
echo ""

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
ABL_CSV="data/results/interventions/${BEHAVIOUR}/runC/intervention_ablation_${BEHAVIOUR}.csv"
UI_RUN="data/ui_offline/20260426-034658_physics_decay_type_train_n108"
GROUPING_DIR="data/analysis/runC/grouping"
CLUSTERING_DIR="data/analysis/runC/clustering_full"
SEMANTICS_DIR="data/analysis/runC/cluster_semantics"

[ -f "$ABL_CSV" ]  || { echo "ERROR: Ablation CSV not found: $ABL_CSV";  exit 1; }
[ -d "$UI_RUN" ]   || { echo "ERROR: UI run not found: $UI_RUN";          exit 1; }
mkdir -p "$GROUPING_DIR" "$CLUSTERING_DIR" "$SEMANTICS_DIR"

# ── Step 19: build grouping CSVs ──────────────────────────────────────────────
echo ""
echo "==== [19] Feature-prompt grouping ===="
date
python -u scripts/19_feature_prompt_analysis.py \
    --behaviour    "$BEHAVIOUR" \
    --split        "$SPLIT" \
    --ui_run       "$UI_RUN" \
    --abl_csv      "$ABL_CSV" \
    --grouping_dir "$GROUPING_DIR"
echo "Done: $(date)"

# ── Step 22: build clustering input matrices ───────────────────────────────────
echo ""
echo "==== [22] Clustering inputs ===="
date
python -u scripts/22_prepare_clustering_inputs.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR"
echo "Done: $(date)"

# ── Step 23: clustering benchmark ─────────────────────────────────────────────
echo ""
echo "==== [23] Clustering benchmark ===="
date
python -u scripts/23_run_clustering_benchmark.py \
    --clustering_dir "$CLUSTERING_DIR"
echo "Done: $(date)"

# ── Step 26: cluster semantics ────────────────────────────────────────────────
echo ""
echo "==== [26] Cluster semantics ===="
date
python -u scripts/26_cluster_semantics.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --out_dir        "$SEMANTICS_DIR"
echo "Done: $(date)"

echo ""
echo "========================================================"
echo "  All done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runC/ data/analysis/runC/"
