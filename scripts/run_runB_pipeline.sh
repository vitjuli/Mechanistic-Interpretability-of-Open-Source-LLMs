#!/bin/bash
# run_runB_pipeline.sh
#
# Local runner for the Run B downstream pipeline (scripts 19-29).
# Works for both V1 (physics_decay_type_probe) and V2 (physics_decay_type_probe_v2).
#
# Usage:
#   bash scripts/run_runB_pipeline.sh [--start STEP] [--behaviour BEHAVIOUR]
# where STEP is 19, 22, 23, 26, 272, 27, 28, 29
# and   BEHAVIOUR defaults to physics_decay_type_probe_v2

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
# UI run stays V1: graph.json (features/communities) is behaviour-independent
UI_RUN="data/ui_offline/20260430-152526_physics_decay_type_probe_train_n108"

# ── Parse args ─────────────────────────────────────────────────────────────────
START_STEP=19
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)      START_STEP=$2; shift 2 ;;
        --behaviour)  BEHAVIOUR=$2;  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ABL_CSV="data/results/interventions/${BEHAVIOUR}/runB/intervention_ablation_${BEHAVIOUR}.csv"

# Analysis dirs namespaced by behaviour
RUNB_BASE="data/analysis/${BEHAVIOUR}/runB"
GROUPING_DIR="${RUNB_BASE}/grouping"
CLUSTERING_DIR="${RUNB_BASE}/clustering"
CS_DIR="${RUNB_BASE}/cluster_semantics"
CJ_DIR="${RUNB_BASE}/cluster_joint_ablation"

# Parse --start flag
START_STEP=19
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_STEP=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================================"
echo "  Run B downstream pipeline"
echo "  Behaviour: $BEHAVIOUR"
echo "  Start step: $START_STEP"
echo "========================================================"

# Create output dirs
mkdir -p "$GROUPING_DIR" "$CLUSTERING_DIR" "$CS_DIR" "$CJ_DIR"

run_step() {
    local step=$1
    local label=$2
    shift 2
    if [ "$step" -ge "$START_STEP" ]; then
        echo ""
        echo "---- Step $step: $label ----"
        date
        python3 -u "$@"
        echo "  Step $step done: $(date)"
    else
        echo "  Skipping step $step ($label)"
    fi
}

# ── Step 19: feature × prompt analysis ────────────────────────────────────────
run_step 19 "feature_prompt_analysis" \
    scripts/19_feature_prompt_analysis.py \
    --behaviour  "$BEHAVIOUR" \
    --split      "$SPLIT" \
    --ui_run     "$UI_RUN" \
    --abl_csv    "$ABL_CSV" \
    --grouping_dir "$GROUPING_DIR"

# ── Step 22: prepare clustering inputs ────────────────────────────────────────
run_step 22 "prepare_clustering_inputs" \
    scripts/22_prepare_clustering_inputs.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR"

# ── Step 23: run clustering benchmark ─────────────────────────────────────────
run_step 23 "run_clustering_benchmark" \
    scripts/23_run_clustering_benchmark.py \
    --clustering_dir "$CLUSTERING_DIR"

# ── Step 26: cluster semantics ────────────────────────────────────────────────
run_step 26 "cluster_semantics" \
    scripts/26_cluster_semantics.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --out_dir        "$CS_DIR"

# ── Step 27: cluster joint ablation ── CSD3/GPU ONLY, run via sbatch ──────────
# Script 27 requires GPU and runs via jobs/run_probe_runB_joint_ablation.sbatch.
# After syncing the output CSV, run --start 272 to continue locally.
# run_step 27 "cluster_joint_ablation" scripts/27_cluster_joint_ablation.py ...

# ── Step 272: analyse joint ablation (local, after CSD3 sync) ─────────────────
# --start 272  ←  use this after syncing joint_ablation CSV from CSD3
run_step 272 "analyse_joint_ablation" \
    scripts/27b_analyse_joint_ablation.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --joint_dir      "$CJ_DIR"

# ── Step 28: enrichment robustness ────────────────────────────────────────────
run_step 28 "enrichment_robustness" \
    scripts/28_enrichment_robustness.py \
    --grouping_dir         "$GROUPING_DIR" \
    --cluster_semantics_dir "$CS_DIR"

# ── Step 29: final cluster validation ─────────────────────────────────────────
run_step 29 "final_cluster_validation" \
    scripts/29_final_cluster_validation.py \
    --grouping_dir           "$GROUPING_DIR" \
    --cluster_semantics_dir  "$CS_DIR" \
    --cluster_joint_dir      "$CJ_DIR" \
    --clustering_dir         "$CLUSTERING_DIR"

echo ""
echo "========================================================"
echo "  Run B pipeline complete: $(date)"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  Grouping:           $GROUPING_DIR/"
echo "  Clustering:         $CLUSTERING_DIR/"
echo "  Cluster semantics:  $CS_DIR/"
echo "  Joint ablation:     $CJ_DIR/"
