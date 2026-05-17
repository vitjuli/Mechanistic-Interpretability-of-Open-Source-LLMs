#!/bin/bash
# run_runC_pipeline.sh
#
# Local downstream pipeline for Run C (top-10-per-layer sign-complete).
# Run AFTER syncing CSD3 outputs from jobs/run_probe_runC_pipeline.sbatch.
#
# Steps:
#   19  feature × prompt analysis
#   22  prepare clustering inputs
#   23  run clustering benchmark
#   26  cluster semantics
#   272 analyse joint ablation (after CSD3 joint-ablation job)
#   28  enrichment robustness
#   29  final cluster validation
#   Compare  runC_comparison_report.py
#
# Usage:
#   bash scripts/run_runC_pipeline.sh [--start STEP]
# STEP: 19, 22, 23, 26, 272, 28, 29, compare
#
# Prereqs (sync from CSD3 first):
#   data/results/interventions/physics_decay_type_probe/runC/intervention_ablation_physics_decay_type_probe.csv
#   data/analysis/runC_top10_sign_complete/runC_manifest.json
#   data/results/attribution_graphs/physics_decay_type/attribution_graph_train_n108_roleaware_static_k10.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BEHAVIOUR=physics_decay_type_probe
SPLIT=train
UI_RUN="data/ui_offline/20260430-152526_physics_decay_type_probe_train_n108"
ABL_CSV="data/results/interventions/${BEHAVIOUR}/runC/intervention_ablation_${BEHAVIOUR}.csv"

RUNC_BASE="data/analysis/runC_top10_sign_complete"
GROUPING_DIR="${RUNC_BASE}/grouping"
CLUSTERING_DIR="${RUNC_BASE}/clustering"
CS_DIR="${RUNC_BASE}/cluster_semantics"
CJ_DIR="${RUNC_BASE}/cluster_joint_ablation"

START_STEP=19
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_STEP=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================================"
echo "  Run C downstream pipeline"
echo "  Start step: $START_STEP"
echo "========================================================"

# Check ablation CSV exists
[ -f "$ABL_CSV" ] || { echo "ERROR: $ABL_CSV not found. Sync from CSD3 first."; exit 1; }

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

run_step 19 "feature_prompt_analysis" \
    scripts/19_feature_prompt_analysis.py \
    --behaviour    "$BEHAVIOUR" \
    --split        "$SPLIT" \
    --ui_run       "$UI_RUN" \
    --abl_csv      "$ABL_CSV" \
    --grouping_dir "$GROUPING_DIR"

run_step 22 "prepare_clustering_inputs" \
    scripts/22_prepare_clustering_inputs.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR"

run_step 23 "run_clustering_benchmark" \
    scripts/23_run_clustering_benchmark.py \
    --clustering_dir "$CLUSTERING_DIR"

run_step 26 "cluster_semantics" \
    scripts/26_cluster_semantics.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --out_dir        "$CS_DIR"

# Step 27 (joint ablation) runs via CSD3:
#   sbatch jobs/run_probe_runC_joint_ablation.sbatch
# After syncing, resume from step 272:
#   bash scripts/run_runC_pipeline.sh --start 272

run_step 272 "analyse_joint_ablation" \
    scripts/27b_analyse_joint_ablation.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --joint_dir      "$CJ_DIR"

run_step 28 "enrichment_robustness" \
    scripts/28_enrichment_robustness.py \
    --grouping_dir          "$GROUPING_DIR" \
    --cluster_semantics_dir "$CS_DIR"

run_step 29 "final_cluster_validation" \
    scripts/29_final_cluster_validation.py \
    --grouping_dir           "$GROUPING_DIR" \
    --cluster_semantics_dir  "$CS_DIR" \
    --cluster_joint_dir      "$CJ_DIR" \
    --clustering_dir         "$CLUSTERING_DIR"

run_step 300 "runC_null_cluster_test" \
    scripts/runC_null_cluster_test.py \
    --runC_base "$RUNC_BASE"

run_step 310 "runC_comparison_report" \
    scripts/runC_comparison_report.py \
    --runB_base   "data/analysis/runB" \
    --runC_base   "$RUNC_BASE" \
    --validation  "data/analysis/runB_validation"

echo ""
echo "========================================================"
echo "  Run C pipeline complete: $(date)"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  Grouping:           $GROUPING_DIR/"
echo "  Clustering:         $CLUSTERING_DIR/"
echo "  Cluster semantics:  $CS_DIR/"
echo "  Joint ablation:     $CJ_DIR/"
echo "  Report:             $RUNC_BASE/RUN_C_COMPARISON_REPORT.md"
