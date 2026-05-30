#!/bin/bash
# Sync sub-cluster (k=30) results from CSD3 and run local analyses.
# Usage: bash scripts/sync_subgroup_results.sh

set -e

REMOTE_USER=iv294
REMOTE_HOST=login.hpc.cam.ac.uk
REMOTE_ROOT=/rds/user/iv294/hpc-work/thesis/project
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$LOCAL_ROOT"

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Rsync results from CSD3 (will prompt for 2FA twice)"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p data/analysis/runD_v2/carrier_stability/subgroup_decomp
mkdir -p data/analysis/runD_v2/cluster_joint_ablation_subgroup

echo ""
echo "→ Rsync 1/2: carrier_stability/subgroup_decomp/"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/data/analysis/runD_v2/carrier_stability/subgroup_decomp/" "data/analysis/runD_v2/carrier_stability/subgroup_decomp/"

echo ""
echo "→ Rsync 2/2: cluster_joint_ablation_subgroup/"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/data/analysis/runD_v2/cluster_joint_ablation_subgroup/" "data/analysis/runD_v2/cluster_joint_ablation_subgroup/"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Step 2: Run 3 local CPU-only analyses"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "→ Condition I.a — ICC (script 38b)"
python3 scripts/38b_compute_icc.py \
    --act_dir data/analysis/runD_v2/activations \
    --grouping_dir data/analysis/runD_v2/grouping \
    --clustering_dir data/analysis/runD_v2/clustering_full \
    --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp \
    --clustering_col agglo_coimp_subgroup_k30

echo ""
echo "→ Condition I.b — Polarity (script 34)"
python3 scripts/34_polarity_consistency.py \
    --cluster_col agglo_coimp_subgroup_k30 \
    --grouping_dir data/analysis/runD_v2/grouping \
    --clustering_dir data/analysis/runD_v2/clustering_full \
    --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp

echo ""
echo "→ Condition III.1 — IR analysis (script 27b)"
python3 scripts/27b_analyse_joint_ablation.py \
    --cluster_col agglo_coimp_subgroup_k30 \
    --joint_dir data/analysis/runD_v2/cluster_joint_ablation_subgroup \
    --grouping_dir data/analysis/runD_v2/grouping \
    --clustering_dir data/analysis/runD_v2/clustering_full

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL DONE — results in:"
echo "    data/analysis/runD_v2/carrier_stability/subgroup_decomp/"
echo "    data/analysis/runD_v2/cluster_joint_ablation_subgroup/"
echo "═══════════════════════════════════════════════════════════════"
