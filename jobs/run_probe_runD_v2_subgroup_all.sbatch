#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=subg-all
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/subg-all-%j.out
#SBATCH --error=logs/subg-all-%j.out
#
# RunD v2 — subgroup decomposition (k=30) verification of all conditions.
# Tests Condition II (IIA, script 37) and Condition III.1 (Joint Ablation, script 27)
# on the new sub-cluster column `agglo_coimp_subgroup_k30` (30 sub-clusters,
# no singletons, derived from HAC on W_coimportance at k_target=60 with
# singleton-merge post-processing).
#
# Cluster count: 14 (coimp_louvain) → 30 (agglo_coimp_subgroup_k30)
# IIA: 269 pairs × 30 clusters × 2 dirs ≈ 16,140 patches (~2x previous)
# Joint ablation: 538 prompts × 30 clusters (~2x previous)
#
# Sequential run; estimated 14-18h total:
#   Step 1: IIA (~8-10h)
#   Step 2: Joint ablation (~6-8h)
#
# Prereqs (must be on CSD3 already):
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv
#       (must include column 'agglo_coimp_subgroup_k30')
#   data/analysis/runD_v2/grouping/feature_prompt_contributions.csv
#   data/analysis/runD_v2/grouping/prompt_metadata.csv
#   data/prompts/physics_decay_type_probe_v2_train.jsonl
#
# Outputs (sync these locally afterward):
#   data/analysis/runD_v2/carrier_stability/subgroup_decomp/iia_probe_clusters.csv
#   data/analysis/runD_v2/cluster_joint_ablation_subgroup/
#
# After CSD3 sync, run locally (CPU-only follow-ups):
#   # Condition I.a — ICC for sub-clusters
#   python3 scripts/38b_compute_icc.py \
#       --act_dir data/analysis/runD_v2/activations \
#       --grouping_dir data/analysis/runD_v2/grouping \
#       --clustering_dir data/analysis/runD_v2/clustering_full \
#       --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp \
#       --clustering_col agglo_coimp_subgroup_k30
#
#   # Condition I.b — Polarity for sub-clusters
#   python3 scripts/34_polarity_consistency.py \
#       --cluster_col agglo_coimp_subgroup_k30 \
#       --grouping_dir data/analysis/runD_v2/grouping \
#       --clustering_dir data/analysis/runD_v2/clustering_full \
#       --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp
#
#   # Condition III.1 — IR analysis from joint ablation CSV
#   python3 scripts/27b_analyse_joint_ablation.py \
#       --cluster_col agglo_coimp_subgroup_k30 \
#       --joint_dir data/analysis/runD_v2/cluster_joint_ablation_subgroup \
#       --grouping_dir data/analysis/runD_v2/grouping \
#       --clustering_dir data/analysis/runD_v2/clustering_full

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  RunD v2 — Sub-cluster (k=30) all-conditions verification"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BEHAVIOUR=physics_decay_type_probe_v2
SPLIT=train
CLUSTER_COL=agglo_coimp_subgroup_k30
GROUPING_DIR="data/analysis/runD_v2/grouping"
CLUSTERING_DIR="data/analysis/runD_v2/clustering_full"
CARRIER_DIR="data/analysis/runD_v2/carrier_stability/subgroup_decomp"
JOINT_DIR="data/analysis/runD_v2/cluster_joint_ablation_subgroup"

mkdir -p "$CARRIER_DIR" "$JOINT_DIR"

# ─── Sanity check: required files exist and column is present ───────────────
for f in "${CLUSTERING_DIR}/cluster_labels.csv" \
         "${GROUPING_DIR}/feature_prompt_contributions.csv" \
         "${GROUPING_DIR}/prompt_metadata.csv" \
         "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

python -c "
import pandas as pd
df = pd.read_csv('${CLUSTERING_DIR}/cluster_labels.csv')
assert '${CLUSTER_COL}' in df.columns, 'Missing column: ${CLUSTER_COL}'
n_clusters = df['${CLUSTER_COL}'].nunique()
n_missing = df['${CLUSTER_COL}'].isna().sum()
print(f'Cluster column ${CLUSTER_COL}: {n_clusters} unique clusters, {n_missing} missing assignments')
assert n_missing == 0, f'{n_missing} features have no sub-cluster assignment'
" || { echo "ERROR: cluster column validation failed"; exit 1; }

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Condition II — IIA (script 37) on sub-clusters
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 1/2: IIA (Condition II) — script 37 on $CLUSTER_COL"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/37_iia_probe_clusters.py \
    --device          cuda \
    --cluster_col     "$CLUSTER_COL" \
    --prompt_metadata "${GROUPING_DIR}/prompt_metadata.csv" \
    --clustering_dir  "$CLUSTERING_DIR" \
    --out_dir         "$CARRIER_DIR"

echo ""
echo "==== STEP 1 done: $(date) ===="
echo ""

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Condition III.1 — Joint cluster ablation (script 27) on sub-clusters
# ════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 2/2: Joint Ablation (Condition III.1) — script 27 on $CLUSTER_COL"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --clusters       all \
    --n_prompts      538 \
    --cluster_col    "$CLUSTER_COL" \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --run_dir        "$JOINT_DIR" \
    --device         cuda

echo ""
echo "==== STEP 2 done: $(date) ===="
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL STEPS DONE: $(date)  Job: ${SLURM_JOB_ID}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sync these locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/carrier_stability/subgroup_decomp/ data/analysis/runD_v2/carrier_stability/subgroup_decomp/"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/cluster_joint_ablation_subgroup/ data/analysis/runD_v2/cluster_joint_ablation_subgroup/"
echo ""
echo "Then locally (CPU-only):"
echo "  # I.a — ICC for sub-clusters"
echo "  python3 scripts/38b_compute_icc.py --act_dir data/analysis/runD_v2/activations --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp --clustering_col $CLUSTER_COL"
echo "  # I.b — Polarity for sub-clusters"
echo "  python3 scripts/34_polarity_consistency.py --cluster_col $CLUSTER_COL --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability/subgroup_decomp"
echo "  # III.1 — IR analysis"
echo "  python3 scripts/27b_analyse_joint_ablation.py --cluster_col $CLUSTER_COL --joint_dir data/analysis/runD_v2/cluster_joint_ablation_subgroup --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full"
