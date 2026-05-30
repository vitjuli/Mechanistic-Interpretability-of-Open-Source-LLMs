#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ir-null-k30
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/ir-null-k30-%j.out
#SBATCH --error=logs/ir-null-k30-%j.out
#
# IR Null Control Experiment for k=30 SUB-CLUSTER partition (Script 27c).
#
# ★ IMPORTANT: this now uses agglo_coimp_subgroup_k30 (30 sub-clusters)
# NOT coimp_louvain (k=14). Previous job 29878330 used the wrong column.
#
# For each unique sub-cluster size k (13 unique sizes: 2, 3, 4, 5, 6, 8, 10,
# 11, 12, 13, 17, 18, 20), samples N_null random feature bundles from the
# 227-feature pool and runs joint ablation on subsampled prompts per bundle.
#
# Reduced budget vs k=14 version to fit in 30h time limit:
#   n_null_per_size = 10 (was 20)
#   n_prompts_null  = 80 (was 100)
#   Total forward passes: 13 sizes × 10 bundles × 80 prompts ≈ 10,400
#   Estimated time: 18-22h on Ampere GPU.
#   Statistical power: 10 bundles per size → enough for null distribution
#   (real IR ≈ 0.5 for n=2 vs expected null IR ≈ 1.0 → wide gap, easy to detect).
#
# Tests whether IR_null ≈ 1 empirically.
# If confirmed, real sub-cluster IR ≤ 0.35 is statistically meaningful.
# If null IR ≈ 0.5 for n=2 too → small-size additivity is a math artefact.
#
# Prereqs (already on CSD3):
#   data/analysis/runD_v2/grouping/feature_prompt_contributions.csv
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv
#     (must have column agglo_coimp_subgroup_k30 — verified before run)
#   data/prompts/physics_decay_type_probe_v2_train.jsonl
#
# Optional (for real vs null comparison):
#   data/analysis/runD_v2/cluster_joint_ablation_subgroup/joint_ablation_*.csv

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  IR Null Control — k=30 sub-clusters (Script 27c)"
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
JOINT_DIR="data/analysis/runD_v2/cluster_joint_ablation_subgroup"
NULL_DIR="data/analysis/runD_v2/ir_null_control_subgroup"

mkdir -p "$NULL_DIR"

# ─── Sanity check: cluster column exists and has 30 unique values ────────────
for f in "${GROUPING_DIR}/feature_prompt_contributions.csv" \
         "${CLUSTERING_DIR}/cluster_labels.csv" \
         "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

python -c "
import pandas as pd
df = pd.read_csv('${CLUSTERING_DIR}/cluster_labels.csv')
assert '${CLUSTER_COL}' in df.columns, 'Missing column: ${CLUSTER_COL}'
n_clusters = df['${CLUSTER_COL}'].nunique()
sizes = df.groupby('${CLUSTER_COL}').size()
n_features = len(df)
n_unique_sizes = sizes.nunique()
print(f'Cluster column ${CLUSTER_COL}: {n_clusters} sub-clusters, {n_features} features')
print(f'Unique sizes: {sorted(sizes.unique().tolist())}')
print(f'Number of unique sizes: {n_unique_sizes}')
assert n_clusters == 30, f'Expected 30 sub-clusters, got {n_clusters}'
assert n_features == 227, f'Expected 227 features, got {n_features}'
print('SANITY CHECK PASSED.')
" || { echo "ERROR: cluster column validation failed"; exit 1; }

echo ""
echo "==== [Script 27c] IR Null Control on $CLUSTER_COL ===="
date

python -u scripts/27c_ir_null_control.py \
    --behaviour        "$BEHAVIOUR" \
    --split            "$SPLIT" \
    --cluster_col      "$CLUSTER_COL" \
    --grouping_dir     "$GROUPING_DIR" \
    --clustering_dir   "$CLUSTERING_DIR" \
    --joint_dir        "$JOINT_DIR" \
    --run_dir          "$NULL_DIR" \
    --n_null_per_size  10 \
    --n_prompts_null   80 \
    --seed             42 \
    --device           cuda

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync and analyse locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/ir_null_control_subgroup/ data/analysis/runD_v2/ir_null_control_subgroup/"
echo "  python3 scripts/27d_ir_null_analysis.py --null_dir data/analysis/runD_v2/ir_null_control_subgroup --joint_dir data/analysis/runD_v2/cluster_joint_ablation_subgroup --cluster_col $CLUSTER_COL"
