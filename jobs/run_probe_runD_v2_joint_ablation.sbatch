#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runD-v2-joint
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/probe-runD-v2-joint-%j.out
#SBATCH --error=logs/probe-runD-v2-joint-%j.out
#
# RunD v2 — Condition III.1 (Coherence): joint cluster ablation (script 27)
# Tests IR = Delta_joint / sum(Delta_k) <= 0.35 per cluster.
#
# 14 clusters (coimp_louvain), 227 features, 538 prompts.
# Estimated time: ~3-5h (538 prompts x 14 clusters, multi-layer simultaneous patch)
#
# Prereqs (already on CSD3 after RunD v2 pipeline):
#   data/analysis/runD_v2/grouping/feature_prompt_contributions.csv
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv  (has coimp_louvain column)
#   data/prompts/physics_decay_type_probe_v2_train.jsonl
#
# Run in parallel with run_iia_probe_clusters.sbatch — independent jobs.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD v2 Joint Ablation (Condition III.1)"
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
GROUPING_DIR="data/analysis/runD_v2/grouping"
CLUSTERING_DIR="data/analysis/runD_v2/clustering_full"
JOINT_DIR="data/analysis/runD_v2/cluster_joint_ablation"

for f in "${GROUPING_DIR}/feature_prompt_contributions.csv" \
         "${CLUSTERING_DIR}/cluster_labels.csv" \
         "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

mkdir -p "$JOINT_DIR"

echo "==== [Script 27] RunD v2 joint cluster ablation ===="
date

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --clusters       all \
    --n_prompts      538 \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --run_dir        "$JOINT_DIR" \
    --device         cuda

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync and run analysis locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/cluster_joint_ablation/ data/analysis/runD_v2/cluster_joint_ablation/"
echo "  python3 scripts/27b_analyse_joint_ablation.py --joint_dir data/analysis/runD_v2/cluster_joint_ablation --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full"
