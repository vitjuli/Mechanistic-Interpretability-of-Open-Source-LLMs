#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ir-null-ctrl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ir-null-ctrl-%j.out
#SBATCH --error=logs/ir-null-ctrl-%j.out
#
# IR Null Control Experiment (Script 27c)
#
# For each unique real cluster size k (among 14 coimp_louvain clusters),
# samples N_null=20 random feature bundles of size k from the 227-feature pool
# and runs joint ablation on 100 prompts per bundle.
#
# Total forward passes: ~14 unique sizes × 20 bundles × 100 prompts ≈ 28,000
# Estimated time: ~3-5h on Ampere GPU.
#
# This tests whether IR_null ≈ 1 empirically (analytical argument only before).
# If confirmed, the real cluster IR ≤ 0.35 is statistically meaningful.
#
# Prereqs (already on CSD3 after RunD v2 pipeline):
#   data/analysis/runD_v2/grouping/feature_prompt_contributions.csv
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv
#   data/prompts/physics_decay_type_probe_v2_train.jsonl
#
# Optional (for real vs null comparison):
#   data/analysis/runD_v2/cluster_joint_ablation/joint_ablation_physics_decay_type_probe_v2_train.csv
#
# Can run in parallel with run_probe_runD_v2_joint_ablation.sbatch.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  IR Null Control — RunD v2 (Script 27c)"
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
NULL_DIR="data/analysis/runD_v2/ir_null_control"

for f in "${GROUPING_DIR}/feature_prompt_contributions.csv" \
         "${CLUSTERING_DIR}/cluster_labels.csv" \
         "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

mkdir -p "$NULL_DIR"

echo "==== [Script 27c] IR Null Control ===="
date

python -u scripts/27c_ir_null_control.py \
    --behaviour        "$BEHAVIOUR" \
    --split            "$SPLIT" \
    --grouping_dir     "$GROUPING_DIR" \
    --clustering_dir   "$CLUSTERING_DIR" \
    --joint_dir        "$JOINT_DIR" \
    --run_dir          "$NULL_DIR" \
    --n_null_per_size  20 \
    --n_prompts_null   100 \
    --seed             42 \
    --device           cuda

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync and analyse locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/ir_null_control/ data/analysis/runD_v2/ir_null_control/"
echo "  python3 scripts/27d_ir_null_analysis.py --null_dir data/analysis/runD_v2/ir_null_control --joint_dir data/analysis/runD_v2/cluster_joint_ablation"
