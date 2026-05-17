#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runC-joint
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/probe-runC-joint-%j.out
#SBATCH --error=logs/probe-runC-joint-%j.out
#
# Run C joint cluster ablation (script 27)
# Run AFTER run_probe_runC_pipeline.sbatch completes.
#
# Prereqs (all on CSD3):
#   data/analysis/runC_top10_sign_complete/grouping/feature_prompt_contributions.csv
#   data/analysis/runC_top10_sign_complete/clustering/cluster_labels.csv
#   data/prompts/physics_decay_type_probe_train.jsonl
#
# Runtime estimate: ~2-3h (more features per cluster than Run B)

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe — Run C joint ablation"
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

BEHAVIOUR=physics_decay_type_probe
SPLIT=train
RUNC_BASE="data/analysis/runC_top10_sign_complete"
GROUPING_DIR="${RUNC_BASE}/grouping"
CLUSTERING_DIR="${RUNC_BASE}/clustering"
JOINT_DIR="${RUNC_BASE}/cluster_joint_ablation"

# Verify prereqs
for f in "${GROUPING_DIR}/feature_prompt_contributions.csv" \
         "${CLUSTERING_DIR}/cluster_labels.csv" \
         "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

mkdir -p "$JOINT_DIR"

echo "==== [Script 27] Run C joint cluster ablation ===="
date

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --clusters       all \
    --n_prompts      470 \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --run_dir        "$JOINT_DIR"

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Next:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:$(pwd)/${JOINT_DIR}/ ${JOINT_DIR}/"
echo "  bash scripts/run_runC_pipeline.sh --start 272"
