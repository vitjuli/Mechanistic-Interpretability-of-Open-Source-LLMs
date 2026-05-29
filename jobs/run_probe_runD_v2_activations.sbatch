#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=collect-act
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/collect-act-%j.out
#SBATCH --error=logs/collect-act-%j.out
#
# Script 38: Collect raw transcoder activations a_k^(ℓ)(p) for all 538 prompts
# and all 227 features (L10-L25). One forward pass per prompt, no intervention.
#
# Output: data/analysis/runD_v2/activations/activation_matrix.npy (227 x 538)
# Estimated time: ~10-15 min on Ampere.
#
# Prereqs:
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv
#   data/prompts/physics_decay_type_probe_v2_train.jsonl

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Collect activations — RunD v2 (Script 38)"
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
CLUSTERING_DIR="data/analysis/runD_v2/clustering_full"
OUT_DIR="data/analysis/runD_v2/activations"

for f in "${CLUSTERING_DIR}/cluster_labels.csv" \
          "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    [ -f "$f" ] || { echo "ERROR: Missing: $f"; exit 1; }
done

mkdir -p "$OUT_DIR"

echo "==== [Script 38] Collecting activations ===="
date

python -u scripts/38_collect_activations.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --clustering_dir "$CLUSTERING_DIR" \
    --out_dir        "$OUT_DIR" \
    --device         cuda

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync and compute ICC locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/activations/ data/analysis/runD_v2/activations/"
echo "  python3 scripts/38b_compute_icc.py --act_dir data/analysis/runD_v2/activations --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability"
