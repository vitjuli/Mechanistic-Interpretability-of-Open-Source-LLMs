#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=iia-probe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/iia-probe-%j.out
#SBATCH --error=logs/iia-probe-%j.out
#
# IIA on 269 random alpha/beta pairs, 14 coimp_louvain clusters, 227 features.
# Script 37: patches feature activations of each cluster from source prompt
# into target prompt, measures sign flip rate (IIA = Condition II test).
#
# Estimated time: ~4-6h (538 baseline passes + 7532 patching passes)
# 269 pairs x 14 clusters x 2 directions = 7532 patches
#

set -euo pipefail

module purge
module load rhel8/default-amp

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

echo "=== IIA probe clusters ==="
echo "Job: $SLURM_JOB_ID   Node: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python3 scripts/37_iia_probe_clusters.py --device cuda \
    --cluster_col     coimp_louvain \
    --prompt_metadata data/analysis/runD_v2/grouping/prompt_metadata.csv \
    --clustering_dir  data/analysis/runD_v2/clustering_full \
    --out_dir         data/analysis/runD_v2/carrier_stability

echo "Done: $(date)"
