#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=iia-probe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/iia-probe-%j.out
#SBATCH --error=logs/iia-probe-%j.out
#
# IIA on 24 contrastive pairs for G_probe (12 agglo clusters).
# Script 37: patches feature activations of each cluster from source prompt
# into target prompt, measures sign flip rate (IIA = Condition II test).
#
# Estimated time: ~60-90 min (48 baseline passes + 576 patched passes)
#

set -euo pipefail

module purge
module load rhel8/default-amp

ATEN_CPU_CAPABILITY=avx2
MKL_ENABLE_INSTRUCTIONS=AVX2
export ATEN_CPU_CAPABILITY MKL_ENABLE_INSTRUCTIONS

cd "$SLURM_SUBMIT_DIR"

echo "=== IIA probe clusters ==="
echo "Job: $SLURM_JOB_ID   Node: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python3 scripts/37_iia_probe_clusters.py --device cuda

echo "Done: $(date)"
