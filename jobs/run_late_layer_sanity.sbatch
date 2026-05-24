#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=late-layer
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/late-layer-%j.out
#SBATCH --error=logs/late-layer-%j.out
#
# Script 55: Late-layer ablation sanity check.
# Verifies that L24 is a genuine causal peak, not an edge-of-range artifact.
#
# Protocol:
#   For each of L26, L28, L30, L33 (4 layers):
#     - Rank features by |mean_act_alpha − mean_act_beta|
#     - Select top-2 discriminative features
#     - Run zero-ablation on all 538 prompts
#     - Compare effect_size and SFR to L24 benchmark (~0.30 SFR, ~0.61 fixed-axis eff)
#
# Runtime estimate:
#   Activation collection + baseline: 4 layers × 538 prompts ≈ 36 min
#   Ablation: 4 layers × 2 features × 538 prompts ≈ 70 min
#   Total: ~1.8h → 1.2h buffer within 3h limit
#
# Output:
#   data/analysis/late_layer_sanity/physics_decay_type_probe_v2/
#       late_layer_sanity_physics_decay_type_probe_v2_train.csv
#       late_layer_sanity_summary_physics_decay_type_probe_v2_train.csv

set -uo pipefail
mkdir -p logs
echo "========================================================"
echo "  Script 55: Late-layer ablation sanity check"
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

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo ""

python -u scripts/55_late_layer_sanity.py \
    --behaviour  physics_decay_type_probe_v2 \
    --split      train \
    --layers     26 28 30 33 \
    --top_k      2 \
    --output_dir data/analysis/late_layer_sanity

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/late_layer_sanity/ data/analysis/late_layer_sanity/"
