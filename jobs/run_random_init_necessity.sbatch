#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=rand-nec
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rand-nec-%j.out
#SBATCH --error=logs/rand-nec-%j.out
#
# Script 56: Random-init control for ablation-based necessity
#
# Measures necessity(G) = Sign Flip Rate under joint zero-ablation of cluster G,
# comparing trained-model activations vs random-init activations.
#
# Key outputs:
#   necessity_trained  — SFR under ablation of G (trained model)
#   necessity_rand     — SFR under ablation of G (random-init model)
#   noop_sfr           — reconstruction noise floor
#   delta_ctrl_nec     — necessity_trained − necessity_rand
#
# These numbers determine the final form of Theorem 1 (T1.2 operational form):
#   If delta_ctrl_nec > 0.1 for key clusters → Variant 2 with real bounds
#   If trivial → Variant 1 qualitative only
#
# Runtime estimate:
#   12 clusters × 538 prompts × 3 conditions × ~1.5s/prompt ≈ 3.5h
#   (random-init forward passes ≈ 2× cheaper; well within 4h limit)
#
# Output:
#   data/analysis/random_init_necessity/physics_decay_type_probe_v2/
#       random_init_necessity_physics_decay_type_probe_v2_train_summary.json
#       random_init_necessity_physics_decay_type_probe_v2_train.csv

set -uo pipefail
mkdir -p logs
echo "========================================================"
echo "  Script 56: Random-init necessity control"
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

python -u scripts/56_random_init_necessity.py \
    --behaviour  physics_decay_type_probe_v2 \
    --split      train \
    --all_clusters \
    --seed       42 \
    --output_dir data/analysis/random_init_necessity

echo ""
echo "==== Done: $(date)  Job: ${SLURM_JOB_ID} ===="
echo ""
echo "Sync:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/random_init_necessity/ data/analysis/random_init_necessity/"
