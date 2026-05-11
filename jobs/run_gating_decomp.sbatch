#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=gate-decomp
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/gate-decomp-%j.out
#SBATCH --error=logs/gate-decomp-%j.out

# sbatch jobs/run_gating_decomp.sbatch
# Per-family L28 transition decomposition (220 prompts × L19-L36).
# ~15-20 min GPU time.

set -euo pipefail

cd /rds/user/iv294/hpc-work/thesis/project
module purge
module load rhel8/default-amp 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Gating Family Decomposition"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -3 || true

python -u scripts/84_gating_family_decomposition.py \
    --device cuda \
    --dtype  bfloat16

echo ""
echo "  Sync:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/gating_probe_v1/ data/results/gating_probe_v1/"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/plots/ plots/"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/docs/gating_family_decomposition.md docs/"
echo "Done — $(date)"
