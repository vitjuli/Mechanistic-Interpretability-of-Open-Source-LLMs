#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-s7
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ie-s7-%j.out
#SBATCH --error=logs/ie-s7-%j.out

# sbatch jobs/run_ie_s7.sbatch
# S7: cluster interventions for physics_intensive_extensive_v1

set -euo pipefail
BEHAVIOUR=physics_intensive_extensive_v1

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

echo "── S7: Cluster interventions — ${BEHAVIOUR} ──"
nvidia-smi | head -3 || true

python -u scripts/72_ie_cluster_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --device    cuda \
    --k         6 \
    --n_prompts 100

echo "Done — $(date)"
