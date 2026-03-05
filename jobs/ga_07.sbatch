#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ga_07
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ga_07-%j.out
#SBATCH --error=logs/ga_07-%j.err

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

# --- Purge Intel/Cascade Lake modules (they inject AVX-512 libs that
#     cause SIGILL on AMD EPYC nodes in the ampere partition) ---
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# CPU instruction safety for AMD EPYC nodes (no AVX-512)
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
python --version
echo

echo "Running interventions..."

python -u scripts/07_run_interventions.py \
  --behaviour grammar_agreement \
  --split train \
  --experiment all \
  --n_prompts 20 \
  --top_k 5 \
  --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
  --graph_n_prompts 80

echo
echo "Finished at $(date)"
