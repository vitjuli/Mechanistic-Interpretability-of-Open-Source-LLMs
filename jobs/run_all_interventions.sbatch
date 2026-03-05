#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -J interv_all
#SBATCH -o logs/interv_all-%j.out
#SBATCH -e logs/interv_all-%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
date

# Purge Intel/Cascade Lake modules (AVX-512 -> SIGILL on AMD EPYC)
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

echo "GPU check:"
nvidia-smi || true

# Go to project + activate venv
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

# CPU instruction safety for AMD EPYC nodes (no AVX-512)
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export TOKENIZERS_PARALLELISM=false

# Verify environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)')"

echo ""
echo "============================================================"
echo "Running ALL EXPERIMENTS: Ablation, Patching, Importance, Steering"
echo "============================================================"
date

# Run EVERYTHING in one go using --experiment all
# Increased n_prompts to 50 for better stats (adjust as needed)
# Added --export_token_examples for thesis visuals
python scripts/07_run_interventions.py \
  --behaviour grammar_agreement \
  --split train \
  --experiment all \
  --model_size 4b \
  --layers 15 16 17 18 19 20 \
  --n_prompts 50 \
  --top_k 20 \
  --export_token_examples

echo ""
echo "All experiments completed."
date
