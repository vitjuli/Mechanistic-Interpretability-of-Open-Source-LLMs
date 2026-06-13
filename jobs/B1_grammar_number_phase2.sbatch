#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=B1gn-p2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/B1gn-p2-%j.out
#SBATCH --error=logs/B1gn-p2-%j.out

set -uo pipefail
mkdir -p logs

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "==== Phase 2: 122 tier 1 broad sweep ===="
date

python -u scripts/122_b1_steering_sweep.py \
    --corpus data/prompts/B1_grammar_number.jsonl \
    --dump_dir data/analysis/runD_v2/B1_grammar_number/field_dump \
    --tier 1 \
    --out_dir data/analysis/runD_v2/B1_grammar_number

echo "==== Phase 2 done ===="
date
