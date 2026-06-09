#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j108-joint
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j108-%j.out
#SBATCH --error=logs/j108-%j.out
#
# 108 — Joint read∩use axis. Pareto front of (AUC, cos(v, u)) — can we find
# a single direction that both DECODES the concept AND aligns with the
# gradient? Two key columns: best AUC at cos(v,u) ≥ 0.5, best cos at AUC ≥ 0.9.
# If both low → read & use are structurally incompatible. ~4-5 min real.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/joint_axis

echo "========================================================"
echo "  108 — joint read∩use Pareto"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
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

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
nvidia-smi || true
date

python -u scripts/108_joint_read_use_axis.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 8 13 17 21 24 28 32 35 \
    --out_dir data/analysis/runD_v2/joint_axis

echo ""
echo "==== Done: $(date) ===="
