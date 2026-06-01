#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=final-batch
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Parametrized runner for the final parallel batch (65/66/67/68).
# Usage (launch four jobs in parallel, each gets its own GPU node):
#
#   cd /rds/user/iv294/hpc-work/thesis/project
#   mkdir -p logs
#   sbatch -J j65 jobs/run_final.sbatch scripts/65_concept_axis_causality.py  "<args65>"
#   sbatch -J j66 jobs/run_final.sbatch scripts/66_carrier_vs_wres_geometry.py "<args66>"
#   sbatch -J j67 jobs/run_final.sbatch scripts/67_carrier_ablation_full.py    "<args67>"
#   sbatch -J j68 jobs/run_final.sbatch scripts/68_wres_stability.py           "<args68>"
#
# $1 = python script path, $2 = quoted argument string for that script.
# See LAUNCH_SHEET.md for the exact arg strings.

set -uo pipefail

SCRIPT="$1"; shift
ARGS="${1:-}"; shift || true

mkdir -p logs data/analysis/runD_v2/geometry_stage1 data/analysis/iia_failure_diagnosis

echo "========================================================"
echo "  Final batch runner — ${SCRIPT}"
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

echo "Python: $(which python)"
nvidia-smi || true
echo ""
date

# -u for unbuffered logs; word-splitting of ARGS is intentional
python -u "${SCRIPT}" ${ARGS}

echo ""
echo "==== Done: $(date) ===="
