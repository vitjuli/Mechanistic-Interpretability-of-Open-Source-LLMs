#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-strengthen
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ie-strengthen-%j.out
#SBATCH --error=logs/ie-strengthen-%j.out

# sbatch jobs/run_ie_strengthen.sbatch
#
# Two strengthening analyses for physics_intensive_extensive_v1:
#   S1: eval_clean_cross_domain  (script 74) — ~45 min
#   S2: readout_failure_causal   (script 75) — ~45 min
#
# Prerequisite: run script 73 locally (no GPU) to generate prompts:
#   python scripts/73_generate_clean_cross_domain_abstraction.py

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

TSTART=$(date +%s)
stamp() { echo "  [$(( ($(date +%s) - TSTART) / 60 ))m elapsed]  $(date)"; }

echo "========================================================"
echo "  IE Strengthening Analyses"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -6 || true
echo ""

# ── S1: Clean cross-domain v2 evaluation ─────────────────────────────────────
echo "── S1: Clean D-v2 evaluation (script 74) ──────────────"
python -u scripts/74_eval_clean_cross_domain.py \
    --device cuda \
    --dtype  bfloat16
stamp

echo ""

# ── S2: Readout failure causal test ──────────────────────────────────────────
echo "── S2: Readout failure causal test (script 75) ─────────"
python -u scripts/75_readout_failure_causal.py \
    --device cuda \
    --dtype  bfloat16 \
    --k      6
stamp

ELAPSED=$(( ($(date +%s) - TSTART) / 60 ))
echo ""
echo "========================================================"
echo "  COMPLETE — ${ELAPSED}m total — $(date)"
echo "  Results:"
echo "    data/results/abstraction_ie/clean_cross_domain_v2/"
echo "    data/results/abstraction_ie/readout_failure_causal/"
echo "    docs/clean_cross_domain_v2_results.md"
echo "    docs/readout_failure_causal_results.md"
echo ""
echo "  Sync locally:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/abstraction_ie/ data/results/abstraction_ie/"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/docs/ docs/"
echo "========================================================"
