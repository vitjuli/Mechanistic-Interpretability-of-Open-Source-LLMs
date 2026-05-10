#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-strengthen2
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ie-strengthen2-%j.out
#SBATCH --error=logs/ie-strengthen2-%j.out

# sbatch jobs/run_ie_strengthen2.sbatch
# S1 data already saved — regenerate report only, then run S2.

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
echo "  IE Strengthen2 — report + S2 causal test"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -3 || true

# ── S1 report only (no GPU needed, data already saved) ───────────────────────
echo "── S1: Regenerate report from saved CSVs ───────────────"
python -u scripts/74_eval_clean_cross_domain.py --report_only
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
echo "  Sync:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/abstraction_ie/ data/results/abstraction_ie/"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/docs/ docs/"
echo "========================================================"
