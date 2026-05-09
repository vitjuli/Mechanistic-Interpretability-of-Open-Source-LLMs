#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=ie-analysis
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ie-analysis-%j.out
#SBATCH --error=logs/ie-analysis-%j.out

# sbatch jobs/run_ie_analysis.sbatch
# RUNS: 70 (layer transition ARI + transfer) → 71 (cluster analysis) → 72 (cluster interventions)
# REQUIRES: run_ie_pipeline.sbatch completed successfully

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

echo "========================================================"
echo "  IE Abstraction Analysis — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID}  $(date)"
echo "========================================================"
nvidia-smi | head -4 || true

# ── Script 70: Layer transition ARI + transfer probe ──────────────────────────
echo ""
echo "── Script 70: Layer ARI transition ──"
python -u scripts/70_ie_layer_transition.py \
    --behaviour "$BEHAVIOUR" \
    --device cuda

echo "Layer transition done: $(date)"

# ── Script 71: Cluster analysis + cross-domain transfer + readout failure ─────
echo ""
echo "── Script 71: Cluster analysis + transfer ──"
python -u scripts/71_ie_cluster_analysis.py \
    --behaviour "$BEHAVIOUR" \
    --device cuda \
    --k 6

echo "Cluster analysis done: $(date)"

# ── Script 72: Cluster interventions ─────────────────────────────────────────
echo ""
echo "── Script 72: Cluster interventions ──"
python -u scripts/72_ie_cluster_interventions.py \
    --behaviour "$BEHAVIOUR" \
    --device cuda \
    --k 6 \
    --n_prompts 100

echo ""
echo "── Analysis complete: $(date) ──"
echo "Results: data/results/abstraction_ie/${BEHAVIOUR}/"
echo "========================================================"
