#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=clust-interv
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clust-interv-%j.out
#SBATCH --error=logs/clust-interv-%j.out

# ── Cluster intervention analysis (script 46) ─────────────────────────────────
# Prerequisites:
#   - Script 45 must have run and produced:
#     data/results/internal_candidate_analysis/physics_internal_candidate_selection_v2/
#       feature_clusters_k6_kmeans.csv
# sbatch jobs/run_cluster_intervention.sbatch

set -euo pipefail

BEHAVIOUR=physics_internal_candidate_selection_v2
SPLIT=train
K=6
METHOD=kmeans

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
echo "  Cluster intervention — ${BEHAVIOUR}"
echo "  Job ${SLURM_JOB_ID}  Node $(hostname)  $(date)"
echo "========================================================"
python --version
nvidia-smi | head -4 || true

CLUSTER_CSV="data/results/internal_candidate_analysis/${BEHAVIOUR}/feature_clusters_k${K}_${METHOD}.csv"
[ ! -f "$CLUSTER_CSV" ] && {
    echo "[FATAL] Cluster CSV not found: $CLUSTER_CSV"
    echo "  Run: python scripts/45_candidate_feature_clustering.py first"
    exit 1; }
echo "  [OK] Cluster CSV: $CLUSTER_CSV"

python -u scripts/46_cluster_intervention_analysis.py \
    --behaviour     "$BEHAVIOUR" \
    --split         "$SPLIT" \
    --k             "$K" \
    --method        "$METHOD" \
    --n_prompts     100 \
    --device        cuda

echo "Script 46 exit code: $?  ($(date))"

echo ""
echo "========================================================"
echo "  DONE — $(date)  Job $SLURM_JOB_ID"
echo "========================================================"
ls -lh data/results/internal_candidate_analysis/${BEHAVIOUR}/cluster_*k${K}*.csv 2>/dev/null || true

echo ""
echo "  Rsync to local Mac:"
BASE="iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
ADIR="data/results/internal_candidate_analysis/${BEHAVIOUR}"
echo "  rsync -av \"${BASE}/${ADIR}/\" /path/to/local/${ADIR}/"
echo "========================================================"
