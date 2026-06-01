#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=concept-proj
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/concept-proj-%j.out
#SBATCH --error=logs/concept-proj-%j.out
#
# 63: Concept projection geometry — CONVERGENCE arbiter.
# After 62 refuted uniform redundancy (carrier dirs are mutually ~orthogonal,
# PR/k = 0.92-0.93, like random), 63 tests the convergence hypothesis:
#   "Write directions are orthogonal to each other BUT all project onto gbar
#    with the same sign — many-to-one read-out reconciling ortho + Jaccard 0.92."
#
# Decision rule (preregistered):
#   maj_sign >> null AND |cos_C(d, gbar)| >> null  →  CONVERGENCE
#   cos_C symmetric and ≈ null                    →  DIFFUSE
#
# PREREQUISITE: concept_directions.npz (gbar + Sigma_inv) from script 60_
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/runD_v2/geometry_stage1
#   sbatch jobs/run_concept_projection.sbatch

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/geometry_stage1

echo "========================================================"
echo "  Concept Projection Geometry (63 — convergence arbiter)"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
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

# ── verify prerequisites ────────────────────────────────────────────────────
GEOM_NPZ=data/analysis/runD_v2/geometry_stage1/concept_directions.npz
CLUSTERS=data/analysis/runD_v2/clustering_full/cluster_labels.csv

[ -f "$GEOM_NPZ" ] || { echo "ERROR: $GEOM_NPZ not found — run 60_ first"; exit 1; }
[ -f "$CLUSTERS" ] || { echo "ERROR: $CLUSTERS not found"; exit 1; }
echo "Prerequisites verified."

# Use SAME feature set as 62 (coimp_louvain k=14: cluster 13=L24, cluster 8=L18)
# so 62 and 63 describe the same carrier.
date
python -u scripts/63_concept_projection_geometry.py \
    --concept_npz    "$GEOM_NPZ" \
    --cluster_labels "$CLUSTERS" \
    --cluster_col    coimp_louvain \
    --clusters       13,8 \
    --layers         24 18 \
    --out_dir        data/analysis/runD_v2/geometry_stage1 \
    --n_null_draws   200 \
    --null_layer_min 10 \
    --null_layer_max 25 \
    --device         cuda \
    --model_size     4b \
    --seed           0

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Read: data/analysis/runD_v2/geometry_stage1/concept_projection.json"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/geometry_stage1/ data/analysis/runD_v2/geometry_stage1/"
