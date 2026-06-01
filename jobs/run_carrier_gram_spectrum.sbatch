#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=gram-spectrum
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gram-spectrum-%j.out
#SBATCH --error=logs/gram-spectrum-%j.out
#
# 62: Carrier Gram spectrum arbiter — UNIFORM REDUNDANCY vs COMBINATORIAL.
# Builds K_ij = <d_i, d_j>_C over L24 (β-carrier) and L18 (α-carrier)
# decoder rows, reports eigenvalue spectrum + null calibration.
#
# Decision rule:
#   PR/k → 0, lambda1 → 1, |cos_C| → 1, below null band  →  UNIFORM REDUNDANCY
#   PR/k ~ null, lambda1 small, flat                     →  COMBINATORIAL
#
# Compute trivial (~minutes); GPU needed only for transcoder loading.
#
# PREREQUISITE: concept_directions.npz with Sigma_inv from script 60_
#
# BEFORE SUBMITTING:
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs data/analysis/runD_v2/geometry_stage1
#   sbatch jobs/run_carrier_gram_spectrum.sbatch

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/geometry_stage1

echo "========================================================"
echo "  Carrier Gram Spectrum (62 — uniform-redundancy arbiter)"
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

# ── L24 (β-carrier, coimp_louvain cluster 13, n=20)
# ── L18 (α-carrier, coimp_louvain cluster 8,  n=17)
# Use the same coimp_louvain k=14 column that matches results_summary_ru.md C13/C8 labels.
date
python -u scripts/62_carrier_gram_spectrum.py \
    --concept_npz   "$GEOM_NPZ" \
    --cluster_labels "$CLUSTERS" \
    --cluster_col   coimp_louvain \
    --clusters      13,8 \
    --layers        24 18 \
    --out_dir       data/analysis/runD_v2/geometry_stage1 \
    --n_null_draws  200 \
    --null_layer_min 10 \
    --null_layer_max 25 \
    --device        cuda \
    --model_size    4b \
    --seed          0

echo ""
echo "==== Done: $(date) ===="
echo ""
echo "Read: data/analysis/runD_v2/geometry_stage1/carrier_gram_spectrum.json"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/geometry_stage1/ data/analysis/runD_v2/geometry_stage1/"
