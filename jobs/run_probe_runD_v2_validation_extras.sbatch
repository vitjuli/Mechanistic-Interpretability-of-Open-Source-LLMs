#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=val-extras
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/val-extras-%j.out
#SBATCH --error=logs/val-extras-%j.out
#
# Validation extras for sub-cluster carrier analysis:
#   STEP A: Per-pair flip data for L24 sub-clusters + L18
#           → check whether L24 sub-detectors flip the SAME pairs (interchangeable)
#             or DIFFERENT pairs (complementary specialised detectors)
#   STEP B: Combined patching tests (L18 + L24 sub-clusters)
#           → tests counterweight hypothesis: does adding L18 to L24 increase IIA?
#
# Targets analysed:
#   C16, C18, C19, C20 — 4 L24 sub-clusters (n=2, 10, 3, 5)
#   C35 — L18 (n=17, α-carrier)
#
# Estimated time: ~2-3h (5 targets in mode A, ~4 combos in mode B; vs 30 targets in full job)

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Validation extras for sub-cluster carriers"
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

CLUSTER_COL=agglo_coimp_subgroup_k30
GROUPING_DIR="data/analysis/runD_v2/grouping"
CLUSTERING_DIR="data/analysis/runD_v2/clustering_full"
OUT_DIR="data/analysis/runD_v2/carrier_stability/subgroup_decomp"

mkdir -p "$OUT_DIR"

# ════════════════════════════════════════════════════════════════════════════
# STEP A: Per-pair flip data — overlap analysis of L24 sub-detectors + L18
# Targets: C16 (L24-a, n=2), C18 (L24-b, n=10), C19 (L24-c, n=3),
#          C20 (L24-d, n=5), C35 (L18, n=17)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP A: Per-pair flip data (L24 sub-detectors + L18)"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/37b_iia_validation.py \
    --mode per_pair \
    --target_clusters "16,18,19,20,35" \
    --cluster_col     "$CLUSTER_COL" \
    --prompt_metadata "${GROUPING_DIR}/prompt_metadata.csv" \
    --clustering_dir  "$CLUSTERING_DIR" \
    --out_dir         "$OUT_DIR" \
    --device          cuda

echo ""
echo "==== STEP A done: $(date) ===="

# ════════════════════════════════════════════════════════════════════════════
# STEP B: Combined patching — counterweight hypothesis test
# Combos:
#   16+35     — L24-a (β-detector core) + L18 (α-counterweight): does adding L18 help?
#   16+18+19+20 — All 4 L24 sub-clusters together (should equal full L24, baseline)
#   16+18+19+20+35 — Full L24 + L18: maximum carrier patching
#   35       — L18 alone (reference)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP B: Combined patching (counterweight hypothesis)"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/37b_iia_validation.py \
    --mode combined \
    --combinations "35;16+35;16+18+19+20;16+18+19+20+35" \
    --cluster_col     "$CLUSTER_COL" \
    --prompt_metadata "${GROUPING_DIR}/prompt_metadata.csv" \
    --clustering_dir  "$CLUSTERING_DIR" \
    --out_dir         "$OUT_DIR" \
    --device          cuda

echo ""
echo "==== STEP B done: $(date) ===="

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE: $(date)  Job: ${SLURM_JOB_ID}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/carrier_stability/subgroup_decomp/ data/analysis/runD_v2/carrier_stability/subgroup_decomp/"
echo ""
echo "Then locally analyze with:"
echo "  python3 scripts/analyse_iia_validation.py  # to be written after seeing output"
