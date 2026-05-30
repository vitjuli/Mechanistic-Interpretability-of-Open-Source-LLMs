#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=graded-iia
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/graded-iia-%j.out
#SBATCH --error=logs/graded-iia-%j.out
#
# Graded patching experiment: tests whether L24 (β-carrier) and L18 (α-carrier)
# show CONTINUOUS or DISCRETE structure under fractional patching.
#
# For each pool ∈ {L24, L18, random_L25 control}:
#   For each α ∈ {0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0}:
#     Sample 5 random subsets of size round(α × |pool|)
#     Run IIA (200 pairs × 2 directions) per subset
#     Record IIA values
#
# Predicted patterns:
#   - L24 (REDUNDANT distributed coding): IIA SATURATES at α≈0.1
#     (we already know 2/20 features = 94% of full L24's flips)
#   - L18 (POPULATION coding): IIA grows LINEARLY with α
#   - random_L25 control: IIA stays near zero (specificity test)
#
# Runtime estimate:
#   3 pools × 7 alphas × 5 subsets = 105 IIA runs
#   Each ~2 min after baseline caching → ~4h total
#   8h limit gives 4h buffer.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  Graded Patching Experiment (population vs distributed)"
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

CLUSTERING_DIR="data/analysis/runD_v2/clustering_full"
OUT_DIR="data/analysis/runD_v2/graded_patching"

mkdir -p "$OUT_DIR"

# ── Sanity check ───────────────────────────────────────────────────────────
[ -f "${CLUSTERING_DIR}/cluster_labels.csv" ] || { echo "Missing cluster_labels.csv"; exit 1; }

python -c "
import pandas as pd
df = pd.read_csv('${CLUSTERING_DIR}/cluster_labels.csv')
assert 'agglo_coimp_subgroup_k30' in df.columns, 'Missing column'
df['layer'] = df['feature_id'].str.extract(r'L(\d+)_').astype(int)
print(f'L24: {(df.layer==24).sum()} features, L18: {(df.layer==18).sum()}, L25: {(df.layer==25).sum()}')
"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running graded patching: L24 + L18 + random L25 control"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/57_graded_patching.py \
    --target_layers    L24 L18 random_l25 \
    --cluster_col      agglo_coimp_subgroup_k30 \
    --clustering_dir   "$CLUSTERING_DIR" \
    --out_dir          "$OUT_DIR" \
    --behaviour        physics_decay_type_probe_v2 \
    --split            train \
    --alphas           0.05 0.10 0.15 0.25 0.50 0.75 1.0 \
    --n_subsets        5 \
    --max_pairs        200 \
    --seed             42 \
    --device           cuda

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE: $(date)  Job: ${SLURM_JOB_ID}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sync results:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/graded_patching/ data/analysis/runD_v2/graded_patching/"
echo ""
echo "Generate visualization locally:"
echo "  python3 scripts/57b_plot_graded_curves.py --in_dir data/analysis/runD_v2/graded_patching"
