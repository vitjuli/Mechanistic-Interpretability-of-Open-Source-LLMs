#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=h3h4-sub
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/h3h4-sub-%j.out
#SBATCH --error=logs/h3h4-sub-%j.out
#
# Tests H3 (backup pathways) and H4 (early vs late) on the k=30 sub-cluster
# partition (agglo_coimp_subgroup_k30). Replicates the analyses from thesis_1_2
# at finer cluster resolution to verify two key hypotheses:
#
# H3 (Wang 2023 — IOI backup heads):
#   For each of 30 sub-clusters: ablate features, measure Δact on remaining 197
#   features. Backup candidates: Δact > 2σ AND mean > 0.5. Last-layer sub-clusters
#   (L25 → C27, C30, C32) should have ZERO backup (testable prediction).
#
# H4 (early vs late propagation):
#   Combine all features from early sub-clusters (layer < L_split) and late
#   (layer ≥ L_split), patch each set, measure IIA. Tests: does early IIA > late?
#
# Steps:
#   1. H3: 30 sub-clusters × 80 prompts × 227 features monitored — ~3-4h
#   2. H4: 3 splits × early/late × 200 pairs + per-layer scan — ~2h
# Total: 5-6h
#
# Prerequisites:
#   data/analysis/runD_v2/clustering_full/cluster_labels.csv
#     (must contain agglo_coimp_subgroup_k30 column)
#   data/prompts/physics_decay_type_probe_v2_train.jsonl
#
# After completion, sync results:
#   rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/h3h4_subcluster/ data/analysis/runD_v2/h3h4_subcluster/

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  H3 (backup) + H4 (early/late) on k=30 sub-clusters"
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
GROUPING_DIR="data/analysis/runD_v2/grouping"
OUT_DIR="data/analysis/runD_v2/h3h4_subcluster"

mkdir -p "$OUT_DIR"

# ── Sanity check ─────────────────────────────────────────────────────────────
[ -f "${CLUSTERING_DIR}/cluster_labels.csv" ] || { echo "Missing cluster_labels.csv"; exit 1; }
python -c "
import pandas as pd
df = pd.read_csv('${CLUSTERING_DIR}/cluster_labels.csv')
assert 'agglo_coimp_subgroup_k30' in df.columns, 'Missing column'
print(f'Sub-clusters: {df.agglo_coimp_subgroup_k30.nunique()}, Features: {len(df)}')
" || exit 1

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running H3 + H4 diagnosis on agglo_coimp_subgroup_k30"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

python -u scripts/55_subcluster_h3h4_diagnosis.py \
    --mode             both \
    --behaviour        physics_decay_type_probe_v2 \
    --split            train \
    --cluster_col      agglo_coimp_subgroup_k30 \
    --clustering_dir   "$CLUSTERING_DIR" \
    --grouping_dir     "$GROUPING_DIR" \
    --out_dir          "$OUT_DIR" \
    --max_prompts_h3   80 \
    --max_pairs_h4     200 \
    --device           cuda

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE: $(date)  Job: ${SLURM_JOB_ID}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sync results locally:"
echo "  rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/h3h4_subcluster/ data/analysis/runD_v2/h3h4_subcluster/"
