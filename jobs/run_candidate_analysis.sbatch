#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cand-analysis
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cand-analysis-%j.out
#SBATCH --error=logs/cand-analysis-%j.out

# ── Internal candidate-state analysis: scripts 41-43 ─────────────────────────
#
# Prerequisites (transcoder features already extracted by run_internal_v2_pipeline):
#   data/results/transcoder_features/layer_{10..25}/
#     physics_internal_candidate_selection_v2_train_top_k_indices.npy
#     physics_internal_candidate_selection_v2_train_top_k_values.npy
#
# Outputs:
#   data/results/internal_candidate_analysis/physics_internal_candidate_selection_v2/
#     candidate_feature_table.csv
#     candidate_feature_summary_by_layer.csv
#     candidate_feature_summary_by_particle.csv
#     top_candidate_features.json
#     per_prompt_candidate_scores.csv
#     report.md
#     candidate_trajectory_by_layer.csv
#     candidate_rank_evolution.csv
#     report_trajectory.md
#     candidate_clustering_metrics.csv
#     candidate_clustering_report.md
#     *.png (trajectory and clustering plots)
#
# sbatch jobs/run_candidate_analysis.sbatch

set -euo pipefail

BEHAVIOUR=physics_internal_candidate_selection_v2
SPLIT=train

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Candidate state analysis — Scripts 41/42/43"
echo "  Job ${SLURM_JOB_ID}  Node $(hostname)  $(date)"
echo "========================================================"
python --version

mkdir -p logs data/results/internal_candidate_analysis/${BEHAVIOUR}

# ── Pre-flight: verify transcoder features exist ──────────────────────────────
MISSING=0
for L in $(seq 10 25); do
    IDX="data/results/transcoder_features/layer_${L}/${BEHAVIOUR}_${SPLIT}_top_k_indices.npy"
    if [ ! -f "$IDX" ]; then
        echo "[WARN] Missing: $IDX"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "[ERROR] $MISSING layers missing transcoder features."
    echo "  Run the full pipeline first: sbatch jobs/run_internal_v2_pipeline.sbatch"
    echo "  Or re-run script 04 specifically:"
    echo "  python -u scripts/04_extract_transcoder_features.py --behaviour $BEHAVIOUR --split $SPLIT"
    exit 1
fi
echo "  [OK] Transcoder features verified for all 16 layers"

# ── Script 41: T/C/B candidate state analysis ────────────────────────────────
echo ""
echo "████  SCRIPT 41: Internal candidate-state analysis  $(date)"

python -u scripts/41_internal_candidate_state_analysis.py \
    --behaviour    "$BEHAVIOUR" \
    --split        "$SPLIT" \
    --feature_set  circuit_only all_graph \
    && RC41=0 || RC41=$?

echo "Script 41 exit code: $RC41  ($(date))"
[ $RC41 -ne 0 ] && echo "[WARN] Script 41 failed — check output"

if [ -f "data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_feature_table.csv" ]; then
    python3 -c "
import pandas as pd
df = pd.read_csv('data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_feature_table.csv')
strong = df['strong_candidate_feature'].sum()
partial = df['partial_candidate_feature'].sum()
ordering = df['ordering_T_gt_C_gt_B'].sum()
print(f'  Features with T>C>B ordering: {ordering}/{len(df)}')
print(f'  Strong candidate features (FDR sig): {strong}')
print(f'  Partial candidate features: {partial}')
print()
print('  Top 5 by competitor_presence:')
print(df.sort_values('competitor_presence', ascending=False).head(5)[
    ['feature_id','particle','layer','target_mean','competitor_mean','background_mean','internal_presence_ratio']
].to_string(index=False))
"
fi

# ── Script 42: Trajectory analysis ───────────────────────────────────────────
echo ""
echo "████  SCRIPT 42: Candidate trajectory analysis  $(date)"

python -u scripts/42_candidate_trajectory_analysis.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    && RC42=0 || RC42=$?

echo "Script 42 exit code: $RC42  ($(date))"
[ $RC42 -ne 0 ] && echo "[WARN] Script 42 failed"

if [ -f "data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_trajectory_by_layer.csv" ]; then
    python3 -c "
import pandas as pd
df = pd.read_csv('data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_trajectory_by_layer.csv')
print('  Trajectory (layer | rank_acc | margin | entropy):')
for _, r in df.iterrows():
    print(f'    L{int(r[\"layer\"]):2d} | {r[\"candidate_rank_accuracy\"]:.3f} | {r[\"mean_margin\"]:.3f} | {r[\"mean_entropy\"]:.3f}')
"
fi

# ── Script 43: Clustering ─────────────────────────────────────────────────────
echo ""
echo "████  SCRIPT 43: Candidate clustering  $(date)"

python -u scripts/43_candidate_clustering.py \
    --behaviour    "$BEHAVIOUR" \
    --split        "$SPLIT" \
    --layer_groups all early mid late retrieval circuit_layers \
    && RC43=0 || RC43=$?

echo "Script 43 exit code: $RC43  ($(date))"
[ $RC43 -ne 0 ] && echo "[WARN] Script 43 failed"

if [ -f "data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_clustering_metrics.csv" ]; then
    python3 -c "
import pandas as pd
df = pd.read_csv('data/results/internal_candidate_analysis/${BEHAVIOUR}/candidate_clustering_metrics.csv')
sub = df[(df['k']==4) & (df['label_type']=='particle_identity')].sort_values('ari', ascending=False)
print('  k=4 clustering vs particle identity:')
print(sub[['layer_group','ari','nmi','purity','silhouette']].to_string(index=False))
"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  DONE — $(date)  Job $SLURM_JOB_ID"
echo "========================================================"
ls -lh data/results/internal_candidate_analysis/${BEHAVIOUR}/ 2>/dev/null || true

echo ""
echo "  ── Rsync outputs to local Mac ──"
BASE="iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
ANALYSIS_DIR="data/results/internal_candidate_analysis"
echo ""
echo "  # Feature files (needed for 41/42/43 locally):"
echo "  for L in {10..25}; do"
echo "    rsync -av \"${BASE}/data/results/transcoder_features/layer_\${L}/${BEHAVIOUR}_${SPLIT}_top_k_\"{indices,values}\".npy\" \\"
echo "      \"data/results/transcoder_features/layer_\${L}/\""
echo "  done"
echo ""
echo "  # Analysis outputs:"
echo "  rsync -av \"${BASE}/${ANALYSIS_DIR}/${BEHAVIOUR}/\" \"${ANALYSIS_DIR}/${BEHAVIOUR}/\""
echo "========================================================"
