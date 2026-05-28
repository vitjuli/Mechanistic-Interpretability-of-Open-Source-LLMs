#!/bin/bash
# Submit the RunD v2 pipeline: ablation with --ablation_sign all (227 features).
#
# s04 and s06 are ALREADY DONE — skip them.
# This script re-runs: abl_p1/p2/p3 (parallel) → merge → semantics
#
# Dependency graph:
#
#   abl_v2_p1 (GPU) ─┐
#   abl_v2_p2 (GPU) ─┤ → merge_v2 (CPU) → semantics_v2 (CPU)
#   abl_v2_p3 (GPU) ─┘
#
# Usage (run from CSD3 project root):
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs
#   bash jobs/submit_runD_v2_pipeline.sh
#
# After semantics completes, sync to local:
#   rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD_v2/ data/analysis/runD_v2/
#   rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/results/interventions/physics_decay_type_probe_v2/runD_v2/ data/results/interventions/physics_decay_type_probe_v2/runD_v2/
#
# Then run locally:
#   K=14
#   python scripts/30_carrier_stability_recovery.py --k $K --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability
#   python scripts/32_permutation_stability.py      --k $K --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability
#   python scripts/34_polarity_consistency.py       --k $K --grouping_dir data/analysis/runD_v2/grouping --clustering_dir data/analysis/runD_v2/clustering_full --out_dir data/analysis/runD_v2/carrier_stability

set -euo pipefail
cd /rds/user/iv294/hpc-work/thesis/project
mkdir -p logs

echo "============================================================"
echo "  Submitting RunD v2 pipeline (ablation_sign=all, 227 features)"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Ablation — three parts in parallel (GPU, ~20h each) ──────────────────────
JOB_ABL1=$(sbatch --parsable jobs/run_probe_runD_v2_abl_p1.sbatch)
echo "  [1a] v2_abl_p1   → job $JOB_ABL1  (GPU ~20h)"

JOB_ABL2=$(sbatch --parsable jobs/run_probe_runD_v2_abl_p2.sbatch)
echo "  [1b] v2_abl_p2   → job $JOB_ABL2  (GPU ~20h)"

JOB_ABL3=$(sbatch --parsable jobs/run_probe_runD_v2_abl_p3.sbatch)
echo "  [1c] v2_abl_p3   → job $JOB_ABL3  (GPU ~20h)"

# ── Merge parts (CPU, ~5 min) ─────────────────────────────────────────────────
JOB_MERGE=$(sbatch --parsable \
    --dependency=afterok:${JOB_ABL1}:${JOB_ABL2}:${JOB_ABL3} \
    jobs/run_probe_runD_v2_merge.sbatch)
echo "  [2] v2_merge     → job $JOB_MERGE  (CPU ~5min, after all 3 abl)"

# ── Semantics: scripts 19 → 22 → 23 → 26 (CPU, ~3h) ─────────────────────────
JOB_SEM=$(sbatch --parsable \
    --dependency=afterok:${JOB_MERGE} \
    jobs/run_probe_runD_v2_semantics.sbatch)
echo "  [3] v2_semantics → job $JOB_SEM   (CPU ~3h, after merge)"

echo ""
echo "============================================================"
echo "  All jobs queued. Monitor with:"
echo "    squeue -u iv294"
echo ""
echo "  Job IDs summary:"
echo "    v2_abl_p1   = $JOB_ABL1"
echo "    v2_abl_p2   = $JOB_ABL2"
echo "    v2_abl_p3   = $JOB_ABL3"
echo "    v2_merge    = $JOB_MERGE"
echo "    v2_semantics= $JOB_SEM"
echo ""
echo "  Estimated wall time: ~20h (abl in parallel) + ~3h (semantics)"
echo "  Total GPU budget: ~60h (3 × 20h)"
echo "============================================================"
