#!/bin/bash
# Submit the full RunD pipeline with automatic SLURM dependencies.
#
# Dependency graph:
#
#   s04 (GPU) → s06 (CPU) → abl_p1 (GPU) ─┐
#                          → abl_p2 (GPU) ─┤ → merge (CPU) → semantics (CPU)
#                          → abl_p3 (GPU) ─┘
#
# All jobs use --dependency=afterok: jobs only start if their predecessor
# exited cleanly (code 0). If any step fails the chain stops automatically.
#
# Usage (run from CSD3 project root):
#   cd /rds/user/iv294/hpc-work/thesis/project
#   git pull && mkdir -p logs
#   bash jobs/submit_runD_pipeline.sh
#
# To submit only from a given step (e.g. if s04 already done), edit and
# comment out earlier sbatch lines, then set JOB_S04 manually to the
# completed job ID:
#   JOB_S04=12345678   # already done
#   JOB_S06=$(sbatch --parsable --dependency=afterok:$JOB_S04 ...)
#

set -euo pipefail
cd /rds/user/iv294/hpc-work/thesis/project
mkdir -p logs

echo "============================================================"
echo "  Submitting RunD pipeline"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Step 1: feature extraction (GPU, ~3-4h) ──────────────────────────────────
JOB_S04=$(sbatch --parsable jobs/run_probe_runD_s04.sbatch)
echo "  [1] s04          → job $JOB_S04   (GPU ~3-4h)"

# ── Step 2: attribution graph (CPU, ~1h) ─────────────────────────────────────
JOB_S06=$(sbatch --parsable \
    --dependency=afterok:${JOB_S04} \
    jobs/run_probe_runD_s06.sbatch)
echo "  [2] s06          → job $JOB_S06   (CPU ~1h, after $JOB_S04)"

# ── Step 3: ablation — three parts in parallel (GPU, ~20h each) ──────────────
JOB_ABL1=$(sbatch --parsable \
    --dependency=afterok:${JOB_S06} \
    jobs/run_probe_runD_abl_p1.sbatch)
echo "  [3a] abl_p1      → job $JOB_ABL1  (GPU ~20h, after $JOB_S06)"

JOB_ABL2=$(sbatch --parsable \
    --dependency=afterok:${JOB_S06} \
    jobs/run_probe_runD_abl_p2.sbatch)
echo "  [3b] abl_p2      → job $JOB_ABL2  (GPU ~20h, after $JOB_S06)"

JOB_ABL3=$(sbatch --parsable \
    --dependency=afterok:${JOB_S06} \
    jobs/run_probe_runD_abl_p3.sbatch)
echo "  [3c] abl_p3      → job $JOB_ABL3  (GPU ~20h, after $JOB_S06)"

# ── Step 4: merge parts (CPU, ~5 min) ────────────────────────────────────────
JOB_MERGE=$(sbatch --parsable \
    --dependency=afterok:${JOB_ABL1}:${JOB_ABL2}:${JOB_ABL3} \
    jobs/run_probe_runD_merge.sbatch)
echo "  [4] merge        → job $JOB_MERGE  (CPU ~5min, after all 3 abl)"

# ── Step 5: grouping + clustering (CPU, ~3h) ─────────────────────────────────
JOB_SEM=$(sbatch --parsable \
    --dependency=afterok:${JOB_MERGE} \
    jobs/run_probe_runD_semantics.sbatch)
echo "  [5] semantics    → job $JOB_SEM   (CPU ~3h, after $JOB_MERGE)"

echo ""
echo "============================================================"
echo "  All jobs queued. Monitor with:"
echo "    squeue -u iv294"
echo "  Or watch the chain:"
echo "    squeue -u iv294 -o '%.10i %.9P %.20j %.8T %.12M %.5D %R'"
echo ""
echo "  Job IDs summary:"
echo "    s04      = $JOB_S04"
echo "    s06      = $JOB_S06"
echo "    abl_p1   = $JOB_ABL1"
echo "    abl_p2   = $JOB_ABL2"
echo "    abl_p3   = $JOB_ABL3"
echo "    merge    = $JOB_MERGE"
echo "    semantics= $JOB_SEM"
echo ""
echo "  After semantics completes, sync to local:"
echo "    rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runD/ \\"
echo "          data/analysis/runD/"
echo ""
echo "  Then inspect data/analysis/runD/clustering_full/ to choose k,"
echo "  and run scripts 30-34 locally."
echo "============================================================"
