#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=probe-runD-v2-merge
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe-runD-v2-merge-%j.out
#SBATCH --error=logs/probe-runD-v2-merge-%j.out
#
# runD v2 Step 4: Merge the three v2 ablation part CSVs into one.
# Reads from runD_v2_p1/p2/p3, writes to runD_v2/.
# This job is submitted automatically by submit_runD_v2_pipeline.sh.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD v2 Merge"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export PYTHONUNBUFFERED=1

IDIR="data/results/interventions/physics_decay_type_probe_v2"
for part in runD_v2_p1 runD_v2_p2 runD_v2_p3; do
    f="$IDIR/$part/intervention_ablation_physics_decay_type_probe_v2.csv"
    [ -f "$f" ] || { echo "ERROR: Missing $f"; exit 1; }
    echo "  OK: $f  ($(wc -l < $f) lines)"
done
echo ""

echo "==== Merging v2 ablation parts ===="
python scripts/merge_runD_v2.py

echo ""
echo "========================================================"
echo "  Done: $(date)  Job: ${SLURM_JOB_ID}"
echo "========================================================"
