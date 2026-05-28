#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=probe-runD-merge
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe-runD-merge-%j.out
#SBATCH --error=logs/probe-runD-merge-%j.out
#
# runD Step 4: Merge the three ablation part CSVs into one.
#
# Reads:
#   data/results/interventions/physics_decay_type_probe_v2/runD_p1/intervention_ablation_*.csv
#   data/results/interventions/physics_decay_type_probe_v2/runD_p2/intervention_ablation_*.csv
#   data/results/interventions/physics_decay_type_probe_v2/runD_p3/intervention_ablation_*.csv
#
# Writes:
#   data/results/interventions/physics_decay_type_probe_v2/runD/intervention_ablation_*.csv
#
# This job is submitted automatically by submit_runD_pipeline.sh with
# --dependency=afterok:<abl_p1>:<abl_p2>:<abl_p3>
# Do NOT submit manually unless all three abl parts have completed successfully.
#

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe_v2 — RunD Merge"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
echo "========================================================"

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export PYTHONUNBUFFERED=1

# Verify all three parts exist before merging
IDIR="data/results/interventions/physics_decay_type_probe_v2"
for part in runD_p1 runD_p2 runD_p3; do
    f="$IDIR/$part/intervention_ablation_physics_decay_type_probe_v2.csv"
    [ -f "$f" ] || { echo "ERROR: Missing $f"; exit 1; }
    echo "  OK: $f  ($(wc -l < $f) lines)"
done
echo ""

echo "==== Merging ablation parts ===="
python scripts/merge_runD.py

echo ""
echo "========================================================"
echo "  Done: $(date)  Job: ${SLURM_JOB_ID}"
echo "========================================================"
