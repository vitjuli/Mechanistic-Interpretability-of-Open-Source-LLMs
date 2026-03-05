#!/bin/bash
# Submit all jobs with dependencies (CPU-only version)
# Run this script from the project directory

cd ~/rds/hpc-work/thesis/project

echo "=== Submitting Full Pipeline (CPU-only) ==="
echo ""

# Step 1: Generate prompts (no dependency)
JOB1=$(sbatch --parsable jobs/01_generate_prompts.sh)
echo "Step 1 - Generate Prompts:  Job $JOB1 (~10 min)"

# Step 2: Baseline (depends on Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/02_run_baseline.sh)
echo "Step 2 - Run Baseline:      Job $JOB2 (after $JOB1, ~4 hrs)"

# Step 3: Capture activations (depends on Step 1)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/03_capture_activations.sh)
echo "Step 3 - Capture Acts:      Job $JOB3 (after $JOB1, ~6 hrs)"

# Step 4: Extract features (depends on Step 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 jobs/04_extract_features.sh)
echo "Step 4 - Extract Features:  Job $JOB4 (after $JOB3, ~6 hrs)"

# Step 5: Attribution graph (depends on Step 4)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/06_attribution_graph.sh)
echo "Step 5 - Attribution Graph: Job $JOB5 (after $JOB4, ~8 hrs)"

# Step 6: Interventions (depends on Step 5)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 jobs/07_interventions.sh)
echo "Step 6 - Interventions:     Job $JOB6 (after $JOB5, ~8 hrs)"

# Step 7: Figures (depends on Step 6 and Step 2)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB6,afterok:$JOB2 jobs/08_figures.sh)
echo "Step 7 - Generate Figures:  Job $JOB7 (after $JOB6, $JOB2, ~30 min)"

echo ""
echo "=== Full Pipeline Submitted (CPU-only) ==="
echo ""
echo "Note: CPU inference is slower than GPU."
echo "Expected total runtime: ~24-36 hours"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/<script>_<jobid>.out"
