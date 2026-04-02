#!/bin/bash
#SBATCH --job-name=qwen3_full_pipeline
#SBATCH --partition=ampere
#SBATCH --account=DAMTP-SL2-GPU    # Adjust to your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # Single A100 sufficient
#SBATCH --mem=64G
#SBATCH --time=48:00:00            # 2 days for full analysis
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@cam.ac.uk

# Complete pipeline from scratch to final figures
# Model: Qwen3-4B-Instruct-2507
# Behaviours: grammar_agreement, factual_recall, sentiment_continuation, arithmetic
#
# WARNING: This is computationally intensive (48h walltime)
# For faster results, use individual SLURM scripts in parallel
#
# Usage: sbatch slurm_scripts/pipeline_full.slurm

set -e  # Exit on error

echo "=========================================="
echo "FULL MECHANISTIC INTERPRETABILITY PIPELINE"
echo "Model: Qwen3-4B-Instruct-2507"
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Load CSD3 modules
module purge
module load rhel8/default-amp
module load python/3.10.5-gcc-11.2.0
module load cuda/12.1
module load cudnn/8.9_cuda-12.1

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Set HuggingFace cache
export HF_HOME=/rds/project/rds-YOUR_PROJECT/hf_cache  # Update with your RDS project
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME/torch

mkdir -p logs

# ===========================================================================
# Phase 0: Generate Prompts
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 0: Generating prompts (4 behaviours)"
echo "=========================================="
python scripts/01_generate_prompts.py

# ===========================================================================
# Phase 1: Baseline Measurements
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 1: Running baseline measurements"
echo "=========================================="
python scripts/02_run_baseline.py --all --split train
python scripts/02_run_baseline.py --all --split test

# ===========================================================================
# Phase 2: Activation Capture
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 2: Capturing activations"
echo "=========================================="
python scripts/03_capture_activations.py --all --split train

# ===========================================================================
# Phase 3: Extract Transcoder Features
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 3: Extracting transcoder features"
echo "=========================================="
python scripts/04_extract_transcoder_features.py \
    --all \
    --model_size 4b \
    --top_k 50

# ===========================================================================
# Phase 4: Attribution Graphs
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 4: Building attribution graphs"
echo "=========================================="
python scripts/06_build_attribution_graph.py \
    --all \
    --split train \
    --n_prompts 30

# ===========================================================================
# Phase 5: Intervention Experiments
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 5: Running causal interventions"
echo "=========================================="
python scripts/07_run_interventions.py \
    --all \
    --split test \
    --intervention all \
    --layer 15 \
    --n_samples 30

# ===========================================================================
# Phase 6: Generate Visualizations
# ===========================================================================
echo ""
echo "=========================================="
echo "Phase 6: Generating visualizations"
echo "=========================================="
python scripts/08_generate_figures.py --all

echo ""
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Results saved to:"
echo "  - Baseline: data/results/"
echo "  - Activations: data/activations/"
echo "  - Transcoder features: data/results/transcoder_features/"
echo "  - Attribution graphs: data/results/attribution_graphs/"
echo "  - Intervention results: data/results/interventions/"
echo "  - Figures: figures/"
echo "=========================================="
