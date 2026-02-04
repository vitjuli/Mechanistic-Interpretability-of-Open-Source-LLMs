#!/bin/bash
# Full Pipeline Execution Script
# Runs all steps sequentially: prompts → baseline → features → graph → interventions → figures

set -e  # Exit on any error

echo "=========================================="
echo "STARTING FULL PIPELINE (6 STEPS)"
echo "=========================================="
echo "Behaviour: grammar_agreement"
echo "Splits: train (prompts/baseline/features/graph), test (interventions)"
echo ""

# Step 1: Generate prompts
echo "=========================================="
echo "STEP 1: Generate Prompts"
echo "=========================================="
python scripts/01_generate_prompts.py
echo "✓ Step 1 complete"
echo ""

# Step 2: Run baseline
echo "=========================================="
echo "STEP 2: Baseline Evaluation"
echo "=========================================="
python scripts/02_run_baseline.py --behaviour grammar_agreement --split train
echo "✓ Step 2 complete"
echo ""

# Step 3: Extract transcoder features (skip activation capture, directly extract)
echo "=========================================="
echo "STEP 3: Extract Transcoder Features"
echo "=========================================="
python scripts/04_extract_transcoder_features.py --behaviour grammar_agreement --split train
echo "✓ Step 3 complete"
echo ""

# Step 4: Build attribution graph
echo "=========================================="
echo "STEP 4: Build Attribution Graph"
echo "=========================================="
python scripts/06_build_attribution_graph.py --behaviour grammar_agreement --split train --n_prompts 20
echo "✓ Step 4 complete"
echo ""

# Step 5: Run interventions (causal validation)
echo "=========================================="
echo "STEP 5: Run Interventions"
echo "=========================================="
python scripts/07_run_interventions.py --behaviour grammar_agreement --split test --n_interventions 50
echo "✓ Step 5 complete"
echo ""

# Step 6: Generate figures
echo "=========================================="
echo "STEP 6: Generate Figures"
echo "=========================================="
python scripts/08_generate_figures.py --behaviour grammar_agreement
echo "✓ Step 6 complete"
echo ""

echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Results saved in data/results/"
echo "Figures saved in data/figures/"
echo ""
