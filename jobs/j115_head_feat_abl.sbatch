#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j115-head-feat-abl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j115-%j.out
#SBATCH --error=logs/j115-%j.out
#
# 115 — Head→feature ablation bridge. Causal bridge from heads to features:
# ablate each top-K answer-writer / axis-writer / random head at layer L
# (writer_layers = 21, 24) and measure activation drop at feature_layers
# (24, 28, 32, 35) for Q1 features (used+concept) and Q2 features
# (decoded-but-unused), from j112b. Hypothesis: answer-writer heads feed
# Q1 features specifically; axis-writer heads feed Q2 features specifically.
#
# Requires: data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv (j112b)
# Expected runtime: ~25-30 min real. n_eval=24, n_random=8.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/head_feat_ablation

echo "========================================================"
echo "  115 — head→feature ablation bridge"
echo "  Job: ${SLURM_JOB_ID:-NA}  Started: $(date)"
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

export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "Python: $(which python)"
python -c "from transcoder_loader import load_transcoder_set; print('transcoder_loader import OK')"
nvidia-smi || true
date

python -u scripts/115_head_feature_ablation_bridge.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --writer_layers 21 24 \
    --feature_layers 24 28 32 35 \
    --feature_csv data/analysis/runD_v2/quad_multimetric/feature_metrics_full.csv \
    --topk_heads 5 \
    --n_random 8 \
    --n_eval 24 \
    --out_dir data/analysis/runD_v2/head_feat_ablation

echo ""
echo "==== Done: $(date) ===="
