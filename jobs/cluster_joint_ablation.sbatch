#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=cja_probe
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cja_probe-%j.out
#SBATCH --error=logs/cja_probe-%j.out

# ── Cluster joint ablation: physics_decay_type_probe ──────────────────────
# Ablates all features in each co-importance Louvain cluster simultaneously
# (multi-layer hook, single forward pass per cluster per prompt).
#
# Input:  data/results/clustering/cluster_labels.csv  (coimp_louvain column)
#         data/results/grouping/feature_prompt_contributions.csv
#         data/prompts/physics_decay_type_probe_train.jsonl
#
# Output: data/results/cluster_joint_ablation/
#           joint_ablation_physics_decay_type_probe_train.csv
#           synergy_summary.csv
#           interaction_by_answer.csv
#           per_prompt_interaction.csv
#           analytic_bounds.csv
#
# After job: rsync output to local, then run 27b locally for plots + dashboard JSON.
#
# sbatch jobs/cluster_joint_ablation.sbatch

set -euo pipefail

BEHAVIOUR=physics_decay_type_probe
SPLIT=train

cd /rds/user/iv294/hpc-work/thesis/project

module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Cluster Joint Ablation: ${BEHAVIOUR} ${SPLIT}"
echo "  Job $SLURM_JOB_ID  Node $(hostname)  $(date)"
echo "========================================================"

python --version
nvidia-smi | head -6 || true

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "==== Pre-flight ===="
PROMPTS="data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"
CLUSTER_CSV="data/results/clustering/cluster_labels.csv"
CONTRIB_CSV="data/results/grouping/feature_prompt_contributions.csv"

[ -f "$PROMPTS" ]     && echo "  Prompts:      $(wc -l < $PROMPTS) lines — OK"  || { echo "  [FATAL] $PROMPTS not found"; exit 1; }
[ -f "$CLUSTER_CSV" ] && echo "  Cluster CSV:  OK"                               || { echo "  [FATAL] $CLUSTER_CSV not found"; exit 1; }
[ -f "$CONTRIB_CSV" ] && echo "  Contrib CSV:  $(wc -l < $CONTRIB_CSV) rows — OK" || { echo "  [FATAL] $CONTRIB_CSV not found"; exit 1; }

mkdir -p data/results/cluster_joint_ablation logs

echo "  Pre-flight OK at $(date)"

# ── Section A: Joint ablation (all clusters × all prompts) ────────────────────
echo ""
echo "████  JOINT ABLATION — all 11 clusters × 470 prompts  $(date)"

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour "$BEHAVIOUR" \
    --split     "$SPLIT" \
    --clusters  all \
    --device    cuda

RC=$?
echo ""
echo "Joint ablation exit code: $RC  ($(date))"
[ $RC -ne 0 ] && { echo "[FATAL] Joint ablation failed"; exit 1; }

# Quick sanity check on output
python3 -c "
import pandas as pd
from pathlib import Path
p = Path('data/results/cluster_joint_ablation/joint_ablation_physics_decay_type_probe_train.csv')
if not p.exists():
    print('[WARN] Output CSV not found after ablation')
    exit()
df = pd.read_csv(p)
print(f'Output: {len(df)} rows, {df.cluster_id.nunique()} clusters, {df.prompt_idx.nunique()} prompts')
print(df.groupby('cluster_id').agg(
    n=('prompt_idx','count'),
    mean_joint_eff=('joint_effect','mean'),
    mean_indiv_sum=('individual_sum','mean'),
    mean_ratio=('interaction_ratio','mean'),
    flip_rate=('sign_flipped_joint','mean'),
).round(3).to_string())
" || true

# ── Section B: Analysis (runs on the output) ──────────────────────────────────
echo ""
echo "████  ANALYSIS (27b)  $(date)"

python -u scripts/27b_analyse_joint_ablation.py

echo ""
echo "========================================================"
echo "  DONE — Job $SLURM_JOB_ID — $(date)"
echo ""
echo "  Key outputs:"
ls -lh data/results/cluster_joint_ablation/ 2>/dev/null || true
echo ""
echo "  After rsync to local:"
echo "    python scripts/27b_analyse_joint_ablation.py   (re-runs for dashboard JSON)"
echo "  Then reload dashboard_probe to see the Cluster Semantics tab update."
echo "========================================================"
