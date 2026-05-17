#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=probe-runB-joint
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probe-runB-joint-%j.out
#SBATCH --error=logs/probe-runB-joint-%j.out
#
# physics_decay_type_probe — Run B cluster joint ablation (script 27)
#
# For each of the 12 co-importance Louvain clusters and all 470 prompts:
#   - baseline forward pass
#   - per-layer MLP input extraction + cluster feature zeroing
#   - joint patched forward pass
# Measures joint_effect, individual_sum, interaction_ratio, sign_flipped_joint.
#
# Reads:  data/analysis/runB/clustering/cluster_labels.csv
#         data/analysis/runB/grouping/feature_prompt_contributions.csv
#         data/prompts/physics_decay_type_probe_train.jsonl
# Writes: data/analysis/runB/cluster_joint_ablation/joint_ablation_physics_decay_type_probe_train.csv
#
# Runtime estimate: scripts 19/22/23 ~2 min; script 27 ~1-2 h
#
# Prereqs:
#   - Run B ablation CSV at data/results/interventions/physics_decay_type_probe/runB/
#     (done — SLURM 29432300)
#   - data/ui_offline/20260430-152526_physics_decay_type_probe_train_n108/ present
#   Scripts 19/22/23 run automatically as a CPU preamble in this job.

set -uo pipefail
mkdir -p logs

echo "========================================================"
echo "  physics_decay_type_probe — Run B joint ablation (script 27)"
echo "  Job: ${SLURM_JOB_ID}  Started: $(date)"
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

echo "Python: $(which python)"
python --version
nvidia-smi || true
echo ""

# ── Paths ──────────────────────────────────────────────────────────────────────
BEHAVIOUR=physics_decay_type_probe
SPLIT=train
N_PROMPTS=470

GROUPING_DIR="data/analysis/runB/grouping"
CLUSTERING_DIR="data/analysis/runB/clustering"
JOINT_DIR="data/analysis/runB/cluster_joint_ablation"

UI_RUN="data/ui_offline/20260430-152526_physics_decay_type_probe_train_n108"
ABL_CSV="data/results/interventions/${BEHAVIOUR}/runB/intervention_ablation_${BEHAVIOUR}.csv"

echo "Parameters:"
echo "  BEHAVIOUR:      $BEHAVIOUR"
echo "  N_PROMPTS:      $N_PROMPTS"
echo "  GROUPING_DIR:   $GROUPING_DIR"
echo "  CLUSTERING_DIR: $CLUSTERING_DIR"
echo "  JOINT_DIR:      $JOINT_DIR"
echo "  clusters:       all (12 coimp-Louvain clusters)"
echo ""

# Verify hard prereqs (ablation CSV + UI run)
for f in "$ABL_CSV" "$UI_RUN/graph.json" "data/prompts/${BEHAVIOUR}_${SPLIT}.jsonl"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing prereq: $f"
        exit 1
    fi
done
echo "Hard prereqs found."
echo ""

mkdir -p "$GROUPING_DIR" "$CLUSTERING_DIR" "$JOINT_DIR"

# ── Ensure CPU analysis dependencies are available ────────────────────────────
python -c "import sklearn" 2>/dev/null || pip install scikit-learn -q
python -c "import community" 2>/dev/null || pip install python-louvain -q

# ── Step 19: feature × prompt analysis (CPU, ~1 min) ─────────────────────────
echo "==== [Step 19] feature_prompt_analysis ===="
date
python -u scripts/19_feature_prompt_analysis.py \
    --behaviour    "$BEHAVIOUR" \
    --split        "$SPLIT" \
    --ui_run       "$UI_RUN" \
    --abl_csv      "$ABL_CSV" \
    --grouping_dir "$GROUPING_DIR"
echo "  Step 19 done: $(date)"
echo ""

# ── Step 22: prepare clustering inputs (CPU, ~10 s) ──────────────────────────
echo "==== [Step 22] prepare_clustering_inputs ===="
date
python -u scripts/22_prepare_clustering_inputs.py \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR"
echo "  Step 22 done: $(date)"
echo ""

# ── Step 23: run clustering benchmark (CPU, ~30 s) ────────────────────────────
echo "==== [Step 23] run_clustering_benchmark ===="
date
python -u scripts/23_run_clustering_benchmark.py \
    --clustering_dir "$CLUSTERING_DIR"
echo "  Step 23 done: $(date)"
echo ""

# ── Run script 27 ──────────────────────────────────────────────────────────────
echo "==== [Script 27] Cluster joint ablation ===="
date

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour      "$BEHAVIOUR" \
    --split          "$SPLIT" \
    --clusters       all \
    --n_prompts      "$N_PROMPTS" \
    --grouping_dir   "$GROUPING_DIR" \
    --clustering_dir "$CLUSTERING_DIR" \
    --run_dir        "$JOINT_DIR"

echo ""
echo "==== Script 27 done: $(date) ===="
echo ""

# ── Quick summary from output CSV ──────────────────────────────────────────────
JOINT_CSV="${JOINT_DIR}/joint_ablation_${BEHAVIOUR}_${SPLIT}.csv"
if [ -f "$JOINT_CSV" ]; then
    python -c "
import pandas as pd
df = pd.read_csv('${JOINT_CSV}')
print(f'Joint ablation summary ({len(df)} rows):')
print(df.groupby('cluster_id').agg(
    n=('prompt_idx','count'),
    joint_eff=('joint_effect','mean'),
    indiv_sum=('individual_sum','mean'),
    interact_ratio=('interaction_ratio','mean'),
    sfr=('sign_flipped_joint','mean'),
).round(3).to_string())
" || true
fi

echo ""
echo "========================================================"
echo "  Run B joint ablation done: $(date)"
echo "  Job: ${SLURM_JOB_ID}"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Sync results to local:"
echo "     rsync -avz iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project/data/analysis/runB/cluster_joint_ablation/ data/analysis/runB/cluster_joint_ablation/"
echo ""
echo "  2. Run script 27b locally (generates synergy/interaction CSVs for script 29):"
echo "     python3 scripts/27b_analyse_joint_ablation.py \\"
echo "         --grouping_dir   data/analysis/runB/grouping \\"
echo "         --clustering_dir data/analysis/runB/clustering \\"
echo "         --joint_dir      data/analysis/runB/cluster_joint_ablation"
echo ""
echo "  3. Run script 29 locally:"
echo "     bash scripts/run_runB_pipeline.sh --start 29"
