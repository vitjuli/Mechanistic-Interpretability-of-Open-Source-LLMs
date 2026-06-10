#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j112b-multimetric
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j112b-%j.out
#SBATCH --error=logs/j112b-%j.out
#
# 112b — Multi-metric concept-importance + selectivity-vs-attribution
# quadrants over ALL 36 layers. Replaces 112's AUC-only definition of
# 'concept-selective' with 4 measures (auc_global / fire_sel / mi_label /
# engagement) that catch:
#   (1) narrow β-subgroup features (auc≈0.5 but mi/fire non-zero)
#   (2) both-class strong firers (auc≈0.5 but engagement=1.0)
# Plus null-calibrated thresholds (no arbitrary 0.3). Two quadrant
# variants side-by-side (CONSERVATIVE = old AUC-only; EXPANDED = honest
# multi-metric), and an "extra caught" column that quantifies how much
# the original 112 picture undercounted concept-features.
#
# Expected runtime: ~10–13 min real (forward+grad over 538 prompts with
# 36 hooks, then ~5–8 s per layer for transcoder encode + 20 selectivity
# null draws + 10 attribution null draws).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/quad_multimetric

echo "========================================================"
echo "  112b — multimetric quadrants — ALL 36 LAYERS"
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

python -u scripts/112b_feature_quadrants_multimetric.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers -1 \
    --tau 0.05 \
    --n_null_sel 20 \
    --n_null_attr 10 \
    --out_dir data/analysis/runD_v2/quad_multimetric

echo ""
echo "==== Done: $(date) ===="
