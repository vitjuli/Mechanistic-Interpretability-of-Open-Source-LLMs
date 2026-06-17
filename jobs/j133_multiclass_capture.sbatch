#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j133-mcap
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/j133-%j.out
#SBATCH --error=logs/j133-%j.out
#
# 133 — Multiclass field-dump capture. K-class generalization of 119:
# stores K per-class gradients d(logit_c)/dh per layer instead of a single
# d(logit_β-logit_α)/dh. Lets the atlas reconstruct u under any usage formula
# (mean_others, max_others, softmax) post-hoc by linear combination.
#
# Default: physics_internal_candidate_selection_v2, 4 single-token classes
# (electron, neutron, photon, proton) subsampled to 80 each = 320 prompts.
# Cost: 320 prompts × 4 backward = 1280 backward passes; expected ~15-25 min
# on A100 fp32. Disk: ~K × 0.4 GB = ~1.6 GB.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/particles4_raw/field_dump

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="src/transcoder:src:${PYTHONPATH:-}"

echo "==== 133: multiclass field-dump capture (K=4) ===="
date

python -u scripts/133_capture_field_dump_multiclass.py \
    --corpus data/prompts/physics_internal_candidate_selection_v2_train.jsonl \
    --out_dir data/analysis/runD_v2/particles4_raw/field_dump \
    --classes electron neutron photon proton \
    --per_class 80

echo ""
echo "==== Done: $(date) ===="
