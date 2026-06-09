#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j109-feat
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/j109-%j.out
#SBATCH --error=logs/j109-%j.out
#
# 109 — Feature alignment: top-K transcoder features aligned with u vs w_res
# per layer. Overlap & per-feature cos correlation. Disjoint top sets + zero
# correlation = "used and read carry different dictionary features". Needs
# transcoder decoders. ~6-8 min real (lazy load per layer).

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/feat_align

echo "========================================================"
echo "  109 — feature alignment u vs w_res"
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

python -u scripts/109_feature_alignment_u_vs_wres.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 13 17 21 24 --topk 30 \
    --out_dir data/analysis/runD_v2/feat_align

echo ""
echo "==== Done: $(date) ===="
