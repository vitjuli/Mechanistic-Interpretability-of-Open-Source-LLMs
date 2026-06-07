#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=j90-fmode
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/j90-%j.out
#SBATCH --error=logs/j90-%j.out
#
# 90 — Forced-mode alignment: does u rotate toward w_res when the shortcut is closed?
# (A) per-layer cos(u, w_res) in BASE vs FORCED few-shot regime (balanced, leak-free).
# (B) intact-flip steering on 4 layers in forced regime vs shuffled+random nulls.
# (C) cue-subset alignment in base regime (excluding mass_A + emission cues).
# ~1.5 h. cue_csv has prompt_idx + cue_type joined from cluster_joint_ablation_subgroup.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/forced_mode

echo "========================================================"
echo "  90 — forced-mode alignment (rotation + intact steering + cue subset)"
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

echo "Python: $(which python)"
nvidia-smi || true
date

CUE_CSV="data/analysis/runD_v2/cluster_joint_ablation_subgroup/per_prompt_cluster_effect_matrix.csv"
[ ! -f "$CUE_CSV" ] && { echo "WARNING: cue_csv missing ($CUE_CSV) — part C will be skipped"; CUE_CSV=""; }

# ── smoke (3-4 min) ───────────────────────────────────────────────────────
echo ""
echo "==== SMOKE (L16 21 35, steer L21, 5 targets) ===="
date
python -u scripts/90_forced_mode_alignment.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --layers 16 21 35 \
    --steer_layers 21 \
    --max_targets 5 \
    --n_random 1 \
    --out_dir data/analysis/runD_v2/forced_mode/smoke
SMOKE_RC=$?
[ $SMOKE_RC -ne 0 ] && { echo "SMOKE FAILED ($SMOKE_RC)"; exit 1; }
echo "==== smoke OK ===="

# ── full run (all 36 layers, steer L16/21/24/35, cue_csv if present) ──────
echo ""
echo "==== FULL RUN (all layers, steer 16 21 24 35, with cue_csv) ===="
date
python -u scripts/90_forced_mode_alignment.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --cue_csv "$CUE_CSV" \
    --out_dir data/analysis/runD_v2/forced_mode

echo ""
echo "==== Done: $(date) ===="
