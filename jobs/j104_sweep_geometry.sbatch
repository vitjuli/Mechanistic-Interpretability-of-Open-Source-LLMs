#!/bin/bash
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --qos=gpu2
#SBATCH --job-name=j104-sg
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j104-%j.out
#SBATCH --error=logs/j104-%j.out
#
# 104 — Single c=4 sweep across all 36 layers + 2D-clouds geometry.
# Five output CSVs feeding final figures of §6:
#   steering_by_layer.csv       — 36 × {w_res, u, random}: intact, mean Δm,
#                                 AUC-A on layer (tautology), AUC-B at L24/L35
#                                 readouts (perturbation reaches the readout?)
#   steering_detail.csv         — per-prompt records
#   geometry_by_layer.csv       — per-layer signed cosines, sigmas, mu-gap
#   clouds_2d.csv               — per-prompt 2D coords in {w_res,γ̄} and {u,w_res}
#   vectors_2d.csv              — unit-vector tips for arrows over the clouds
#
# Honest caveat: AUC-A is tautological (we shift the residual by c·σ along the
# probe axis itself); AUC-B is the substantive readout-arrival metric.

set -uo pipefail
mkdir -p logs data/analysis/runD_v2/sweep104

echo "========================================================"
echo "  104 — c=4 sweep across 36 layers + 2D-clouds geometry"
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
nvidia-smi || true
date

echo ""
echo "==== Sweep + geometry run (c=4, readouts L24 L35, n_eval=24) ===="
date
python -u scripts/104_single_sigma_sweep_geometry.py \
    --prompts data/prompts/physics_decay_type_probe_v2_train.jsonl \
    --c 4 --readouts 24 35 --n_eval 24 \
    --out_dir data/analysis/runD_v2/sweep104

echo ""
echo "==== Done: $(date) ===="
