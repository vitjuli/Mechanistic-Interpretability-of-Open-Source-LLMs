#!/bin/bash
#SBATCH -J decode_chk
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -o logs/decode-%j.out
#SBATCH -e logs/decode-%j.out
set -uo pipefail
module purge
module load rhel8/default-amp 2>/dev/null || true
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate
export ATEN_CPU_CAPABILITY=avx2 PYTHONPATH="scripts:src/transcoder:src:${PYTHONPATH:-}"
mkdir -p logs; date
# eg_V2: delta gives garbage by intact (0.21); usage clean (1.0). Decode top-8 to SEE the tokens.
P=data/analysis/runD_v2/particle_pairs/electron_vs_photon
python -u scripts/steering_decode_check.py \
  --dump $P/field_dump_119 \
  --prompts data/prompts/particle_pairs/particles_electron_vs_photon.jsonl \
  --steering_csv $P/steering_delta/steering_sweep_tier1.csv \
  --layer 24 --c 16 --n 8
echo "=== done $(date) ==="
