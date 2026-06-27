#!/bin/bash
#SBATCH -J part_27c
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH -o logs/part27c-%j.out
#SBATCH -e logs/part27c-%j.out
#
# Particle-cluster 27c: ablate each co-importance cluster, measure LENGTH & ROTATION of
# delta(photon-neutron) vs same-size null. Corpus = neutron_vs_photon (delta = the axis clusters encode).
# carrier_geom_core loads model+transcoders itself (works; ran for decay 27c). n_null=5 (NOT 20 -> avoids timeout).
# CAVEAT for thesis: clusters built on full multiclass (correct-vs-incorrect); delta here = photon-neutron pair.
set -uo pipefail
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="scripts:src/transcoder:src:${PYTHONPATH:-}"
mkdir -p logs; nvidia-smi || true; date

PAIR=data/prompts/particle_pairs/particles_neutron_vs_photon.jsonl
python -u scripts/27c_particle_crossmatrix.py \
  --membership data/results/clustering_particles/cluster_membership_particles.csv \
  --pool data/results/clustering_particles/particle_pool_by_layer.json \
  --prompts $PAIR \
  --label_key _pair_class --pos_value 1 \
  --want_u --n_null 5 \
  --out particle_crossmatrix.csv
echo "=== done $(date) ==="
