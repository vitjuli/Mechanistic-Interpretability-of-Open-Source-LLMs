#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=fliplaw131
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fliplaw-%j.out
#SBATCH --error=logs/fliplaw-%j.out
#
# Flip-law completion:
#   131 (GPU) delta+usage tier-2 sweep  ->  cells_tier2_delta.csv
#   132 (CPU) end-to-end assembly       ->  flip_law_{pool,train}/numbers_for_thesis.json
#
# Split params VERIFIED against the real 122 runs (jobs/B1_*_phase3.sbatch pass no
# --split_seed/--train_frac/--shrink -> defaults): seed=0, frac=0.6, shrink=0.1.
# sigma = std(H_train @ w_res), so matching these three keeps w_res/sigma identical.

set -uo pipefail
mkdir -p logs

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2

ROOT=data/analysis/runD_v2
SPLIT_SEED=0
TRAIN_FRAC=0.6
SHRINK=0.1

for CPT in B1_alpha_beta B1_grammar_number; do
  echo "=== 131 delta sweep: $CPT (GPU) ==="
  python -u scripts/131_delta_sweep_tier2.py \
      --dump_dir    $ROOT/$CPT/field_dump \
      --corpus      data/prompts/$CPT.jsonl \
      --match_cells $ROOT/$CPT/cells_tier2.csv \
      --out_dir     $ROOT/$CPT \
      --dirs delta,usage \
      --split_seed $SPLIT_SEED --train_frac $TRAIN_FRAC --shrink $SHRINK

  echo "=== 132 assembly: $CPT (CPU, in-sample F + held-out F) ==="
  for FS in pool train; do
    python -u scripts/132_flip_law_assembly.py \
        --dump_dir $ROOT/$CPT/field_dump \
        --cells    $ROOT/$CPT/cells_tier2.csv $ROOT/$CPT/cells_tier2_delta.csv \
        --concept  $CPT --F_split $FS \
        --out_dir  $ROOT/$CPT/flip_law_$FS \
        --heldout_reps 50 \
        --split_seed $SPLIT_SEED --train_frac $TRAIN_FRAC --shrink $SHRINK
  done

  echo "=== 133 whitening theorem: $CPT (CPU) ==="
  python -u scripts/133_whitening_theorem.py \
      --dump_dir $ROOT/$CPT/field_dump \
      --out_dir  $ROOT/$CPT/whitening \
      --concept  $CPT \
      --split_seed $SPLIT_SEED --train_frac $TRAIN_FRAC --shrink $SHRINK
done

echo "DONE. flip-law: $ROOT/*/flip_law_*/numbers_for_thesis.json"
echo "      whitening: $ROOT/*/whitening/whitening_numbers.json"
