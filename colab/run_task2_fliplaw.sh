#!/usr/bin/env bash
# Task 2 (re-run): flip-law measured-delta sweep (131) -> assembly (132) -> whitening (133),
# both concepts. Copies outputs to Drive at the end so they survive a disconnect.
#   !bash colab/run_task2_fliplaw.sh
set -e
cd /content/project 2>/dev/null || cd "$(dirname "$0")/.."
ROOT=data/analysis/runD_v2; SEED=0; TF=0.6; SH=0.1

for CPT in B1_alpha_beta B1_grammar_number; do
  echo "################## $CPT — 131 delta sweep ##################"
  python -u scripts/131_delta_sweep_tier2.py \
      --dump_dir    $ROOT/$CPT/field_dump \
      --corpus      data/prompts/$CPT.jsonl \
      --match_cells $ROOT/$CPT/cells_tier2.csv \
      --out_dir     $ROOT/$CPT \
      --dirs delta,usage \
      --split_seed $SEED --train_frac $TF --shrink $SH

  for FS in pool train; do
    echo "########## $CPT — 132 assembly ($FS) ##########"
    python -u scripts/132_flip_law_assembly.py \
        --dump_dir $ROOT/$CPT/field_dump \
        --cells    $ROOT/$CPT/cells_tier2.csv $ROOT/$CPT/cells_tier2_delta.csv \
        --concept  $CPT --F_split $FS \
        --out_dir  $ROOT/$CPT/flip_law_$FS \
        --heldout_reps 50 \
        --split_seed $SEED --train_frac $TF --shrink $SH
  done

  echo "########## $CPT — 133 whitening ##########"
  python -u scripts/133_whitening_theorem.py \
      --dump_dir $ROOT/$CPT/field_dump \
      --out_dir  $ROOT/$CPT/whitening \
      --concept  $CPT --n_null 50 \
      --split_seed $SEED --train_frac $TF --shrink $SH
done

echo "======== DONE ========"
# persist to Drive so results survive a disconnect
if [ -d /content/drive/MyDrive ]; then
  D=/content/drive/MyDrive/task2_out; mkdir -p "$D"
  for CPT in B1_alpha_beta B1_grammar_number; do
    mkdir -p "$D/$CPT"
    cp -r $ROOT/$CPT/flip_law_pool $ROOT/$CPT/flip_law_train $ROOT/$CPT/whitening "$D/$CPT/" 2>/dev/null || true
    cp $ROOT/$CPT/cells_tier2_delta.csv "$D/$CPT/" 2>/dev/null || true
  done
  echo "✓ outputs copied to Drive/task2_out (safe to disconnect)"
fi
