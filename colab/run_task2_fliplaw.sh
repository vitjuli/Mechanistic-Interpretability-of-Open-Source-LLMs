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
      --model_name Qwen/Qwen3-4B \
      --split_seed $SEED --train_frac $TF --shrink $SH

  echo "########## $CPT — continuity gate (usage: 131 vs 122) ##########"
  # HARD GATE: same model, same split -> usage flips must reproduce 122 within +-0.03.
  # set -e stops the whole run if this fails, so no delta numbers get assembled from a bad sweep.
  python -u scripts/check_continuity_131_122.py \
      --cells_122   $ROOT/$CPT/cells_tier2.csv \
      --cells_131   $ROOT/$CPT/cells_tier2_delta.csv \
      --tol 0.03 --concept $CPT

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

  # 133 is model-free (numpy over the dump) and its outputs already match the thesis,
  # so it is OFF by default here. Re-enable with: RUN_133=1 bash colab/run_task2_fliplaw.sh
  if [ "${RUN_133:-0}" = "1" ]; then
    echo "########## $CPT — 133 whitening ##########"
    python -u scripts/133_whitening_theorem.py \
        --dump_dir $ROOT/$CPT/field_dump \
        --out_dir  $ROOT/$CPT/whitening \
        --concept  $CPT --n_null 50 \
        --split_seed $SEED --train_frac $TF --shrink $SH
  else
    echo "########## $CPT — 133 whitening SKIPPED (RUN_133=1 to enable) ##########"
  fi

  # Persist THIS concept before starting the next one: if the second concept fails the gate,
  # set -e kills the script and a copy that only ran at the end would lose the first one.
  if [ -d /content/drive/MyDrive ]; then
    D=/content/drive/MyDrive/task2_out/$CPT; mkdir -p "$D"
    cp -r $ROOT/$CPT/flip_law_pool $ROOT/$CPT/flip_law_train "$D/" 2>/dev/null || true
    cp -r $ROOT/$CPT/whitening "$D/" 2>/dev/null || true
    cp $ROOT/$CPT/cells_tier2_delta.csv "$D/" 2>/dev/null || true
    echo "✓ $CPT copied to Drive/task2_out (safe to disconnect)"
  fi
done

echo "======== DONE ========"
