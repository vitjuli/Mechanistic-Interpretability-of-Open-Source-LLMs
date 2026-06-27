#!/usr/bin/env bash
# Pull the 6-pair results (small CSVs only, NOT the big .npy dumps) from CSD3 to local.
# Usage:  bash scripts/sync_pairs_results.sh
set -e
cd "$(dirname "$0")/.."
HOST=${SYNC_HOST:-iv294@login.hpc.cam.ac.uk}   # override: SYNC_HOST=iv294@login-q-1.hpc.cam.ac.uk
REMOTE=/rds/user/iv294/hpc-work/thesis/project
DEST=data/analysis/runD_v2/particle_pairs
mkdir -p "$DEST"

echo "=== [1/2] realized_*.csv (delta-rotation, from project root) ==="
rsync -avz "$HOST:$REMOTE/realized_*.csv" "$DEST/"

echo "=== [2/2] steering_sweep_tier*.csv (skip big npy dumps) ==="
rsync -avz --include='*/' --include='steering_sweep_tier*.csv' --exclude='*' \
  "$HOST:$REMOTE/data/analysis/runD_v2/particle_pairs/" "$DEST/"

echo "DONE -> $DEST  (6 realized_*.csv + 6 steering_sweep_tier1.csv)"
