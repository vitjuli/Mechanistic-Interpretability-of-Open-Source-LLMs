#!/usr/bin/env bash
# Sync everything needed for the 6-pair particles priority run to CSD3.
# Plain rsync (no SSH multiplexing -> reliable, will ask password+TOTP per transfer).
# Usage:  bash scripts/sync_particles_pairs.sh
set -e
cd "$(dirname "$0")/.."
HOST=iv294@login.hpc.cam.ac.uk
REMOTE=/rds/user/iv294/hpc-work/thesis/project

echo "=== [1/3] scripts (fixed 154 + slicer + checker) ==="
rsync -avz \
  scripts/make_particle_pairs.py scripts/154_dump_realized_writes.py \
  scripts/119_capture_field_dump.py scripts/122_b1_steering_sweep.py \
  scripts/155_realized_write_decomposition.py scripts/preflight_check.py \
  "$HOST:$REMOTE/scripts/"

echo "=== [2/3] sbatch ==="
rsync -avz jobs/run_particle_pairs_all.sbatch "$HOST:$REMOTE/jobs/"

echo "=== [3/3] 6 sliced pair files ==="
rsync -avz data/prompts/particle_pairs/ "$HOST:$REMOTE/data/prompts/particle_pairs/"

echo "DONE -> scripts + sbatch + 6 pair files on CSD3. Now run on CSD3:  python scripts/preflight_check.py"
