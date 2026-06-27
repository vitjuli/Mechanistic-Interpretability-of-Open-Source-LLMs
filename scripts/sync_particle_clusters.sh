#!/usr/bin/env bash
# Sync everything the particle-clusters GPU job needs to CSD3 (one rsync -> one auth).
# scripts 04/06/07 + config + corpus + the sbatch. Uses -R to preserve paths into subdirs.
# Usage:  bash scripts/sync_particle_clusters.sh
set -e
cd "$(dirname "$0")/.."
HOST=iv294@login.hpc.cam.ac.uk
REMOTE=/rds/user/iv294/hpc-work/thesis/project

rsync -avzR \
  scripts/04_extract_transcoder_features.py \
  scripts/06_build_attribution_graph.py \
  scripts/07_run_interventions.py \
  configs/experiment_config.yaml \
  data/prompts/physics_internal_candidate_selection_v2_train.jsonl \
  jobs/run_particle_clusters_gpu.sbatch \
  "$HOST:$REMOTE/"

echo "DONE -> 04/06/07 + config + corpus + sbatch on CSD3"
