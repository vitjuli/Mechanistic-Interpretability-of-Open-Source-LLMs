#!/usr/bin/env bash
# Sync steering_delta/ (tier1 CSVs) for all 4 concepts from CSD3 to local.
# Authenticates ONCE (SSH multiplexing), then reuses the connection for all rsyncs.
# Usage:  bash scripts/sync_steering.sh
set -e
cd "$(dirname "$0")/.."
HOST=iv294@login.hpc.cam.ac.uk
REMOTE=/rds/user/iv294/hpc-work/thesis/project
RUND=data/analysis/runD_v2

CM=/tmp/cm_csd3_$$
SSH="ssh -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=300"

echo "Authenticating once (enter password + TOTP) ..."
$SSH "$HOST" true     # single auth; subsequent rsyncs reuse this connection

for C in B1_alpha_beta raw_suffix B1_grammar_number particles4_binary/electron_vs_photon; do
  echo "=== syncing $C/steering_delta ==="
  rsync -avz -e "$SSH" "$HOST:$REMOTE/$RUND/$C/steering_delta/" "$RUND/$C/steering_delta/"
done

ssh -O exit -o ControlPath=$CM "$HOST" 2>/dev/null || true
echo "DONE -> all 4 steering_sweep_tier1.csv local"
