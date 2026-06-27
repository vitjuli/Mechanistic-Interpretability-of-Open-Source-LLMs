#!/usr/bin/env bash
# Run the 6-pair particles discriminability+geometry matrix on existing binary slices.
# Usage:  bash scripts/run_pair_matrix.sh
set -e
cd "$(dirname "$0")/.."
ROOT=data/analysis/runD_v2/particles4_binary
PAIRS="electron_vs_neutron electron_vs_photon electron_vs_proton neutron_vs_photon neutron_vs_proton photon_vs_proton"

.venv/bin/python scripts/particles_pair_matrix.py \
  --slices_root "$ROOT" \
  --pairs $PAIRS \
  --out particles_pair_matrix.csv

echo "DONE -> particles_pair_matrix.csv"
