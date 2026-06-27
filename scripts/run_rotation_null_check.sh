#!/usr/bin/env bash
# ABSOLUTE per-layer angle (mean_cos) for delta vs usage + random null floor. Dumps only.
# Decides: is u really more fixed than delta (contrast real), or do both rotate alike (scale-free
# metrics hid nothing / artifact)?  Usage:  bash scripts/run_rotation_null_check.sh
set -e
cd "$(dirname "$0")/.."
SD=data/analysis/runD_v2

.venv/bin/python scripts/rotation_null_check.py --from_dump \
  --pair scaffold:$SD/B1_alpha_beta/realized_writes \
         rawsuffix:$SD/raw_suffix/realized_writes \
         grammar:$SD/B1_grammar_number/realized_writes \
         particles:$SD/particles4_binary/electron_vs_photon/realized_writes
