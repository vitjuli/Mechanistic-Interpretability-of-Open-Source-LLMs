#!/usr/bin/env bash
# Length(norm) vs causality + HYSTERESIS test for delta and u, 4 concepts. From dumps + steering.
# Usage:  bash scripts/run_norm_causality.sh
set -e
cd "$(dirname "$0")/.."
SD=data/analysis/runD_v2
.venv/bin/python scripts/norm_causality.py --from_dump \
  --pair scaffold:$SD/B1_alpha_beta/realized_writes:$SD/B1_alpha_beta/steering_delta/steering_sweep_tier1.csv \
         rawsuffix:$SD/raw_suffix/realized_writes:$SD/raw_suffix/steering_delta/steering_sweep_tier1.csv \
         grammar:$SD/B1_grammar_number/realized_writes:$SD/B1_grammar_number/steering_delta/steering_sweep_tier1.csv \
         particles:$SD/particles4_binary/electron_vs_photon/realized_writes:$SD/particles4_binary/electron_vs_photon/steering_delta/steering_sweep_tier1.csv \
  --flip_col flip_norm_intact
