#!/usr/bin/env bash
# EXACT rotation/length vs causality for the USAGE axis u (gradient-based) on the 4 concepts.
# Contrast against the delta run (scripts/run_rotation_causality_v2.sh).
# Usage:  bash scripts/run_rotation_causality_usage.sh
set -e
cd "$(dirname "$0")/.."
SD=data/analysis/runD_v2

.venv/bin/python scripts/rotation_causality_v2.py --from_dump --axis usage \
  --pair scaffold:$SD/B1_alpha_beta/realized_writes:$SD/B1_alpha_beta/steering_delta/steering_sweep_tier1.csv \
         rawsuffix:$SD/raw_suffix/realized_writes:$SD/raw_suffix/steering_delta/steering_sweep_tier1.csv \
         grammar:$SD/B1_grammar_number/realized_writes:$SD/B1_grammar_number/steering_delta/steering_sweep_tier1.csv \
         particles:$SD/particles4_binary/electron_vs_photon/realized_writes:$SD/particles4_binary/electron_vs_photon/steering_delta/steering_sweep_tier1.csv \
  --flip_col flip_norm_intact
