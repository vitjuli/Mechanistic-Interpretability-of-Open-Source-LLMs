#!/usr/bin/env bash
# Rotation<->causality temporal-structure analysis on the 4 concepts (local, ready CSVs).
# Usage:  bash scripts/run_rotation_causality.sh
set -e
cd "$(dirname "$0")/.."
SD=data/analysis/runD_v2

.venv/bin/python scripts/rotation_causality_link.py \
  --pair scaffold:realized_scaffold.csv:$SD/B1_alpha_beta/steering_delta/steering_sweep_tier1.csv \
         rawsuffix:realized_rawsuffix.csv:$SD/raw_suffix/steering_delta/steering_sweep_tier1.csv \
         grammar:realized_grammar.csv:$SD/B1_grammar_number/steering_delta/steering_sweep_tier1.csv \
         particles:realized_particles.csv:$SD/particles4_binary/electron_vs_photon/steering_delta/steering_sweep_tier1.csv \
  --flip_col flip_norm_intact \
  --out rotation_causality_link.csv
