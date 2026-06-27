#!/usr/bin/env bash
# Run 155 (realized-write decomposition) on all 4 concept dumps. Usage:  bash scripts/run_155_all.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=data/analysis/runD_v2

$PY scripts/155_realized_write_decomposition.py --dump "$R/B1_alpha_beta/realized_writes"                  --out realized_scaffold.csv
$PY scripts/155_realized_write_decomposition.py --dump "$R/raw_suffix/realized_writes"                     --out realized_rawsuffix.csv
$PY scripts/155_realized_write_decomposition.py --dump "$R/B1_grammar_number/realized_writes"              --out realized_grammar.csv
$PY scripts/155_realized_write_decomposition.py --dump "$R/particles4_binary/electron_vs_photon/realized_writes" --out realized_particles.csv

echo "DONE -> realized_scaffold.csv realized_rawsuffix.csv realized_grammar.csv realized_particles.csv"
