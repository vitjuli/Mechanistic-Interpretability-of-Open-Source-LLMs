#!/usr/bin/env bash
# ||delta||/||bulk|| ratio + centroid alignment vs layer (construction-aware). Dumps only.
# Usage:  bash scripts/run_axis_residual_alignment.sh
set -e
cd "$(dirname "$0")/.."
SD=data/analysis/runD_v2
.venv/bin/python scripts/axis_residual_alignment.py --from_dump \
  --pair scaffold:$SD/B1_alpha_beta/realized_writes \
         rawsuffix:$SD/raw_suffix/realized_writes \
         grammar:$SD/B1_grammar_number/realized_writes \
         particles:$SD/particles4_binary/electron_vs_photon/realized_writes
