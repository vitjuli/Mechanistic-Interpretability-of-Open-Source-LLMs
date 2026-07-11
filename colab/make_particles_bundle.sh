#!/usr/bin/env bash
# Bundles the particle-pair dumps + prompts for the overnight semantic-steering run.
# The B1_alpha_beta / B1_grammar_number dumps are already in the first bundle (on Colab);
# this adds only the 6 particle pairs.
#   bash colab/make_particles_bundle.sh   ->  ~/colab_particles_bundle.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=~/colab_particles_bundle.tar.gz
tar -czf "$OUT" \
  data/analysis/runD_v2/particles4_binary/*/field_dump \
  data/prompts/particle_pairs/*.jsonl
echo "✓ $(du -h "$OUT" | cut -f1)  ->  $OUT"
echo "Upload to Google Drive (same place as the first bundle)."
