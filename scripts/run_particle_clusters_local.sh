#!/bin/bash
# CPU clustering for particle latent states. Run ON CSD3 (data is there) AFTER the GPU job (04/06/07).
# Chain (all CPU, minutes):  09 (ui_run+graph.json) -> 19 (pivot ablation -> matrices) -> 22 -> 23 -> 26
# set -e: stop at first error so we see exactly where a handoff breaks.
set -e
cd "$(dirname "$0")/.."

BEH=physics_internal_candidate_selection_v2
GRAPH_N=120
GRAPH_SUFFIX=_roleaware
GROUP=data/results/grouping
CLUST=data/results/clustering_particles
ABL_CSV=data/results/interventions/$BEH/particles/intervention_ablation_$BEH.csv

echo "=== 09: build ui_run (graph.json + raw_sources) from the n120_roleaware graph + interventions ==="
python scripts/09_prepare_offline_ui.py \
  --behaviour $BEH --split train \
  --graph_suffix $GRAPH_SUFFIX --graph_n_prompts $GRAPH_N

echo "=== 19: pivot ablation -> feature_prompt_effect_matrix.csv (+ abs/group/sfr) ==="
python scripts/19_feature_prompt_analysis.py \
  --behaviour $BEH --split train --abl_csv $ABL_CSV --grouping_dir $GROUP

echo "=== 22: prepare clustering inputs (co-importance matrices) ==="
python scripts/22_prepare_clustering_inputs.py --grouping_dir $GROUP --clustering_dir $CLUST

echo "=== 23: clustering benchmark -> cluster_labels.csv (incl. coimp_louvain) ==="
python scripts/23_run_clustering_benchmark.py --clustering_dir $CLUST

echo "=== 26: cluster semantics (top-token labels) ==="
python scripts/26_cluster_semantics.py --clustering_dir $CLUST --grouping_dir $GROUP --no_dashboard

echo ""
echo "DONE -> $CLUST/cluster_labels.csv (coimp_louvain = particle C0..Cn)"
