# Snapshot: grammar_agreement train n80 (2026-02-23)

Frozen baseline of the grammar_agreement offline-UI dataset.
Source run: `data/ui_offline/20260223-193226_grammar_agreement_train_n80/`

## Parameters

| Key | Value |
|---|---|
| behaviour | grammar_agreement |
| split | train |
| graph_n_prompts | 80 |
| community_method | louvain |
| effect_clusters | 0 (auto) |
| git commit | `128c551` |

## Graph statistics

- Feature nodes: 61
- Total nodes: 64 (61 features + input + output_correct + output_incorrect)
- Edges: 183
- Layers: 10-25 (16 layers)

## Intervention data

- interventions.csv: 960 rows (header + 960 data rows)
- Experiment types: ablation_zero, patching, steering

## Input artifacts (from run_manifest.json)

- `data/results/attribution_graphs/grammar_agreement/attribution_graph_train_n80.json` (53 KB)
- `data/results/interventions/grammar_agreement/intervention_ablation_grammar_agreement.csv` (60 KB)
- `data/results/interventions/grammar_agreement/intervention_patching_grammar_agreement.csv` (143 KB)
- `data/results/interventions/grammar_agreement/intervention_steering_grammar_agreement.csv` (54 KB)
- Feature importance CSVs for layers 15-20

## CLI commands used

```bash
# Step 1: Prepare offline UI data
python -u scripts/09_prepare_offline_ui.py \
    --behaviour grammar_agreement \
    --split train \
    --graph_n_prompts 80 \
    --community_method louvain

# Step 2: Sanity check
python -u scripts/09_sanity_check_ui_data.py \
    --run_dir data/ui_offline/20260223-193226_grammar_agreement_train_n80
```

## Notes

- Graph source was JSON (attribution_graph_train_n80.json), not GraphML
- Louvain community detection used abs(weight) to handle negative attribution edges
- Effect clustering produced 0 clusters (sklearn/scipy not available at time of run)
