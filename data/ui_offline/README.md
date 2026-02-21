# Offline UI Dataset

Pre-computed artifacts for building interactive dashboards (Neuronpedia-style explorer) over mechanistic interpretability results. No GPU or model access required to use these files.

Produced by: `python scripts/09_prepare_offline_ui.py`

## Directory structure

Each run produces `data/ui_offline/<run_id>/` containing:

### Core data

| File | Format | Description |
|---|---|---|
| `interventions.parquet` | Parquet | Unified table of all intervention experiments (ablation, patching, steering) with flattened metadata columns (`meta.*`). Primary data source for all filtering/drill-down. |
| `interventions.csv` | CSV | Same data in CSV for portability. |
| `common_prompt_idx.json` | JSON | Intersection of prompt_idx and layers across all experiment types. Use to filter to comparable subsets. |

### Aggregated metrics (for quick plotting)

| File | Format | Description |
|---|---|---|
| `interventions_layer_agg.parquet` | Parquet | Per (experiment_type, layer): n, mean/median/std of effect sizes, sign flip rate. Use for layer-level bar charts. |
| `interventions_prompt_agg.parquet` | Parquet | Per (experiment_type, prompt_idx): same metrics. Use for prompt-level heatmaps. |
| `interventions_feature_agg.parquet` | Parquet | Per (experiment_type, layer, feature_id): exploded from feature_indices, aggregated across prompts. Use for feature importance rankings. |
| `feature_importance.parquet` | Parquet | Concatenated feature importance sweep results (correlation with logit diff). |

### Graph and supernodes

| File | Format | Description |
|---|---|---|
| `graph.json` | JSON | Node-link format attribution graph. Load directly for graph visualisation (d3, vis.js, etc). Nodes have `id`, `layer`, `feature_idx`, `type` attributes. Edges have `source`, `target`, `weight`. |
| `supernodes.json` | JSON | S1: Graph-community grouping. `{community_id: [node_ids]}`. Use to colour/group nodes in graph view. |
| `supernodes_summary.parquet` | Parquet | Summary per community: size, top nodes by degree, method used. |
| `supernodes_effect.json` | JSON | S2: Effect-similarity clusters. `{cluster_id: ["L15_F12345", ...]}`. Groups features with similar causal effect profiles. |
| `supernodes_effect_summary.parquet` | Parquet | Summary per cluster: size, representative feature, method. |

### Audit and reproducibility

| File | Format | Description |
|---|---|---|
| `audit.json` | JSON | Coverage diagnostics: prompt_idx per experiment type, missing indices, duplicate prompts, per-summary stats. Check `missing_prompt_idx` to see which experiments have uneven coverage. |
| `run_manifest.json` | JSON | Git commit, input file sizes/timestamps, parameters, output file list. For reproducibility. |

## How to use in v0 / frontend

1. Load `interventions.parquet` as the primary dataset
2. Use columns for filtering:
   - `experiment_type`: "ablation_zero", "steering", "patching"
   - `layer`: transformer layer number
   - `prompt_idx`: prompt index
   - `in_common_prompt_set`: boolean flag for prompts present in ALL experiment types
3. Use `interventions_layer_agg.parquet` for overview charts (effect size by layer)
4. Use `graph.json` + `supernodes.json` for graph visualisation with community colouring
5. Use `audit.json` to show data quality warnings in the UI

## Regenerating

```bash
python scripts/09_prepare_offline_ui.py \
    --behaviour grammar_agreement \
    --split train \
    --graph_n_prompts 80

# Verify:
python scripts/09_sanity_check_ui_data.py --run_dir data/ui_offline/<run_id>
```
