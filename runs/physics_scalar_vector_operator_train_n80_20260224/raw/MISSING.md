# Raw Source Artifacts — NOT AVAILABLE ON DISK

The intermediate result files recorded in `run_manifest.json` no longer exist on disk.
They were produced on CSD3 (2026-02-24), used to build the `ui_offline/` export, and
were not separately archived before deletion.

## Files that were here (sizes from run_manifest.json)

| File | Size at run time | Contents |
|---|---|---|
| `attribution_graph_train_n80.json` | 21 464 bytes | Raw attribution graph (JSON, 80-prompt union) |
| `intervention_ablation_physics_scalar_vector_operator.csv` | 81 189 bytes | Per-row ablation results |
| `intervention_ablation_physics_scalar_vector_operator_summary.json` | 880 bytes | Ablation summary stats |
| `intervention_patching_physics_scalar_vector_operator.csv` | 183 275 bytes | Per-row patching results |
| `intervention_patching_physics_scalar_vector_operator_summary.json` | 880 bytes | Patching summary stats |
| `intervention_steering_physics_scalar_vector_operator.csv` | 41 052 bytes | Per-row steering results |
| `intervention_steering_physics_scalar_vector_operator_summary.json` | 902 bytes | Steering summary stats |
| `importance/feature_importance_layer_*.csv` (×16 layers) | ~200–1 450 bytes each | Per-layer feature importance |

## Where the content is preserved

The `ui_offline/` folder in this snapshot contains processed versions of these files:

| ui_offline/ file | Corresponds to |
|---|---|
| `graph.json` | Processed (node-link JSON) version of `attribution_graph_train_n80.json` |
| `interventions.csv` | All three intervention CSVs merged + enriched columns |
| `interventions_layer_agg.csv` | Aggregated from `interventions.csv` |
| `interventions_prompt_agg.csv` | Aggregated from `interventions.csv` |
| `interventions_feature_agg.csv` | Aggregated from `interventions.csv` |
| `feature_importance.csv` | All `importance/feature_importance_layer_*.csv` merged |
| `audit.json` | Summaries extracted from all three `_summary.json` files |

## To re-generate these files

Re-run Steps 3–5 from the README run commands using the prompt JSONL and config
in this snapshot. Results will be functionally identical (see README for
nondeterminism notes).
