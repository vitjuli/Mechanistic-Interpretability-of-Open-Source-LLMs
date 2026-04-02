# Reproducibility Snapshot: physics_scalar_vector_operator (train, n=80)

**Run date:** 2026-02-24
**Git commit:** `2f6879598443600bdc8c3143f278477de9d8fa6d` (message: "fix,gpu")
**Model:** `Qwen/Qwen3-4B-Instruct-2507`
**Transcoders:** per-layer, model_size=4b (circuit-tracer PLTs)
**Layer range captured:** 15–20 (config `layer_range: [15, 21]`)
**Attribution graph layers with features:** 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25

---

## Snapshot Contents

```
physics_scalar_vector_operator_train_n80_20260224/
├── README.md                              ← this file
├── physics_scalar_vector_operator_train.jsonl   (80 prompts, source of truth)
├── physics_scalar_vector_operator_test.jsonl    (20 prompts)
├── baseline_metrics_train.json            ← teacher-forced baseline results
├── raw/                                   ← intended for original pipeline outputs
│   └── MISSING.md                        ← explains why files are absent (see below)
└── ui_offline/                            ← complete UI export (primary artifact)
    ├── audit.json                         ← per-experiment summaries + coverage
    ├── graph.json                         ← attribution graph (23 feature nodes, 69 links)
    ├── interventions.csv                  ← all intervention rows (all 3 experiments merged)
    ├── interventions_layer_agg.csv        ← layer-level aggregation
    ├── interventions_prompt_agg.csv       ← prompt-level aggregation
    ├── interventions_feature_agg.csv      ← feature-level aggregation
    ├── feature_importance.csv             ← all layers merged
    ├── common_prompt_idx.json
    ├── supernodes.json / supernodes_summary.csv
    ├── supernodes_effect.json / supernodes_effect_summary.csv
    └── run_manifest.json                  ← exact input/output paths + parameters
```

### raw/ — source artifacts status

The intermediate pipeline outputs (attribution graph JSON, per-experiment intervention CSVs,
per-layer feature importance CSVs) were produced on CSD3, consumed to build `ui_offline/`,
and were not separately archived before deletion. `raw/MISSING.md` lists each file,
its size at run time (from `run_manifest.json`), and where its content is preserved in
`ui_offline/`.

**Summary mapping `raw/` → `ui_offline/`:**

| Would-be raw/ file | Preserved in ui_offline/ as |
|---|---|
| `attribution_graph_train_n80.json` (21 KB) | `graph.json` (node-link JSON format) |
| `intervention_ablation_*.csv` (81 KB) | rows with `experiment_type=ablation` in `interventions.csv` |
| `intervention_patching_*.csv` (183 KB) | rows with `experiment_type=patching` in `interventions.csv` |
| `intervention_steering_*.csv` (41 KB) | rows with `experiment_type=steering` in `interventions.csv` |
| `importance/feature_importance_layer_*.csv` (×16) | `feature_importance.csv` (merged) |

---

## Key Metrics

| Metric | Value |
|---|---|
| Baseline accuracy | **0.8375** (67/80 prompts) |
| Baseline accuracy (sign) | 0.9875 |
| Mean logprob diff (baseline) | **4.440** |
| Median logprob diff | 4.000 |
| Std logprob diff | 2.638 |
| Score method | teacher_forced_logprob (normalized_per_token) |
| Attribution graph: feature nodes | **23** |
| Attribution graph: links | **69** |
| Attribution graph: feature layers | 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25 |
| Ablation: n_experiments | 320, mean_effect = −0.310 |
| Patching: n_experiments | 320, mean_effect = −0.325 |
| Steering: n_experiments | 240, mean_effect = −0.436 |

---

## Exact Run Commands

```bash
# Step 1 — Prompts (already generated; do NOT regenerate to preserve exact seed)
# python scripts/01_generate_prompts.py --behaviour physics_scalar_vector_operator
# Output: data/prompts/physics_scalar_vector_operator_{train,test}.jsonl

# Step 2 — Baseline
python scripts/02_run_baseline.py \
    --behaviour physics_scalar_vector_operator \
    --split train
# Output: data/results/baseline_metrics_train.json

# Step 3 — Extract transcoder features
python scripts/04_extract_transcoder_features.py \
    --behaviour physics_scalar_vector_operator \
    --split train
# Output: data/results/transcoder_features/layer_*/physics_scalar_vector_operator_train_*

# Step 4 — Build attribution graph (n_prompts=80 = full train set)
python scripts/06_build_attribution_graph.py \
    --behaviour physics_scalar_vector_operator \
    --split train \
    --n_prompts 80
# Output: data/results/attribution_graphs/physics_scalar_vector_operator/attribution_graph_train_n80.json

# Step 5 — Interventions (strict mode; graph-driven only)
python scripts/07_run_interventions.py \
    --behaviour physics_scalar_vector_operator \
    --split train \
    --graph_n_prompts 80
# Output: data/results/interventions/physics_scalar_vector_operator/

# Step 6 — UI offline export
python scripts/09_prepare_ui_data.py \
    --behaviour physics_scalar_vector_operator \
    --split train \
    --graph_n_prompts 80
# Output: data/ui_offline/<timestamp>_physics_scalar_vector_operator_train_n80/
```

---

## Seeds and Determinism

| Setting | Value |
|---|---|
| `seeds.prompt_generation` (config) | 42 |
| `seeds.intervention_sampling` (config) | 456 |
| `seeds.torch_seed` (config) | 789 |
| Graph community detection | Louvain (non-deterministic; re-runs may give different communities but same graph structure) |
| Transcoder features | deterministic (no sampling) |

**Nondeterministic elements:**
- Louvain community detection assigns nodes to communities randomly; supernodes may differ on re-run, but the underlying graph (nodes + edges) is deterministic.
- Float precision differences on different hardware may shift the 6 "failed" prompts (accuracy 83.75%) slightly, but the mean logprob diff should be stable to ±0.1.

---

## Opening the Offline UI

```bash
# Copy snapshot ui_offline/ to the live ui_offline/ directory
cp -r runs/physics_scalar_vector_operator_train_n80_20260224/ui_offline/ \
    data/ui_offline/20260224-115852_physics_scalar_vector_operator_train_n80/

# Or load from snapshot directly by pointing the dashboard to:
#   runs/physics_scalar_vector_operator_train_n80_20260224/ui_offline/
# (edit dashboard/src/config or use --data-dir flag if supported)

cd dashboard && npm run dev
# Then load run: 20260224-115852_physics_scalar_vector_operator_train_n80
```
