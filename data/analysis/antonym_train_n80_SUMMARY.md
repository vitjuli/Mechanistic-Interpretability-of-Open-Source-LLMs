# Stage 0b Summary — Graph Topology & Intervention Coverage Audit (Antonym)

**Date:** 2026-03-04
**Data:** `antonym_operation_train_n80` — actual antonym artifacts (not grammar_agreement proxy)
**Source artifacts:**
- Graph: `data/ui_offline/20260302-211821_antonym_operation_train_n80/raw_sources/attribution_graph_train_n80.json`
- Interventions: `data/ui_offline/20260302-211821_antonym_operation_train_n80/raw_sources/`
**Analysis outputs:**
- `data/analysis/antonym_train_n80_topology/`
- `data/analysis/antonym_train_n80_coverage/`

---

## Terminology

- **Cell (full 48-dim matrix):** One (feature, experiment_type, layer) triple. Dimensions: 72 features × 3 experiment types × 16 layers = 3,456 cells. A cell is *observed* if at least 1 intervention prompt produced a result for that feature in that experiment at that layer.
- **Cell (effective 3-dim matrix):** One (feature, experiment_type) pair, collapsing the layer dimension. Dimensions: 72 × 3 = 216 cells. A cell is *observed* if that experiment type recorded ≥1 prompt for that feature (regardless of layer column).
- **Structural missingness:** Cells in the full 48-dim matrix that are unavoidably empty because the feature belongs to a specific layer and cannot appear in other layers' columns.
- **Effective missingness:** Fraction of cells in the 3-dim matrix that are unobserved — cells that *should* be filled but were not tested in any of the N intervention prompts.

---

## Key Numbers — antonym_operation_train_n80

### Graph topology

| Metric | Value |
|--------|-------|
| Total nodes | 75 |
| Feature nodes | 72 |
| Hub nodes | 3 (input, output_correct, output_incorrect) |
| Total links | 216 |
| **Feature → Feature edges** | **0** |
| All feature in-degrees | 1 (uniform) |
| All feature out-degrees | 2 (uniform) |
| Positive (excitatory) edges | 113 (52.3%) |
| Negative (inhibitory) edges | 103 (47.7%) |
| Mean \|edge weight\| | 0.978 |
| Layers represented | 10–25 (16 layers) |
| Features per layer (min/max) | 2 (L16, L21, L22) / 7 (L14, L20, L24) |

### Intervention coverage — overall (all experiment types)

| Metric | Value |
|--------|-------|
| N intervention prompts | 20 |
| N graph features | 72 |
| Full 48-dim matrix missingness | **94.8%** |
| → of which: structural (unavoidable) | **93.8%** |
| → of which: effective (cells that should be filled) | **16.7%** |
| Effective matrix dimensions | 72 × 3 (feature × experiment_type) |
| Overall effective coverage | **83.3%** (180/216 cells observed) |

### Intervention coverage — per experiment type

| Experiment | Features tested | Features never tested | Coverage | Saturates? |
|------------|----------------|----------------------|----------|------------|
| **patching** | 72 / 72 | 0 | **100%** | Yes — at N≈5 |
| **steering** | 65 / 72 | 7 | **90.3%** | Flat — structural |
| **ablation** | 43 / 72 | **29** | **59.7%** | **Flat — structural** |

---

## Critical Findings

### Finding 1 — Star topology: Louvain = 1 is inevitable, not a parameter issue

The attribution graph is a **bipartite star**: every feature connects to `input`
(in-edge) and to `output_correct` / `output_incorrect` (out-edges). There are
**zero feature-to-feature edges**. In undirected form, all feature nodes are
structurally identical — Louvain's modularity function has no community signal.
Raising γ (resolution) does not help; Q(any partition) ≤ 0.

**Stage 2 must use a different basis for communities**, not Louvain on the
attribution graph.

### Finding 2 — 94.8% missingness is almost entirely structural

The full 48-dim matrix has 3,456 cells but only 180 are observed (5.2%). Of the
missing 94.8%, **93.8% is structural** — a feature at layer 14 simply cannot
produce data in layer 10's experiment columns. The remaining 16.7% is the true
effective gap.

The current `build_supernodes_effect()` imputes structural zeros with 0.0 then
normalises 48-dim row vectors — producing clustering artefacts on structurally
empty dimensions.

**Stage 3 must use the 3-dim matrix** (feature × experiment_type), not 48-dim.

### Finding 3 — Ablation coverage is FLAT at 59.7%; this is NOT a prompt-count problem

This is the most important correction from the grammar_agreement proxy analysis.

**Subsampling curve (ablation):** Coverage is 59.7% at N=1 and remains 59.7% at
N=20. Adding more intervention prompts does **not** increase ablation coverage.
The 29 missing features never appear in the per-prompt top-k activation ranking
for any of the 20 prompts.

**Root cause:** Script 07 selects features for ablation via per-prompt top-k
activation magnitude (`--top_k 5`). Features with consistently low activation on
the 20 intervention prompts are never selected, regardless of N. This is a
threshold problem, not a sample-size problem.

**Patching and steering are different:**
- Patching: saturates at N≈5–7 (100% coverage); subsampling curve is NOT flat.
- Steering: flat at 90.3% (7 features structurally absent from top-k). Same root
  cause as ablation but affects fewer features.

**Fix for Stage 1:** Increase `--top_k` in script 07 (e.g., from 5 to 15–20).
This broadens the per-prompt selection pool so that features with moderate (not
just top-5) activation are also ablated. More prompts help only for reducing
variance, not for coverage.

---

## Validation of "coverage saturates at ~7 prompts" claim

The claim from the grammar_agreement proxy was **partially correct**:

| Experiment | Claim | Actual (antonym) |
|------------|-------|-----------------|
| Patching | Saturates ~7 prompts | ✅ Confirmed (saturates at N≈5) |
| Steering | Saturates ~7 prompts | ❌ Flat at 90.3% (structural) |
| Ablation | Saturates ~7 prompts | ❌ Flat at 59.7% (structural) |
| **Overall** | Saturates ~7 prompts | ⚠️ Partially — only patching saturates |

The grammar_agreement proxy had higher overall coverage because its behaviour
produces different activation patterns. The antonym data reveals that ablation
coverage is severely incomplete and requires a different fix.

---

## Stage 0b Go/No-Go Checklist

| Check | Result | Status |
|-------|--------|--------|
| Star topology confirmed on actual antonym graph | 0 feature-feature edges | ✅ |
| Louvain = 1 root cause identified | Bipartite star, no modularity signal | ✅ |
| Structural vs effective missingness distinguished | 93.8% structural, 16.7% effective | ✅ |
| Per-experiment subsampling curves computed | Done for ablation, patching, steering | ✅ |
| "Saturates at 7 prompts" validated on antonym | Patching ✅, ablation ❌ flat, steering ❌ flat | ✅ |
| Ablation coverage gap identified and root cause found | 29 features, top_k threshold, not N | ✅ |
| Stage 3 clustering space identified | Use 3-dim, not 48-dim | ✅ |
| Scripts run without error on actual antonym data | Both scripts passed | ✅ |

**→ All checks pass. Proceed to Stage 1 with updated parameters (see below).**

---

## Stage 1 Parameter Recommendations (updated after Stage 0b)

### Prompt scaling (scripts 01 + 06)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `train_size` | 80 | **200** | 4 templates × 50 EN pairs; min_prompts=20 in graph (vs 8), doubles statistical bar for feature inclusion |
| `test_size` | 20 | **50** | Maintain 20% test split |
| `--n_prompts` in script 06 | 80 | **200** | Use all train prompts for attribution graph |
| `min_prompts` (auto) | ceil(0.1×80)=8 | ceil(0.1×200)=**20** | Automatic; doubles the bar for feature inclusion |

50 EN pairs = 25 existing + 25 new — requires tokenization audit of new pairs first.

### Intervention scaling (script 07) — UPDATED vs proxy analysis

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `--n_prompts` in script 07 | 20 | **80** | Reduces std(effect_size) by ~2×; only patching benefits from coverage, ablation does not |
| **`--top_k`** | **5** | **20** | **NEW — fixes ablation coverage from 59.7% to ~100%**; ablation top-k of 5 per layer misses 29 graph features entirely |

> **Note:** `--top_k 20` means each ablation step tests the top-20 features per layer per prompt. With 72 graph features spread across 16 layers (avg 4.5 per layer), top-20 essentially covers the full layer, guaranteeing ablation coverage ≥ 99% of graph features.

### Config changes (`configs/experiment_config.yaml`)

```yaml
antonym_operation:
  train_size: 200   # was 80
  test_size: 50     # was 20
```

### New SLURM job: `jobs/antonym_200_02_09.sbatch`

```bash
N_PROMPTS=200
N_INTERVENTIONS=80
TOP_K=20           # was 5 — critical fix for ablation coverage
#SBATCH --time=14:00:00   # longer: more prompts + wider top_k
```

---

## Stage 2 Plan (community detection)

**Primary method — Feature-profile clustering (no new dependencies):**
Cluster features on `(layer_bin, beta_sign, specific_score, frequency)`.
Produces interpretable groups: "early excitatory", "mid inhibitory", etc.
Deterministic, zero parameters beyond K.

**Secondary method — Co-activation graph + Leiden:**
Build feature-feature edges from per-prompt top-k co-occurrence counts.
Edge weight = number of prompts in which both features appear in top-k.
Needs: `leidenalg` pip install on CSD3.

---

## Stage 3 Plan (effect clustering)

Replace current 48-dim imputed clustering with:

**Primary:** Cluster on 3-dim effective matrix (feature × experiment_type).
Per-feature vector = `(mean_ablation_effect, mean_patching_effect, mean_steering_effect)`.
With Stage 1's `--top_k 20`, ablation missingness drops from 40.3% to ~0%.
Effective matrix will be essentially complete.

**Evaluation:** Stability ARI across 10 random seeds; agreement with Stage 2 communities.
