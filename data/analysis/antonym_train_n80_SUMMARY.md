# Stage 0 Summary — Graph Topology & Intervention Coverage Audit

**Date:** 2026-03-02
**Local test data:** `grammar_agreement_train_n80` (same format as antonym; identical pipeline version)
**Antonym CSD3 paths:** see Section 5 for CSD3 commands

---

## Key Numbers (grammar_agreement_train_n80 as structural proxy)

### Graph topology

| Metric | Value |
|--------|-------|
| Total nodes | 64 |
| Feature nodes | 61 |
| Hub nodes | 3 (input, output_correct, output_incorrect) |
| Total links | 183 |
| **Feature → Feature edges** | **0** |
| All feature in-degrees | 1 (uniform) |
| All feature out-degrees | 2 (uniform) |
| Positive (excitatory) edges | 84 (45.9%) |
| Negative (inhibitory) edges | 99 (54.1%) |
| Mean \|edge weight\| | 1.48 |

### Intervention coverage

| Metric | Value |
|--------|-------|
| N intervention prompts | 20 |
| Full 48-dim matrix missingness | **94.8%** |
| → of which: structural (unavoidable) | **93.8%** |
| → of which: effective (cells that should be filled) | **17.5%** |
| Effective matrix dimensions | 61 × 3 (feature × experiment_type) |
| Min prompts to reach 80% effective coverage | **1** |
| Min prompts to reach 90% effective coverage | **1** |
| Coverage curve saturation point | **~7 prompts** |

---

## Critical Findings

### Finding 1 — Star topology: Louvain = 1 is inevitable, not a parameter issue

The attribution graph is a **bipartite star**: every feature connects to `input`
(in-edge) and to `output_correct` / `output_incorrect` (out-edges). There are
**zero feature-to-feature edges**. In undirected form, all feature nodes are
structurally identical — Louvain's modularity function has no community signal to
exploit. Raising γ (resolution) does not help.

**Stage 2 must use a different basis for communities**, not Louvain on the
attribution graph.

### Finding 2 — 94.8% missingness is almost entirely structural

The 48-dim effect matrix used by `prepare.py` has a feature at layer 14 filling
only 3/48 cells (one per experiment type at its own layer). The remaining 45 are
structural zeros (feature cannot appear in a different layer's experiment).

Correct effective missingness = **17.5%** (for grammar_agreement, 20 prompts).
The current `build_supernodes_effect()` imputes structural zeros with 0.0 then
normalises 48-dim row vectors — producing clustering artefacts on structurally
empty dimensions.

**Stage 3 must use the 3-dim matrix** (feature × experiment_type), not 48-dim.

### Finding 3 — Coverage saturates at ~7 prompts

The subsampling curve reaches 100% effective coverage within 7 prompts. Adding
more intervention prompts **does not improve coverage** (all graph features are
already tested by the first handful of prompts in strict mode). The benefit of
more prompts is **lower variance in mean_effect_size estimates** (std ∝ 1/√N).

Current N=20 → N=80 reduces effect-size std by ~2×. This matters for clustering
stability and for detecting subtle inhibitory effects.

---

## Stage 0 Go/No-Go Checklist for Stage 1

| Check | Result | Required |
|-------|--------|----------|
| Star topology confirmed (ff_edges = 0) | ✅ 0 feature-feature edges | Must be 0 |
| Louvain = 1 root cause identified | ✅ Bipartite star, no modularity signal | Must identify |
| Structural vs effective missingness distinguished | ✅ 93.8% structural, 17.5% effective | Must distinguish |
| Coverage curve computed | ✅ Saturates at ~7 prompts | Must compute |
| Stage 3 clustering space identified | ✅ Use 3-dim, not 48-dim | Must identify |
| Scripts run without error | ✅ Both scripts passed | Must pass |
| Output artifacts present (reports + PNGs) | ✅ 14 files generated | Must be present |

**→ All checks pass. Stage 1 is approved to proceed.**

---

## Stage 1 Parameter Recommendations

Based on Stage 0 findings, the motivation for Stage 1 is **statistical power
and graph quality**, not coverage (which is already adequate at N=20).

### Prompt scaling (scripts 01 + 06)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `train_size` | 80 | **200** | 4 templates × 50 EN pairs; gives min_prompts=20 in graph construction (vs 8), so only features appearing in ≥10% of prompts survive — much cleaner graph |
| `test_size` | 20 | **50** | Maintain 20% test split; enables robust test-set evaluation |
| `--n_prompts` in script 06 | 80 | **200** | Use all train prompts for attribution graph |
| `min_prompts` (auto) | ceil(0.1×80)=8 | ceil(0.1×200)=**20** | Automatic via existing formula; doubles the statistical bar for feature inclusion |

**50 EN pairs = 25 existing + 25 new** — requires tokenization audit first.

### Intervention scaling (script 07)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `--n_prompts` in script 07 | 20 | **80** | Reduces std(effect_size) by ~2× vs current; provides 4× more data points per feature for mean estimation |

Coverage is already saturated, so 80 (not 200) is sufficient for intervention sampling.
The remaining 120 train prompts serve graph construction only.

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
#SBATCH --time=12:00:00
```

---

## Stage 2 Plan (community detection)

**Primary method — Feature-profile clustering (no new dependencies):**
Cluster features on `(layer_bin, beta_sign, specific_score, frequency)`.
Produces interpretable groups: "early excitatory", "mid inhibitory", etc.
Deterministic, zero parameters beyond K.

**Secondary method — Co-activation graph + Leiden:**
Build feature-feature edges from the per-prompt top-k lists stored in
`graph["union_params"]["prompt_indices"]` + the raw feature extraction data
(step 04 outputs). Edge weight = number of co-occurring prompts.
Needs: `leidenalg` pip install on CSD3.

---

## Stage 3 Plan (effect clustering)

Replace current 48-dim imputed clustering with:

**Primary:** Cluster on 3-dim effective matrix (feature × experiment_type).
Per-feature vector = `(mean_ablation_effect, mean_patching_effect, mean_steering_effect)`.
This has ~17% missingness (tractable) and is physically meaningful.

**Evaluation:** Stability ARI across 10 random seeds; agreement with Stage 2 communities.

---

## CSD3 Commands to Run Stage 0 on Antonym Data

After syncing `data/ui_offline/20260302-211821_antonym_operation_train_n80/` and
`data/results/interventions/antonym_operation/` to local:

```bash
python scripts/a_analyze_graph_topology.py \
    --graph data/ui_offline/20260302-211821_antonym_operation_train_n80/graph.json \
    --out_dir data/analysis/antonym_train_n80_topology

python scripts/a_analyze_interventions_coverage.py \
    --interventions_dir data/results/interventions/antonym_operation \
    --behaviour antonym_operation \
    --out_dir data/analysis/antonym_train_n80_coverage
```

Or run directly on CSD3 (no GPU required).
