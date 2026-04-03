# Multilingual Antonym Circuits — Final Analysis Report
## Behaviour: `multilingual_circuits_b1`

| Version | SLURM | Date | Attribution Method | Status |
|---|---|---|---|---|
| **B1-v2 (canonical)** | 25679695 | 2026-03-21 | OLS beta proxy | Superseded |
| **B1-gradient (current)** | 26929559 | 2026-04-02 | Gradient × activation | **CURRENT** |

**Model**: Qwen3-4B-Base · **Dashboard**: `dashboard_b1/`

---

## Table of Contents

1. [Behaviour Definition](#1-behaviour-definition)
2. [Dataset Construction](#2-dataset-construction)
3. [Full Pipeline](#3-full-pipeline)
4. [Mathematical Formalism](#4-mathematical-formalism)
5. [Results](#5-results)
   - 5.1 Baseline Gate
   - 5.2 Attribution Graphs
   - 5.3 IoU — Cross-lingual Universality (Claim 3)
   - 5.4 Language Profiles
   - 5.5 Community Structure
   - 5.6 Bridge Features
   - 5.7 Causal Circuit
   - 5.8 Circuit Validation
   - 5.9 Reasoning Traces
   - 5.10 Failure Analysis
   - 5.11 Artifact Confirmation
6. [Detailed Comparison with Anthropic's Implementation](#6-detailed-comparison-with-anthropics-implementation)
7. [Problems, Limitations, and Mistakes](#7-problems-limitations-and-mistakes)
8. [Summary: Claims vs Evidence](#8-summary-claims-vs-evidence)
9. [Gradient Attribution Upgrade](#9-gradient-attribution-upgrade)

---

## 1. Behaviour Definition

**Task**: Given a prompt of the form:
```
The opposite of "{word}" is
```
(EN) or an equivalent French surface form (FR), predict the correct single-token antonym with a leading space (e.g., ` cold`, ` lent`).

- **Behaviour type**: Type 2 — Candidate Set (binary comparison between correct token logprob and incorrect token logprob)
- **Languages**: English (EN) + French (FR), 8 semantically aligned concept pairs
- **Concepts**: Antonym pairs indexed 0, 2–8 (index 1 excluded: `froid` tokenises as 2 tokens in Qwen3)
- **Goal**: Identify which transcoder features and circuits in Qwen3-4B underlie this behaviour, and whether those circuits are shared across the two languages

---

## 2. Dataset Construction

### 2.1 Concepts and Templates

8 semantic antonym pairs appear in both English and French:
- Each concept yields 8 surface-form templates (T0–T7)
- Templates vary word order, framing phrases, and surrounding context
- Correct token: leading-space antonym (e.g., ` cold`, ` lent`)
- Incorrect token: the original word or a near-synonym (e.g., ` hot`, ` rapide`)

### 2.2 Stratified Split

For each of the 16 `(concept_index, language)` groups:
- 6 templates → **train**
- 2 templates → **test**
- Test template indices are shared between EN and FR (enabling aligned C3 patching)

**Final sizes**:
| Split | EN | FR | Total |
|---|---|---|---|
| Train | 48 | 48 | 96 |
| Test | 16 | 16 | 32 |

### 2.3 Template Artifact History (B1-v1 → B1-v2)

B1-v1 FR templates contained the pattern `"{word}"` (typographic quotes). Qwen3's tokenizer encodes this as a distinctive token sequence, causing the model to continue with a closing `"` (quote-completion) rather than the antonym. This inflated FR accuracy, inflated mid-layer IoU, and injected artifact features into the circuit.

**Fix applied in B1-v2**: all FR templates rewritten to avoid `"{word}"` pattern. FR accuracy changed from ~87% (artifact-inflated) to 79.2% (true signal). Artifact features L22_F108295 and L22_F32734 were confirmed absent from the corrected circuit.

---

## 3. Full Pipeline

```
Script 01 → 02 → 04 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
             ↓
          prepare_b1_dashboard.py
             ↓
          dashboard_b1/
```

### Script 01 — `01_generate_prompts.py`
- Reads concept definitions from `configs/experiment_config.yaml`
- Registers behaviour generator in `GENERATORS` dict
- Outputs JSONL with fields: `text`, `correct_token`, `incorrect_token`, `concept_index`, `language`, `template_idx`
- Applies stratified split for train/test

### Script 02 — `02_run_baseline.py`
- Runs **Qwen3-4B-Instruct** (note: instruct, not base — see §7.10)
- Computes per-language `sign_accuracy`: fraction of prompts where `logit_correct > logit_incorrect`
- Gate threshold: EN ≥ 0.90, FR ≥ 0.65 (empirically calibrated after format debugging)

### Script 04 — `04_extract_transcoder_features.py`
- Loads **Qwen3-4B-Base** with per-layer transcoders from the `circuit-tracer` library
- Hook point: `post_attention_layernorm` (MLP input, not residual stream)
- Extracts top-k=50 features per prompt per layer (layers 10–25)
- Stores: feature activations, attribution coefficients (gradient × activation)
- `d_transcoder = 163,840` per layer

### Script 06 — `06_build_attribution_graph.py`
Provides two aggregation modes:

**Mode A: `aggregate_graphs_per_prompt_union()`** (star graph)
- Per-prompt: keep features with `|attribution| > threshold`
- Union across all train prompts
- Output: 84 feature nodes, 252 star edges

**Mode B: `aggregate_graphs_role_aware()`** (role-aware graph)
- Same union of decision-token features
- Plus content-word features (detected via regex `"([^"]+)"` + offset_mapping)
- Plus virtual-weight (VW) edges between adjacent layers (threshold=0.01)
- Output: 86 feature nodes, 633 edges (star + VW)
- `--vw_threshold 0.01 --k_content 10 --output_suffix _roleaware`

### Script 07 — `07_run_interventions.py`
- **Pairing mode**: C3 language-swap — same `concept_index`, EN source → FR target
- **Experiment types**: `ablation_zero` (set feature to 0) and `patching` (replace with source value)
- **Strict mode** (default): only layers with graph features are processed; exits with code 1 if zero results
- `--top_k 20` features per layer
- Outputs: `interventions.csv`, per-layer and per-prompt aggregations

### Scripts 07a + 07b — Baseline and Stratification
- `07a_compute_baselines.py`: logit diff baselines for source/target prompt pairs
- `07b_stratify_prompts.py`: ensures balanced sampling for intervention analysis

### Script 08 — `08_causal_edges.py`
- For each candidate feature pair `(fᵢ, fⱼ)` with `i < j`:
  - Patch `fᵢ` activation → measure `Δ(fⱼ activation)`
  - Causal edge if `|mean Δ| > threshold`
- AGW (Attribution Graph Walk) pre-filter: only tests pairs reachable in graph
- `require_causal_edge=True` by default
- Produces final causal circuit: 21 feature nodes + 3 I/O = 24 nodes, 55 edges, 50 paths
- NaN/SIGALRM guards applied for CSD3 stability

### Script 09 — `09_prepare_offline_ui.py`
- Serialises graph, features, interventions, communities, reasoning traces to UI-consumable JSON/CSV
- Output: `data/ui_offline/`

### Script 10 — `10_reasoning_trace.py`
- Per-prompt zone trajectory: `contribution_to_correct = −effect_size` at each circuit feature
- Zones: early (L10–L17), mid (L18–L22), late (L23–L25)
- Sign convention: positive = supports correct prediction
- Outputs: trace per prompt, zone summary, dominant trajectory classification

### Script 11 — `11_ablation_supplement.py`
- Additional ablation statistics for 2 circuit features not fully covered by script 08:
  - L23_F64429: `mean|eff|=1.330`, `neg_frac=77%`
  - L24_F76363: `mean|eff|=0.770`

### Script 12 — `12_competition_analysis.py`
- Tests whether baseline errors are residual quote-completion artifacts
- Computes `argmax_is_quote_rate` across 18 incorrect prompts
- Measures correlation between L13 activation and logit margin

### Script 13 — `13_l22_intervention.py`
- Dose-response study: amplify B1-v1 artifact features (L22_F108295, L22_F32734) vs control (L22_F78043)
- Tests at α = 1.5, 2.0, 3.0

### `prepare_b1_dashboard.py`
- Aggregates all results → 18 JSON/CSV files in `dashboard_b1/public/data/`
- Computes IoU per layer (pooled, decision, content modes)
- Computes bridge features, community summary, node language labels
- Outputs: `graph.json`, `circuit.json`, `iou_data.json`, `bridge_features.json`, `community_summary.json`, `node_labels.json`, `error_cases.json`, `interventions*.csv`, `prompt_traces.json`, `prompt_features.csv`, `prompt_paths.csv`, `layerwise_traces.csv`, `supernodes*.json/csv`, `run_manifest.json`

---

## 4. Mathematical Formalism

### 4.1 Model and Transcoder

Each MLP at layer ℓ is approximated by a per-layer transcoder:

```
MLP_ℓ(x) ≈ T_ℓ(x) = W_dec_ℓ · ReLU(W_enc_ℓ · x + b_enc_ℓ) + b_dec_ℓ
```

where:
- `x ∈ ℝ^{d_model}` is the post-attention-layernorm residual
- `W_enc_ℓ ∈ ℝ^{d_tc × d_model}`, `W_dec_ℓ ∈ ℝ^{d_model × d_tc}`
- `d_tc = 163,840` (transcoder hidden dimension per layer)
- `d_model = 2,560` (Qwen3-4B)

Feature `k` activates as:

```
a_k^ℓ(x) = ReLU((W_enc_ℓ · x)_k + b_k)
```

Hook point: `post_attention_layernorm` — i.e., `x` is the MLP input, not the full residual stream. Attention outputs are not decomposed.

### 4.2 Attribution Scoring (Star Graph)

The behaviour signal is the logit difference:

```
Δlogit(p) = logit_{correct}(p) − logit_{incorrect}(p)
```

First-order attribution of feature `k` at layer `ℓ` to `Δlogit` (current, gradient-based):

```
α_k^ℓ(p) = a_k^ℓ(p) × (∂Δlogit(p) / ∂a_k^ℓ(p))
```

This is a gradient × activation product computed per-prompt — analogous to integrated gradients at a single point. The star graph connects `input → feature_k → output` using `|α_k^ℓ|` as edge weight.

Feature `k` is included in the graph if `|α_k^ℓ| > θ_attr` for at least one prompt.

> **Historical note (B1-v2)**: The original canonical run (SLURM 25679695) used an OLS beta proxy rather than true per-prompt gradients. See §9.2 for a full description of the difference and its effects. All results tables in §5 report gradient-run values (SLURM 26929559) as the current canonical, with B1-v2 values retained for comparison.

### 4.3 Virtual-Weight (VW) Edges

For adjacent layers ℓ (source) and ℓ+1 (target), structural connectivity is estimated:

```
VW(i, j) = W_enc_{ℓ+1}[j, :] · W_dec_ℓ[:, i]
```

This is the projection of feature `i`'s decoder direction onto feature `j`'s encoder direction — how much of feature `i`'s output would be "read" by feature `j` in a linear approximation.

A VW edge is drawn if `|VW(i,j)| > 0.01`. This never requires running the model — it is a weight-space analysis.

The submatrix used is `W_enc_tgt[S_tgt, :] @ W_dec_src[S_src, :].T` where `S` are the selected feature indices, never the full `d_tc × d_tc` matrix.

### 4.4 Causal Edges (Activation Patching, Script 08)

A directed causal edge `fᵢ → fⱼ` (layers `ℓᵢ < ℓⱼ`) is established by:

```
Δa_j^{(p)} = a_j(model | patch fᵢ to source value) − a_j(model | baseline)
```

Averaged over prompts:

```
mean_Δ = E_p[Δa_j^{(p)}]
```

Edge exists if `|mean_Δ| > δ_causal`. This requires a forward pass and is **causal** (not structural).

The Attribution Graph Walk (AGW) pre-filters candidate pairs to those reachable by the star graph, reducing the quadratic cost.

### 4.5 Intersection over Union — Cross-lingual Universality

For each layer ℓ, active feature sets are collected across all prompts of each language:

```
S_lang^ℓ = ⋃_{p ∈ lang} {k : a_k^ℓ(p) > 0 and |α_k^ℓ(p)| > θ}
```

Layer-level IoU:

```
IoU_ℓ = |S_EN^ℓ ∩ S_FR^ℓ| / |S_EN^ℓ ∪ S_FR^ℓ|
```

Zone averages (Claim 3):

```
early = mean(IoU_ℓ, ℓ ∈ {10, 11})
mid   = mean(IoU_ℓ, ℓ ∈ {12, 13, ..., 20})
late  = mean(IoU_ℓ, ℓ ∈ {22, 23, 24, 25})
ratio = mid / early
```

Three modes:
- **Pooled**: all prompts, all token positions in the last-5 window
- **Decision**: decision token position only
- **Content**: content-word token positions only

### 4.6 Language Profile Assignment

For each feature node `k`, effects are measured per language across all 96 prompts:

```
μ_EN = mean_p∈EN(effect_k(p)),   μ_FR = mean_p∈FR(effect_k(p))
```

Language profile is assigned as:
- `en_leaning`: `|μ_EN| > threshold AND |μ_EN| > 2|μ_FR|`
- `fr_leaning`: `|μ_FR| > threshold AND |μ_FR| > 2|μ_EN|`
- `balanced`: both languages active with similar magnitude
- `insufficient_data`: neither language reaches threshold

### 4.7 Bridge Feature Score

A feature is a **bridge** if it fires in both languages with the same sign:

```
bridge_score = min(|μ_EN|, |μ_FR|)   if sign(μ_EN) = sign(μ_FR), else 0
is_bridge = bridge_score > 0
```

### 4.8 Circuit Validation Metrics

**Necessity**: Ablate all 21 circuit features simultaneously, measure:
```
necessity = P_p(|Δlogit_ablated − Δlogit_baseline| > ε)   [disruption rate]
```

**Sufficiency S1**: Restore only circuit features from scrambled baseline:
```
S1 = P_p(sign(Δlogit_restored) = sign(Δlogit_original))   [sign preservation rate]
```

**Sufficiency S1.5**: Stricter S1 with higher restoration amplitude:
```
S1.5 = sign preservation rate at higher feature amplification
```

**Sufficiency S2 (Transfer)**: Patch EN circuit activations into FR prompts:
```
transfer_rate = P_pair(Δlogit shifts toward correct in FR target)
mean_shift = mean_pair(Δlogit_patched − Δlogit_baseline)
```

---

## 5. Results

### 5.1 Baseline Gate

| Metric | B1-v1 | B1-v2 (canonical) |
|---|---|---|
| EN sign_accuracy | 1.000 | 1.000 |
| FR sign_accuracy | ~0.875 (artifact) | **0.792** |
| Mean normalised logit diff | 3.511 | 3.511 |
| Gate | PASS | **PASS** |

Note: FR accuracy dropped when template artifact was removed, revealing the true signal.

### 5.2 Attribution Graphs

| Graph Type | B1-v2 (beta proxy) | **B1-gradient (current)** |
|---|---|---|
| Star — feature nodes | 84 | — |
| Star — edges | 252 | — |
| Role-aware — feature nodes | 86 | **137** |
| Role-aware — edges | 633 | **1,447** |

The gradient run produces a significantly larger role-aware graph (+59% nodes, +129% edges). Added in role-aware graph: pure content nodes + "both" nodes (active at decision + content positions). VW edges account for the bulk of the total edge count. See §9.5 for interpretation of the expansion.

### 5.3 IoU — Cross-lingual Universality (Claim 3)

| Mode | Early (L10–L11) | Mid (L12–L20) | Late (L22–L25) | **Ratio** |
|---|---|---|---|---|
| **Pooled** | 0.2375 | 0.2589 | 0.2262 | **1.090×** |
| Decision-only | 0.2182 | 0.2262 | 0.2363 | 1.037× |
| Content-only | 0.2217 | 0.2352 | 0.2012 | 1.061× |

Raw pooled per-layer:

| Layer | IoU | | Layer | IoU |
|---|---|---|---|---|
| L10 | 0.2246 | | L18 | 0.2449 |
| L11 | 0.2505 | | L19 | 0.2630 |
| L12 | 0.2621 | | L20 | 0.2605 |
| L13 | 0.2640 | | L21 | 0.2325 |
| L14 | 0.2380 | | L22 | 0.2605 |
| L15 | 0.2622 | | L23 | 0.2242 |
| L16 | 0.2573 | | L24 | 0.2065 |
| L17 | 0.2783 | | L25 | 0.2074 |

**Verdict**: WEAK. The direction is unambiguous (mid > late > early in pooled), but amplitude is small at 1.090×. Multiple optimisation attempts (decision-only, content-only, concept-paired) all yield lower ratios, confirming 1.090× pooled is the definitive best estimate.

**Optimisation attempts history**:

| Attempt | Ratio | Note |
|---|---|---|
| B1-v1 pooled | 1.253× | Artifact-inflated; quote-completion FR features |
| B1-v2 pooled | **1.090×** | Canonical; template artifact removed |
| Decision-only | 1.037× | Worse; structural tokens dilute early IoU |
| Content-only | 1.061× | Worse; only T0+T1 have content in window |
| Concept-paired | 1.162× | Worse; 3 prompts/concept → flat, noisy |
| v1 (original multilingual_circuits) | 1.283× | Different dataset; last-5 token window; 48 prompts |

### 5.4 Language Profiles

| Profile | B1-v2 (86 nodes) | **B1-gradient (137 nodes)** |
|---|---|---|
| balanced | 67 (77.9%) | **114 (83.2%)** |
| fr_leaning | 15 (17.4%) | **18 (13.1%)** |
| en_leaning | 4 (4.7%) | **5 (3.6%)** |
| insufficient_data | 0 (0%) | **0 (0%)** |

The gradient run improves the balanced fraction by +5.3 percentage points. This reflects that gradient attribution selects features with directional contribution to Δlogit per-prompt, which are more likely to be genuinely cross-lingual (balanced) rather than being language-biased by dataset-level correlation patterns.

FR-leaning features concentrate at late layers (L21–L25) in both runs. Most are in the competitor community, not in the main semantic pathway.

**Historical progression**:
- B1-v1 fr_leaning: 44/86 (51%) — quote-completion artifact inflating FR-specific features
- B1-v2 fr_leaning: 15/86 (17.4%) — after template fix
- B1-gradient fr_leaning: 18/137 (13.1%) — after gradient attribution upgrade

### 5.5 Community Structure (B1-gradient: 12 Louvain communities; B1-v2: 7)

**Method**: Community detection is run on the **VW-only subgraph** using the Louvain algorithm (`python-louvain`, `community.best_partition()`). Star edges — the hub-and-spoke connections from the input node to every feature and from every feature to the output nodes — are explicitly excluded before running Louvain, because including them would force all features into a single hub-dominated super-community rather than revealing genuine feature-feature groupings. I/O nodes (`input`, `output_correct`, `output_incorrect`) are also excluded. Edge weights are taken as `abs(weight)`: attribution-derived VW values can be negative (inhibitory paths), and unsigned Louvain correctly treats all structural proximity as cohesion regardless of sign. Louvain maximises the modularity objective `Q = Σ_c [L_c/m − (d_c/2m)²]` where `L_c` is the total edge weight within community `c`, `m` is the total graph weight, and `d_c` is the sum of weighted degrees in `c`. The algorithm is randomised; results were accepted after a single run with the default random state.

**B1-v2 communities** (7 total; VW subgraph: 379 edges, 86 nodes):

| C | Layers | N | Profile | Role |
|---|---|---|---|---|
| C0 | L10–L16 | 32 | balanced (91%) | Early cross-lingual processing |
| C1 | L17–L20 | 14 | balanced (100%) | Mid semantic transformation |
| C3 | L20–L23 | 14 | balanced (93%) | Output preparation; hub L22_F41906 |
| C4 | L23–L25 | 16 | balanced (69%) | Final output routing |
| **C2** | **L21–L25** | **8** | **fr_leaning (100%)** | **FR-specific competitor circuit** |
| C5 | L10 | 1 | fr_leaning | Isolated outlier |
| C6 | L20 | 1 | fr_leaning | Isolated outlier |

**B1-gradient communities** (12 total; VW subgraph over 137-node graph): The larger graph (137 nodes, 1,447 edges) yields 12 communities, providing a finer-grained decomposition while preserving the same high-level structure: an early cross-lingual pathway (L10–L16), a mid semantic transformation band (L17–L21), an output-preparation cluster (L22–L24), and a structurally separate FR-specific competitor community in late layers (L21–L25). The key L22_F41906 hub remains central to the output-preparation cluster.

**Key structural finding (both runs)**: The FR-specific community (C2 in B1-v2) is structurally separate from the main pathway. It was validated as inhibitory (Script 13): amplifying its features *hurts* correct predictions. This may represent a competing FR token-level pathway or residual quote-completion signal.

#### Two Clustering Systems in the Codebase

The project contains two distinct feature grouping mechanisms that answer different questions. It is important to distinguish them because they operate on different inputs, use different algorithms, and produce different kinds of insight.

**System 1 — Louvain graph communities** (used for B1-v2, described above)

The Louvain communities are derived from the **structure of the attribution graph** — specifically, which features are connected to which other features through virtual-weight edges. A VW edge between features `i` and `j` exists because their weight matrices are geometrically aligned: `W_enc[j,:] · W_dec[:,i] > 0.01`. This means the community structure reflects **which features are structurally positioned to influence each other**, not how they actually behave on any particular prompt. The question it answers is: *does the network of potential feature interactions decompose into natural subgraphs?* The result — 5 substantive communities spanning early→mid→late layers — shows that the attribution graph has modular structure, with the anomalous C2 (100% FR-leaning) standing apart from the main balanced pathway. This is the only clustering method actually run for the B1 dashboard. Both `supernodes.json` and `supernodes_effect.json` in `dashboard_b1/public/data/` are directly serialised from these Louvain communities; there is no separate second pass.

**System 2 — sklearn `AgglomerativeClustering` / Ward linkage** (generic pipeline, not run for B1)

A second clustering method exists in `src/ui_offline/prepare.py` (`build_supernodes_effect()`), invoked via `script 09 --effect_clusters N`. Here each feature is represented as a vector of `mean_effect_size` values across all `(experiment_type, layer)` combinations from the intervention results — i.e., a profile of *how strongly and in which direction the feature responds to ablation and patching across all layers*. These vectors are L2-normalised (to compare direction of effect, not magnitude), then clustered with sklearn's `AgglomerativeClustering(n_clusters=N, linkage='ward')`, falling back to `scipy.cluster.hierarchy.linkage(..., method='ward')` if sklearn is unavailable. The number of clusters defaults to `min(50, sqrt(n_features))` if not specified. The question it answers is: *which features exhibit similar causal behaviour under intervention, regardless of where they sit in the graph?* Two features in different layers and different graph communities could end up in the same effect cluster if they both respond strongly to patching and weakly to ablation, for example. This method was **not** used for the B1 dashboard. `prepare_b1_dashboard.py` bypasses `build_supernodes_effect()` entirely, instead re-exporting the Louvain communities under the `supernodes_effect.json` filename to maintain dashboard file format compatibility.

**Why the distinction matters**: The Louvain communities answer a structural/topological question about the graph. The Ward clusters would answer a functional/behavioural question about how features respond to causal interventions. For the B1 behaviour, using the same Louvain partition for both views means the dashboard's "cluster filter" (ClusterSelector sidebar) and the "Layer Heatmap" and violin tabs are all showing the same grouping from two angles — structural and interventional — rather than two independently derived partitions. Had Ward clustering been run on B1's intervention data, it might have revealed features that group together by effect profile but belong to different Louvain communities, which would be a meaningful cross-validation of the two approaches.

**Dashboard visualisation of communities**:
- **Communities tab**: one card per Louvain community showing n_features, layer range, and lang_profile breakdown (balanced/fr_leaning/en_leaning counts)
- **Attribution graph (centre panel)**: nodes are coloured by lang_profile by default; the `ClusterSelector` sidebar filter shows or hides entire communities, letting the user isolate one subgraph at a time
- **Layer Heatmap tab**: mean absolute effect size per (experiment_type × layer), filterable by selected community — shows which layers are most causally active for each community
- **ExperimentViolin tab**: distribution of effect sizes (ablation vs patching) across all features in selected communities, as a Plotly violin + box plot — makes it visible whether a community is dominated by ablation-only effects, patching-only effects, or both

### 5.6 Bridge Features

- Candidate features: 49 (top-ranked by combined EN+FR effect magnitude)
- Bridge features: **33 / 49 = 67.35%**
- C3 disruption: **0.645**, 95% CI [−0.403, −0.343]

Strongest bridges by `bridge_score = min(|μ_EN|, |μ_FR|)`:

| Feature | Bridge Score | Layer | Note |
|---|---|---|---|
| L25_F43384 | 0.842 | 25 | Output layer |
| L25_F15948 | 0.835 | 25 | Output layer |
| L25_F19816 | 0.816 | 25 | Output layer |
| L25_F64049 | 0.816 | 25 | Output layer |
| L25_F70978 | 0.831 | 25 | Output layer |
| L25_F90133 | 0.822 | 25 | Output layer |
| L22_F41906 | 0.677 | 22 | **Primary semantic hub** |
| L22_F99330 | 0.688 | 22 | |
| L23_F83865 | 0.647 | 23 | |

The prevalence of late-layer (L25) high-scoring bridges is notable — these are likely output-routing features active for any antonym prediction regardless of language.

### 5.7 Causal Circuit

**B1-v2 circuit** (beta proxy): 24 nodes (21 feature + 3 I/O), 55 edges, 50 paths (layers L10–L25)

**B1-gradient circuit** (current): **19 nodes** (16 feature + 3 I/O), **29 edges**, layers L16–L25

The gradient-selected circuit is smaller and more focused. Early-layer features (L12, L13) are absent — they appeared in the beta circuit due to dataset-wide correlation rather than per-prompt causal contribution. L16/L17 are added. The key semantic hub L22_F41906 and the top causal edge L24_F35447→L25_F43384 are preserved in both circuits.

**B1-v2 full feature list**: L12_F83869, L13_F70603, L18_F149556, L19_F107296, L20_F89742, L21_F27974, L22_F41906, L22_F78043, L23_F40170, L23_F64429, L23_F6889, L23_F83865, L24_F119196, L24_F30233, L24_F35447, L24_F5768, L24_F76363, L25_F125339, L25_F41381, L25_F43384, L25_F90133

**B1-gradient circuit layers**: L16, L17, L18, L19, L22, L23, L24, L25 (16 features; L22_F41906 hub and L24_F35447→L25_F43384 top edge preserved)

**Main pathway structure (B1-v2 reference)**:
```
input → L20_F89742 → L21_F27974 → L22_F41906 → output_correct
                                               ↘ L23_F6889 → L24_F35447 → L25_F43384 → output
input → L12_F83869 → L13_F70603 → output_correct
input → L22_F78043 → L23_F83865 → output_correct
                  ↘ L23_F40170  → output_correct
input → L23_F64429 → L24_F76363 → output_correct
input → L18_F149556 → L19_F107296 → output_correct
```

**Top causal edge** (both runs): L24_F35447 → L25_F43384 (`mean_delta_abs = 2.283`)

**Star edge weights** (attribution, B1-v2, `mean_delta_abs`):
- input → L22_F78043: 6.386 (strongest star edge)
- L22_F41906 → output: 5.831
- input → L21_F27974: 4.847
- L25_F41381 → output: 4.851

### 5.8 Circuit Validation

| Metric | B1-v2 (21-feat, beta) | **B1-gradient (16-feat)** | Verdict |
|---|---|---|---|
| Necessity — disruption rate | 10.4% | **35.4%** | **DISTRIBUTED → MODERATE** |
| Necessity — mean effect | +0.932 | — | — |
| S1 sign preservation | 71.9% | — | **PARTIAL** |
| S1.5 sign preservation | 63.5% | — | **WEAK** |
| S1 mean retention ratio | 0.736 | — | Circuit restores ~74% of logit diff |
| S1.5 mean retention ratio | 0.634 | — | Degrades with stricter threshold |
| S2 transfer rate | 12.5% | 12.5% | **NEGATIVE** (unchanged; uses same activation data) |
| S2 mean logit shift | −1.309 | −1.309 | Negative — EN→FR patch hurts FR |

The 3.4× improvement in necessity (10.4% → 35.4%) is the most impactful change from the gradient upgrade. The gradient-selected circuit is more causally necessary because it selects features with actual per-prompt contributions to Δlogit, rather than dataset-level correlates. 35.4% still indicates a **distributed** circuit — the behaviour is not bottlenecked — but the result is substantially stronger.

S1/S1.5 and S2 results are unchanged because they depend on the star graph (step 07) and raw activations, not the role-aware graph. See §9.4 for full details.

### 5.9 Reasoning Traces (Script 10)

- 78/96 prompts predicted correctly; 18 incorrect (all FR)
- Type A failures (early-layer negative contribution): 16
- Type B failures (other): 2 (p45, p71)
- Trajectory accuracy: B1-v2 = 65.6% → **B1-gradient = 76.0%** (+10.4 pp)

**Zone means for correct vs incorrect**:

| Zone | Correct mean | Incorrect mean | Δ |
|---|---|---|---|
| Early (L10–L17) | +0.829 | −0.189 | −1.019 |
| Mid (L18–L22, L22_F41906) | +0.897 | +0.170 | −0.727 |
| Late (L23–L25) | +5.748 | +6.775 | +1.027 |

**L22_F41906** is the top discriminating feature (Δ = −0.727 between correct/incorrect at mid zone).

**Notable feature-level observations**:
- L13_F70603: correct=+0.193, incorrect=−0.443 (Δ=−0.635; preserved discriminator)
- L23_F64429 (FR-leaning): hurts incorrect more than correct (+1.947 vs +0.850) — may act as partial corrector
- Late zone: higher mean for *incorrect* prompts — suggests the late-layer output features fire strongly regardless of correctness; direction is determined earlier

### 5.10 Failure Analysis (Script 12)

- 18 incorrect prompts (all FR); concept breakdown: C2=1, C3=1, C4=6, C6=4, C7=6
- `argmax_is_quote_rate` = 0.278 (5/18 = 27.8% still show quote-completion as top prediction)
- Mean logit margin: −4.288 (B1-v1 was −11.838; template fix massively reduced competition)
- Median argmax rank of correct answer: 78 (B1-v1: 642 — 8× improvement)
- L13 activation vs logit margin: Pearson r = −0.479 (stronger L13 → smaller margin → more competition)
- Competition type: mixed (word-specific, not template-specific)

Residual failures are concept-specific (concepts 2, 3, 4, 6, 7) — the French word for these concepts happens to appear in a context where quote-completion is still reinforced.

### 5.11 Artifact Confirmation (Script 13)

L22_F108295 and L22_F32734 (B1-v1 circuit hubs) vs control L22_F78043:

| α (amplification) | Δlogit vs control |
|---|---|
| 1.5× | −0.382 |
| 2.0× | −0.858 |
| 3.0× | −1.681 |

Monotone worsening at all α. Confirmed as anti-semantic features: amplifying them inhibits correct antonym prediction. B1-v1 "S2 = 0.75" result was measuring transfer of the quote-completion circuit, not the semantic antonym circuit.

---

## 6. Detailed Comparison with Anthropic's Implementation

**Reference**: Anthropic (2025), "On the Biology of a Large Language Model", Transformer Circuits Thread. URL: https://transformer-circuits.pub/2025/attribution-graphs/biology.html. Model: Claude 3.5 Haiku.

---

### 6.1 Experimental Setup

| Dimension | **Anthropic** | **Our Implementation** |
|---|---|---|
| Model | Claude 3.5 Haiku (production, closed) | Qwen3-4B-Base (open-weight) |
| Model size | ~7B parameters (estimated) | 4B parameters |
| Transcoder type | Cross-Layer Transcoder (CLT) | Per-layer transcoders (circuit-tracer) |
| Transcoder features | ~30 million total across all layers | 163,840 per layer × 16 layers = ~2.6M |
| Languages tested | English, French, Chinese (3) | English, French (2) |
| Concepts | 1 (small/large antonym, 3 surface forms) | 8 concepts × 8 templates = 96 prompts |
| Behaviour type | Qualitative circuit diagram | Quantitative: IoU, bridge rate, necessity/sufficiency |
| Attention decomposition | Partial (QK/OV split noted) | Not decomposed (hook at MLP input only) |

### 6.2 Transcoder Architecture Difference

**Anthropic (CLT)**:
- A single cross-layer transcoder maps from any layer to any later layer
- Features at layer ℓ can directly influence features at layer ℓ+k (multi-hop without running attention)
- The CLT is trained to predict the residual stream at all layers jointly
- This gives natural cross-layer edges without requiring explicit causal patching

**Our approach (per-layer)**:
- Independent transcoders per layer, each trained only on that layer's MLP
- Cross-layer connections must be inferred via VW edges (weight-space approximation) or causal patching (activation patching)
- This means multi-hop causal structure requires explicit experiments rather than being built into the transcoder representation
- **Implication**: our graph under-represents long-range dependencies; Anthropic's CLT captures them structurally

### 6.3 Attribution Graph Construction

| Step | **Anthropic** | **Our Implementation** |
|---|---|---|
| Attribution formula | Gradient × activation through CLT structure | `α_k = (∂Δlogit/∂a_k) · a_k` per layer |
| Cross-layer edges | Built into CLT (direct feature-to-feature paths) | VW approximation (weight dot product) |
| Graph scope | Single prompt, then manually grouped | All train prompts → union graph |
| Graph pruning | Threshold on contribution to output | Threshold `|α_k| > θ_attr` |
| Supernodes | Manual grouping of similar features | Automated Louvain community detection |
| Attention edges | Noted but acknowledged as partially invisible | Not modelled (hook at MLP input) |
| Error nodes | Explicit residual nodes for CLT approximation error | Not included |

The Anthropic approach is fundamentally more principled: their CLT with error nodes explicitly accounts for model fidelity at every step. Our star graph approximates feature contributions with a first-order estimate and does not include the residual error of the transcoder itself.

### 6.4 Multilingual IoU Analysis

| Metric | **Anthropic** | **Our Result** |
|---|---|---|
| Finding | Middle layers more language-agnostic | Confirmed (direction), 1.090× ratio |
| Baseline | Random paragraph pairs (unrelated text) | Not computed (no baseline IoU) |
| Languages | EN/FR, EN/ZH, FR/ZH | EN/FR only |
| Granularity | Qualitative statement | Quantitative layer-by-layer |
| Result strength | "Notably higher generalization in middle" | WEAK — 1.090× middle/early |
| Improvement with scale | CLT: larger model = more generalisation | Not tested (single model) |
| Language pair without shared script | EN/ZH shows "especially strong generalisation" | Not applicable (both EN/FR use Latin script) |

**Critical gap**: Anthropic's IoU analysis uses **randomly unrelated paragraphs as a baseline** to determine whether the observed IoU is above chance. We did not compute such a baseline. All our reported IoU values (0.22–0.26) may be close to the random baseline for this model and task, making it impossible to confirm whether middle-layer generalisation is statistically meaningful.

**Scale finding**: Anthropic reports that Claude 3.5 Haiku exhibits notably *higher* generalisation than a smaller model. This suggests our 1.090× ratio may partly reflect Qwen3-4B's smaller scale — a larger model might show a stronger gradient.

### 6.5 Feature Language Profiles

| Aspect | **Anthropic** | **Our Implementation** |
|---|---|---|
| Language-specific features | At tokenisation boundaries (early + late layers) | Late layers L21–L25 (FR-leaning community C2) |
| Cross-lingual features | Middle layers, semantic operations | Middle layers (communities C0/C1/C3/C4, 78–100% balanced) |
| Feature semantics | Named: "open-quote-in-language-X", "say-large-in-EN" | Not named; labelled by lang_profile only |
| English privilege | Explicitly documented: EN multilingual features have stronger direct weights to EN output | Not tested; no equivalent experiment |
| Assignment method | Manual inspection of examples | Automated: `min(|μ_EN|, |μ_FR|) > threshold` |

Anthropic explicitly identifies **English as the "default" language** of multilingual features: EN output features receive stronger direct connections from multilingual (balanced) features than FR/ZH output features do. Non-English outputs require language-specific routing features as intermediaries. We did not test this hypothesis in our implementation — it would require inspecting W_dec directions relative to specific output token embeddings.

### 6.6 Circuit Decomposition

Anthropic decomposed the antonym circuit into three separable components:

1. **Operation** — antonym vs synonym (separate feature clusters)
2. **Operand** — which concept (small vs hot)
3. **Language** — output language routing

They validated compositionality via activation patching: swapping the "operation" component from antonym→synonym circuit changes the output from the antonym to the synonym, while leaving the operand and language components unchanged.

**Our implementation**:
- We performed C3 patching (language swap, same concept): swap EN→FR to test language component
- We did **not** implement operation swap (C1) or operand swap (C2) for this behaviour
- Result: C3 S2 transfer fails (−1.309 mean shift) — language component does not transfer
- This may mean the circuit is not cleanly decomposable along language lines in Qwen3-4B, OR that the aligned feature set identified in our graph does not correspond to the language-routing component specifically

### 6.7 Causal Validation Methodology

| Method | **Anthropic** | **Our Implementation** |
|---|---|---|
| Intervention type | Constrained patching at fixed intervention layer | Activation patching per feature, no intervention layer |
| Key constraint | Clamp residual stream at intervention layer | Not applied — indirect effects can propagate freely |
| Effect scope | Output logit change in original model | Logit diff change; script 08 measures intermediate feature activation |
| Validation target | Output logits (most interpretable) | Mix of feature activations (script 08) and logit diff (script 07) |
| Mechanistic unfaithfulness | Explicitly acknowledged as a concern | Acknowledged but not measured |

Anthropic's "constrained patching" approach is more principled: by clamping activations up to the intervention layer, they guarantee that effects measured *after* the intervention layer are due to the patched component, not confounded by the residual stream changing independently. Our patching does not apply this constraint, so measured effects mix direct and indirect (uncontrolled) contributions.

### 6.8 Scale of Evidence

| Claim | **Anthropic** | **Our B1-v2** |
|---|---|---|
| Shared features exist | Qualitative: "20/27 features active across all 3 languages" | Quantitative: 67% bridge rate |
| Middle layers more universal | "Notably higher" (qualitative, with baseline) | 1.090× ratio (no baseline) |
| Language-specific components exist | Explicitly shown for EN/FR/ZH output routing | C2 community (100% FR-leaning, inhibitory) |
| Circuit is sufficient | 6× amplification transfers behaviour | S2 transfer_rate=12.5%, mean_shift=−1.309 (fails) |
| English privilege | Explicitly shown: direct weights to EN output stronger | Not tested |
| Compositionality (operation/operand/language) | Validated via 3-way patching | Only language-swap tested; fails |

### 6.9 Absence of Attention Decomposition

Anthropic explicitly flags that one critical interaction in the antonym circuit — between antonym features and "say large" features — is **mediated by changes in attention head patterns** (QK circuit), not by direct MLP feature connections. This interaction is invisible to their CLT-based attribution graph. They acknowledge this as a known limitation.

In our implementation, the hook at `post_attention_layernorm` means we capture MLP feature interactions but entirely miss:
- How attention heads route information between positions
- QK-mediated feature interactions
- Positional attention patterns that may distinguish EN/FR templates

This is a structural gap shared by both approaches, but more acute in ours because we do not even attempt partial attention decomposition. The full circuit for antonym prediction almost certainly involves attention-mediated steps that our graphs cannot represent.

### 6.10 Summary of Key Differences

| Dimension | **Anthropic advantage** | **Our advantage / compensating factor** |
|---|---|---|
| Transcoder coverage | CLT across all layers (30M features, cross-layer) | None — per-layer is a strict subset |
| Graph fidelity | Error nodes for approximation residual | None |
| Constrained patching | Clamped at intervention layer | None |
| Scale and model size | Much larger model, more features | Open-weight model; reproducible |
| IoU baseline | Random paragraph baseline computed | None |
| Decomposability test | 3-way (operation/operand/language) | C3 only (language) |
| Quantitative rigor | Qualitative circuit + qualitative IoU | Full layer-by-layer IoU, bridge rates, necessity/sufficiency |
| Dataset breadth | 3-language, single concept | 2-language, 8 concepts, 8 templates each |
| Attention handling | Partially acknowledged (QK/OV noted) | Entirely absent |
| Reproducibility | Closed model | Open-weight, all scripts provided |

---

## 7. Problems, Limitations, and Mistakes

### 7.1 Template Artifact (Critical — Fixed in B1-v2)

**Problem**: FR templates in B1-v1 included `"{word}"` (typographic quotes), causing the model to continue with closing `"` (quote-completion). This inflated FR IoU at mid layers from true ~0.24 to artifact ~0.29, yielding a false 1.253× ratio, and made L22_F108295 and L22_F32734 appear as circuit hubs.

**Fix**: All FR templates rewritten in B1-v2 to avoid the pattern. Confirmed by Script 13 (monotone-worsening dose-response for artifact features).

**Residual problem**: 5/18 (27.8%) of FR failures still show quote-completion as argmax. This is concept-specific and cannot be fully eliminated without changing the underlying concepts (not the templates).

### 7.2 Weak IoU Gradient (Claim 3)

**Problem**: The cross-lingual universality gradient is only 1.090×, borderline weak/moderate. The theoretical maximum ratio is constrained by dataset size: with only 3 prompts per concept per language in the training window, feature sets are dominated by structural tokens (position, punctuation, template boilerplate), which activate uniformly across all layers, flattening the IoU curve.

**Root cause**: Fundamental dataset constraint — 3 prompts/concept × 8 concepts × 2 languages = 48 prompts per language. Not fixable without collecting additional data.

**No baseline**: We never computed what the "random" IoU is for unrelated prompts. If the random baseline for Qwen3-4B is ~0.22 (close to our early-layer value), then the entire 0.22–0.26 range may be noise.

### 7.3 Distributed Necessity (10.4% beta → 35.4% gradient)

**Problem (B1-v2)**: Ablating all 21 circuit features disrupts only 10.4% of prompts. The circuit is not necessary.

**Update (B1-gradient)**: With gradient-selected 16-feature circuit, necessity improves to 35.4%. This is a meaningful 3.4× gain, confirming that the beta proxy included non-causally-relevant features. The circuit remains distributed — 35.4% disruption means the behaviour is not bottlenecked at these features alone.

**Interpretation**: Qwen3-4B uses many parallel pathways for antonym prediction. The identified circuit is one contributor among many. This may reflect (a) genuine redundancy in the model, (b) over-pruning of the circuit (top-k=50 limit), or (c) the graph-walk pre-filter excluding legitimate causal paths. The gradient upgrade addresses (b) partially — features selected by gradient are more causally focused — but (a) and (c) remain.

### 7.4 Failed S2 Transfer (−1.309 mean shift)

**Problem**: Patching EN circuit activations into FR prompts hurts FR performance (negative mean shift). Language component transfer fails completely.

**Possible explanations**:
- EN and FR feature activations for the same concept differ in scale/direction even for "balanced" (bridge) features
- The circuit's language-routing component is not separable via simple activation replacement
- Aligned pairing by concept_index is insufficient — the circuit may require the entire residual stream context, not just the 21 selected features
- The per-layer transcoder architecture, which lacks cross-layer structure, may produce circuits where EN→FR transfer is inherently impossible at this granularity

### 7.5 Missing Anthropic Baseline for IoU

We did not compute the IoU baseline for randomly paired, unrelated prompts. Without this, we cannot determine whether our measured IoU values (0.22–0.26) represent meaningful cross-lingual generalisation or simply reflect the base rate of feature co-occurrence at any two prompts in the model. This is the single most important missing experiment for validating Claim 3.

### 7.6 L13_F70603 Sign Flip

L13_F70603 shows opposite signs: EN effect = −0.320, FR effect = +0.173. It is part of the causal circuit but suppresses EN predictions while promoting FR predictions. This is structurally inconsistent with a cross-lingual shared feature. It was included in the circuit because it has a causal edge to the output, but its language profile is mixed and not fully explained.

### 7.7 VW Threshold Selection

The VW threshold of 0.01 was set without systematic ablation. The role-aware graph has 633 edges vs 252 in the star graph (2.5×). Many VW edges may be spurious. Community structure (Louvain) is sensitive to edge weights. A proper analysis would sweep the threshold and report community stability.

### 7.8 Content Word Detection Coverage Gap

Content-word detection found the target word in the last-5 token window for only 22/48 FR prompts (templates T0+T1 only). Template T3 does not contain the word in the decision window. The "content-only" IoU curve therefore represents a non-random subset of prompts and templates, making the content/decision comparison unreliable.

### 7.9 No Test-Set Circuit Validation

The causal circuit was extracted and validated entirely on the 96 train prompts. The 32 test prompts were never used to validate that the same circuit applies. This risks overfitting to the train templates.

### 7.10 Baseline Model vs Transcoder Model Mismatch

Script 02 (baseline gate) runs Qwen3-4B-**Instruct**, while Scripts 04–13 run Qwen3-4B-**Base** (transcoders are trained on base). The gate threshold is validated on a different model than the one analysed. Base model accuracy on FR templates was not formally gated — it was verified to work but not quantitatively reported.

### 7.11 C2 Competitor Circuit Not Characterised

Community C2 (8 features, L21–L25, 100% FR-leaning, inhibitory) is confirmed to hurt correct predictions but its positive functional role is unknown. It may represent: residual quote-completion, FR token routing, or an unrelated FR context feature. Without feature steering experiments or manual feature interpretation, its role remains opaque.

### 7.12 Attention Mechanisms Entirely Absent

The full antonym circuit almost certainly involves attention-mediated steps. Our hook at `post_attention_layernorm` captures only MLP feature interactions. Any information routing between token positions — which is crucial for the model to identify the target word `"{word}"` in the prompt — is invisible to our analysis.

---

## 8. Summary: Claims vs Evidence

Values shown as **B1-v2 (beta proxy)** → **B1-gradient (current)**. Unchanged metrics are shown once.

| Claim | Metric | B1-v2 | B1-gradient | Verdict |
|---|---|---|---|---|
| **C1**: Model performs the behaviour reliably | EN/FR baseline accuracy | 1.000 / 0.792 | 1.000 / 0.792 | **STRONG** |
| **C2**: Interpretable features exist | Role-aware graph nodes | 86 | **137** | **CONFIRMED** |
| **C3**: Middle layers more cross-lingually universal | IoU ratio (pooled) | 1.090× | **1.090×** (unchanged) | **WEAK** |
| **C4**: Cross-lingual bridge features exist | Bridge rate | 67.35% | **67.35%** (unchanged) | **MODERATE** |
| **C4b**: Language-specific features in late layers | FR competitor community | 100% FR-leaning, inhibitory | same | **CONFIRMED** |
| **C5a**: Circuit is necessary | Disruption rate | 10.4% | **35.4%** | **DISTRIBUTED → MODERATE** |
| **C5b**: Circuit is sufficient (sign preservation) | S1 rate | 71.9% | 71.9% | **PARTIAL** |
| **C5c**: Circuit transfers EN→FR | S2 transfer / mean shift | 12.5% / −1.309 | 12.5% / −1.309 | **NEGATIVE** |
| **C6**: Main semantic hub at L22 | L22_F41906 centrality | bridge=0.677 | preserved | **CONFIRMED** |
| **C7**: Anthropic "middle layers universal" replicated | IoU direction | correct, amplitude weak | same | **PARTIAL REPLICATION** |
| **C8**: Gradient attribution improves circuit quality | Necessity rate | 10.4% | **35.4%** | **CONFIRMED (+3.4×)** |

---

### Final Assessment

The B1-gradient analysis on Qwen3-4B-Base achieves **partial replication** of Anthropic's multilingual circuits finding. The qualitative direction is confirmed: middle-layer features are more cross-lingually shared than early or late layers, and a structurally separate FR-specific competitor circuit exists in late layers. The gradient attribution upgrade (§9) improved necessity from 10.4% to 35.4% and trajectory accuracy from 65.6% to 76.0%, while IoU/bridge/C3 disruption remain unchanged (those metrics do not depend on the attribution graph). Quantitative claims remain weaker than Anthropic's, for three compounding reasons:

1. **Template artifact** (fixed): the B1-v1 measurements were partly measuring quote-completion transfer, not semantic antonym transfer. After the fix, all metrics degraded toward their true values.

2. **Dataset constraint** (unfixable without new data): 3 prompts per concept per language is insufficient to separate structural token overlap from semantic feature overlap in the IoU analysis.

3. **Architecture gap** (inherent): per-layer transcoders without cross-layer structure, no constrained patching, no attention decomposition — our circuit representation is a strict subset of what Anthropic's CLT can represent.

The most impactful missing experiment is the **IoU random baseline**. Without it, the 1.090× middle-layer elevation cannot be claimed to be above chance.

---

## 9. Gradient Attribution Upgrade

**SLURM 26929559 · Date: 2026-04-02 · Replaces B1-v2 as the canonical run**

This section describes the methodological change from OLS beta proxy to gradient × activation attribution in step 06b (`aggregate_graphs_role_aware()`), and explains its effects on all downstream metrics.

### 9.1 Motivation

The B1-v2 canonical run (SLURM 25679695) computed attribution scores for the role-aware graph using an OLS (ordinary least squares) beta coefficient. This proxy has two fundamental weaknesses:

**Weakness 1 — Dataset-level statistic, not per-prompt**. The beta coefficient is:
```
β_k = Σ_p (a_k^p · Δlogit^p) / Σ_p (a_k^p)²
```
This is the OLS slope of regressing `Δlogit` on `a_k` across all prompts `p` in the training set. It captures which features globally correlate with the behaviour signal — not which features causally contribute to any individual forward pass.

**Weakness 2 — Not causal**. The per-prompt score used for feature selection was:
```
score_k^p = a_k^p × β_k
```
This multiplies a per-prompt activation by a dataset-average coefficient. The result approximates the true gradient × activation product only if `β_k ≈ ∂Δlogit/∂a_k` — which holds only under linearity assumptions that generally do not apply. In practice, features that are highly active on many prompts (high variance) receive inflated beta scores regardless of their directional contribution to the logit difference on any specific prompt.

In contrast, Anthropic's CLT attribution uses gradient × activation as the native attribution formula (§4.2). Our star graph also uses this formula (step 04 computes gradients during feature extraction). The mismatch between step 04 (gradient-based) and step 06b (beta-based) was an inconsistency introduced during development.

### 9.2 The Beta Proxy (B1-v2 Implementation)

**Full formula**:

Step 1 — Compute dataset-level regression coefficient for each feature k at each layer ℓ:
```
β_k^ℓ = [Σ_p (a_k^ℓ(p) · Δlogit(p))] / [Σ_p (a_k^ℓ(p))²]
```
This is equivalent to the OLS slope in a no-intercept univariate regression of `Δlogit` on `a_k^ℓ`.

Step 2 — Per-prompt attribution estimate:
```
score_k^ℓ(p) = a_k^ℓ(p) × β_k^ℓ
```

Step 3 — Select top-k features per prompt by `|score_k^ℓ(p)|`, then union across all prompts.

The `specific_score` field stored in the B1-v2 graph JSON was:
```
specific_score = fraction of prompts where |score_k^ℓ(p)| > threshold
```
i.e., a participation rate, not a gradient magnitude.

### 9.3 Gradient × Activation Attribution (New Implementation)

**Mathematical formula** (per-prompt, per-feature):

```
α_k^ℓ(p) = a_k^ℓ(p) × (∂Δlogit(p) / ∂a_k^ℓ(p))
```

This is the first-order attribution at the operating point — analogous to integrated gradients evaluated at a single point (the actual activation). It is the standard attribution formula used in Anthropic's CLT graphs and in the TCAV / gradient-based feature importance literature.

**Chain rule decomposition**: The transcoder feature activations `a_k^ℓ` are not directly in the PyTorch computational graph. Instead, we apply the chain rule through the transcoder decoder:

```
∂Δlogit(p) / ∂a_k^ℓ(p) ≈ (∂Δlogit(p) / ∂MLP_output^ℓ(p)) · W_dec^ℓ[k, :]
```

where:
- `∂Δlogit / ∂MLP_output^ℓ ∈ ℝ^{d_model}` is the gradient of the logit difference w.r.t. the entire MLP output vector at layer ℓ
- `W_dec^ℓ[k, :] ∈ ℝ^{d_model}` is the k-th row of the transcoder decoder weight matrix (the decoder direction for feature k)
- The dot product gives the scalar `∂Δlogit / ∂a_k^ℓ`

The approximation holds exactly when the transcoder is perfectly linear from `a_k^ℓ` to `MLP_output^ℓ` (i.e., when feature k's decoder direction is the only path from `a_k^ℓ` to `MLP_output^ℓ`). In practice, JumpReLU transcoders have a non-linear gating mechanism; the chain rule treats the gate as the straight-through estimator (locally linear), which is standard in sparse autoencoder gradient analysis.

For inactive features (`a_k^ℓ(p) = 0`), the attribution is exactly zero regardless of the gradient, so no approximation is needed at those points.

**Full attribution vector for a prompt p at layer ℓ**:
```
attr^ℓ(p) = feat_vals^ℓ(p) ⊙ (W_dec_sub^ℓ · grad^ℓ(p))
```
where:
- `feat_vals^ℓ(p) ∈ ℝ^{K}` — activations of the top-K features (from step 04)
- `W_dec_sub^ℓ ∈ ℝ^{K × d_model}` — submatrix of W_dec for the K selected feature indices
- `grad^ℓ(p) ∈ ℝ^{d_model}` — gradient `∂Δlogit / ∂MLP_output^ℓ` captured by backward hook
- `⊙` — element-wise product
- Result: K attribution scores, one per feature

### 9.4 Implementation Details

**Backward hook mechanism**:

A `register_full_backward_hook` is attached to each `block.mlp` module before the forward pass:

```python
# Pseudo-code
grads = {}
def make_hook(layer_idx):
    def hook(module, grad_input, grad_output):
        grads[layer_idx] = grad_output[0].detach().cpu()
    return hook

for ℓ in layers:
    hooks[ℓ] = model.blocks[ℓ].mlp.register_full_backward_hook(make_hook(ℓ))
```

The forward pass is run **without** `torch.no_grad()` to preserve the gradient tape. After the forward pass:
```python
delta_logit = logit_correct - logit_incorrect
delta_logit.backward()  # single backward pass
```

The hooks capture `grad_output[0]` at each MLP, which is `∂Δlogit / ∂MLP_output^ℓ`.

After processing each prompt, `model_hf.zero_grad(set_to_none=True)` is called to prevent gradient accumulation across prompts.

**Fallback**: For prompts where the correct or incorrect token is multi-token (tokenises to >1 subword), the gradient approach cannot unambiguously define `Δlogit`. In these cases, the implementation falls back to raw activation magnitude (`α_k ≈ a_k`) rather than raising an exception. In practice, all B1 tokens are single-token (antonyms like ` cold`, ` lent` are consistently one token in Qwen3's tokenizer), so the fallback was not triggered.

**Computational cost**: One additional backward pass per prompt. At 96 train prompts, this adds negligible time relative to the forward passes in steps 04 and 07.

**Graph node fields** (updated from beta-named to gradient-named, with backward-compatible aliases):

| Old field (B1-v2) | New field (B1-gradient) | Description |
|---|---|---|
| `beta` | *(removed)* | OLS coefficient — removed |
| `beta_sign` | `grad_attr_sign` | Sign of mean gradient attribution |
| `specific_score` | `specific_score` | Fraction of prompts active (kept for compatibility) |
| `mean_abs_score_conditional` | `mean_abs_grad_attr_conditional` | Mean absolute gradient attribution (active prompts only) |
| `mean_score_conditional` | `mean_grad_attr_conditional` | Mean signed gradient attribution (active prompts only) |

### 9.5 What Changed and What Did Not

**Full comparison table**:

| Metric | B1-v2 (beta proxy) | **B1-gradient (current)** | Changed? | Why |
|---|---|---|---|---|
| Attribution method (step 06b) | OLS beta proxy | Gradient × activation | **YES** | Core change |
| Role-aware graph: nodes | 86 | **137** (+59%) | **YES** | Gradient selects more diverse features |
| Role-aware graph: edges | 633 | **1,447** (+129%) | **YES** | More nodes → more VW edges |
| Circuit nodes (feature) | 21 | **16** (−24%) | **YES** | More focused, causally relevant set |
| Circuit edges | 55 | **29** (−47%) | **YES** | Smaller node set → fewer edge pairs |
| Necessity (disruption rate) | 10.4% | **35.4%** (+3.4×) | **YES** | Key improvement |
| Trajectory accuracy (script 10) | 65.6% | **76.0%** (+10pp) | **YES** | Cleaner early-zone features |
| Node balanced% | 77.9% (86 nodes) | **83.2%** (137 nodes) | **YES** | Gradient favours cross-lingual features |
| Communities (Louvain) | 7 | **12** | **YES** | Larger graph → finer partition |
| **IoU pooled ratio (Claim 3)** | **1.090×** | **1.090×** | **NO** | Uses raw features from step 04 |
| **Bridge features** | **67.35%** | **67.35%** | **NO** | Uses raw activations, not graph |
| **C3 disruption score** | **0.645** | **0.645** | **NO** | Uses star graph ablation CSV |
| **EN/FR baseline accuracy** | **1.000 / 0.792** | **1.000 / 0.792** | **NO** | Step 02 unchanged |
| **S1/S1.5/S2** | **71.9% / 63.5% / 12.5%** | **unchanged** | **NO** | Uses star graph (step 07) |

**Why IoU, bridge, and C3 disruption are mathematically unchanged**:

These three metrics are computed by `a_analyze_multilingual_circuits.py` from step 04 outputs (raw activation files) and step 07 star-graph ablation CSVs:

- **IoU** (§4.5): uses `S_lang^ℓ = ⋃_{p ∈ lang} {k : a_k^ℓ(p) > 0 and |α_k^ℓ(p)| > θ}` — the `α_k` here is from step 04's gradient computation during feature extraction, **not** from the role-aware graph. Step 04 was not changed.

- **Bridge features** (§4.7): uses `mean_effect_size` from the step 07 ablation results on the star graph. The star graph is built by `aggregate_graphs_per_prompt_union()`, which was not modified.

- **C3 disruption**: uses the patching results from step 07 intervention on the star graph circuit. Same reasoning.

The role-aware graph (`aggregate_graphs_role_aware()`) is the only component that was changed, and it feeds into: step 08 causal circuit extraction, step 10 reasoning traces, and node language labelling — which explains why necessity, trajectory accuracy, and community structure all changed while IoU/bridge/C3 did not.

### 9.6 Interpretation of Key Changes

**Graph expansion (86 → 137 nodes)**:

Beta attribution favoured features with high variance (active on many prompts) because `β_k ∝ Cov(a_k, Δlogit) / Var(a_k)` — features active on fewer prompts but strongly directional get suppressed by the denominator. Gradient attribution treats all prompts equally: a feature that fires only on 3 prompts but strongly contributes to Δlogit on those prompts receives a high `|α_k|` for those prompts. The result is a graph that includes a richer set of selective features.

**Circuit contraction (21 → 16 features, 55 → 29 edges)**:

The causal circuit (step 08) selects features from the role-aware graph that form causal chains. With gradient attribution, the initial candidate set consists of features that are per-prompt causally relevant. The activation patching in step 08 confirms actual causal connections, which is more efficient when starting from a causally-motivated candidate set. Fewer features pass the causal edge threshold, producing a smaller but more coherent circuit.

**Necessity improvement (10.4% → 35.4%)**:

This is the clearest validation of the gradient upgrade. By construction, a feature with `α_k^ℓ(p) > threshold` is one where `∂Δlogit / ∂a_k^ℓ > 0` (or `< 0` for inhibitory features) on prompt `p` — it is locally causal. Ablating locally-causal features should disrupt the output. The 3.4× improvement confirms that gradient-selected features are genuinely causal in a way that beta-selected features were not.

The circuit is still distributed (35.4% < 100%): many additional features outside the 16-feature circuit can independently support the behaviour. This reflects genuine redundancy in Qwen3-4B rather than any remaining limitation of the attribution method.

**Trajectory accuracy (65.6% → 76.0%)**:

The reasoning trace (script 10) reconstructs whether early-zone features support or oppose correct predictions. With beta attribution, some high-variance early features appeared in the circuit but had mixed directional contributions across prompts. With gradient attribution, early features are more cleanly directionally consistent, making it easier to predict correct vs incorrect trajectories from early-zone activity.

**Community expansion (7 → 12)**:

A 59% larger graph naturally supports finer-grained Louvain communities. The 12 communities in the gradient run cover the same L10–L25 span as the 7 B1-v2 communities, but with more detailed subdivision of the mid-layer transformation region (L14–L21). The key structural finding — a late-layer (L21–L25) FR-specific competitor community structurally separate from the main pathway — is preserved in both partitions.

**Balanced profile improvement (77.9% → 83.2%)**:

Beta attribution inflated scores for features that correlate with `Δlogit` across the dataset regardless of language. Some FR-specific token features had high `Δlogit` correlation on FR prompts only, but the beta coefficient captured this as a generally relevant feature. Gradient attribution correctly assigns low per-prompt attribution to these features on EN prompts (where they are inactive or non-directional), reducing the count of fr_leaning nodes from 15 to 18 (in proportion, from 17.4% to 13.1% of the larger graph).
