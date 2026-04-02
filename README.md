# Mechanistic Interpretability of LLM Behaviours via Transcoders

**Author:** Iuliia Vitiugova
**Affiliation:** DAMTP, University of Cambridge — MPhil Dissertation
**Model:** Qwen3-4B (base for transcoder work; instruct for baseline evaluation only)
**Inspired by:** [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 2025)

---

## Overview

This project investigates whether the circuit-level interpretability methods pioneered by Anthropic on Claude 3.5 Haiku can be reproduced on an **open-source model** (Qwen3-4B). We use **transcoders** (sparse autoencoders trained on MLP activations) to extract interpretable features, build **attribution graphs** tracing causal interactions between features across layers, and validate circuits through three types of **causal interventions**: ablation, activation patching, and feature steering.

The project introduces a formal definition of **behaviour** as a unit of analysis, a **four-type typology** of LLM computations, and an intervention-based **causal edge pipeline** that identifies which feature-to-feature connections are functionally causal (not just correlated).

**Primary focus behaviour:** `multilingual_circuits_b1` — EN+FR antonym prediction across 8 cross-lingual concepts, directly replicating Anthropic's multilingual circuit claims with a larger template set (8 templates per group vs 4 in the original study).

---

## Authoritative Results — `multilingual_circuits_b1`

> **Reference run: SLURM 25307123 (graph), SLURM 25408631 (causal edges), SLURM 25518258 (circuit validation)**
> **Data directory:** `data/results/causal_edges/multilingual_circuits_b1/`

### Baseline

| Metric | Value |
|---|---|
| EN accuracy | 1.000 (48/48 train) |
| FR accuracy | 0.729 (35/48 train) |
| mean_norm_diff | 3.736 |
| Status | **PASS** |

### Attribution Graph

| Metric | Value |
|---|---|
| Graph type | Role-aware (decision + content-word nodes) |
| Feature nodes | 94 |
| Edges | 851 (273 star + 578 VW, threshold=0.01) |
| Layers | 10–25 |
| Communities | 10 (VW-only subgraph) |
| lang_profile | fr_leaning=46.8%, balanced=45.7%, en_leaning=5.3%, insuff=2.1% |

### Causal Edge Pipeline (Script 08)

| Metric | Value |
|---|---|
| Total VW edges | 575 |
| AGW-selected candidates (top 60%) | 345 |
| Causal edges found (τ=0.10) | 98 |
| Causal edge δ range | [−1.164, +1.373] |
| Top edge | L12_F83869→L13_F70603 (δ=1.37, ε=1.92) |
| VW sign concordance | 98% (96/98 match) |

### Circuit (27-Feature, Path-Traced, K=50)

| Metric | Value |
|---|---|
| Features | 27 |
| Edges | 69 (20 causal + 49 star) |
| Paths | 50 |
| Layer span | L10–L25 (14 layers covered) |
| Stable core (K=20) | 14 features |
| K-saturation | K=50 = K=100 (circuit complete at K=50) |
| Tau invariance | Identical circuit for τ ∈ {0.05, 0.10, 0.15, 0.20} |

**14-feature stable core:**
`L10_F141643, L11_F56459, L12_F83869, L13_F70603, L14_F57525, L15_F127839, L16_F45664, L23_F64429, L24_F136810, L24_F29680, L24_F48363, L25_F111603, L25_F138698, L25_F34754`

**Central hub:** `L23_F64429` — 7/8 outgoing causal edges positive; fans out to 5 distinct L24 targets.

### Circuit Validation (SLURM 25518258)

| Phase | Metric | Value | Verdict |
|---|---|---|---|
| **Necessity (Phase 7)** | disruption_rate | 0.375 | MODERATE |
| | mean_effect | +0.178 | Positive (ablation degrades output) |
| **S1 linear (Phase 8)** | sign_preserved | 67.7% | WEAK |
| | mean_retention | 0.263 | WEAK |
| **S1.5 layerwise (Phase 8)** | sign_preserved | 54.2% | WEAK |
| | mean_retention | 0.157 | WEAK |
| **S2 cross-lingual (Phase 9)** | transfer_rate | **0.75** (6/8 pairs) | **STRONG** |
| | mean_shift | **0.371** | **STRONG** (7× vs degenerate circuit) |

> **Note on WEAK S1/S1.5:** This is mechanistically expected for a distributed 16-layer pipeline. The circuit features are a causal relay — they require residual stream context from non-circuit features at each layer. S2 (cross-lingual injection) is the correct sufficiency test for this circuit type and is STRONG.

### IoU Analysis (Claim 3: Middle Layers Dominant)

| Variant | early | mid | late | ratio (mid/early) |
|---|---|---|---|---|
| Pooled (all-vs-all) | 0.289 | 0.362 | 0.317 | **1.253×** |
| Decision-only | 0.413 | 0.465 | 0.437 | 1.126× |
| Content-only | 0.282 | 0.345 | 0.306 | 1.220× |
| **Status** | | | | **Borderline moderate** |

> Pooled 1.253× is the definitive Claim 3 metric. Direction is unambiguous (middle > late > early in all variants); gradient is shallow due to fundamental 3-prompts/concept constraint.

### Community Structure (B1)

| Community | Size | Layer range | Dominant lang_profile |
|---|---|---|---|
| Late FR-specific (C5) | ~19 | L22–L25 | 84% fr_leaning |
| Late cross-lingual (C8) | ~18 | L22–L25 | 89% balanced |
| Early balanced (C1) | ~27 | L10–L13 | mostly balanced |
| Mid-transition (C0) | ~10 | L13–L18 | mixed |
| Semantic transform (C4) | ~21 | L18–L22 | mixed |

---

## Authoritative Results — `multilingual_circuits` (MC, reference)

| Metric | Value | Notes |
|---|---|---|
| EN accuracy | 1.000 | 24/24 |
| FR accuracy | 0.667 | gate threshold 0.65 |
| mean_norm_diff | 3.511 | |
| IoU pooled ratio | 1.283× | best Claim 3 signal |
| Bridge features | 32/53 = 60.4% | |
| C3 disruption | 0.588 | CI [−0.202, −0.126] |
| Reference SLURM | 25058380 (v2, last_5) | USE v2 AS REFERENCE |

---

## Theoretical Framework

### Definition of Behaviour

> **Behaviour** = a repeatable, internally structured, intervenable latent computation that may not be explicitly verbalised.

Three testable properties:

1. **Causal directionality** — intervening on feature Z changes output: `y_abl = y(do(Z=0))`
2. **Persistency** — surface-level changes preserve the mechanism: `x→x' ⟹ same features`
3. **Substitutability** — replacing Z with Z' produces predictable output change: `y' = y(do(Z=z'))`

### Four-Type Behaviour Typology

| Type | Pathway | Examples |
|---|---|---|
| **Type 1: Latent States** | Input → latent → latent → output | Dallas→Texas→Austin; decay chain classification |
| **Type 2: Candidate Set** | Input → {candidates} → filter → output | Grammar agreement; antonym selection |
| **Type 3: Abstraction** | Input_surface → abstract → output_surface | **Multilingual circuits**; gauge equivalence |
| **Type 4: Gating** | Input → classifier → allow/block/redirect | Refusal circuits; thermodynamic constraints |

---

## Behaviours

### Currently Implemented

| Behaviour | Type | Status | Prompts | Key result |
|---|---|---|---|---|
| `grammar_agreement` | Type 2 | **COMPLETE** | 80 train / 100 test | Acc=85%, mean_logit_diff=4.11 |
| `physics_scalar_vector_operator` | Type 3 | **COMPLETE** | 80 train / 20 test | Scalar/vector classification |
| `antonym_operation` | Type 2 | **COMPLETE** | 80 train / 20 test | EN-only, single-token antonyms |
| `multilingual_circuits` (MC) | Type 3 | **COMPLETE** | 48 train / 16 test | IoU ratio=1.283×, bridge=60.4% |
| `multilingual_circuits_b1` (B1) | Type 3 | **COMPLETE** | 96 train / 32 test | IoU ratio=1.253×, S2=0.75 |
| `multilingual_antonym` | Type 2 | SUPERSEDED | — | Baseline FAILED (58.8% — synonyms are one-to-many) |

### Behaviour Configuration

Defined in `configs/experiment_config.yaml` under `behaviours:`.
Prompt generators registered in `GENERATORS` dict in `scripts/01_generate_prompts.py`.

### `multilingual_circuits_b1` — Dataset Details

- **Concepts:** 8 cross-lingual concept pairs (idx 0, 2–8; concept_1 excluded: `froid` is 2 tokens in FR)
- **FR concept 2:** `rapide→lent` (fixed from `vite→lent` which had word-class mismatch)
- **Templates:** T0–T3 (original MC templates) + T4–T7 (new surface variants), 8 total per group
- **Split:** Stratified 6 train + 2 test per (concept, language) group → 96 train, 32 test
- **Baseline thresholds:** EN ≥ 0.90, FR ≥ 0.65

---

## Pipeline

### Architecture

```
Qwen3-4B (base)
    ↓  hook at post_attention_layernorm (MLP input, NOT residual stream)
Transcoder (per-layer, mwhanna/qwen3-4b-transcoders)
    d_model=2560, d_transcoder=163840, layers 10–25
    ↓
Sparse features z^(l) ∈ R^163840
```

### Full Pipeline (Steps 01–08)

```
01 → Generate prompts
02 → Baseline evaluation
04 → Extract transcoder features (top-k per layer)
06 → Build attribution graph  [star / VW / role-aware modes]
07 → Causal interventions (ablation, patching, steering)
08 → Causal edges + circuit tracing + validation
09 → Prepare offline UI data
```

### Step-by-Step Commands

#### Step 01 — Generate Prompts

```bash
python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1
```

Output: `data/prompts/multilingual_circuits_b1_{train,test}.jsonl`

#### Step 02 — Baseline Evaluation

```bash
python scripts/02_run_baseline.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --model_size 4b
```

Output: `data/results/baselines/baseline_multilingual_circuits_b1_train.csv`

#### Step 04 — Extract Transcoder Features

```bash
python scripts/04_extract_transcoder_features.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --model_size 4b \
    --top_k 50          # top-k features per layer per prompt
```

Output: `data/results/transcoder_features/multilingual_circuits_b1/`

#### Step 06a — Build Star Attribution Graph

```bash
python scripts/06_build_attribution_graph.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_node_mode decision_only
```

Output: `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96.json`

#### Step 06b — Build Role-Aware Graph (with VW edges)

```bash
python scripts/06_build_attribution_graph.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_node_mode role_aware \
    --vw_threshold 0.01 \
    --k_content 10 \
    --output_suffix _roleaware
```

Output: `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json`

**VW edge computation:** `W_enc_tgt[tgt_feats,:] @ W_dec_src[src_feats,:]^T` (submatrix only, never full d_tc×d_tc)

#### Step 07 — Causal Interventions

```bash
# Ablation (importance)
python scripts/07_run_interventions.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --experiment importance \
    --model_size 4b \
    --top_k 20 \
    --per_feature         # REQUIRED for _b1 suffix

# Activation patching
python scripts/07_run_interventions.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --experiment patching \
    --model_size 4b \
    --top_k 20 \
    --per_feature
```

> **Critical:** `--per_feature` must be explicit for `multilingual_circuits_b1`. The default resolves to `False` for the `_b1` suffix.

#### Step 08 — Causal Edges + Circuit Tracing (Full Run)

```bash
GRAPH_PATH="data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json"

python scripts/08_causal_edges.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --agw_top_frac 0.6 \
    --tau_causal 0.10 \
    --n_paths 50 \
    --top_n_io_edges 10
```

**Phases run:**
- Phase 0: Load role-aware graph (I/O nodes, graph_layers)
- Phase 1: AGW prefilter (top 60% of VW edges → candidates)
- Phase 2: Load model + transcoders (layers 10–25)
- Phase 3: Baseline cache (96 forward passes)
- Phase 4: Ablation loop (~71 sources × 96 prompts ≈ 6,800 passes)
- Phase 5: Aggregate → causal edges (τ=0.10 filter)
- Phase 6: Path tracing → circuit JSON (top-50 paths, require_causal_edge=True)
- Phase 7: Necessity (group ablation, ~96 passes)
- Phase 8: Sufficiency S1 (linear) + S1.5 (layerwise)
- Phase 9: Sufficiency S2 (cross-prompt injection)
- Phase 10: Presentation graph (Graph B)

**Estimated wall time:** ~2.5h on A100 (5h SLURM slot recommended)

#### Step 08 — Revalidate Existing Circuit (Fast Mode)

```bash
CIRCUITS_PATH="data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json"
GRAPH_PATH="data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json"

python scripts/08_causal_edges.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --from_circuit "$CIRCUITS_PATH" \
    --top_n_io_edges 10
```

Skips Phases 1–6. Runs Phases 7–9 only (~100 min on A100). Writes results back in-place.

Available flags for Phase 8/9:
- `--skip_sufficiency` — skip Phases 8+9
- `--skip_s1_5` — skip S1.5 layerwise (S1 linear only)
- `--skip_s2` — skip cross-prompt injection
- `--debug_sanity` — log S1.5 hook sanity on first prompt
- `--max_prompts N` — truncate to N prompts (debug only)

#### Step 08 — Retrace Circuit from Existing Causal Edges (CPU-only)

```bash
python scripts/08_causal_edges.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_json "$GRAPH_PATH" \
    --model_size 4b \
    --from_causal_edges data/results/causal_edges/multilingual_circuits_b1/causal_edges_multilingual_circuits_b1_train.json \
    --n_paths 50
```

No model needed. Reruns Phases 6+10 only (path tracing + presentation graph). Useful for changing `n_paths` or `tau_causal` without GPU.

#### Step 09 — Prepare Offline UI

```bash
python scripts/09_prepare_offline_ui.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --n_prompts 96
```

Output: `data/ui_offline/<timestamp>_multilingual_circuits_b1_train_n96/`

#### Analysis — Multilingual Circuits

```bash
python scripts/a_analyze_multilingual_circuits.py \
    --behaviour multilingual_circuits_b1 \
    --split train \
    --graph_json "$GRAPH_PATH" \
    --node_labels \
    --community_summary \
    --per_feature
```

Outputs:
- `data/analysis/multilingual_circuits_b1/node_language_labels.csv`
- `data/analysis/multilingual_circuits_b1/community_summary.{json,md}`
- `data/analysis/multilingual_circuits_b1/iou_results.json`

### CSD3 / SLURM Jobs

| Job Script | Purpose | Estimated Time |
|---|---|---|
| `jobs/multilingual_circuits_b1_02_09.sbatch` | Full pipeline (steps 02–09, n=96) | ~3h |
| `jobs/multilingual_circuits_b1_08_causal_edges.sbatch` | Script 08 full run (causal edges + validation) | ~2.5h (5h slot) |
| `jobs/multilingual_circuits_b1_08_from_circuit.sbatch` | Script 08 revalidation only (Phases 7–9) | ~1h40m (3h slot) |
| `jobs/multilingual_circuits_b1_08_s15_debug.sbatch` | S1.5 sanity check (1 prompt, debug) | <5min |

**CSD3 environment variables (AMD EPYC / Ampere nodes):**

```bash
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512VL AVX512BW AVX512DQ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
```

**Module loading:**

```bash
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
```

---

## Key Data Files

### Authoritative Outputs — `multilingual_circuits_b1`

| File | Description | Size |
|---|---|---|
| `data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json` | **27-feature validated circuit** (paths, edges, necessity, S1/S1.5/S2) | 22 KB |
| `data/results/causal_edges/multilingual_circuits_b1/causal_edges_multilingual_circuits_b1_train.json` | 98 causal edges with δ, std, effect_size, AGW, VW scores | 37 KB |
| `data/results/causal_edges/multilingual_circuits_b1/presentation_graph_multilingual_circuits_b1_train.json` | Graph B for visualization | 28 KB |
| `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json` | Role-aware graph (94 nodes, 851 edges) | large |
| `data/analysis/multilingual_circuits_b1/node_language_labels.csv` | Per-node lang_profile labels | — |
| `data/analysis/multilingual_circuits_b1/community_summary.md` | Community analysis report | — |

### Frozen Snapshots

| Path | Purpose |
|---|---|
| `data/ui_offline_snapshots/grammar_agreement_train_n80_2026-02-23/` | Frozen baseline grammar UI |
| `data/runs/multilingual_circuits_v2_last5/` | MC v2 reference (SLURM 25058380) |

---

## Circuit Mechanistic Structure

The 27-feature circuit has three functionally distinct zones:

### Zone A — Early Encoding (L10–L16, 8 features)

Input concept encoding. Includes a **disinhibition gate**:

```
L10_F141643 →(−0.85)→ L11_F56459          [inhibitory: L10 suppresses L11]
L11_F56459  →(+0.28)→ L12_F28960
L12_F83869  →(+1.37)→ L13_F70603          [strongest edge; top scoring path]
L13_F70603  →(+0.13)→ L14_F57525

L14_F57525  →(−0.81)→ L15_F127839         ─┐ double-negative
L15_F127839 →(−1.16)→ L16_F45664  ε=−4.14  ─┘ = disinhibition gate
```

### Zone B — Mid-Layer Modulation (L18–L22, 6 features)

```
L18_F152973 →(−0.44)→ L19_F135640         [pure inhibitor: all 3 outputs negative]
L20_F149117 →(+0.22)→ L21_F145615
L21_F129759 →(+0.39)→ L22_F108295         [pure excitor: all 3 outputs positive]
L21_F129759 →(+0.31)→ L22_F32734
```

### Zone C — Late-Layer Output Preparation (L23–L25, 13 features)

```
L22_F108295 →(+0.46)→ L23_F64429          [main late hub activator]
                  L23_F64429 →(+0.82)→ L24_F136810 →(+)→ L25_F138698, F34754, F59053, F104408, F111603
                  L23_F64429 →(+0.52)→ L24_F134204
                  L23_F64429 →(+0.51)→ L24_F29680
                  L23_F64429 →(+0.45)→ L24_F76363
                  L23_F64429 →(+0.41)→ L24_F131457
L24_F48363  →(−0.64)→ L25_F111603         [inhibitory late path]
```

**Sign statistics:** 62% excitatory (60/98), 38% inhibitory (38/98). VW sign concordance = 98%.

**Pure inhibitors:** `L18_F152973`, `L24_F48363`, `L24_F60321`, `L24_F119196`, `L24_F36045`
**Pure excitors:** `L21_F129759`, `L22_F108295`, `L24_F136810`, `L24_F76363`

---

## Graph Modes

Script 06 supports three graph construction modes:

| Mode | CLI flag | Description |
|---|---|---|
| **Star only** | `--graph_node_mode decision_only` | Decision-token features only; star edges (attribution correlation) |
| **VW graph** | `--vw_threshold 0.01` | Adds virtual-weight edges W_enc @ W_dec^T between adjacent-layer features |
| **Role-aware** | `--graph_node_mode role_aware` | Adds content-word nodes detected from prompt text; dual decision+content attrs |

### Role-Aware Node Types

| causal_status | position_role | Description |
|---|---|---|
| `output_attributed` | `decision` | Standard decision-token feature |
| `upstream_candidate` | `content` | Content-word feature; no output edges |
| `both` | `both` | Feature active at both decision + content positions |

---

## Intervention Modes (Script 07)

| Flag | Mode | Description |
|---|---|---|
| *(default)* | STRICT | Layers with no graph features are skipped; exits code 1 if zero results |
| `--control_fallback` | CONTROL | First-K fallback for control experiments; adds `feature_source` column |

Output CSVs include `feature_source` ("graph"\|"control") and `layer_has_graph_features` bool when using control mode. Every `_summary.json` records `control_fallback: bool` for audit.

---

## Method Details

### Transcoders

Pre-trained from [`mwhanna/qwen3-4b-transcoders`](https://huggingface.co/mwhanna/qwen3-4b-transcoders). Hook at `post_attention_layernorm` (MLP input, not residual stream).

```
d_model    = 2560
d_tc       = 163840  (transcoder hidden dim per layer)
layers     = 10–25   (16 layers for multilingual_circuits behaviours)
```

### Attribution Graphs

Per-prompt union: features active across all prompts are merged; edge weights are beta coefficients from regression of transcoder feature activations. Graph saved as JSON (not GraphML — GraphML had a 0-edge conversion bug).

**VW edges:** For features (src at layer l, tgt at layer l+1):
```
vw_weight = W_enc_tgt[tgt_feat, :] @ W_dec_src[src_feat, :]^T
```
Only the relevant submatrix is computed (never the full 163840×163840 product).

### Causal Edge Detection (Script 08)

1. **AGW prefilter:** Rank all VW edges by AGW score (attribution graph weight product). Keep top `agw_top_frac` fraction as candidates.
2. **Ablation loop:** For each source feature: zero-ablate across all prompts, measure change in target feature activation. `mean_delta = E[baseline_act − ablated_act]`.
3. **Threshold:** Edge is causal if `|mean_delta| ≥ τ_causal` (default 0.10).
4. **Path tracing:** DFS from high-attribution input features through causal edges to output. Score = product of star-edge weights × causal-edge weights. Top K paths → circuit.

**Sign convention:** `mean_delta > 0` means ablating source reduces target (source is excitatory). `mean_delta < 0` means ablating source increases target (source is inhibitory).

### Interventions

| Type | Formula | Measures |
|---|---|---|
| **Ablation** | `z_i = 0` | Causal directionality |
| **Patching** | `z_i = z'_i` (from paired prompt) | Substitutability |
| **Steering** | `z_i += α · d` | Steerability |

**Pairing logic:** `create_prompt_pairs()` in `scripts/07_run_interventions.py`. Mode A = direction-swap (same concept, fwd↔rev). Mode B = cross-lingual (by concept_index).

---

## Notation Reference

| Symbol | Meaning |
|---|---|
| T | Sequence length |
| V | Vocabulary size |
| d | Model dimension (d_model=2560) |
| d_tc | Transcoder hidden dimension (163840) |
| L | Number of transformer layers (28 total; 10–25 used) |
| h^(l) | MLP hidden activations at layer l |
| z^(l) | Sparse transcoder features at layer l |
| E_l, D_l | Transcoder encoder/decoder at layer l |
| δ | mean_delta: causal effect size (ablation change in target) |
| ε | effect_size: δ / std_delta (standardised) |
| τ | tau_causal: threshold for declaring an edge causal |
| AGW | Attribution graph weight (proxy for causal importance) |
| VW | Virtual weight (W_enc @ W_dec^T dot product) |
| IoU | Intersection-over-Union of EN and FR feature sets |

### Forward Pass with Interventions

```
X^(0) = TokEmb(tokens) + PosEmb(positions)

For each layer l = 1..L:
    X  = X + Attn_l(LN_1(X))              # attention + residual
    h  = MLP_input(X)                     # = LN_2(X) @ W1 + b1 before nonlinearity

    if l in intervention_layers:
        z  = E_l(h)                       # transcoder encode → sparse features
        z  = Intervene(z)                 # ablation / patch / steer / zero-ablate
        h' = D_l(z)                       # transcoder decode → reconstruct MLP input
        h  = h'

    X = X + MLP_output(h)                 # MLP output + residual

logits = LN_final(X_last) @ W_U          # next-token prediction
```

---

## Dashboard

Interactive React + Vite + Plotly + D3-force visualization at `dashboard/`.

```bash
cd dashboard
npm install
npm run dev          # local dev server
npm run build        # production build → dist/
```

Serves offline UI data from `data/ui_offline/<run>/`. Load run: point `VITE_DATA_DIR` to the run directory.

---

## Environment Setup

### Local (Conda)

```bash
conda env create -f environment.yml
conda activate qwen-circuits
pip install -e .    # install package in editable mode
```

### CSD3 (HPC)

```bash
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate
```

See `docs/CSD3_SETUP.md` for full venv rebuild instructions.

---

## Proposed Next Research Directions

This section documents candidate extensions planned after the B1 circuit validation.

### 1 — New Theories

**Disinhibition as a computation primitive.** The L14→L15→L16 double-negative chain (ε=−4.14) is the strongest-effect edge in the circuit and forms a disinhibition gate. Hypothesis: this pattern recurs across behaviour types as a universal gating mechanism.

**Hub-and-spoke late-layer topology.** L23_F64429 fans out to 5 L24 targets with all-positive edges. Hypothesis: late-layer output preparation generalises to a hub-and-spoke topology where a single cross-lingual hub coordinates multiple language-specific output features.

**Inhibitory intermediate features.** L18_F152973 is a pure inhibitor (3/3 negative outgoing edges) with no circuit role in direct excitation. Hypothesis: such features act as signal suppressors that enforce selectivity in the mid-layer transition.

### 2 — New Behaviours

New behaviours should be registered in `configs/experiment_config.yaml` and `scripts/01_generate_prompts.py`.

**Physics-domain proposals** (from `docs/novel_behaviours.md`):

| # | Behaviour | Type | Difficulty |
|---|---|---|---|
| 1 | Decay type classification | Type 1 | Medium |
| 2 | Radioactive decay chain | Type 1 | High |
| 3 | Quantum selection rules | Type 2 | Medium |
| 4 | Relativistic vs classical regime | Type 2 | Medium |
| 5 | Gauge equivalence | Type 3 | High |
| 6 | Intensive/extensive quantity | Type 3 | Low |
| 7 | 2nd law of thermodynamics gating | Type 4 | Medium |
| 8 | Work sign classification | Type 4 | Low |

**Selection criteria for new behaviours:**
1. Baseline accuracy ≥ 85% (EN) or ≥ 65% (FR) before proceeding
2. Single-token or near-single-token target (check with `scripts/b_tokenize_audit_multilingual.py`)
3. Binary or small candidate set (makes IoU analysis tractable)
4. At least 2 surface variants per (concept, language) group

### 3 — Prompt Scaling

Current B1 uses 96 train prompts (48 EN + 48 FR, 8 templates × 8 concepts × 2 languages).

**Scaling axes to explore:**

| Axis | Current | Candidate | Expected effect |
|---|---|---|---|
| Templates per group | 8 (T0–T7) | 12–16 | More robust bridge/IoU estimates |
| Concepts | 8 | 12–16 | Wider cross-concept generalization |
| Prompts in ablation loop | 96 | 192 | Reduce variance in δ estimates |
| Token positions | decision only | decision + last 3 content | More complete picture |

**Key reference:** With 3 prompts/concept (MC baseline), the IoU standard deviation is high and the gradient is flattened. Moving to 6 prompts/concept (B1) improved bridge rate from 60.4% to 67.4%. Scaling to 12 prompts/concept may stabilize the Claim 3 gradient further.

**Practical limit:** Phase 4 ablation (71 sources × N_prompts) is the bottleneck. At N=192 it becomes ~13,600 forward passes (~5h on A100). Use `--max_prompts` for diagnostic runs.

### 4 — Community Detection Rearchitecture

Current approach: Louvain on VW-only subgraph (no star edges, no I/O nodes). Produces 10 communities for B1.

**Proposed changes:**

| Variant | Description | Expected finding |
|---|---|---|
| **Weighted Louvain** | Use `abs(vw_weight)` as edge weight | Better separation of strong/weak communities |
| **Signed community detection** | Use SPONGE or SIGNED-LOUVAIN for +/− edges | Distinguish excitatory vs inhibitory sub-circuits |
| **Hierarchical** | Girvan-Newman or multi-resolution Louvain | Reveal nested community structure |
| **Layer-constrained** | Only allow edges within ±2 layers | Clean layer-progression communities |
| **Full graph (star + VW)** | Include all edge types | May show I/O hub dominance — compare with VW-only |
| **Causal-edge-only graph** | Only the 98 confirmed causal edges | Community structure of the verified circuit only |

**Implementation note:** Louvain with negative edges requires `abs(weight)` or a signed algorithm (current bug fix in MEMORY). NetworkX `community.louvain_communities()` uses `weight` parameter directly; ensure positive values.

### 5 — Effect of Different Clustering Methods

Current: Louvain (stochastic, resolution=1.0). Single resolution gives one partition.

**Alternative methods to benchmark:**

| Method | Library | Key parameter | Strength |
|---|---|---|---|
| **Louvain** (current) | `networkx.community` | resolution γ | Fast; standard baseline |
| **Leiden** | `leidenalg` | resolution | More stable than Louvain; fewer disconnected communities |
| **Spectral clustering** | `sklearn` | n_clusters | Deterministic; works on non-sparse graphs |
| **Label propagation** | `networkx.community` | — | Very fast; good for large graphs |
| **Infomap** | `cdlib` | — | Flow-based; captures directional information flow |
| **Stochastic Block Model** | `graph-tool` | n_blocks | Principled probabilistic model |
| **DBSCAN on embeddings** | `sklearn` | eps, min_samples | Density-based; good for outlier detection |

**Evaluation metrics:**

- Modularity Q (higher = better community separation)
- Normalized Mutual Information (NMI) vs lang_profile labels (do communities correlate with language?)
- Community size distribution (Gini coefficient)
- Layer purity: fraction of community members from same layer range
- Stability across random seeds (run × 10, compute ARI)

**Key research question:** Do the C5 (84% fr_leaning) and C8 (89% balanced) communities at L22–L25 remain stably separated across clustering methods? If yes, they are robust structural features. If they merge, the finding is method-dependent.

---

## Implementation Checklist for Next Steps

When adding a new behaviour or new analysis:

```
[ ] Register in configs/experiment_config.yaml
[ ] Register generator in scripts/01_generate_prompts.py
[ ] Run tokenization audit (scripts/b_tokenize_audit_multilingual.py)
[ ] Verify baseline passes (EN ≥ 0.85, FR ≥ 0.65)
[ ] Run script 02 (baseline) → confirm pass
[ ] Run script 04 (features, --top_k 50)
[ ] Run script 06b (role-aware graph, --vw_threshold 0.01)
[ ] Run analysis script (a_analyze_multilingual_circuits.py)
[ ] Run script 07 (interventions, --top_k 20, --per_feature)
[ ] Run script 08 full (causal edges + circuit, N_PATHS=50)
[ ] Inspect: features, causal edges, circuit, validation
[ ] If validation weak: run --from_circuit revalidation
[ ] Update MEMORY.md with authoritative numbers
```

---

## Project Structure

```
project/
├── scripts/                        # Pipeline scripts (run in order 01→08)
│   ├── 01_generate_prompts.py          # Prompt generation (all behaviours)
│   ├── 02_run_baseline.py              # Baseline accuracy + logit diff
│   ├── 04_extract_transcoder_features.py  # Top-k feature extraction per layer
│   ├── 06_build_attribution_graph.py   # Star / VW / role-aware attribution graphs
│   ├── 07_run_interventions.py         # Ablation, patching, steering
│   ├── 08_causal_edges.py              # Causal edges + circuit tracing + validation
│   ├── 08_generate_figures.py          # Analysis figure generation
│   ├── 09_prepare_offline_ui.py        # Offline UI data packaging
│   ├── a_analyze_multilingual_circuits.py  # IoU, lang profiles, communities, bridges
│   └── b_tokenize_audit_multilingual.py    # Token audit for multilingual behaviours
│
├── src/                            # Core library
│   ├── model_utils.py                  # ModelWrapper: Qwen3 loading, hooks
│   └── transcoder/                     # Transcoder loading and extraction
│
├── configs/
│   ├── experiment_config.yaml          # All behaviour definitions and params
│   └── transcoder_config.yaml          # Transcoder repo and layer config
│
├── data/
│   ├── prompts/                        # JSONL prompt sets per behaviour
│   ├── results/
│   │   ├── attribution_graphs/         # JSON graphs (star, VW, role-aware)
│   │   ├── causal_edges/               # Causal edges, circuits, presentation graphs
│   │   │   └── multilingual_circuits_b1/   # AUTHORITATIVE B1 outputs
│   │   ├── transcoder_features/        # Per-layer feature metadata
│   │   └── interventions/              # Ablation/patching/steering CSVs
│   ├── analysis/                       # Analysis outputs (IoU, labels, communities)
│   ├── ui_offline/                     # Timestamped UI data packages
│   └── runs/                           # Versioned experiment snapshots
│
├── jobs/                           # SLURM sbatch scripts (CSD3)
│   ├── multilingual_circuits_b1_02_09.sbatch       # Steps 02–09
│   ├── multilingual_circuits_b1_08_causal_edges.sbatch  # Causal edges (full)
│   ├── multilingual_circuits_b1_08_from_circuit.sbatch  # Revalidation (fast)
│   └── multilingual_circuits_b1_08_s15_debug.sbatch     # S1.5 debug
│
├── dashboard/                      # React + Vite + Plotly + D3-force UI
├── docs/                           # Extended documentation
│   ├── PIPELINE_GUIDE.md               # Detailed pipeline guide
│   ├── behaviours.md                   # Behaviour definitions and typology
│   ├── PAPER_COMPARISON.md             # Comparison to Anthropic Biology paper
│   ├── novel_behaviours.md             # 8 physics-domain behaviour proposals
│   └── CSD3_SETUP.md                   # HPC setup instructions
├── figures/                        # Generated analysis figures
├── environment.yml                 # Conda environment (Python 3.10, PyTorch 2.2)
└── pyproject.toml                  # Package metadata
```

---

## References

- Lindsey, J., Gurnee, W., et al. (2025). *On the Biology of a Large Language Model*. Anthropic Transformer Circuits Thread. [Link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- Qwen Team (2025). *Qwen3 Technical Report*. [Link](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread.
- Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*.
