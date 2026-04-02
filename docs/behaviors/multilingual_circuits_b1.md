# `multilingual_circuits_b1` — Canonical Reference (B1-v2)

**Status:** CANONICAL — B1-v2 clean-template run (2026-03-21)
**Supersedes:** B1-v1 (SLURM 25307123, 2026-03-18) — contaminated by template artifact
**Pipeline run:** SLURM 25679695
**Analysis data:** `data/analysis/multilingual_circuits_b1/`

---

## 1. Behavior Overview

### What it is
Antonym completion in two languages (English + French) for the same set of semantic concepts.
The model is given a prompt ending in a partial antonym frame and must predict the correct antonym token.

### Task type
Type 2: Candidate Set — the model must rank two specific tokens (correct antonym vs. distractors).
Evaluation: binary sign accuracy (`logit(correct) > logit(incorrect)`).

### Language setup
- **EN**: 48 train prompts, 16 test prompts
- **FR**: 48 train prompts, 16 test prompts
- Both share the same 8 semantic concepts

### Concepts (8 cross-lingual pairs)
| Idx | EN pair | FR pair |
|-----|---------|---------|
| 0 | hot→cold | chaud→froid |
| 2 | fast→slow | rapide→lent |
| 3 | big→small | grand→petit |
| 4 | long→short | long→court |
| 5 | strong→weak | fort→faible |
| 6 | old→new | vieux→nouveau |
| 7 | clean→dirty | propre→sale |
| 8 | light→dark | clair→sombre |

(Concept 1 excluded: froid=2 tokens, incompatible with single-token evaluation)

### Prompt family (B1-v2: 8 templates per group)
**EN templates (T0–T7):**
- T0: `The opposite of "{word}" is`
- T1: `The antonym of "{word}" is`
- T2: `"{word}" is the opposite of`
- T3: `"{word}" is the antonym of`
- T4: `The contrary of "{word}" is`
- T5: `"{word}" is the contrary of`
- T6: `The word opposite to "{word}" is`
- T7: `"{word}" is an antonym of`

**FR templates (T0–T7, B1-v2, clean):**
- T0: `Le contraire de {word} est`
- T1: `L'antonyme de {word} est`
- T2: `L'opposé de {word} est`
- T3: `Le mot opposé à {word} est`
- T4: `Le terme opposé de {word} est`
- T5: `En français, le contraire de {word} est`
- T6: `Le contraire de {word} :`
- T7: `L'antonyme de {word} :`

### Train/test split
Stratified: 2 test + 6 train per (concept, language) group.
→ 96 train (48 EN + 48 FR), 32 test (16 EN + 16 FR)

### Why this behavior matters
- Directly replicates Anthropic's multilingual circuits paper (Wendler et al., 2024)
- Tests cross-lingual generalization in a controlled, binary setup
- B1 extends the base MC behavior with 2× more surface variation per concept
- Acts as the **anchor case** for all multilingual mechanistic interpretability claims

---

## 2. B1-v1 → B1-v2 History

### What was wrong in B1-v1
All 8 FR templates used a quoted-word pattern: `Le contraire de "{word}" est`.
This caused the model to predict the closing `"` (quote-completion bias) on failing FR prompts.

**Evidence:**
- 18/18 failing FR prompts in B1-v1 predicted `' "'` as argmax
- Mean logit margin for failing FR prompts: −11.838
- Median correct token rank: 642 (range 55–5732)

### What happened to the analysis in B1-v1
- The attribution graph captured features from the **quote-completion circuit**, not the semantic antonym circuit
- Identified "gateway features" (L22_F108295, L22_F32734) were specific to closing a quote, not computing antonyms
- S2 transfer result (0.75) measured transfer of quote-completion behavior, not semantic computation
- fr_leaning node profile inflation: 44/94 nodes (46.8%) were fr_leaning, primarily because quote-completion is FR-specific
- The apparent circuit structure was partly real, partly artifact

### What changed in B1-v2 (2026-03-21)
- All 8 FR templates rewritten to remove `"{word}"` pattern (called B1-v2 templates)
- No `de`-ending FR templates retained (article-attraction risk removed)
- Prompts regenerated: `python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1`
- Full pipeline rerun: scripts 02→04→06→07→08→09 on HPC (SLURM 25679695, 25743140, 25765101, 25773917)
- Analysis rerun locally with B1-v2 graph + B1-v2 .npy features

### Why B1-v2 is canonical
- FR accuracy increased: 72.9% → 79.2%
- Quote rate dropped from 100% (18/18) to 27.8% (5/18), and remaining 5 are word-specific model priors
- The circuit identified in B1-v2 uses L22_F41906 (which was the **negative control** in B1-v1) as its primary L22 hub
- B1-v1's "gateway features" (L22_F108295, L22_F32734) are absent from the B1-v2 circuit, and scaling them actively hurts correct predictions
- Node language profiles are now predominantly balanced (77.9% vs 45.7% in B1-v1), consistent with a genuine cross-lingual task

---

## 3. Final Authoritative Results (B1-v2)

### Behavioral accuracy
| | EN | FR | Combined |
|--|--|--|--|
| Sign accuracy | **1.000** (48/48) | **0.792** (38/48) | **0.896** |
| Failing prompts | 0 | 10 | 10 |
| Failing concepts (FR) | — | c2(1), c3(1), c4(6), c6(4), c7(6) | — |

### Graph structure (roleaware)
| Property | B1-v2 | B1-v1 |
|----------|-------|-------|
| Feature nodes | **86** | 94 |
| Total edges | **633** | 851 |
| VW edges (community subgraph) | **379** | 575 |

### Causal circuit
| Property | B1-v2 | B1-v1 |
|----------|-------|-------|
| Feature nodes | **21** (+3 I/O = 24 total) | 27 |
| Edges | **55** | 69 |
| Paths (n=50) | **50** | 50 |
| Layer range | **L10–L25** | L10–L25 |
| Primary L22 hub | **L22_F41906** | L22_F108295 (ARTIFACT) |
| Top causal edge | **L24_F35447→L25_F43384 (δ=2.283)** | L12→L13 (δ=1.37) |
| L12→L13 edge | **preserved (δ=1.464)** | δ=1.37 |

**Circuit features (21):**
`L12_F83869, L13_F70603, L18_F149556, L19_F107296, L20_F89742, L21_F27974, L22_F41906, L22_F78043, L23_F40170, L23_F64429, L23_F6889, L23_F83865, L24_F119196, L24_F30233, L24_F35447, L24_F5768, L24_F76363, L25_F125339, L25_F41381, L25_F43384, L25_F90133`

### Causal validation
| Metric | B1-v2 | Interpretation |
|--------|-------|----------------|
| Necessity disruption_rate | **0.1042** | Circuit is distributed/redundant |
| Necessity mean_effect | **+0.932** | Positive when disrupted |
| S1 linear sufficiency | **WEAK** | Expected for distributed circuit |
| S1.5 layerwise sufficiency | **WEAK** | Same |
| S2 transfer_rate | **0.125** | Weak cross-lingual injection |
| S2 mean_shift | **−1.309** | Injection does not help FR |

### Competition analysis (failing FR prompts)
| Metric | B1-v2 | B1-v1 |
|--------|-------|-------|
| argmax_is_quote_rate | **0.278** (5/18) | 1.000 (18/18) |
| Mean logit margin | **−4.288** | −11.838 |
| Median correct rank | **78** | 642 |
| Top failing token | `' "'` (5), `' le'` (5), `' court'` (4) | `' "'` (18) |

### Cross-lingual generalization
| Metric | B1-v2 | B1-v1 |
|--------|-------|-------|
| Bridge features | **33/49 = 67.35%** | 33/49 = 67.35% |
| C3 disruption_rate | **0.645** | 0.645 |
| C3 mean_effect | **−0.372** | −0.372 |
| C3 CI | **[−0.403, −0.343]** | same |

### IoU (EN vs FR feature activation overlap per layer)
| | Early (L10-11) | Middle (L12-20) | Late (L21-25) | Ratio |
|--|--|--|--|--|
| **Pooled** | 0.2375 | 0.2589 | 0.2262 | **1.090×** |
| **Decision** | 0.2182 | 0.2262 | 0.2363 | **1.037×** |
| **Content** | 0.2217 | 0.2352 | 0.2012 | **1.061×** |

B1-v1 reference: pooled 1.253×, decision 1.126×.

### Node language profiles (86 nodes)
| Profile | B1-v2 | B1-v1 |
|---------|-------|-------|
| balanced | **67 (77.9%)** | 43 (45.7%) |
| fr_leaning | **15 (17.4%)** | 44 (46.8%) |
| en_leaning | **4 (4.7%)** | 5 (5.3%) |
| insufficient_data | **0** | 2 (2.1%) |

### Community structure (7 communities, 379 VW edges)
| C | N | Layers | Profile | Role |
|---|---|--------|---------|------|
| C0 | 32 | L10–L16 | 91% balanced | Early processing |
| C1 | 14 | L17–L20 | 100% balanced | Mid-layer transition |
| C2 | 8 | L21–L25 | **100% fr_leaning** | FR-specific COMPETING pathway |
| C3 | 14 | L20–L23 | 93% balanced | Semantic transformation hub |
| C4 | 16 | L23–L25 | 69% balanced | Output circuit |
| C5 | 1 | L10 | fr_leaning | Isolated |
| C6 | 1 | L20 | fr_leaning | Isolated |

**C2 contains:** L22_F108295, L23_F64429, L24_F74929, L24_F76363, L24_F81698, L25_F104408
**C3 contains:** L22_F41906, L21_F27974, L22_F78043, L22_F99330, L23_F83865

---

## 4. Canonical Mechanistic Interpretation

### Main pathway (genuine, cross-lingual)
`C0 (L10–L16) → C1 (L17–L20) → C3 (L20–L23) → C4 (L23–L25)`

All predominantly balanced. Processes the antonym relationship independently of language.
This pathway is active for both EN and FR correct predictions.

**Key nodes:**
- **L12_F83869** (C0, balanced): early semantic encoder; top bridge (bridge_score=0.436); L12→L13 causal edge (δ=1.464) preserved across B1-v1 and B1-v2
- **L13_F70603** (C0, balanced): early discriminator; contribution correct=+0.193, incorrect=−0.443, Δ=−0.635; appears in both runs
- **L22_F41906** (C3, balanced): genuine L22 semantic hub; top path node (input→L21→L22_F41906→output); bridge_score=0.677; was negative control in B1-v1

### Competing pathway (FR-specific, inhibitory)
C2: all fr_leaning, L21–L25. Activated more by FR prompts, but does NOT help correct output. L23_F64429 has neg_frac=77% (ablating it IMPROVES correct output for 77% of prompts). This community likely represents an FR-language-specific pathway that competes with the genuine semantic pathway.

**Key nodes:**
- **L22_F108295** (C2, fr_leaning): artifact "gateway" in B1-v1; absent from B1-v2 circuit; scaling it actively hurts correct predictions
- **L23_F64429** (C2, fr_leaning): inhibitory hub; mean|effect|=1.330 but neg_frac=77%; contributes MORE to incorrect prompts (+1.947) than correct (+0.850)

### What the FR failure pattern tells us
18/18 failing prompts are FR. They cluster in concepts 4 (long→court), 6 (vieux→nouveau), 7 (propre→sale). The mid-zone trajectory is the key discriminator: correct prompts have L22_F41906 contribution +0.897; incorrect prompts have +0.170 (Δ=−0.727). This makes L22_F41906 the main decision point distinguishing success from failure.

### Bridge features (cross-lingual computation)
33/49 graph features (67.35%) are bridges — their ablation harms both EN and FR. Top bridges are all at L25, suggesting the output-preparation stage is shared across languages. The bridge analysis confirms that cross-lingual computation exists but is concentrated at the output layer, not distributed uniformly.

### Caveats
- Necessity is low (0.1042): the circuit is not the only pathway; many parallel routes exist
- S2 is weak (0.125): injecting EN features into FR prompts does not reliably produce correct FR output
- Residual 5/18 quote predictions are word-specific model priors (concepts chaud/propre), not fixable by template design
- Claim 3 (middle > early IoU gradient) is directionally correct but numerically weak (1.090×)

---

## 5. What Changed Relative to B1-v1

### Survived (robust across template fix)
| Finding | Status |
|---------|--------|
| Bridge rate 67.35% | ✓ Preserved |
| C3 disruption_rate 0.645 | ✓ Preserved |
| L12→L13 causal edge | ✓ Preserved (δ=1.37→1.464) |
| L13_F70603 as early discriminator | ✓ Preserved (Δ=−0.635) |
| 18/18 incorrect prompts are FR | ✓ Preserved (same indices) |
| IoU direction (middle > early, pooled) | ✓ Preserved |
| Late L25 as top bridge layer | ✓ Preserved |

### Downgraded
| Finding | B1-v1 | B1-v2 |
|---------|-------|-------|
| Claim 3 (IoU gradient) | Borderline moderate (1.253×) | **Very weak (1.090×)** |
| Claim 5 (S2 transfer) | Strong (0.75) | **Weak (0.125)** |
| FR accuracy | 72.9% | 79.2% (improvement) |

### Rejected (confirmed artifacts)
| Finding | Reason |
|---------|--------|
| L22_F108295 as gateway feature | Absent from B1-v2 circuit; scaling it hurts correct predictions |
| L22_F32734 as gateway feature | Same |
| S2=0.75 as cross-lingual transfer evidence | Was measuring quote-completion transfer |
| fr_leaning majority (46.8%) | Inflation from quote-completion features |
| 10-community structure | 9 (B1-v1 corrected) → 7 (B1-v2) |

### Became stronger
| Finding | Why |
|---------|-----|
| C2 as competing FR pathway | Now structurally isolated community (100% fr_leaning), confirmed as inhibitory |
| L22_F41906 as genuine hub | Was control in B1-v1; now top path node with bridge rank 8 |
| Circuit identity shift | The B1-v1 circuit was a different circuit (quote-completion), not a contaminated version of the same circuit |

---

## 6. Strength of Evidence

| Finding | Strength | Evidence |
|---------|----------|---------|
| EN=100% accuracy, circuit works for EN | **Strong** | 48/48 sign_acc in both B1-v1 and B1-v2 |
| FR=79.2% accuracy | **Strong** | Direct measurement; gate PASS |
| Bridge features (67.35%) | **Strong** | Consistent across B1-v1/v2; robust to template change |
| C3 disruption (0.645) | **Strong** | Consistent across B1-v1/v2; concentrated at L20–L25 |
| L12→L13 early motif | **Strong** | Preserved across template fix; top causal edge in both runs |
| L22_F41906 as genuine hub | **Moderate** | Top path node; bridge_score=0.677; mid discriminator Δ=−0.727; but necessity is low |
| L22_F108295/F32734 as artifacts | **Strong** | Absent from B1-v2 circuit; scaling HURTS; all evidence consistent |
| C2 as competing FR pathway | **Moderate** | 100% fr_leaning, inhibitory L23_F64429; but only 8 features, no direct ablation of C2 as a unit |
| Competing pathway causes FR failures | **Moderate** | L23_F64429 contributes more to incorrect; but direct causal link not tested |
| IoU middle > early gradient (Claim 3) | **Weak** | Pooled 1.090×, direction correct but near-flat; decision IoU inverted |
| S2 cross-lingual transfer (Claim 5) | **Weak** | 0.125 transfer rate; previous 0.75 was artifact |
| Residual 5/18 FR failures are word-specific | **Moderate** | Quote bias for chaud/propre; not fixable by templates; plausible but not causally verified |

---

## 7. Authoritative Files

### Canonical B1-v2 outputs

| File | Contents | Status |
|------|----------|--------|
| `data/prompts/multilingual_circuits_b1_train.jsonl` | 96 B1-v2 prompts, 0 quoted-word patterns | **CANONICAL** |
| `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json` | 86 feature nodes, 633 edges, B1-v2 graph | **CANONICAL** |
| `data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96.json` | 84 nodes, 252 edges, star graph | **CANONICAL** |
| `data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json` | 24-node circuit, 55 edges, 50 paths | **CANONICAL** |
| `data/results/interventions/multilingual_circuits_b1/intervention_ablation_multilingual_circuits_b1.csv` | Ablation results, 96 prompts, per-feature | **CANONICAL** |
| `data/results/interventions/multilingual_circuits_b1/intervention_patching_C3_multilingual_circuits_b1.csv` | C3 patching results | **CANONICAL** |
| `data/results/baseline_multilingual_circuits_b1_train.csv` | EN=1.000, FR=0.792 baseline | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/iou_per_layer.csv` | Per-layer IoU (pooled) — B1-v2 | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/iou_per_layer_decision.csv` | Per-layer IoU (decision token) | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/iou_per_layer_content.csv` | Per-layer IoU (content position) | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/bridge_features.csv` | 33/49 bridge features | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/node_language_labels.csv` | 86 nodes with lang_profile | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/community_summary.json` | 7 communities, B1-v2 | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/community_summary.md` | Human-readable community summary | **CANONICAL** |
| `data/analysis/multilingual_circuits_b1/c3_patching_per_feature.csv` | Per-feature C3 disruption | **CANONICAL** |
| `data/results/reasoning_traces/multilingual_circuits_b1/error_cases_train.json` | 18 failure cases, zone analysis | **CANONICAL** |
| `data/results/transcoder_features/multilingual_circuits_b1_train_position_map.json` | Token position map, 480 rows | **CANONICAL** |
| `data/results/transcoder_features/layer_*/multilingual_circuits_b1_train_top_k_indices.npy` | Top-k feature indices per layer (L10–L25) | **CANONICAL** |

### Legacy / archival only (do not cite as canonical)

| File | Issue |
|------|-------|
| B1-v1 UI snapshot `data/ui_offline/20260318-013309_multilingual_circuits_b1_train_n96/` | B1-v1 data, artifact circuit |
| Any `competition_analysis_train.json` or `l22_intervention_results_train.json` from the B1-v1 run | Pre-template-fix, not reflective of genuine circuit |
| `docs/behaviors/multilingual_circuits_b1.md` if it shows B1-v1 numbers | This file is B1-v2; earlier analysis files are superseded |

---

## 8. How to Use B1-v2 as an Anchor Case

### What B1-v2 teaches us

1. **Template artifacts can recruit entirely different circuits.** The artifact in B1-v1 was not a noisy version of the genuine circuit — it was a structurally distinct circuit for a different sub-task (quote-completion). Future behaviors: always run a quote-rate / argmax-is-artifact check before interpreting circuit results.

2. **The genuine cross-lingual antonym pathway is balanced, not FR-leaning.** With clean prompts, 77.9% of graph features are balanced. Language-specific features do exist (C2) but they form a competing pathway, not the main one. When investigating a new multilingual behavior, the expectation should be: main pathway ≈ balanced, language-specific nodes ≈ minority at late layers.

3. **S2 injection is task-specific, not a general property.** B1-v2 shows S2=0.125; the previous S2=0.75 was artifact. Do not assume that cross-lingual feature sharing implies strong injection sufficiency. Test S2 explicitly.

4. **Bridge rate (67.35%) and C3 disruption (0.645) are robust.** These numbers survived the template fix unchanged. They measure something real about the task structure and can be used as reference benchmarks for new multilingual behaviors.

5. **Early motifs (L12→L13) can be a stable anchor.** The L12→L13 causal edge survived template fix. When looking for homologous structures in new behaviors, start here.

### Safe assumptions to transfer to new behaviors
- Bridge rate > 50% is expected for genuinely cross-lingual tasks
- Balanced features will dominate if templates are clean
- Late layers (L20+) are where language-specific divergence happens
- L12→L13 may be a general antonym-processing motif (test explicitly)
- If argmax_is_quote_rate > 0 on FR failing prompts, stop and fix templates before analyzing the circuit

### Assumptions that must NOT be transferred
- Do not assume the same circuit features will appear in other antonym behaviors
- Do not assume L22_F41906 is a general semantic hub — it is specific to this concept/model
- Do not assume IoU ratio ≈ 1.09 is the right number for other behaviors
- Do not assume FR accuracy will be lower than EN in all multilingual tasks
- Do not use B1-v1 as a reference for anything other than illustrating template artifacts

---

## 9. Open Questions

1. **Why does concept 4 (long→court) have 6/8 FR train failures?** This is the highest failure rate per concept. Is it a tokenization issue (multi-meaning of "long" in FR), a template interaction, or a genuine model knowledge gap? Not resolved.

2. **Is the C2 competing pathway genuinely causally responsible for FR failures?** The evidence is correlational (C2 features are more active on failing prompts, L23_F64429 is inhibitory). A direct ablation of the C2 community as a whole has not been done.

3. **What drives the residual 5/18 quote predictions?** They are tied to concepts chaud/propre. Are these word-specific tokenization priors, or does the model have a structural tendency to quote certain French adjectives? This could be tested with different French concepts from the same semantic category.

4. **Does the L12→L13 motif generalize?** This is the most preserved finding across B1-v1 and B1-v2. Whether it appears in other antonym behaviors, other language pairs, or other semantic inversion tasks is unknown.

5. **Why is IoU so flat in B1-v2?** The pooled gradient (1.090×) is much weaker than B1-v1 (1.253×). Is this because the B1-v2 genuine circuit uses a different set of features with more uniform cross-layer overlap? Or is the IoU metric simply too noisy with n=48 prompts per language?

6. **What would S2 look like with better concept coverage?** Currently only 8 pairs, 6 prompts each. The S2 test has 8 injection pairs. A larger concept set might show clearer transfer patterns.

7. **Is L22_F41906 specific to this transcoder run, or is it reproducible?** Feature indices in transcoders are run-specific. This question matters if comparing across model versions.

---

## 10. Minimal Reproduction Instructions

### Prerequisites
- CSD3 HPC access (SLURM)
- Project checked out at `/rds/user/iv294/hpc-work/thesis/project`
- Python venv with required packages
- B1-v2 templates committed (after 2026-03-21)

### Step-by-step (train split)

```bash
# 0. Verify templates (must show 0 FR prompts with quoted-word pattern)
cd /rds/user/iv294/hpc-work/thesis/project
git pull
python scripts/01_generate_prompts.py --behaviour multilingual_circuits_b1
grep '"' data/prompts/multilingual_circuits_b1_train.jsonl | grep '"fr"' | head -3
# Must return 0 lines

# 1. Baseline gate (fast check, 20min, GPU)
sbatch jobs/multilingual_circuits_b1_02_gate.sbatch
# Expected: GATE PASS, EN=1.000, FR≥0.75, argmax_is_quote_rate=0.00

# 2. Full pipeline (16h, GPU)
sbatch jobs/multilingual_circuits_b1_02_09.sbatch
# Runs: 02→04→06a→06b→07(ablation)→07(patching)→09

# 3. Causal edges + circuit (3h, GPU)
sbatch jobs/multilingual_circuits_b1_08_causal_edges.sbatch

# 4. Ablation supplement (GPU)
sbatch jobs/multilingual_circuits_b1_11_ablation_supplement.sbatch

# 5. Reasoning traces (CPU-only)
sbatch jobs/multilingual_circuits_b1_10_reasoning_trace_v2.sbatch

# 6. Sync to local (see sync instructions below)

# 7. Run analysis locally (CPU, ~5 min)
python scripts/a_analyze_multilingual_circuits.py \
  --behaviour multilingual_circuits_b1 \
  --split train \
  --node_labels \
  --community_summary
```

### Files to sync from HPC (after pipeline)
```
data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96_roleaware.json
data/results/attribution_graphs/multilingual_circuits_b1/attribution_graph_train_n96.json
data/results/baseline_multilingual_circuits_b1_train.csv
data/results/causal_edges/multilingual_circuits_b1/circuits_multilingual_circuits_b1_train.json
data/results/interventions/multilingual_circuits_b1/intervention_ablation_multilingual_circuits_b1.csv
data/results/interventions/multilingual_circuits_b1/intervention_patching_C3_multilingual_circuits_b1.csv
data/results/reasoning_traces/multilingual_circuits_b1/error_cases_train.json
data/results/transcoder_features/multilingual_circuits_b1_train_position_map.json
data/results/transcoder_features/layer_{10..25}/multilingual_circuits_b1_train_top_k_indices.npy
```

### Key reference parameters
```
--top_k 50          (script 04: feature extraction)
--top_k 20          (script 07: intervention)
--vw_threshold 0.01 (script 06b: roleaware graph)
--k_content 10      (script 06b)
--per_feature        (script 07: must be explicit for _b1 suffix)
n_prompts=96
```
