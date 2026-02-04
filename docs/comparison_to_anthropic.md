# Side-by-Side Comparison with Anthropic's Results

## Key Findings Comparison

### 1. Multi-Step Reasoning (Factual Recall)

**Anthropic (Claude 3.5 Haiku):**
- "Dallas" -> activates Texas features -> activates Austin features
- Clear multi-hop attribution path through features
- Intervention on Texas features blocks Austin output

**This Work (Qwen3-4B-Instruct-2507):**
- Country -> Capital attribution path
- [Results TBD]
- [Intervention results TBD]

### 2. Arithmetic Circuits

**Anthropic:**
- Found addition lookup table features
- Features respond to specific operand combinations
- Diagonal activation patterns in operand x operand heatmaps
- Generalize across contexts (citations, astronomy data)

**This Work:**
- Two-digit addition prompts
- [Feature structure TBD]
- [Generalization TBD]

### 3. Sentiment Processing

**Anthropic:**
- Identified sentiment aggregation features
- Clear positive/negative separation in feature space
- Context-dependent sentiment features

**This Work:**
- Sentiment continuation with explicit positive/negative contexts
- [Feature separation TBD]
- [Attribution structure TBD]

### 4. Grammatical Circuits

**Anthropic:**
- Not a primary focus but mentioned in context of language processing
- Subject-number tracking through attention patterns

**This Work:**
- Primary focus on subject-verb agreement
- [Circuit structure TBD]
- [Intervention results TBD]

---

## Methodology Comparison

### Feature Extraction

| Aspect | Anthropic CLT | Our PLTs (circuit-tracer) |
|--------|---------------|---------------------------|
| Type | Cross-layer transcoder | Per-layer transcoder |
| Output | Multi-layer decoder | Single layer |
| Sparsity | JumpReLU | JumpReLU |
| Features | 30M | ~200K per layer |
| Cross-layer | Native (multi-layer decoder) | Approximated via virtual weights |
| Training | Custom on proprietary data | Pre-trained (circuit-tracer) |
| Source | Proprietary | HuggingFace (mwhanna/*) |

**Impact:** We use pre-trained PLTs from circuit-tracer rather than training our own.
PLTs operate on single layers, so cross-layer attribution requires computing virtual
weight matrices: W_enc_target @ W_dec_source.T to approximate pathways.

### Attribution Method

| Aspect | Anthropic | This Work |
|--------|-----------|-----------|
| Edge computation | Backward Jacobian with stop-gradients | Gradient-based approximation |
| Nonlinearity handling | Frozen nonlinearities | Standard backprop |
| Error nodes | Explicit | Not included |
| Attention | Frozen patterns | Not decomposed |

**Impact:** Our attribution is a first-order approximation of their method.
We expect noisier graphs with potentially more false edges.

### Validation

| Method | Anthropic | This Work |
|--------|-----------|-----------|
| Feature suppression | -M * activation | Zero ablation |
| Constrained patching | Layer-range limited | Full forward pass |
| Swap experiments | Feature-level swaps | Feature-level swaps |
| Effect measurement | Output change | Logit difference change |

---

## Scale Analysis

### What Changes at 4B vs 70B+

**Hypotheses:**
1. Fewer distinct features per concept (less redundancy)
2. More polysemantic features (multi-use)
3. Shallower circuits (fewer intermediate steps)
4. Less robust to intervention (single-point failures)

**Evidence required:**
- Feature interpretation statistics
- Circuit depth measurements
- Intervention effect magnitudes
- Polysemanticity scores

---

## Figure Comparison

### Planned Side-by-Side Figures

| Figure | Anthropic | This Work |
|--------|-----------|-----------|
| Attribution graph | Fig 1 (Dallas->Texas->Austin) | Country->Capital path |
| Addition features | Fig 3 (operand heatmaps) | 2-digit addition heatmaps |
| Sentiment circuit | Fig 5 (sentiment aggregation) | Sentiment continuation path |
| Intervention effects | Bar charts of % effect | Bar charts of logit diff change |

---

*Last updated: [Date]*
*Status: Template prepared, results pending*
