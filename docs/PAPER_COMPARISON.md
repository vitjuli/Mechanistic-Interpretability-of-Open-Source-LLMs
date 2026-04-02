# Systematic Comparison to Anthropic's "On the Biology of a Large Language Model"

This document provides a detailed comparison between Anthropic's original work and our reproduction study on Qwen3-4B-Instruct-2507.

## Overview

| Aspect | Anthropic (2025) | This Work |
|--------|------------------|-----------|
| **Model** | Claude 3.5 Haiku (~70B params) | Qwen3-4B-Instruct-2507 (4B params) |
| **Model Access** | Proprietary, internal | Open weights, HuggingFace |
| **Scale Ratio** | ~18x larger | Baseline |
| **Feature Extraction** | Cross-layer transcoders (CLT) | Standard SAEs |
| **Feature Count** | 30 million | ~200K (15 layers x 14K) |

---

## Methodology Comparison

### Feature Extraction

| Component | Anthropic | This Work | Notes |
|-----------|-----------|-----------|-------|
| Architecture | Cross-layer transcoder | Standard SAE | CLT enables cross-layer attribution |
| Input | Residual stream | MLP activations | We focus on MLP for simplicity |
| Expansion | Variable (300K-30M) | 4x fixed | Computational constraints |
| Sparsity | JumpReLU | ReLU + L1 | Similar effect |
| Training data | Diverse web text | Behaviour-specific prompts | More focused |

### Attribution Graphs

| Component | Anthropic | This Work | Notes |
|-----------|-----------|-----------|-------|
| Edge computation | Backward Jacobians | Gradient-based | Simplified approximation |
| Pruning | 0.8 threshold | Top-k per node | Different strategies |
| Node types | Input, CLT features, Error, Output | Input, SAE features, Output | No error nodes |
| Visualization | Interactive, supernodes | Static NetworkX | Simpler tooling |

### Intervention Methods

| Method | Anthropic | This Work | Notes |
|--------|-----------|-----------|-------|
| Ablation | Clamping to negative values | Zero ablation | Similar effect |
| Patching | Constrained to layer range | Full forward pass | Less precise |
| Steering | Feature injection | Not implemented | Future work |

---

## Behaviours Compared

### 1. Grammatical Agreement

**Anthropic findings:**
- Clear subject-number features in middle layers
- Verb agreement computed via feature interactions
- Robust across diverse sentence structures

**Our investigation:**
- [ ] Subject-number features identified?
- [ ] Layer localization matches?
- [ ] Intervention effects similar?

### 2. Factual Recall

**Anthropic findings:**
- Multi-hop reasoning (Dallas -> Texas -> Austin)
- Entity features activate in early-middle layers
- Knowledge recall in later layers

**Our investigation:**
- [ ] Similar multi-hop structure?
- [ ] Entity features present at 4B scale?
- [ ] Recall mechanism comparable?

### 3. Sentiment Continuation

**Anthropic findings:**
- Sentiment features aggregate from context
- Clear positive/negative feature separation
- Continuation driven by sentiment features

**Our investigation:**
- [ ] Sentiment features identifiable?
- [ ] Clear separation at smaller scale?
- [ ] Causal relationship verified?

### 4. Arithmetic

**Anthropic findings:**
- Addition lookup tables in features
- Generalizes across contexts (citations, tables, etc.)
- Operand-specific and sum-specific features

**Our investigation:**
- [ ] Lookup table features present?
- [ ] Context generalization?
- [ ] Feature structure comparable?

---

## Expected Differences

### Scale-Related

1. **Feature granularity**: Smaller models may have more polysemantic features
2. **Circuit depth**: Simpler circuits expected at 4B scale
3. **Robustness**: Less consistent behaviour across prompts

### Methodology-Related

1. **SAE vs CLT**: Standard SAEs may miss cross-layer interactions
2. **Training data**: Behaviour-specific training may reduce generalization
3. **Attribution approximation**: Simplified gradients may miss indirect effects

---

## Metrics for Comparison

### Quantitative

| Metric | Anthropic Reported | Our Target | Actual |
|--------|-------------------|------------|--------|
| SAE R² | ~78% (Haiku) | >85% | TBD |
| L0 sparsity | ~235 features/token | <100 | TBD |
| Dead features | <15% | <20% | TBD |
| Intervention effect | >50% logit change | >30% | TBD |

### Qualitative

- [ ] Features interpretable by inspection?
- [ ] Attribution graphs match expected structure?
- [ ] Interventions produce predicted effects?

---

## Key Figures to Reproduce

### From Biology Paper

1. **Fig 1**: Dallas -> Texas -> Austin attribution graph
   - Our equivalent: Country -> Capital graph for factual recall

2. **Fig 3**: Addition lookup tables
   - Our equivalent: Arithmetic feature heatmaps

3. **Fig 5**: Sentiment feature aggregation
   - Our equivalent: Sentiment continuation attribution

4. **Fig 7**: Grammatical agreement circuit
   - Our equivalent: Subject-verb agreement features

---

## Interpretation Guidelines

### Positive Results
If we find similar circuits:
- Suggests circuits are fundamental, not scale-dependent
- Validates Anthropic's methodology on open models
- Contributes to circuit taxonomy

### Negative Results
If circuits differ significantly:
- Documents scale threshold for circuit emergence
- Identifies model-specific vs universal patterns
- Guides future interpretability research

### Partial Results
Most likely outcome:
- Some behaviours show clear circuits
- Others are more diffuse/polysemantic
- Provides nuanced picture of scale effects

---

## References

1. Lindsey, J., Gurnee, W., et al. (2025). On the Biology of a Large Language Model. Anthropic.
2. Anthropic (2025). Attribution Graphs: Methods.
3. Bricken, T., et al. (2023). Towards Monosemanticity. Anthropic.
4. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features.

---

*Last updated: [Date of analysis]*
