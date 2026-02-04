# Negative Results and Lessons Learned

This document records experiments that did not produce expected results,
which is an essential part of rigorous scientific analysis.

As noted in the project requirements: "Negative or partial reproductions
are acceptable if analysed rigorously."

---

## Template for Recording Negative Results

### [Experiment Name]

**Date:** YYYY-MM-DD

**Hypothesis:** What we expected to find.

**Method:** What we did.

**Result:** What actually happened.

**Analysis:** Why we think this occurred.

**Implications:** What this means for the broader research question.

---

## Anticipated Failure Modes

### 1. Feature Polysemanticity at 4B Scale

**Expectation:** SAE features may be more polysemantic (encoding multiple concepts)
at 4B scale compared to ~70B+ models.

**Evidence to look for:**
- Top-activating prompts for a feature span multiple unrelated categories
- Low feature-category correlation
- High feature co-activation rates

**If confirmed:** This supports the hypothesis that model scale affects
interpretability, not just capability.

### 2. Weak Attribution Signals

**Expectation:** Gradient-based attributions may be noisier with simplified
methodology (standard SAEs vs cross-layer transcoders).

**Evidence to look for:**
- Attribution graphs are dense (many weak edges, few strong ones)
- Top features change significantly across prompts
- Intervention effects are smaller than expected

**If confirmed:** Document the methodological gap between standard SAEs
and Anthropic's CLT approach.

### 3. Behaviour-Specific Failures

Some behaviours may not produce clear circuits at 4B scale:
- **Arithmetic:** May rely more on memorization than structured circuits
- **Factual recall:** Knowledge may be distributed rather than localized
- **Sentiment:** May be too diffuse across layers

### 4. SAE Training Instability

**Potential issues:**
- Dead features (>20% never activate)
- Low reconstruction quality (R^2 < 0.85)
- Feature collapse (many features encoding similar information)

---

## Actual Negative Results

*To be populated during experiments*

### [Example entry - to be replaced]

**Date:** TBD

**Behaviour:** TBD

**Expected:** TBD

**Observed:** TBD

**Analysis:** TBD

---

## Summary

| Behaviour | Expected Circuit | Found? | Notes |
|-----------|-----------------|--------|-------|
| Grammar Agreement | Number feature -> verb selection | TBD | |
| Factual Recall | Entity -> fact -> output | TBD | |
| Sentiment | Context -> sentiment -> continuation | TBD | |
| Arithmetic | Operand features -> sum lookup | TBD | |

---

*This document is updated throughout the experimental process.*
