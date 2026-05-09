# Abstraction Probe Designs — Exploratory Phase

> **Date**: 2026-05-09 | Goal: identify the cleanest abstraction behaviour for full mechanistic analysis

---

## 1. Core Hypothesis

We test whether Qwen3 forms an **invariant abstract variable** — an internal representation
that is shared across different surface expressions, domains, or notations:

```
x_surface  →  z_abstract  →  y_surface
```

The specific invariance we test: **does a quantity change when the system scales?**

This is distinct from latent-state retrieval (previous candidate-selection project).
Here we ask whether the model's internal representation respects a mathematical/physical
invariant regardless of how the question is phrased.

---

## 2. Five Probe Families

### A. Intensive vs Extensive (Primary Candidate)

**Abstraction**: scale invariance — does a quantity change with system size?

| Class | Examples | Scaling rule |
|---|---|---|
| Intensive | temperature, pressure, density, pH | Q(2S) = Q(S) — invariant |
| Extensive | mass, volume, energy, charge | Q(2S) = 2Q(S) — linear |

**Wording families** (8 variants per property):

| Code | Template | Intensive answer | Extensive answer |
|---|---|---|---|
| W0_combine_doubles | "Two identical containers combined — does X double?" | No | Yes |
| W1_scale_amount | "Take twice as much — does X double?" | No | Yes |
| W2_split_preserves | "Split in half — does each half equal X?" | Yes | No |
| W3_sample_size | "Does X depend on how large a sample?" | No | Yes |
| W4_additive | "Is X additive (total = sum of parts)?" | No | Yes |
| W5_expert_claim | "A physicist says: X doubles when amount doubles. Correct?" | No | Yes |
| W6_symbolic | "Does Q(2S) = 2Q(S) hold for X?" | No | Yes |
| W7_adv_claim | "Since larger systems have more material, X must be larger. Correct?" | No | Yes |

**Adversarial properties**: voltage (sounds like it should scale), specific gravity,
heat capacity (easy to confuse with specific heat capacity).

**Total**: 25 properties × 8 wording families = 200 prompts.

**Strength of design**:
- Binary answer format (' Yes'/' No')
- Rich cross-domain analogs available (→ Family D)
- Clear mechanistic hypothesis: feature clusters encode scale-invariance class
- Many properties have confusable surface features (adversarial test)

**Potential confounds**:
- Model may have memorised "temperature is intensive" as a fact
- Wording involving "double" or "scale" may trigger arithmetic reasoning
- "Expert claim" wording may trigger authority-based acceptance

---

### B. Mathematical Scaling Laws

**Abstraction**: power-law exponent under uniform length scaling (x → kx).

| Quantity | Exponent | Example |
|---|---|---|
| Angle, ratio, density | 0 (invariant) | angles don't scale with length |
| Perimeter, circumference | 1 | linear with k |
| Area, surface area | 2 | quadratic in k |
| Volume, mass | 3 | cubic in k |

**Wording families** (4 variants per quantity):

| Code | Template |
|---|---|
| SB0_double_invariant | "If all lengths doubled, does X remain unchanged?" |
| SB1_scale_k | "If lengths scaled by k, does X scale by exactly k?" |
| SB2_scale_k2 | "If lengths scaled by k, does X scale by k²?" |
| SB3_double_eight | "If all lengths doubled, does X increase by factor 8?" |

**Total**: 13 quantities × 4 wording families = 52 prompts.

**Strength**: purely mathematical, no domain knowledge required.
**Weakness**: binary questions per exponent value may be too degenerate (most pairs are No/No).

---

### C. Representation Equivalence

**Abstraction**: mathematical identity across surface notation.

Tests whether the model treats `x²`, `x·x`, `x squared`, `x to the power of 2` as the
same mathematical object. Includes non-equivalent pairs as adversarial cases:
- `√(x+y) ≠ √x + √y` (common student error)
- `(x+y)² ≠ x² + y²`

**Wording families** (4 variants):

| Code | Template |
|---|---|
| RC0_direct | "Is 'x²' the same mathematical expression as 'x·x'?" |
| RC1_function | "If f(x) = x² and g(x) = x·x, are f and g identical functions?" |
| RC2_student | "A student writes x² = x·x. Is the student correct?" |
| RC3_notation | "Do 'x²' and 'x·x' represent the same mathematical object?" |

**Total**: 16 concept pairs × 4 wording families = 64 prompts.

**Strength**: directly tests notation-independence of mathematical concepts.
**Weakness**: the model may simply pattern-match on well-known identities rather than
representing them abstractly.

---

### D. Cross-Domain Abstraction

**Abstraction**: same intensive/extensive structure across physics, economics,
information theory, statistics, biology.

| Domain | Intensive analog | Extensive analog |
|---|---|---|
| Economics | price per unit, profit margin | total revenue, total cost |
| Information | bits per symbol, compression ratio | total bits, file size |
| Statistics | mean, variance | sum, total squared deviation |
| Biology | population density, metabolic rate/kg | total population, total metabolic rate |

**Wording families** (4 variants):

| Code | Template |
|---|---|
| CD0_scale_up | "If you scale up a system by 2×, does X double?" |
| CD1_combine | "If two identical systems combine, does X of the combined system equal 2×?" |
| CD2_subsystem | "A subsystem half the size — does it have the same X?" |
| CD3_proportional | "Is X proportional to system size?" |

**Total**: 17 cross-domain properties × 4 wording families = 68 prompts.

**Strength**: tests genuine generalisation across domains. If the model has learned
intensive/extensive as an abstract concept (not just physics facts), it should transfer.
**Weakness**: economic/statistical properties may be underrepresented in training data.

---

### E. Conservation Laws

**Abstraction**: temporal invariance — is a quantity conserved in an isolated system?

| Class | Examples |
|---|---|
| Conserved | total energy, linear momentum, angular momentum, electric charge |
| Non-conserved | temperature, entropy (can increase), kinetic energy (inelastic), mechanical energy (friction) |

**Wording families** (4 variants):

| Code | Template |
|---|---|
| CL0_isolated | "In a perfectly isolated system, does X remain constant?" |
| CL1_process | "During any physical process in a closed system, is X always conserved?" |
| CL2_symmetry | "Is X a conserved quantity — its total value cannot change due to internal interactions?" |
| CL3_collision | "Two objects collide in empty space. Is X the same before and after?" |

**Total**: 12 quantities × 4 wording families = 48 prompts.

**Strength**: clear Noether-theorem basis; mechanistically distinct from size-scaling.
**Weakness**: smaller dataset; conservation laws may be memorised as facts.

---

## 3. Evaluation Pipeline (Script 61)

### Behavioral metrics

| Metric | Formula | Good threshold |
|---|---|---|
| Accuracy | fraction correct predictions | ≥ 0.75 |
| Consistency | fraction of properties with consistent prediction across all wording families | ≥ 0.80 |
| ND margin | logp(' Yes') − logp(' No') | class-separated |
| ND-AUC | AUC of ND predicting abstraction class | ≥ 0.70 |
| Adversarial gap | accuracy_non_adv − accuracy_adv | < 0.15 (small = no shortcut) |
| Domain std | std of accuracy across domains (Family D only) | < 0.10 (low = generalises) |

### Representation metrics (per-layer hidden states)

| Metric | Description |
|---|---|
| Linear probe acc | 5-fold CV logistic regression on normalised hidden states |
| ARI(abstraction class) | k-means ARI relative to true class labels |
| ARI(wording family) | k-means ARI relative to wording family labels |
| Ratio | ARI(cls) / ARI(wording) — >1 means abstraction > surface encoding |

Key signal: does any layer show a **form→abstraction transition** analogous to the
form→content transition observed in `physics_internal_candidate_selection_v2`?

---

## 4. Ranking Criteria

Composite score = 0.35·accuracy + 0.30·consistency + 0.20·ND-AUC + 0.15·domain-gen

**Additional gates for full mechanistic pipeline:**
1. Accuracy ≥ 0.75 — model reliably knows the abstract rule
2. Consistency ≥ 0.80 — abstraction is robust to surface form
3. Adversarial gap < 0.15 — model is not using lexical shortcuts
4. Domain generalization (Family D) — abstraction is domain-independent
5. Linear probe peak layer — clear form→abstraction transition exists

---

## 5. Anticipated Outcomes and Confounds

### Expected results

| Family | Expected accuracy | Expected risk |
|---|---|---|
| A (intensive/extensive) | 0.75–0.90 | Some memorization of common properties |
| B (scaling law) | 0.65–0.80 | May struggle with exponents 2 and 3 |
| C (representation equivalence) | 0.80–0.95 | High for well-known identities |
| D (cross-domain) | 0.65–0.80 | Lower for economics/biology domains |
| E (conservation) | 0.70–0.85 | May confuse kinetic vs total energy |

### Confound tests

**Lexical shortcut detection**:
- W7_adv_claim (Family A): misleading framing using "more material"
- heat capacity vs specific heat capacity (Family A): similar names, different classes
- Non-equivalent pairs in Family C: common student errors

**Memorization detection**:
- Temperature, mass, volume — well-known examples (likely memorised)
- Electrical resistivity, specific gravity — less common (tests genuine abstraction)
- Economics/statistics properties (Family D) — tests transfer beyond physics training

**Template exploitation detection**:
- W6_symbolic uses mathematical notation — tests if symbolic form changes answer
- Consistency across W0–W7 for the same property is the main signal

---

## 6. Mechanistic Analysis Potential

### Best candidate: Family A (Intensive vs Extensive)

**Why Family A is most mechanistically tractable**:

1. **Clean binary label** — intensive vs extensive is the abstraction class
2. **Rich feature space** — 25 properties × 8 wordings = 200 prompts (enough for attribution graph)
3. **Cross-domain validation** — Family D uses same label with different surface forms
4. **Known physics** — ground truth is well-defined (unlike subjective tasks)
5. **Form→content transition** — expected layer where features shift from encoding "combine/scale/split" 
   to encoding "intensive/extensive" class

**Mechanistic hypothesis**:
Features in L10–L16 encode wording structure (combine, double, split).
Features in L22–L25 encode the abstract class (intensive vs extensive).
Ablating L22–L25 clusters should flip intensive↔extensive predictions more often than they
flip wording-family predictions.

### Second candidate: Family D (Cross-domain)

If Family A shows strong layer-level abstraction, Family D is the cleanest test of
domain-independence: same abstract label, different surface domain. If the same feature
clusters activate for both physics-intensive and economics-intensive, that is strong
evidence for a domain-independent representation.

### Fallback: Family E (Conservation)

Uses a different invariance type (temporal vs spatial/scaling). If Families A and D
show similar circuits, Family E would test whether the same abstraction mechanism
handles different types of physical invariance.

---

## 7. Recommended Next Steps (after baselines)

1. **If Family A passes all gates**: proceed to full attribution graph pipeline
   (`scripts/04`, `scripts/06`, `scripts/07`) on `physics_intensive_extensive_v1`.

2. **If Family D shows strong cross-domain consistency**: run joint attribution on
   physics+economics prompts to test for shared feature clusters.

3. **If linear probe shows clear transition layer** (ratio >1 at L22+): design targeted
   ablation of L22–L25 clusters and measure which class is disrupted.

4. **If adversarial gap > 0.15**: strengthen adversarial prompt set before mechanistic
   analysis to avoid confounded circuits.

5. **If Family A fails (< 0.70 accuracy)**: try a simpler intensive/extensive framing
   with explicit "yes/no" question structure and reduce property set to unambiguous cases.

---

## 8. Files

| File | Description |
|---|---|
| `scripts/60_generate_abstraction_probe_datasets.py` | Dataset generator (5 families, 432 prompts) |
| `scripts/61_run_abstraction_baselines.py` | Baseline evaluator + representation analysis |
| `jobs/run_abstraction_baseline.sbatch` | CSD3 SLURM job (Ampere GPU, 3h, 4B default) |
| `data/prompts/abstraction/A_intensive_extensive_train.jsonl` | Family A (161 train) |
| `data/prompts/abstraction/abstraction_all_train.jsonl` | All families combined (350 train) |
| `data/results/abstraction_probe/{model_tag}/` | Evaluation outputs per model |
| `docs/abstraction_probe_designs.md` | This document |
