# Abstraction Probe Baseline Results

> **Model**: Qwen3-4B | **SLURM**: 29122008 | **Date**: 2026-05-09 | **Split**: train

---

## 1. Summary Table

| Family | Accuracy | Consistency | ND-AUC | Verdict |
|---|---|---|---|---|
| A — Intensive/Extensive | **0.758** | 0.784 | 0.251 | ✓ **Proceed** (marginal pass) |
| B — Scaling Law | 0.857 | 0.763 | — | ~ Simplify then proceed |
| C — Repr. Equivalence | 0.923 | 0.907 | 0.067 | ✗ Ceiling — not interesting |
| D — Cross-domain | 0.455 | 0.873 | 0.401 | ✗ Systematically wrong — important failure |
| E — Conservation Laws | 0.950 | 0.972 | 0.025 | ✗ Ceiling — memorised facts |

---

## 2. Family A — Intensive vs Extensive

### Behavioral results

| Wording family | Accuracy | Assessment |
|---|---|---|
| W0_combine_doubles | **94.7%** | Very strong — direct scaling question |
| W4_additive | **90.9%** | Strong — additive/non-additive framing |
| W2_split_preserves | **86.4%** | Strong — halving preserves intensive |
| W3_sample_size | 76.2% | Adequate |
| W6_symbolic | 71.4% | Adequate — Q(2S)=2Q(S) notation |
| W1_scale_amount | 68.8% | Weak — "take twice as much" less clear |
| W7_adv_claim | 66.7% | Weak — adversarial "more material → more X" |
| **W5_expert_claim** | **47.4%** | **Below chance — authority override** |

**Overall**: 75.8% accuracy (passes ≥75% gate), 78.4% consistency (just below ≥80% gate).

### Key finding: W5 authority override

`W5_expert_claim` template: *"A physicist states: '[property] doubles when the amount of material doubles.' Is this statement correct?"*

The model answers Yes (agrees with the physicist) 52.6% of the time for intensive properties where the physicist's claim is **wrong**. This is a genuine mechanistic signal: the model's intensive/extensive knowledge can be overridden by the `"A physicist states"` authority framing. 

This is **not** a flaw in the probe — it reveals that the abstraction is fragile under social/authority context. For mechanistic analysis: W5 should be excluded from the clean probe set.

### Recommended clean wording set for mechanistic pipeline

Use **W0 + W2 + W4** only (combined accuracy ~90.7%). These share the same logic (combine/split/additive) and produce consistent, high-confidence answers. Exclude W5 (authority override confound) and W7 (adversarial, 66.7% only).

### Adversarial properties

Heat capacity (extensive, confused with specific heat which is intensive): needs verification from the saved CSV.
Voltage (intensive, might be expected to scale): needs verification.

---

## 3. Family B — Mathematical Scaling Laws

### Behavioral results

| Wording family | Accuracy |
|---|---|
| SB0_double_invariant | **100%** — perfect |
| SB3_double_eight | **91.7%** — cubic scaling well-known |
| SB1_scale_k | 80.0% — linear scaling mostly correct |
| SB2_scale_k2 | 70.0% — quadratic scaling hardest |

**Overall**: 85.7% accuracy, 76.3% consistency.

### Assessment

The model handles simple mathematical scaling reasonably well. The difficulty gradient is:
invariance (100%) > cubic (91.7%) > linear (80%) > quadratic (70%).

**Why this family is not ideal for mechanistic analysis**:
- Multi-class problem (5 exponent values): harder to define clean binary classification
- Small dataset (42 train prompts)
- Accuracy variation is mainly about mathematical knowledge, not abstract representation

**Potential**: simplify to binary (scale-invariant vs scales) for a cleaner probe. The
"invariant" (exponent=0) vs "scales" (exponent≥1) split would give 20 invariant vs 28 scaling
prompts with binary ND signal.

---

## 4. Family C — Representation Equivalence

### Behavioral results

| Wording family | Accuracy |
|---|---|
| RC1_function | **100%** |
| RC3_notation | 92.9% |
| RC0_direct | 91.7% |
| RC2_student | 85.7% |

**Overall**: 92.3% accuracy, 90.7% consistency. ND-AUC = **0.067** (near-zero).

### Why this family is not useful for mechanistic analysis

High accuracy with near-zero ND-AUC means the model is very confident about everything —
equivalent and non-equivalent pairs alike. The model has memorised these mathematical
identities as facts and applies them with equal confidence regardless of the answer.

This pattern (high accuracy + low ND-AUC) is the signature of **fact retrieval**, not
abstract reasoning. The model is not computing equivalence — it is retrieving a stored
relation. Mechanistically, this would produce a "lookup" circuit rather than an
"abstraction" circuit.

The non-equivalent adversarial pairs (RC2_student: 85.7%) are interesting — the model
correctly rejects `√(x+y) ≠ √x + √y` at 85.7% — but this is still fact retrieval.

---

## 5. Family D — Cross-Domain Abstraction

### Behavioral results

| Wording family | Accuracy | Notes |
|---|---|---|
| CD0_scale_up | **15.4%** | Catastrophic — domain-specific confound |
| CD2_subsystem | 42.9% | Below chance |
| CD1_combine | 58.3% | Near chance |
| CD3_proportional | 62.5% | Near chance |

**Overall**: 45.5% accuracy (below chance), 87.3% consistency, ND-AUC = 0.401.

### The central scientific finding: no cross-domain abstraction

Family D is the most important result of this exploratory phase.

The model has **87.3% consistency** at **45.5% accuracy** — it is reliably, consistently
giving the **wrong** answer for cross-domain intensive/extensive questions. ND-AUC = 0.401
(just below 0.5) confirms this: the model's confidence (logit ND) slightly predicts the
class but in the **inverted** direction. If we flipped the label assignment, we'd get
AUC ≈ 0.60 — meaning the model genuinely distinguishes the classes, but reverses them.

**Interpretation**: Qwen3-4B's intensive/extensive knowledge is **domain-specific to physics**.
When the same concept appears in economics or statistics, the model applies different
reasoning that systematically misclassifies:

- "price per unit" (intensive) → model says it scales with system size (extensive behaviour)
- "total revenue" (extensive) → model says it doesn't scale (intensive behaviour)

The model likely has separate knowledge stores for physics quantities and economic quantities,
and neither refers to a shared abstract "intensive/extensive" concept.

### CD0 special case

`CD0_scale_up` is worst (15.4%) because the phrase "scale up a system" is semantically
different across domains:
- Physics: "scale up" = take more material at same density → intensive quantities unchanged
- Economics: "scale up" = grow the business → affects both price and revenue in complex ways
- Biology: "scale up" = grow the population → density changes due to carrying capacity

The model applies domain-specific knowledge of what "scaling up" means rather than the
abstract mathematical operation. This is **not a dataset bug** — it is a genuine finding
that the model lacks a domain-invariant "scale by λ" operator.

### What this means

Family D provides strong evidence **against** the hypothesis that Qwen3-4B has formed a
domain-independent abstract representation of scale invariance. The model's intensive/
extensive knowledge exists as domain-specific facts, not as an abstract invariant.

However, this **does not** mean there is nothing interesting to find mechanistically.
The question becomes: does the model have a **physics-specific** intensive/extensive
circuit that it applies within physics but not across domains? If so, that is a distinct
(narrower) form of abstraction worth studying.

---

## 6. Family E — Conservation Laws

### Behavioral results

| Wording family | Accuracy |
|---|---|
| CL1_process | **100%** |
| CL2_symmetry | **100%** |
| CL0_isolated | 90.0% |
| CL3_collision | 87.5% |

**Overall**: 95.0% accuracy, 97.2% consistency. ND-AUC = **0.025** (near-zero).

### Assessment

Same pattern as Family C: ceiling accuracy with near-zero ND-AUC.
Conservation laws (total energy, momentum, charge) are extremely well-memorised facts
in Qwen3-4B. The model applies them consistently across all four wording styles.

This is too easy to be mechanistically interesting — there is little variation to explain.
The near-zero ND-AUC means both conserved and non-conserved quantities are answered with
equal high confidence. This is pure fact retrieval.

---

## 7. Representation Analysis (Family A)

**Warning**: The reported "Peak probe: L1 = 0.969" is **spurious**.

Layer 1 (and likely layers 0–2) show a KMeans convergence warning: "Number of distinct
clusters (1) found smaller than n_clusters". This means the hidden states at early layers
are nearly degenerate — all prompts map to essentially the same point in activation space.
Any linear probe fitted here overfits to tiny numerical noise rather than meaningful signal.

The fix (new `degenerate` flag, variance threshold < 1.0) has been applied and will be
active in the next run. Re-run with `--rep_analysis` to get valid layer curves.

The scientifically interesting question for Family A:
> At which layer does the hidden state representation transition from encoding
> "which wording family is this?" to encoding "is this intensive or extensive?"

This is the analogue of the form→content transition found in `physics_internal_candidate_selection_v2`
(particle ARI > wording ARI at L24). We expect to see this transition in the range L20–L28 for
intensive/extensive, but the analysis needs to be re-run.

---

## 8. Mechanistic Tractability Assessment

### Primary recommendation: Family A with restricted wording (W0 + W2 + W4)

**Rationale**:
- Accuracy 75.8% → high enough that the model knows the rule
- Consistency 78.4% → not perfect, but enough variation to study (unlike C/E at >90%)
- ND-AUC 0.251 → some logit separation between classes (unlike C/E at ~0.05)
- Rich wording variation → can study which surface features affect internal representations
- W5 authority override → identified confound that is itself a mechanistic signal
- 25 properties × up to 8 wording families = sufficient dataset for attribution graphs

**Proposed behaviour name**: `physics_intensive_extensive_v1`

### Secondary test: Family D as generalization probe

Use Family D **after** the full mechanistic pipeline on Family A, not as the primary dataset.
The question to answer: do the intensive/extensive feature clusters found in Family A
physics prompts also activate for Family D economics/statistics prompts?

If yes → domain-independent abstraction (strong claim).
If no → physics-specific memorization circuit (still interesting, but narrower claim).

The current Family D accuracy (45.5%) does not disqualify it as a probe — it means the
model's **output** doesn't generalise, but the **internal representation** might still
show clustering by abstract class. This needs to be tested with hidden-state analysis.

### Do NOT proceed with:
- Family C (ceiling, fact retrieval)
- Family E (ceiling, fact retrieval)
- Family B (multi-class, small dataset, needs redesign as binary)

---

## 9. Next Steps

### Immediate (before full pipeline):

1. **Re-run representation analysis** with fixed degenerate-layer filtering:
   ```bash
   sbatch jobs/run_abstraction_baseline.sbatch   # pushed fix to 61_run_abstraction_baselines.py
   ```

2. **Inspect worst-performing properties** in Family A from the CSV:
   - Which specific intensive properties fail most? (heat capacity? voltage?)
   - Are adversarial properties systematically harder?

3. **Design `physics_intensive_extensive_v1`** dataset:
   - Use W0 + W2 + W4 wording families only
   - Use 12 intensive + 9 extensive properties (exclude most adversarial)
   - Target: ~60–80 train prompts, ~20 test
   - Format: compatible with existing `scripts/01_generate_prompts.py`

4. **Run 0.6B and 8B models** to establish scale effect:
   - Does 0.6B show the same domain-specificity failure?
   - Does 8B generalise cross-domain (Family D)?

### Full mechanistic pipeline (once representation analysis confirms signal):

5. Run `scripts/04_extract_transcoder_features.py` on Family A clean set
6. Run `scripts/06_build_attribution_graph.py` → identify key features
7. Run `scripts/07_run_interventions.py` → verify clusters
8. Test: do Family A clusters activate for Family D prompts? → domain-independence test

---

## 10. Files

| File | Description |
|---|---|
| `data/results/abstraction_probe/4b/results_A_4b.csv` | Family A per-prompt predictions |
| `data/results/abstraction_probe/4b/results_{B,C,D,E}_4b.csv` | Other families |
| `data/results/abstraction_probe/4b/family_ranking_4b.csv` | Family ranking table |
| `data/results/abstraction_probe/4b/baseline_report_4b.md` | Auto-generated report |
| `data/results/abstraction_probe/4b/representation_A.csv` | Layer probe/ARI (re-run needed) |
| `data/results/abstraction_probe/4b/summary_*_4b.png` | Per-family accuracy/ND plots |
| `docs/abstraction_probe_designs.md` | Design rationale (5 families) |
| `docs/abstraction_baseline_results.md` | This document |
