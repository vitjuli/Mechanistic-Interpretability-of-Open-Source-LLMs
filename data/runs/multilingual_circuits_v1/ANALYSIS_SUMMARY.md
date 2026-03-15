# Analysis Summary — multilingual_circuits v1

**Analysis date:** 2026-03-12
**Reference run:** SLURM 24386104 (pipeline) + CSD3 analysis script (2026-03-12)
**Status:** FINAL reference — all bugs fixed, correct prompts, correct language assignment

---

## Strongest Result

**C3 patching (Claim 4): disruption_rate = 0.588, mean_effect = −0.166 ± 0.019, CI [−0.202, −0.126]**

Patching EN antonym features into FR contexts degrades FR model confidence in 58.8% of
(feature × pair × layer) interventions. The 95% CI is fully negative and does not cross zero.
62.1% of individual graph features have a negative mean cross-lingual effect across all pairs.

This is the most direct evidence that EN and FR antonym computation shares a common set of
transcoder features: suppressing those features in one language also disrupts the other.
The effect is consistent and well-powered given the small dataset.

Best C3 layers: L12 (0.854), L20 (0.849), L22 (0.797), L25 (0.750), L23 (0.701).
Bridge features (32/53 = 60.4%) independently confirm: the majority of graph features
harm the correct answer in BOTH languages when ablated.

---

## Weakest Result

**Claim 3 (middle-layer concentration of shared features): IoU middle = 0.431 vs early/late = 0.412**

The IoU ratio is only 1.047x (4.7% higher in middle layers). While the direction matches
Anthropic (middle > early AND late), the contrast is too small to draw a strong conclusion.

The root cause is the measurement: IoU is computed on top-50 features at the **decision token
only** (last token position). At a single token position, EN and FR prompts of the same concept
activate very similar features regardless of layer — there is no strong reason for language-
specific features to appear more at early/late layers and shared features to appear more in
the middle, because the "language-specific" processing that happens earlier (reading the language,
building the lexical representation) is over by the time we reach the last token.

Anthropic uses feature activations across **all tokens in a passage** — early-layer features
are strongly tied to surface form (language), whereas middle-layer features are more semantic.
Collapsing to a single decision token loses this signal.

---

## Exact Limitations

| # | Limitation | Source | Impact on claims |
|---|---|---|---|
| 1 | Decision-token-only IoU | Script 04 `token_positions="decision"` | Eliminates Claim 3 signal; weakens Claims 1–2 |
| 2 | Transcoder features ≠ SAE features | Different feature geometry/sparsity | IoU values not directly comparable to Anthropic |
| 3 | Star graph topology (no feature–feature edges) | Script 06 per-prompt union method | Community structure trivial (1 community); cannot replicate circuit-level topology |
| 4 | 48 prompts (8 concepts × 6 templates) | Small dataset | All estimates have wide CIs; C3 pairs = 24 only |
| 5 | FR accuracy 66.7% (8 failures) | vide/plein (0/3), propre/sale (1/3) | C3 for failing concepts (4, 7) may be noise |
| 6 | IoU computed on attribution graph features, not raw top-50 activations | Different from Anthropic's activation-based IoU | Lower absolute IoU values; graph filter creates selection bias |

**Limitation 1 is the primary cause of the weak Claim 3 result.**
Limitations 2–4 affect absolute comparability but do not invalidate the directional findings.
Limitation 5 is minor: disruption is still above target even for failing FR concepts.

---

## Claim-Level Assessment

| Anthropic Claim | Our evidence | Assessment |
|---|---|---|
| **(1) Language-specific features exist** | Min per-layer IoU = 0.360; 7 EN-specific ablation features (harm EN only) | **Weakly supported.** 36–49% feature sharing per layer leaves room for language-specific features, but we cannot directly identify them from IoU alone. |
| **(2) Shared cross-lingual features exist** | Max per-layer IoU = 0.493; 32/53 bridge features; 62% of features have negative mean C3 effect | **Moderately supported.** Multiple independent measures (IoU, bridge ablation, C3 patching) converge on the same conclusion. |
| **(3) Shared features concentrated in middle layers** | IoU middle=0.431 > early=0.390, late=0.421 — correct direction, but 1.05x ratio | **Weakly supported — direction only.** The decision-token measurement cannot replicate the contrast Anthropic observed. Cannot make a strong claim. |
| **(4) Bridge features degrade both EN and FR** | 32/53 (60.4%) bridge features; C3 disruption_rate=0.588, CI [−0.202, −0.126] | **Strongly supported.** Three independent operationalizations (bridge ablation, C3 disruption, lang-swap strength) all exceed targets. The CI is fully negative. |

**Overall:** Claims 2 and 4 are sufficiently supported for thesis use as proxy evidence.
Claims 1 and 3 require qualification. Claim 3 requires multi-token feature extraction to
reproduce meaningfully.

---

## Next step
See `NEXT_IMPROVEMENT.md` for the single highest-value pipeline improvement.
