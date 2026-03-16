# v1 vs v2 Comparison — multilingual_circuits

**v1:** Decision-token IoU (1 sample/prompt, 48 total)
**v2:** Multi-token IoU (last 5 positions/prompt, 240 total) via `--context_tokens 5`

---

## Pipeline

| Step | v1 | v2 |
|---|---|---|
| 02 Baseline | identical | identical |
| 04 Feature extraction | `decision` (48 samples) | `last_5` (240 samples) |
| 06 Attribution graph | identical | identical (uses `is_decision_position`) |
| 07 Ablation | identical | identical |
| 07 Patching C3 | identical | identical |
| 09 UI prep | identical | identical |
| SLURM job | 24386104 | 25058380 |

---

## Baseline Gate

| Metric | v1 | v2 |
|---|---|---|
| EN sign_accuracy | 1.000 | 1.000 |
| FR sign_accuracy | 0.667 | 0.667 |
| mean_norm_logprob_diff | 3.511 | 3.511 |
| Gate | PASS | PASS |

---

## Attribution Graph

| Metric | v1 | v2 |
|---|---|---|
| Feature nodes | 95 | 95 |
| Edges | 285 | 285 |
| Top feature | L25_F103245 | L25_F103245 |

Graph is identical because step 06 uses only the decision token (via `is_decision_position`) regardless of how many tokens step 04 extracted.

---

## IoU per Layer

| Layer | v1 (decision) | v2 (last_5) | Change |
|-------|--------------|-------------|--------|
| 10    | 0.393        | 0.269       | −0.124 |
| 11    | 0.387        | 0.266       | −0.121 |
| 12    | 0.432        | 0.299       | −0.133 |
| 13    | 0.416        | 0.315       | −0.101 |
| 14    | 0.432        | 0.337       | −0.095 |
| 15    | 0.450        | 0.343       | −0.107 |
| 16    | **0.493**    | 0.358       | −0.135 |
| 17    | 0.435        | 0.350       | −0.085 |
| 18    | 0.420        | 0.344       | −0.076 |
| 19    | 0.421        | 0.363       | −0.058 |
| 20    | 0.449        | **0.379**   | −0.070 |
| 21    | 0.439        | 0.324       | −0.115 |
| 22    | 0.447        | 0.330       | −0.117 |
| 23    | 0.436        | 0.306       | −0.130 |
| 24    | 0.430        | 0.276       | −0.154 |
| 25    | 0.360        | 0.248       | −0.112 |

v1 mean: 0.423 | v2 mean: 0.319

The absolute IoU values are lower in v2 because the feature set for each language is now larger (5× more samples → ~5× more unique features) while the union grows faster than the intersection. This is expected. The critical comparison is the **layer-wise profile shape**, not absolute IoU.

---

## IoU Summary (Early / Middle / Late)

| Region | Layers | v1 | v2 (pooled) | Change |
|--------|--------|----|----|--------|
| Early  | 10–11  | 0.390 | 0.267 | −0.123 |
| Middle | 12–20  | 0.439 | 0.343 | −0.096 |
| Late   | 21–25  | 0.422 | 0.297 | −0.125 |
| **Middle / Early ratio** | — | **1.125×** | **1.283×** | +0.158 |
| Middle > Late? | — | yes (0.017) | yes (0.046) | stronger |

**Formula correction (2026-03-15):** The v1 ratio was previously recorded as 1.047×.
That figure used a non-standard formula: middle / mean(early∪late layers), i.e.
0.431 / 0.412 = 1.046×. The correct and consistent formula is middle / early:
0.439 / 0.390 = **1.125×**. The v2 ratio 1.283× was always computed with the
correct formula. The v1 middle mean also corrects from 0.431 to 0.439 (the prior
figure used rounded per-region means rather than per-layer averages).

The v2 pooled profile shows a clearer gradient (early < late < middle) than v1.
Genuine improvement with consistent formula: 1.125× → 1.283× (+0.158×).

---

## Bridge Features

| Metric | v1 | v2 |
|---|---|---|
| Bridge features | 32/53 (60.4%) | 32/53 (60.4%) |
| Top bridge feature | L25_F43384 | L25_F43384 |

Identical — bridge analysis uses intervention CSVs, not IoU.

---

## C3 Patching

| Metric | v1 | v2 |
|---|---|---|
| disruption_rate | 0.588 | 0.588 |
| mean_effect_size | −0.166 ± 0.019 | −0.166 ± 0.019 |
| 95% CI | [−0.202, −0.126] | [−0.202, −0.126] |
| Best layers | L12, L20, L22 | L12, L20, L22 |

Identical — C3 uses the same graph features and intervention CSVs.

---

## Claim 3 Upgrade

| | v1 | v2 (pooled) | v2 content-only | v2 decision-only |
|---|---|---|---|---|
| IoU ratio (middle/early) | 1.125× | **1.283×** | 1.257× | 1.106× |
| Middle > late? | yes (0.017) | yes (0.046) | yes (0.046) | yes (0.011) |
| Assessment | Weak | Borderline moderate | Weak | Flat (structural token) |

The decision-only curve (1.106×) closely reproduces v1 (1.125×) — validating that both
measure the same thing. The pooled curve (1.283×) remains the best available Claim 3 signal.
Content-only (1.257×) does not improve on pooled because the last_5 window includes
structural tokens (" , of/de) that have high EN↔FR IoU at all layers, diluting the
early-layer dip. Only ~1 of 4 non-decision positions is the content word itself.

---

## Phase 1 Position-Separated IoU (2026-03-15)

| IoU curve | Early (10–11) | Middle (12–20) | Late (21–25) | Middle/Early |
|---|---|---|---|---|
| Pooled (all 5 positions) | 0.267 | 0.343 | 0.297 | **1.283×** |
| Decision token only      | 0.390 | 0.431 | 0.421 | 1.106× |
| Content positions (×4)   | 0.261 | 0.328 | 0.282 | 1.257× |

New outputs in `data/analysis/multilingual_circuits/`:
`iou_per_layer_decision.csv`, `iou_per_layer_content.csv`, `iou_position_comparison.png`

---

## Conclusion

v2 improves on v1's Claim 3 ratio from 1.125× to 1.283× (using a consistent formula).
The pooled IoU provides the strongest gradient; position-separation confirms the direction
but does not sharpen it, because prompt length (8 tokens) limits content-word isolation.
Claim 3 is **borderline moderate/weak** — the direction is unambiguous (middle > late > early
in all three curves) but the gradient is shallow.

The graph (star topology, 95 nodes, 285 edges) is unchanged and remains the primary
limitation for mechanistic interpretation. **Phase 2 (virtual-weight DAG) is the next step.**
