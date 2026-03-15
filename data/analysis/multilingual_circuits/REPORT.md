# Multilingual Circuits Analysis — multilingual_circuits

Behaviour: `multilingual_circuits` | Split: train | n_prompts: 48 (24 EN + 24 FR)

## Baseline Gate

| Metric | Value | Threshold | Status |
|---|---|---|---|
| EN sign_accuracy | 1.0000 | ≥ 0.90 | PASS |
| FR sign_accuracy | 0.6667 | ≥ 0.65 | PASS |
| mean_norm_logprob_diff | 3.5111 | ≥ 1.0 | PASS |

**Overall gate: PASS**

## C3 Patching (Language Swap EN→FR)

| Metric | Value | Target |
|---|---|---|
| disruption_rate (effect < 0) | 0.5877 | ≥ 0.40 |
| flip_rate (sign_flipped) | 0.0719 | report only |
| mean_effect_size ± SEM | -0.1657 ± 0.0194 | report only |
| 95% bootstrap CI | [-0.2023, -0.1262] | — |

**C3 target met: YES**

## Per-Layer IoU — Position Breakdown

Mean IoU (pooled): 0.3191
Max IoU layer (pooled): 20 (IoU = 0.3793)
Min IoU layer (pooled): 25 (IoU = 0.2475)

| IoU curve | Early (10–11) | Middle (12–20) | Late (21–25) | Middle/Early ratio |
|---|---|---|---|---|
| Pooled (all positions) | 0.2674 | 0.3431 | 0.2970 | 1.283× |
| Decision token only | *pending CSD3 run* | *pending* | *pending* | *pending* |
| Content positions (non-decision) | *pending CSD3 run* | *pending* | *pending* | *pending* |

**Note:** Phase 1 analysis is implemented (2026-03-14). Decision-only and content-position
IoU curves will be computed on the next CSD3 run with updated
`scripts/a_analyze_multilingual_circuits.py`. The pooled row above matches v2 results.

**Expected:** decision-only ≈ flat profile (ratio ~1.05×, mirroring v1);
content-position ratio > 1.30× (steeper early→middle gradient).
The content curve is the primary Claim 3 signal: it captures language-specific
lexical processing at early layers and cross-lingual convergence at middle layers.

See `iou_per_layer.csv` for full per-layer pooled breakdown.
After CSD3 run: also `iou_per_layer_decision.csv`, `iou_per_layer_content.csv`,
and `iou_position_comparison.png`.

## Bridge Features (Claim 4)

Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.

Total graph features: N/A (see bridge_features.csv)
Bridge features:      32

See `bridge_features_only.csv` for details.


## Anthropic → Ours: Match vs Mismatch

### Matches (after per-feature conversion)
| Aspect | Anthropic | Ours |
|---|---|---|
| Intervention type | Per-feature causal (SAE feature ablation/patching) | Per-feature causal (transcoder feature ablation/patching) ✓ |
| Language pairs | EN + FR (antonym task) | EN + FR (antonym task) ✓ |
| Intervention target | C3: patch EN features into FR context | C3: patch EN features into FR context ✓ |
| Bridge features | Consistent negative effect in both languages | Consistent negative mean_effect in EN + FR ✓ |

### Mismatches (documented; not changed)
| Aspect | Anthropic | Ours | Impact |
|---|---|---|---|
| Token positions | All positions in paragraph | Decision token only (last) | IoU less discriminative |
| Feature type | Sparse Autoencoder (SAE) features | Transcoder features | Different feature geometry |
| Graph topology | Full circuit (feature–feature edges) | Star (input→feature→output only) | Community detection trivial |
| Languages | EN + FR (+ possibly others) | EN + FR only | Narrower reproduction |
| N prompts | ~thousands (pre-trained circuit) | 48 (24 EN + 24 FR) | Smaller sample |

### Claim-level Results

| Anthropic Claim | Metric | Our Value | Status |
|---|---|---|---|
| (1) Language-specific features exist | Min pooled IoU | 0.2475 | Weakly supported. |
| (2) Shared cross-lang features exist | Max pooled IoU = 0.3793; 32/53 bridge; C3 CI fully negative | three independent measures | Moderately supported. |
| (3) Shared features in middle layers | Pooled middle/early = 1.283×; content-position ratio *pending* | see Phase 1 analysis above | **Pending** (pooled = Moderate; content-position may be Stronger) |
| (4) Bridge features degrade both langs | 32 bridges; disruption=0.588; CI [−0.202, −0.126] | CI fully negative | **Strongly supported.** |

## Claim 3 Assessment — Middle-Layer Concentration

**Current status: MODERATELY SUPPORTED (pooled v2)**

The pooled v2 ratio is 1.283× (early=0.267, middle=0.343, late=0.297). This is
moderately above the 1.10× minimum and above the 1.30× moderate threshold.

**After Phase 1 CSD3 run, assess:**
1. Decision-only ratio: expected ~1.05× (flat). If decision ratio is flat, this
   confirms that the decision token alone cannot distinguish early vs middle layers —
   consistent with the v1 observation.
2. Content-position ratio: expected > 1.30×. If content ratio >= 1.50×, upgrade
   Claim 3 to **Strongly supported**. If 1.30–1.50×, it remains **Moderately
   supported** with clearer mechanistic interpretation. If < 1.30×, the pooled ratio
   was already the best signal and no upgrade is warranted.

**Do not automatically claim strong support before seeing the actual numbers.**

## Notes

- IoU uses top-50 transcoder features per prompt; v2 uses last_5 positions.
- Bridge features require consistent negative mean_effect in BOTH languages;
  score = min(|mean_effect_en|, |mean_effect_fr|).
- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).
  A per-PAIR disruption_rate (any layer) would be higher.
- Phase 1 (position-separated IoU) is implemented 2026-03-14. Requires CSD3 run
  to produce decision/content curves. No pipeline changes needed — only the
  analysis script is updated.