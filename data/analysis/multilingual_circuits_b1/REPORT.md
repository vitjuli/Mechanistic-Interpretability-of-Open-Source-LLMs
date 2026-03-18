# Multilingual Circuits Analysis — multilingual_circuits_b1

Behaviour: `multilingual_circuits_b1` | Split: train | n_prompts: 48 (24 EN + 24 FR)

## Baseline Gate

| Metric | Value | Threshold | Status |
|---|---|---|---|
| EN sign_accuracy | 1.0000 | ≥ 0.90 | PASS |
| FR sign_accuracy | 0.7292 | ≥ 0.65 | PASS |
| mean_norm_logprob_diff | 3.7360 | ≥ 1.0 | PASS |

**Overall gate: PASS**

## C3 Patching (Language Swap EN→FR)

| Metric | Value | Target |
|---|---|---|
| disruption_rate (effect < 0) | 0.6447 | ≥ 0.40 |
| flip_rate (sign_flipped) | 0.0957 | report only |
| mean_effect_size ± SEM | -0.3721 ± 0.0149 | report only |
| 95% bootstrap CI | [-0.4026, -0.3426] | — |

**C3 target met: YES**

## Per-Layer IoU — Position Breakdown

Mean IoU (pooled): 0.3385
Max IoU layer (pooled): 19 (IoU = 0.3895)
Min IoU layer (pooled): 25 (IoU = 0.2766)

| IoU curve | Early (10–11) | Middle (12–20) | Late (21–25) | Middle/Early ratio |
|---|---|---|---|---|
| Pooled (all positions) | 0.2885 | 0.3616 | 0.3170 | 1.253× |
| Decision token only | 0.4131 | 0.4651 | 0.4374 | 1.126× |
| Content positions (non-decision) | 0.2823 | 0.3445 | 0.3060 | 1.220× |

**Note on curves:**
- *Pooled*: all 5 token positions per prompt combined (v2 default).
- *Decision*: final token only (one per prompt). Expected flat layer profile —
  this token is already semantic; EN and FR share the same features here.
- *Content*: non-decision positions (content word + context). Expected steep
  early→middle gradient — early layers process language-specific lexical features;
  middle layers show cross-lingual convergence on shared antonym semantics.

See `iou_per_layer.csv`, `iou_per_layer_decision.csv`, `iou_per_layer_content.csv`.
See `iou_position_comparison.png` for the comparison figure.

## Claim 3 Assessment — Middle-Layer Concentration

**Status: WEAKLY SUPPORTED**

```
Primary signal: content-position middle/early ratio = 1.220× (threshold ≥ 1.10). Direction is correct but gradient is small. Cannot firmly distinguish from noise.
  Pooled ratio:   1.253×
  Decision ratio: 1.126× (expected flat — semantic token, no language contrast)
  Content ratio:  1.220× (expected steep — lexical token, language-specific early layers)
```

**Interpretation note:** Do not conflate Claim 3 support with overall
evidence strength. Claim 3 specifically tests whether shared features are
MORE concentrated in middle layers than early/late. A weak ratio does not
invalidate Claims 1, 2, or 4 — it only means the layer gradient is shallow.

## Bridge Features (Claim 4)

Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.

Total graph features: N/A (see bridge_features.csv)
Bridge features:      33

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
| Token positions | All positions in paragraph | Decision token only (graph); last_5 (IoU v2) | IoU less discriminative |
| Feature type | Sparse Autoencoder (SAE) features | Transcoder features | Different feature geometry |
| Graph topology | Full circuit (feature–feature edges) | Star (input→feature→output only) | Community detection trivial |
| Languages | EN + FR (+ possibly others) | EN + FR only | Narrower reproduction |
| N prompts | ~thousands (pre-trained circuit) | 48 (24 EN + 24 FR) | Smaller sample |


## Claim-Level Summary

| Anthropic Claim | Our evidence | Assessment |
|---|---|---|
| **(1) Language-specific features exist** | Min IoU = 0.2766. Weakly supported — room for language-specific features at early/late layers. | Weakly supported. |
| **(2) Shared cross-lingual features exist** | Max IoU = 0.3895 (layer 19). Moderately supported — three independent measures converge (IoU, bridge, C3). | Moderately supported. |
| **(3) Shared features concentrated in middle layers** | Content-position middle/early = 1.220× (if available) | **WEAKLY SUPPORTED** |
| **(4) Bridge features degrade both EN and FR** | 33 bridges; C3 disrupt=0.645; CI [-0.403, -0.343] | Strongly supported. CI fully negative. |

## Notes

- IoU uses top-50 transcoder features per prompt; multi-token mode uses last_5 positions.
- Bridge features require consistent negative mean_effect in BOTH languages;
  score = min(|mean_effect_en|, |mean_effect_fr|).
- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).
  A per-PAIR disruption_rate (any layer) would be higher.
- Position-separated IoU (Phase 1 of redesign plan) uses `is_decision_position`
  from `position_map.json` to split rows by token role. Content-position IoU is
  the more discriminative Claim 3 signal.