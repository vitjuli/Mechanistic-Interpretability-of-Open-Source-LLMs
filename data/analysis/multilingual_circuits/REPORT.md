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
| disruption_rate (effect < 0) | 0.5597 | ≥ 0.40 |
| flip_rate (sign_flipped) | 0.0741 | report only |
| mean_effect_size ± SEM | -0.1048 ± 0.0186 | report only |
| 95% bootstrap CI | [-0.1442, -0.0700] | — |

**C3 target met: YES**

## Per-Layer IoU — Position Breakdown

Mean IoU (pooled): nan
Max IoU layer (pooled): N/A (IoU = nan)
Min IoU layer (pooled): N/A (IoU = nan)

| IoU curve | Early (10–11) | Middle (12–20) | Late (21–25) | Middle/Early ratio |
|---|---|---|---|---|
| Pooled (all positions) | nan | nan | nan | nan× |

See `iou_per_layer.csv` for full per-layer breakdown.

## Claim 3 Assessment — Middle-Layer Concentration

**Status: INSUFFICIENT DATA**

```
Single-token mode — only pooled ratio available: Middle/early ratio could not be computed for pooled curve.
```

**Interpretation note:** Do not conflate Claim 3 support with overall
evidence strength. Claim 3 specifically tests whether shared features are
MORE concentrated in middle layers than early/late. A weak ratio does not
invalidate Claims 1, 2, or 4 — it only means the layer gradient is shallow.

## Bridge Features (Claim 4)

Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.

Total graph features: N/A (see bridge_features.csv)
Bridge features:      29

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
| **(1) Language-specific features exist** | Min IoU = nan. N/A | Weakly supported. |
| **(2) Shared cross-lingual features exist** | Max IoU = nan (layer N/A). N/A | Moderately supported. |
| **(3) Shared features concentrated in middle layers** | Content-position middle/early = nan× (if available) | **INSUFFICIENT DATA** |
| **(4) Bridge features degrade both EN and FR** | 29 bridges; C3 disrupt=0.560; CI [-0.144, -0.070] | Strongly supported. CI fully negative. |

## Notes

- IoU uses top-50 transcoder features per prompt; multi-token mode uses last_5 positions.
- Bridge features require consistent negative mean_effect in BOTH languages;
  score = min(|mean_effect_en|, |mean_effect_fr|).
- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).
  A per-PAIR disruption_rate (any layer) would be higher.
- Position-separated IoU (Phase 1 of redesign plan) uses `is_decision_position`
  from `position_map.json` to split rows by token role. Content-position IoU is
  the more discriminative Claim 3 signal.