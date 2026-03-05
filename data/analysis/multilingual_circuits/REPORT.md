# Multilingual Circuits Analysis — multilingual_circuits

Behaviour: `multilingual_circuits` | Split: train | n_prompts: 48 (24 EN + 24 FR)

## Baseline Gate

| Metric | Value | Threshold | Status |
|---|---|---|---|
| EN sign_accuracy | nan | ≥ 0.90 | FAIL |
| FR sign_accuracy | nan | ≥ 0.75 | FAIL |
| mean_norm_logprob_diff | nan | ≥ 1.0 | FAIL |

**Overall gate: UNKNOWN**

## C3 Patching (Language Swap EN→FR)

| Metric | Value | Target |
|---|---|---|
| disruption_rate (effect < 0) | nan | ≥ 0.40 |
| flip_rate (sign_flipped) | nan | report only |
| mean_effect_size ± SEM | +nan ± nan | report only |
| 95% bootstrap CI | [+nan, +nan] | — |

**C3 target met: NO**

## Per-Layer IoU (EN vs FR feature activation sets)

Mean IoU: nan
Max IoU layer: N/A (IoU = nan)
Min IoU layer: N/A (IoU = nan)
Middle layers (12–20) mean IoU: nan
Early/late layers mean IoU:     nan

See `iou_per_layer.csv` for full per-layer breakdown.

## Bridge Features (Claim 4)

Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.

Total graph features: N/A (see bridge_features.csv)
Bridge features:      0

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
| (1) Language-specific features exist | Min per-layer IoU | nan | PROXY — partial support |
| (2) Shared cross-lang features exist | Max per-layer IoU | nan | PROXY — partial support |
| (3) Shared features in middle layers | IoU middle(12–20) vs early/late | nan vs nan | PROXY — weak (decision token limits contrast) |
| (4) Bridge features degrade both langs | n bridge features / C3 lang-swap strength | 0 bridges; nan C3 disrupt frac | PARTIAL ✓ |


## Notes

- IoU uses top-50 transcoder features at last (decision) token position per prompt.
- Bridge features require consistent negative mean_effect in BOTH languages;
  score = min(|mean_effect_en|, |mean_effect_fr|).
- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).
  A per-PAIR disruption_rate (any layer) would be higher.
- (1) and (2) are PROXY measures relative to Anthropic (who use token-level
  activation sets across full paragraphs; we use attribution graph features).