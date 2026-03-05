# Multilingual Circuits Analysis — multilingual_circuits

Behaviour: `multilingual_circuits` | Split: train | n_prompts: 48 (24 EN + 24 FR)

## Baseline Gate

| Metric | Value | Threshold | Status |
|---|---|---|---|
| EN sign_accuracy | 1.0000 | ≥ 0.90 | PASS |
| FR sign_accuracy | 0.7500 | ≥ 0.75 | PASS |
| mean_norm_logprob_diff | 3.5834 | ≥ 1.0 | PASS |

**Overall gate: PASS**

## C3 Patching (Language Swap EN→FR)

| Metric | Value | Target |
|---|---|---|
| disruption_rate (effect < 0) | 0.5286 | ≥ 0.40 |
| flip_rate (sign_flipped) | 0.0677 | report only |
| mean_effect_size ± SEM | -0.0958 ± 0.0387 | report only |
| 95% bootstrap CI | [-0.1766, -0.0189] | — |

**C3 target met: YES**

## Per-Layer IoU (EN vs FR feature activation sets)

Mean IoU: 0.4133
Max IoU layer: 16.0 (IoU = 0.4797)
Min IoU layer: 15.0 (IoU = 0.3407)
Middle layers (12–20) mean IoU: 0.4192
Early/late layers mean IoU:     0.4057

See `iou_per_layer.csv` for full per-layer breakdown.

## Bridge Features (Claim 4)

Bridge = feature where mean ablation effect < 0 in BOTH EN and FR.

Total graph features: N/A (see bridge_features.csv)
Bridge features:      9

See `bridge_features_only.csv` for details.


## Anthropic → Ours Mapping

| Anthropic Claim | Metric | Our Value | Status |
|---|---|---|---|
| (1) Language-specific features | Min IoU across layers | 0.3407 | PROXY |
| (2) Shared cross-lang features | Max IoU across layers | 0.4797 | PROXY |
| (3) Shared features in middle layers | IoU middle(12–20) vs early/late | 0.4192 vs 0.4057 | PROXY |
| (4) Bridge features degrade both languages | n_bridges with consistent negative effect | 9 | PARTIAL |


## Notes

- IoU uses top-50 transcoder features at last (decision) token position per prompt.
- Bridge features require consistent negative mean_effect in BOTH languages;
  score = min(|mean_effect_en|, |mean_effect_fr|).
- C3 disruption_rate is per-row (each row = one feature × one pair × one layer).
  A per-PAIR disruption_rate (any layer) would be higher.
- (1) and (2) are PROXY measures relative to Anthropic (who use token-level
  activation sets across full paragraphs; we use attribution graph features).