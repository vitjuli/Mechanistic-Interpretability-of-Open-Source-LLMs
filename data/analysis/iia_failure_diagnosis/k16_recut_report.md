# k=16 recut report

**Date:** 2026-05-29  
**Method:** agglo average-linkage on W_coimportance, cut at k=16  
**Source:** runD_v2 (227 features, 538 prompts)  

## Summary

- 5 α-supporting clusters (orient_delta < −0.05)
- 11 β-supporting clusters (orient_delta > +0.05)
- 0 mixed clusters (|orient_delta| ≤ 0.05)

## Per-cluster polarity (sorted by orient_delta)

| cid | n | layers | σ̃_α | σ̃_β | orient_Δ | polarity | top feature |
|-----|---|--------|-----|------|----------|----------|-------------|
| C7 | 17 | L18 | -0.393 | +0.503 | -0.896 | alpha | L18_F100148 |
| C11 | 4 | L13 | -0.300 | +0.309 | -0.609 | alpha | L13_F58969 |
| C13 | 8 | L10 | -0.197 | +0.286 | -0.483 | alpha | L10_F97799 |
| C15 | 20 | L19 | -0.190 | +0.220 | -0.410 | alpha | L19_F37376 |
| C3 | 18 | L15 | -0.084 | +0.038 | -0.122 | alpha | L15_F64857 |
| C12 | 9 | L12 | +0.045 | -0.014 | +0.059 | beta | L12_F60 |
| C9 | 17 | L17 | -0.040 | -0.175 | +0.135 | beta | L17_F124365 |
| C14 | 8 | L11 | +0.073 | -0.087 | +0.161 | beta | L11_F144890 |
| C5 | 19 | L20 | +0.056 | -0.189 | +0.244 | beta | L20_F117855 |
| C10 | 8 | L14 | +0.043 | -0.227 | +0.270 | beta | L14_F89354 |
| C0 | 14 | L22 | +0.306 | -0.102 | +0.408 | beta | L22_F136259 |
| C8 | 13 | L16 | +0.244 | -0.172 | +0.417 | beta | L16_F78897 |
| C6 | 12 | L25 | +0.142 | -0.303 | +0.445 | beta | L25_F15948 |
| C2 | 20 | L21 | +0.428 | -0.280 | +0.709 | beta | L21_F22609 |
| C1 | 20 | L23 | +0.688 | -0.275 | +0.962 | beta | L23_F83556 |
| C4 | 20 | L24 | +0.518 | -0.815 | +1.334 | beta | L24_F51976 |

## Notes
- L18 cluster (strongest α): preserved (single cluster, all 17 features)
- L24 cluster (strongest β): preserved (single cluster, all 20 features)
- L14 and L17 now separate clusters (resolved k=14 problematic merger)
- L22 and L23 now separate clusters (resolved k=14 problematic merger)

## Files updated
- `data/analysis/iia_failure_diagnosis/cluster_semantics_v2.json`
- `data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json`
- `data/analysis/runD_v2/clustering_full/cluster_labels_k16.csv`

## H2 priority pairs (recommend for sbatch)
- **Strongest α+β pair**: C7 (L18, orient_Δ=-0.896) + C4 (L24, orient_Δ=+1.334)
