# Internal Candidate-State Analysis Report
## physics_internal_candidate_selection_v2 | train split | 447 prompts

## 1. Summary of findings

- Total (feature × particle) tests: 308
- Features with **T > C > B** ordering: 9 / 308 (2.9%)
- **Strong candidate features** (T>C>B + both MW tests significant, FDR α=0.05): 1
- **Partial candidate features** (T>>B only, C≈B): 3

### Evidence classification: **WEAK / NEGATIVE** — no clear internal candidate representation found

## 2. Per-particle summary

### electron
- Strong features: 0  |  Partial: 0
- Mean candidate_specificity: nan
- Mean competitor_presence:   nan
- Mean IPR (competitor/target): 1.0506

### neutron
- Strong features: 1  |  Partial: 3
- Mean candidate_specificity: 0.0390
- Mean competitor_presence:   0.1220
- Mean IPR (competitor/target): 1.0591

### photon
- Strong features: 0  |  Partial: 0
- Mean candidate_specificity: -0.1935
- Mean competitor_presence:   nan
- Mean IPR (competitor/target): nan

### proton
- Strong features: 0  |  Partial: 0
- Mean candidate_specificity: nan
- Mean competitor_presence:   nan
- Mean IPR (competitor/target): 1.0252

## 3. Per-layer summary (candidate feature emergence)

| Layer | n_features | n_strong | n_partial | mean_spec | mean_comp_presence | mean_IPR |
|---|---|---|---|---|---|---|
| L10 | 5 | 0 | 0 | -0.0665 | 0.0643 | 0.9409 |
| L11 | 5 | 0 | 0 | -0.0677 | -0.0006 | 0.9827 |
| L12 | 5 | 0 | 0 | -0.0005 | 0.0828 | 0.9854 |
| L13 | 5 | 0 | 0 | -0.0393 | 0.0046 | 0.9990 |
| L14 | 5 | 0 | 0 | -0.0180 | 0.1305 | 1.0689 |
| L15 | 5 | 0 | 0 | -0.0338 | 0.0850 | 0.9717 |
| L16 | 2 | 0 | 0 | 0.0399 | -0.0267 | 1.0759 |
| L17 | 5 | 0 | 1 | -0.0635 | 0.0349 | 0.9889 |
| L18 | 5 | 0 | 0 | -0.1794 | 0.0858 | 0.9447 |
| L19 | 5 | 0 | 1 | -0.1569 | 0.0480 | 0.9550 |
| L20 | 5 | 0 | 0 | -0.1784 | -0.1317 | 0.9714 |
| L21 | 5 | 0 | 1 | -0.1673 | -0.0786 | 0.9729 |
| L22 | 5 | 0 | 0 | -0.0228 | 0.0756 | 1.0000 |
| L23 | 5 | 0 | 0 | -0.0322 | 0.2237 | 1.0748 |
| L24 | 5 | 1 | 0 | -0.1488 | 0.8564 | 0.9465 |
| L25 | 5 | 0 | 0 | -0.0298 | 0.4091 | 1.8593 |

## 4. Top strong candidate features (T > C > B, FDR-significant)

| Feature | Particle | Layer | target_μ | competitor_μ | background_μ | IPR | cohend_TC | cohend_CB |
|---|---|---|---|---|---|---|---|---|
| L24_F84940 | neutron | 24 | 10.274 | 8.002 | 5.275 | 0.779 | 0.623 | 0.779 |

## 5. Key scientific questions

### Q1: Which layers first show candidate presence?
- Early (L10-L13): best layer L10 with 0 strong features, mean_IPR=0.941
- Mid (L14-L18): best layer L14 with 0 strong features, mean_IPR=1.069
- Late (L22-L25): best layer L24 with 1 strong features, mean_IPR=0.947

### Q2: Are L19-L21 candidate retrieval layers?
- L19: 0 strong features, mean_IPR=0.955, mean_comp_presence=0.0480
- L20: 0 strong features, mean_IPR=0.971, mean_comp_presence=-0.1317
- L21: 0 strong features, mean_IPR=0.973, mean_comp_presence=-0.0786

### Q3: Are L24/L25 output-selection (partial > strong)?
- L24: strong=1, partial=0, mean_IPR=0.947
- L25: strong=0, partial=0, mean_IPR=1.859
