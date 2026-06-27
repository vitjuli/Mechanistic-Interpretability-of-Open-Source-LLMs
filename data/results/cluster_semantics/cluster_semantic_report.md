# Cluster Semantic Report — `physics_decay_type_probe`

**Method:** co-importance Louvain (11 clusters, composite rank #1)
**Date:** 2026-05-01

All interpretations are **tentative**. Labels reflect the strongest data-driven evidence 
but are not causal proofs. Use 'candidate semantic direction' language in thesis.

---

## Cluster 0 — Early β-Routing Module

**Tentative semantic label:** early-layer routing

| Property | Value |
|----------|-------|
| Members (n) | 5 |
| Feature IDs | L10_F116205, L10_F41460, L10_F56424, L10_F72085, L10_F9316 |
| Layers | 10 (span 0.0) |
| Dominant role | — (0/5) |
| Role distribution | {} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | early |
| Mean abs cosine (within) | 0.997 |
| Mean co-importance (within) | 0.9273 |
| Mean feature-to-centroid cosine | 0.9988 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0141 | 4.172 | 4.172 | 1.00 | AUX | ? | electron |
| p0434 | 1.762 | 1.762 | 1.00 | AUX | ? | neutron |
| p0379 | 1.406 | 1.406 | 1.00 | AUX | ? | proton |
| p0081 | 1.300 | 1.300 | 1.00 | AUX | ? | neutron |
| p0367 | 1.062 | 1.062 | 1.00 | AUX | ? | neutron |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0292 | -6.800 | 6.800 | 1.00 | AUX | ? | photon |
| p0145 | -6.519 | 6.519 | 1.00 | AUX | ? | photon |
| p0288 | -6.312 | 6.312 | 1.00 | AUX | ? | photon |
| p0418 | -5.812 | 5.812 | 1.00 | AUX | ? | photon |
| p0075 | -5.219 | 5.219 | 1.00 | AUX | ? | photon |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -0.988 | 1.139 | 0.00 | 0.02 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | photon | 17 | 6.2 | 2.72× | 0.000 |

### Evidence summary

- high co-importance coherence (0.927) — features share decisive prompts
- strong abs-cosine coherence (0.997)
- significant enrichment for correct_answer=photon (lift=2.72, p=0.000)

### Caveats

- Cluster size n=5.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 1 — Early α-Attribution Pair (L11)

**Tentative semantic label:** cross-layer convergence

| Property | Value |
|----------|-------|
| Members (n) | 8 |
| Feature IDs | L11_F134274, L11_F9723, L22_F118125, L22_F139040, L22_F27223, L22_F39641, L22_F41906, L22_F63642 |
| Layers | 11, 22 (span 11.0) |
| Dominant role | α-attr (1/8) |
| Role distribution | {'α-attr': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | multi-layer |
| Mean abs cosine (within) | 0.9022 |
| Mean co-importance (within) | 0.5472 |
| Mean feature-to-centroid cosine | 0.9562 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0023 | 1.492 | 1.492 | 1.00 | AUX | ? | proton |
| p0172 | 0.531 | 0.531 | 1.00 | AUX | ? | neutron |
| p0038 | 0.461 | 0.461 | 1.00 | AUX | ? | neutron |
| p0186 | 0.445 | 0.688 | 0.25 | AUX | ? | photon |
| p0359 | 0.289 | 0.336 | 0.75 | AUX | ? | proton |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0413 | -5.195 | 5.195 | 1.00 | AUX | ? | photon |
| p0050 | -4.656 | 5.203 | 0.75 | AUX | ? | neutron |
| p0081 | -3.961 | 4.570 | 0.75 | AUX | ? | neutron |
| p0108 | -3.777 | 3.777 | 1.00 | AUX | ? | neutron |
| p0305 | -3.551 | 3.551 | 1.00 | AUX | ? | photon |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -1.580 | 1.745 | 0.00 | 0.01 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | neutron | 15 | 5.5 | 2.71× | 0.000 |

### Evidence summary

- high co-importance coherence (0.547) — features share decisive prompts
- strong abs-cosine coherence (0.902)
- significant enrichment for correct_answer=neutron (lift=2.71, p=0.000)

### Caveats

- Cluster size n=8.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 2 — Singleton L12

**Tentative semantic label:** output-stage decision

| Property | Value |
|----------|-------|
| Members (n) | 5 |
| Feature IDs | L24_F104198, L24_F143388, L24_F153576, L24_F163125, L24_F38741 |
| Layers | 24 (span 0.0) |
| Dominant role | α-attr (1/5) |
| Role distribution | {'α-attr': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | late |
| Mean abs cosine (within) | 0.9965 |
| Mean co-importance (within) | 0.9273 |
| Mean feature-to-centroid cosine | 0.9986 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0379 | 1.200 | 1.200 | 1.00 | AUX | ? | proton |
| p0038 | 0.900 | 0.900 | 1.00 | AUX | ? | neutron |
| p0422 | 0.825 | 0.825 | 1.00 | AUX | ? | neutron |
| p0023 | 0.816 | 0.816 | 1.00 | AUX | ? | proton |
| p0177 | 0.394 | 0.394 | 1.00 | AUX | ? | photon |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0050 | -7.850 | 7.850 | 1.00 | AUX | ? | neutron |
| p0305 | -6.875 | 6.875 | 1.00 | AUX | ? | photon |
| p0419 | -5.394 | 5.394 | 1.00 | AUX | ? | photon |
| p0081 | -5.263 | 5.263 | 1.00 | AUX | ? | neutron |
| p0093 | -5.112 | 5.112 | 1.00 | AUX | ? | neutron |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -1.562 | 1.593 | 0.00 | 0.02 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | neutron | 13 | 5.5 | 2.35× | 0.000 |

### Evidence summary

- high co-importance coherence (0.927) — features share decisive prompts
- strong abs-cosine coherence (0.997)
- significant enrichment for correct_answer=neutron (lift=2.35, p=0.000)

### Caveats

- Cluster size n=5.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 3 — Early α-Attribution Pair (L13)

**Tentative semantic label:** cross-layer convergence

| Property | Value |
|----------|-------|
| Members (n) | 9 |
| Feature IDs | L13_F132297, L13_F13278, L23_F118675, L23_F1264, L23_F157706, L23_F29730, L23_F71325, L25_F149731, L25_F15948 |
| Layers | 13, 23, 25 (span 12.0) |
| Dominant role | α-attr (2/9) |
| Role distribution | {'α-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | multi-layer |
| Mean abs cosine (within) | 0.8033 |
| Mean co-importance (within) | 0.3843 |
| Mean feature-to-centroid cosine | 0.9084 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0141 | 0.927 | 1.233 | 0.78 | AUX | ? | electron |
| p0292 | 0.750 | 1.306 | 0.56 | AUX | ? | photon |
| p0379 | 0.705 | 0.753 | 0.78 | AUX | ? | proton |
| p0359 | 0.611 | 0.903 | 0.78 | AUX | ? | proton |
| p0443 | 0.611 | 0.792 | 0.78 | AUX | ? | photon |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0050 | -3.872 | 3.872 | 1.00 | AUX | ? | neutron |
| p0148 | -3.660 | 3.660 | 1.00 | AUX | ? | photon |
| p0413 | -2.993 | 2.993 | 1.00 | AUX | ? | photon |
| p0166 | -2.943 | 3.172 | 0.78 | AUX | ? | neutron |
| p0301 | -2.802 | 2.802 | 1.00 | AUX | ? | neutron |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -0.885 | 1.046 | 0.00 | 0.02 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | photon | 11 | 6.2 | 1.76× | 0.021 |
| correct_answer | neutron | 9 | 5.5 | 1.62× | 0.070 |

### Evidence summary

- high co-importance coherence (0.384) — features share decisive prompts
- significant enrichment for correct_answer=photon (lift=1.76, p=0.021)

### Caveats

- Cluster size n=9.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 4 — Mid-Layer α-Pair (L20)

**Tentative semantic label:** cross-layer convergence

| Property | Value |
|----------|-------|
| Members (n) | 6 |
| Feature IDs | L14_F18110, L20_F101109, L20_F111138, L20_F111313, L20_F117855, L20_F24337 |
| Layers | 14, 20 (span 6.0) |
| Dominant role | α-attr (2/6) |
| Role distribution | {'α-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | multi-layer |
| Mean abs cosine (within) | 0.8935 |
| Mean co-importance (within) | 0.6773 |
| Mean feature-to-centroid cosine | 0.9546 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0023 | 1.370 | 1.370 | 1.00 | AUX | ? | proton |
| p0061 | 1.240 | 1.240 | 1.00 | AUX | ? | neutron |
| p0359 | 0.922 | 0.974 | 0.83 | AUX | ? | proton |
| p0119 | 0.802 | 0.990 | 0.83 | AUX | ? | neutron |
| p0285 | 0.698 | 0.740 | 0.83 | AUX | ? | neutron |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0148 | -5.073 | 5.094 | 0.83 | AUX | ? | photon |
| p0301 | -4.865 | 4.865 | 1.00 | AUX | ? | neutron |
| p0325 | -4.828 | 4.828 | 1.00 | AUX | ? | photon |
| p0138 | -4.760 | 4.760 | 1.00 | AUX | ? | neutron |
| p0247 | -4.490 | 4.490 | 1.00 | AUX | ? | photon |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -1.192 | 1.327 | 0.00 | 0.02 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | photon | 11 | 6.2 | 1.76× | 0.021 |
| correct_answer | neutron | 7 | 5.5 | 1.26× | 0.304 |

### Evidence summary

- high co-importance coherence (0.677) — features share decisive prompts
- strong abs-cosine coherence (0.893)
- significant enrichment for correct_answer=photon (lift=1.76, p=0.021)

### Caveats

- Cluster size n=6.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 5 — Singleton L15

**Tentative semantic label:** cross-layer convergence

| Property | Value |
|----------|-------|
| Members (n) | 10 |
| Feature IDs | L12_F100195, L12_F159359, L12_F50000, L15_F131368, L15_F142865, L15_F45839, L15_F76750, L16_F35329, L17_F119141, L18_F71234 |
| Layers | 12, 15, 16, 17, 18 (span 6.0) |
| Dominant role | α-attr (3/10) |
| Role distribution | {'α-attr': 3, 'β-attr': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | multi-layer |
| Mean abs cosine (within) | 0.7575 |
| Mean co-importance (within) | 0.2524 |
| Mean feature-to-centroid cosine | 0.8842 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0151 | 3.394 | 3.394 | 1.00 | AUX | ? | photon |
| p0443 | 2.625 | 2.625 | 1.00 | AUX | ? | photon |
| p0328 | 2.169 | 2.284 | 0.70 | AUX | ? | neutron |
| p0417 | 1.962 | 1.962 | 1.00 | AUX | ? | photon |
| p0292 | 1.750 | 1.887 | 0.70 | AUX | ? | photon |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0367 | -1.228 | 1.303 | 0.90 | AUX | ? | neutron |
| p0413 | -0.803 | 1.253 | 0.90 | AUX | ? | photon |
| p0221 | -0.787 | 1.062 | 0.80 | AUX | ? | photon |
| p0434 | -0.700 | 0.812 | 0.90 | AUX | ? | neutron |
| p0083 | -0.662 | 0.688 | 0.90 | AUX | ? | neutron |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | 0.409 | 0.626 | 1.00 | 0.00 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | photon | 14 | 6.2 | 2.24× | 0.000 |

### Evidence summary

- high co-importance coherence (0.252) — features share decisive prompts
- significant enrichment for correct_answer=photon (lift=2.24, p=0.000)

### Caveats

- Cluster size n=10.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 6 — L16 β-Processing Module

**Tentative semantic label:** mid-to-late processing

| Property | Value |
|----------|-------|
| Members (n) | 4 |
| Feature IDs | L19_F39488, L19_F41536, L19_F54110, L19_F59757 |
| Layers | 19 (span 0.0) |
| Dominant role | β-attr (2/4) |
| Role distribution | {'β-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | mid-late |
| Mean abs cosine (within) | 0.9772 |
| Mean co-importance (within) | 0.4866 |
| Mean feature-to-centroid cosine | 0.9914 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0443 | 1.695 | 1.695 | 1.00 | AUX | ? | photon |
| p0407 | 1.531 | 1.531 | 1.00 | AUX | ? | photon |
| p0186 | 1.523 | 1.523 | 1.00 | AUX | ? | photon |
| p0017 | 1.461 | 1.461 | 1.00 | AUX | ? | photon |
| p0340 | 1.328 | 1.328 | 1.00 | AUX | ? | photon |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0246 | -3.742 | 3.742 | 1.00 | AUX | ? | neutron |
| p0121 | -2.938 | 2.938 | 1.00 | AUX | ? | neutron |
| p0142 | -2.719 | 2.719 | 1.00 | AUX | ? | neutron |
| p0406 | -2.473 | 2.473 | 1.00 | AUX | ? | neutron |
| p0061 | -2.188 | 2.188 | 1.00 | AUX | ? | neutron |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -0.429 | 0.680 | 0.00 | 0.00 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | neutron | 11 | 5.5 | 1.98× | 0.008 |

### Evidence summary

- high co-importance coherence (0.487) — features share decisive prompts
- strong abs-cosine coherence (0.977)
- significant enrichment for correct_answer=neutron (lift=1.98, p=0.008)

### Caveats

- Cluster size n=4.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 7 — Multi-Layer Convergence Module

**Tentative semantic label:** output-stage decision

| Property | Value |
|----------|-------|
| Members (n) | 6 |
| Feature IDs | L21_F122057, L21_F144598, L21_F15419, L21_F20252, L21_F27974, L21_F31790 |
| Layers | 21 (span 0.0) |
| Dominant role | α-attr (4/6) |
| Role distribution | {'α-attr': 4} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | mixed (Δ = +0.000) |
| Depth zone | late |
| Mean abs cosine (within) | 0.981 |
| Mean co-importance (within) | 0.6357 |
| Mean feature-to-centroid cosine | 0.9921 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0401 | 1.792 | 1.792 | 1.00 | AUX | ? | neutron |
| p0247 | 1.646 | 1.646 | 1.00 | AUX | ? | photon |
| p0391 | 1.615 | 1.615 | 1.00 | AUX | ? | photon |
| p0359 | 1.599 | 1.599 | 1.00 | AUX | ? | proton |
| p0400 | 1.583 | 1.583 | 1.00 | AUX | ? | photon |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| p0305 | -4.234 | 4.234 | 1.00 | AUX | ? | photon |
| p0413 | -4.000 | 4.000 | 1.00 | AUX | ? | photon |
| p0127 | -3.537 | 3.537 | 1.00 | AUX | ? | photon |
| p0358 | -3.042 | 3.042 | 1.00 | AUX | ? | photon |
| p0366 | -2.958 | 2.958 | 1.00 | AUX | ? | photon |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| ? | -0.513 | 0.820 | 0.00 | 0.01 | AUX | electron |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| correct_answer | photon | 16 | 6.2 | 2.56× | 0.000 |

### Evidence summary

- high co-importance coherence (0.636) — features share decisive prompts
- strong abs-cosine coherence (0.981)
- significant enrichment for correct_answer=photon (lift=2.56, p=0.000)

### Caveats

- Cluster size n=6.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---
