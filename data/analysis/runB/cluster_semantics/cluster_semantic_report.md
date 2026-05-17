# Cluster Semantic Report — `physics_decay_type_probe`

**Method:** co-importance Louvain (11 clusters, composite rank #1)
**Date:** 2026-05-01

All interpretations are **tentative**. Labels reflect the strongest data-driven evidence 
but are not causal proofs. Use 'candidate semantic direction' language in thesis.

---

## Cluster 0 — Early β-Routing Module

**Tentative semantic label:** early-layer routing · β-oriented · global β-discriminator

| Property | Value |
|----------|-------|
| Members (n) | 5 |
| Feature IDs | L10_F128064, L10_F35580, L10_F68680, L10_F80002, L10_F83063 |
| Layers | 10 (span 0.0) |
| Dominant role | β-discrim (4/5) |
| Role distribution | {'β-discrim': 4, 'β-attr': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 4 |
| Orientation | beta (Δ = -0.452) |
| Depth zone | early |
| Mean abs cosine (within) | 0.9759 |
| Mean co-importance (within) | 0.6802 |
| Mean feature-to-centroid cosine | 0.9903 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-P5-v01 | 0.938 | 0.938 | 1.00 | AUX | AUX-P5 | alpha |
| L2-BR1-v02 | 0.787 | 0.787 | 1.00 | 2 | L2-BR1 | beta |
| L1-B2-v08 | 0.762 | 0.762 | 1.00 | 1 | L1-B2 | beta |
| L2-BR4-v04 | 0.750 | 0.750 | 1.00 | 2 | L2-BR4 | beta |
| L2-BR4-v03 | 0.750 | 0.750 | 1.00 | 2 | L2-BR4 | beta |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-06-v01 | -0.675 | 0.675 | 1.00 | 2 | CP-L2-06 | alpha |
| L3-A1-v06 | -0.669 | 0.669 | 1.00 | 3 | L3-A1 | alpha |
| L2-AR7-v05 | -0.662 | 0.662 | 1.00 | 2 | L2-AR7 | alpha |
| AUX-A10-v01 | -0.613 | 0.613 | 1.00 | AUX | AUX-A10 | alpha |
| L3-A2-v04 | -0.575 | 0.575 | 1.00 | 3 | L3-A2 | alpha |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-P5 | 0.938 | 0.938 | 1.00 | 0.00 | AUX | alpha |
| AUX-B9 | 0.738 | 0.738 | 1.00 | 0.80 | AUX | beta |
| AUX-B3 | 0.588 | 0.588 | 1.00 | 0.00 | AUX | beta |
| L2-BR5 | 0.546 | 0.546 | 1.00 | 0.40 | 2 | beta |
| KW-B1 | 0.475 | 0.475 | 1.00 | 0.00 | 1 | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | charge_plus_a_unchanged | 3 | 0.3 | 11.75× | 0.001 |
| relation_type | charge_plus_a_unchanged | 3 | 0.3 | 11.75× | 0.001 |
| cue_label | charge_plus_z_change | 2 | 0.3 | 5.88× | 0.041 |
| relation_type | charge_plus_z_change | 2 | 0.3 | 5.88× | 0.041 |
| relation_type | z_plus1_a_unchanged | 2 | 0.4 | 4.70× | 0.063 |
| cue_label | z_plus1_a_unchanged | 2 | 0.4 | 4.70× | 0.063 |

### Evidence summary

- high co-importance coherence (0.680) — features share decisive prompts
- strong abs-cosine coherence (0.976)
- clear orientation bias (Δ=-0.452)
- significant enrichment for cue_label=charge_plus_a_unchanged (lift=11.75, p=0.001)
- significant enrichment for relation_type=charge_plus_a_unchanged (lift=11.75, p=0.001)
- significant enrichment for cue_label=charge_plus_z_change (lift=5.88, p=0.041)
- significant enrichment for relation_type=charge_plus_z_change (lift=5.88, p=0.041)

### Caveats

- Cluster size n=5.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 1 — Early α-Attribution Pair (L11)

**Tentative semantic label:** early-layer routing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 4 |
| Feature IDs | L11_F114815, L11_F144890, L11_F151375, L11_F35425 |
| Layers | 11 (span 0.0) |
| Dominant role | β-attr (2/4) |
| Role distribution | {'β-attr': 2, 'α-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.168) |
| Depth zone | early |
| Mean abs cosine (within) | 0.9604 |
| Mean co-importance (within) | 0.7086 |
| Mean feature-to-centroid cosine | 0.9851 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L2-GRAD-A-v01 | 0.734 | 0.734 | 1.00 | 2 | L2-GRAD-A | alpha |
| L3-A3-v04 | 0.703 | 0.703 | 1.00 | 3 | L3-A3 | alpha |
| CP-L2-08-v02 | 0.578 | 0.578 | 1.00 | 2 | CP-L2-08 | beta |
| KW-A2-v02 | 0.500 | 0.500 | 1.00 | 1 | KW-A2 | alpha |
| L3-A1-v06 | 0.430 | 0.430 | 1.00 | 3 | L3-A1 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-01-v02 | -0.578 | 0.578 | 1.00 | 2 | CP-L2-01 | beta |
| L3-B1-v04 | -0.562 | 0.562 | 1.00 | 3 | L3-B1 | beta |
| L3-B1-v01 | -0.531 | 0.531 | 1.00 | 3 | L3-B1 | beta |
| CP-L3-01-v02 | -0.531 | 0.531 | 1.00 | 3 | CP-L3-01 | beta |
| AUX-B7-v01 | -0.469 | 0.469 | 1.00 | AUX | AUX-B7 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-A1 | 0.297 | 0.297 | 1.00 | 0.00 | AUX | alpha |
| CP-L2-08 | 0.289 | 0.289 | 1.00 | 0.00 | 2 | alpha |
| AUX-A5 | 0.281 | 0.281 | 1.00 | 0.00 | AUX | alpha |
| KW-A2 | 0.281 | 0.281 | 1.00 | 0.00 | 1 | alpha |
| AUX-P9 | 0.281 | 0.281 | 1.00 | 0.00 | AUX | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | lepton_family | 4 | 0.3 | 11.75× | 0.000 |
| cue_label | heavy_nuclear_fragment | 1 | 0.3 | 3.92× | 0.231 |
| level | 3 | 11 | 3.4 | 3.23× | 0.000 |
| inference_steps | 3 | 12 | 3.8 | 3.13× | 0.000 |
| cue_label | helium4_equivalence | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | composite_nuclear_object | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.709) — features share decisive prompts
- strong abs-cosine coherence (0.960)
- clear orientation bias (Δ=+0.168)
- significant enrichment for cue_label=lepton_family (lift=11.75, p=0.000)
- significant enrichment for level=3 (lift=3.23, p=0.000)
- significant enrichment for inference_steps=3 (lift=3.13, p=0.000)

### Caveats

- Cluster size n=4.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 2 — Singleton L12

**Tentative semantic label:** early-layer routing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 3 |
| Feature IDs | L12_F2451, L12_F60, L12_F71226 |
| Layers | 12 (span 0.0) |
| Dominant role | β-attr (2/3) |
| Role distribution | {'α-attr': 1, 'β-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.079) |
| Depth zone | early |
| Mean abs cosine (within) | 0.9411 |
| Mean co-importance (within) | 0.5446 |
| Mean feature-to-centroid cosine | 0.9802 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L3-A3-v04 | 0.552 | 0.552 | 1.00 | 3 | L3-A3 | alpha |
| L2-BR6-v02 | 0.438 | 0.438 | 1.00 | 2 | L2-BR6 | beta |
| L2-BR6-v08 | 0.354 | 0.354 | 1.00 | 2 | L2-BR6 | beta |
| L1-A2-v11 | 0.312 | 0.312 | 1.00 | 1 | L1-A2 | alpha |
| AUX-B3-v01 | 0.312 | 0.312 | 1.00 | AUX | AUX-B3 | beta |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-B2-v01 | -0.438 | 0.438 | 1.00 | AUX | AUX-B2 | beta |
| L1-B2-v06 | -0.438 | 0.438 | 1.00 | 1 | L1-B2 | beta |
| L1-B5-v08 | -0.417 | 0.417 | 1.00 | 1 | L1-B5 | beta |
| L1-B6-v02 | -0.396 | 0.396 | 1.00 | 1 | L1-B6 | beta |
| L1-B6-v07 | -0.354 | 0.354 | 1.00 | 1 | L1-B6 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-B3 | 0.312 | 0.312 | 1.00 | 0.00 | AUX | beta |
| AUX-CF3 | 0.271 | 0.271 | 1.00 | 0.00 | AUX | alpha |
| AUX-P4 | 0.229 | 0.229 | 1.00 | 0.00 | AUX | alpha |
| AUX-P3 | 0.208 | 0.208 | 1.00 | 0.00 | AUX | alpha |
| L2-BR6 | 0.208 | 0.208 | 1.00 | 0.12 | 2 | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | z_plus1_with_antineutrino | 2 | 0.3 | 5.88× | 0.041 |
| cue_label | daughter_n_minus1 | 2 | 0.3 | 5.88× | 0.041 |
| relation_type | z_plus1_with_antineutrino | 2 | 0.3 | 5.88× | 0.041 |
| cue_label | heavy_nuclear_fragment | 1 | 0.3 | 3.92× | 0.231 |
| cue_label | emitted_mass4 | 2 | 0.5 | 3.92× | 0.088 |
| cue_label | daughter_z_plus1 | 2 | 0.5 | 3.92× | 0.088 |

### Evidence summary

- high co-importance coherence (0.545) — features share decisive prompts
- strong abs-cosine coherence (0.941)
- significant enrichment for cue_label=z_plus1_with_antineutrino (lift=5.88, p=0.041)
- significant enrichment for cue_label=daughter_n_minus1 (lift=5.88, p=0.041)
- significant enrichment for relation_type=z_plus1_with_antineutrino (lift=5.88, p=0.041)

### Caveats

- Cluster size n=3.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 3 — Early α-Attribution Pair (L13)

**Tentative semantic label:** early-layer routing · β-oriented

| Property | Value |
|----------|-------|
| Members (n) | 3 |
| Feature IDs | L13_F45942, L13_F57499, L13_F58969 |
| Layers | 13 (span 0.0) |
| Dominant role | α-attr (2/3) |
| Role distribution | {'α-attr': 2, 'β-attr': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | beta (Δ = -0.612) |
| Depth zone | early |
| Mean abs cosine (within) | 0.9874 |
| Mean co-importance (within) | 0.3968 |
| Mean feature-to-centroid cosine | 0.9958 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-B3-v01 | 0.729 | 0.729 | 1.00 | AUX | AUX-B3 | beta |
| L2-BR2-v10 | 0.604 | 0.604 | 1.00 | 2 | L2-BR2 | beta |
| CP-L1-05-v02 | 0.583 | 0.583 | 1.00 | 1 | CP-L1-05 | beta |
| L1-B1-v02 | 0.562 | 0.562 | 1.00 | 1 | L1-B1 | beta |
| AUX-B8-v01 | 0.562 | 0.562 | 1.00 | AUX | AUX-B8 | beta |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L1-A6-v06 | -0.583 | 0.583 | 1.00 | 1 | L1-A6 | alpha |
| L3-A2-v08 | -0.583 | 0.583 | 1.00 | 3 | L3-A2 | alpha |
| L1-A3-v12 | -0.562 | 0.562 | 1.00 | 1 | L1-A3 | alpha |
| L2-AR7-v06 | -0.521 | 0.521 | 1.00 | 2 | L2-AR7 | alpha |
| L1-A4-v08 | -0.521 | 0.521 | 1.00 | 1 | L1-A4 | alpha |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-B3 | 0.729 | 0.729 | 1.00 | 0.00 | AUX | beta |
| AUX-P11 | 0.562 | 0.562 | 1.00 | 1.00 | AUX | beta |
| AUX-B8 | 0.562 | 0.562 | 1.00 | 0.00 | AUX | beta |
| AUX-P7 | 0.562 | 0.562 | 1.00 | 1.00 | AUX | beta |
| AUX-B10 | 0.479 | 0.479 | 1.00 | 0.00 | AUX | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | emitted_charge_minus1 | 2 | 0.5 | 3.92× | 0.088 |
| cue_label | not_nuclear_fragment | 1 | 0.3 | 3.92× | 0.231 |
| cue_label | narrative_ejection | 1 | 0.3 | 3.92× | 0.231 |
| relation_type | narrative_ejection | 1 | 0.3 | 3.92× | 0.231 |
| cue_label | composite_nuclear_object | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | charge_plus_z_change | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.397) — features share decisive prompts
- strong abs-cosine coherence (0.987)
- clear orientation bias (Δ=-0.612)

### Caveats

- Cluster size n=3.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 4 — Mid-Layer α-Pair (L20)

**Tentative semantic label:** output-stage decision · α-oriented · includes 2 circuit feature(s)

| Property | Value |
|----------|-------|
| Members (n) | 10 |
| Feature IDs | L22_F110496, L22_F113311, L22_F133148, L22_F28148, L22_F93236, L23_F109767, L23_F140107, L23_F161196, L23_F71067, L23_F83556 |
| Layers | 22, 23 (span 1.0) |
| Dominant role | β-attr (4/10) |
| Role distribution | {'α-circuit': 2, 'β-attr': 4, 'α-attr': 4} |
| Circuit features | 2 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.701) |
| Depth zone | late |
| Mean abs cosine (within) | 0.8687 |
| Mean co-importance (within) | 0.3115 |
| Mean feature-to-centroid cosine | 0.9391 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| KW-A2-v01 | 1.169 | 1.169 | 1.00 | 1 | KW-A2 | alpha |
| CP-L1-04-v01 | 1.169 | 1.169 | 1.00 | 1 | CP-L1-04 | alpha |
| L1-A2-v01 | 1.169 | 1.169 | 1.00 | 1 | L1-A2 | alpha |
| L2-AR1-v07 | 0.991 | 0.991 | 1.00 | 2 | L2-AR1 | alpha |
| L1-A6-v02 | 0.941 | 0.941 | 1.00 | 1 | L1-A6 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| KW-B5-v02 | -1.134 | 1.134 | 1.00 | 1 | KW-B5 | beta |
| KW-B2-v02 | -0.981 | 0.981 | 1.00 | 1 | KW-B2 | beta |
| CP-L2-01-v02 | -0.906 | 0.906 | 1.00 | 2 | CP-L2-01 | beta |
| AUX-B1-v01 | -0.891 | 0.891 | 1.00 | AUX | AUX-B1 | beta |
| L2-BR2-v07 | -0.875 | 0.875 | 1.00 | 2 | L2-BR2 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| KW-A2 | 0.861 | 0.867 | 1.00 | 0.00 | 1 | alpha |
| AUX-ISO5 | 0.800 | 0.800 | 1.00 | 0.00 | AUX | alpha |
| AUX-ISO1 | 0.781 | 0.781 | 1.00 | 0.00 | AUX | alpha |
| AUX-ISO2 | 0.775 | 0.775 | 1.00 | 0.00 | AUX | alpha |
| AUX-P5 | 0.756 | 0.756 | 1.00 | 0.00 | AUX | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | full_emitted_alpha_spec | 3 | 0.3 | 8.81× | 0.003 |
| relation_type | full_emitted_alpha_spec | 3 | 0.3 | 8.81× | 0.003 |
| cue_label | emitted_charge_plus2 | 3 | 0.5 | 5.88× | 0.011 |
| cue_label | emitted_mass4 | 2 | 0.5 | 3.92× | 0.088 |
| cue_label | composition_plus_daughter | 1 | 0.3 | 2.94× | 0.296 |
| relation_type | composition_plus_daughter | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.311) — features share decisive prompts
- strong abs-cosine coherence (0.869)
- clear orientation bias (Δ=+0.701)
- significant enrichment for cue_label=full_emitted_alpha_spec (lift=8.81, p=0.003)
- significant enrichment for relation_type=full_emitted_alpha_spec (lift=8.81, p=0.003)
- significant enrichment for cue_label=emitted_charge_plus2 (lift=5.88, p=0.011)

### Caveats

- Cluster size n=10.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 5 — Singleton L15

**Tentative semantic label:** mid-layer processing · β-oriented

| Property | Value |
|----------|-------|
| Members (n) | 3 |
| Feature IDs | L15_F15507, L15_F64857, L15_F89707 |
| Layers | 15 (span 0.0) |
| Dominant role | β-attr (2/3) |
| Role distribution | {'α-attr': 1, 'β-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | beta (Δ = -0.147) |
| Depth zone | mid |
| Mean abs cosine (within) | 0.9601 |
| Mean co-importance (within) | 0.5079 |
| Mean feature-to-centroid cosine | 0.9866 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-08-v02 | 1.000 | 1.000 | 1.00 | 2 | CP-L2-08 | beta |
| L3-A2-v05 | 0.917 | 0.917 | 1.00 | 3 | L3-A2 | alpha |
| L3-A2-v02 | 0.635 | 0.635 | 1.00 | 3 | L3-A2 | alpha |
| AUX-P7-v01 | 0.479 | 0.479 | 1.00 | AUX | AUX-P7 | beta |
| L1-A6-v07 | 0.438 | 0.438 | 1.00 | 1 | L1-A6 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L2-AR6-v08 | -0.458 | 0.458 | 1.00 | 2 | L2-AR6 | alpha |
| L3-B3-v06 | -0.417 | 0.417 | 1.00 | 3 | L3-B3 | beta |
| L3-A4-v01 | -0.375 | 0.375 | 1.00 | 3 | L3-A4 | alpha |
| CP-L3-04-v01 | -0.375 | 0.375 | 1.00 | 3 | CP-L3-04 | alpha |
| L3-A4-v04 | -0.375 | 0.375 | 1.00 | 3 | L3-A4 | alpha |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| CP-L2-08 | 0.490 | 0.510 | 1.00 | 0.00 | 2 | alpha |
| AUX-P7 | 0.479 | 0.479 | 1.00 | 1.00 | AUX | beta |
| AUX-P5 | 0.375 | 0.375 | 1.00 | 0.00 | AUX | alpha |
| AUX-P12 | 0.354 | 0.354 | 1.00 | 0.00 | AUX | beta |
| AUX-B6 | 0.333 | 0.333 | 1.00 | 0.00 | AUX | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | baryon_number_4 | 2 | 0.3 | 7.83× | 0.023 |
| cue_label | composite_nuclear_object | 2 | 0.3 | 5.88× | 0.041 |
| cue_label | neutron_to_proton | 2 | 0.4 | 4.70× | 0.063 |
| relation_type | neutron_to_proton | 2 | 0.4 | 4.70× | 0.063 |
| inference_steps | 4 | 3 | 0.6 | 4.70× | 0.022 |
| cue_label | heavy_nuclear_fragment | 1 | 0.3 | 3.92× | 0.231 |

### Evidence summary

- high co-importance coherence (0.508) — features share decisive prompts
- strong abs-cosine coherence (0.960)
- clear orientation bias (Δ=-0.147)
- significant enrichment for cue_label=baryon_number_4 (lift=7.83, p=0.023)
- significant enrichment for cue_label=composite_nuclear_object (lift=5.88, p=0.041)
- significant enrichment for inference_steps=4 (lift=4.70, p=0.022)

### Caveats

- Cluster size n=3.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 6 — L16 β-Processing Module

**Tentative semantic label:** mid-to-late processing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 5 |
| Feature IDs | L16_F14097, L16_F33974, L16_F46637, L16_F53285, L16_F82587 |
| Layers | 16 (span 0.0) |
| Dominant role | β-attr (5/5) |
| Role distribution | {'β-attr': 5} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.396) |
| Depth zone | mid-late |
| Mean abs cosine (within) | 0.9834 |
| Mean co-importance (within) | 0.7154 |
| Mean feature-to-centroid cosine | 0.9933 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-04-v01 | 0.769 | 0.769 | 1.00 | 2 | CP-L2-04 | alpha |
| L2-AR1-v08 | 0.762 | 0.762 | 1.00 | 2 | L2-AR1 | alpha |
| L2-AR1-v02 | 0.738 | 0.738 | 1.00 | 2 | L2-AR1 | alpha |
| L3-FA-v06 | 0.713 | 0.713 | 1.00 | 3 | L3-FA | alpha |
| L2-AR1-v10 | 0.713 | 0.713 | 1.00 | 2 | L2-AR1 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-B1-v01 | -1.175 | 1.175 | 1.00 | AUX | AUX-B1 | beta |
| L2-GRAD-B-v01 | -0.775 | 0.775 | 1.00 | 2 | L2-GRAD-B | beta |
| L2-BR2-v06 | -0.719 | 0.719 | 1.00 | 2 | L2-BR2 | beta |
| L2-BR2-v07 | -0.656 | 0.656 | 1.00 | 2 | L2-BR2 | beta |
| L2-BR2-v01 | -0.594 | 0.594 | 1.00 | 2 | L2-BR2 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-A6 | 0.550 | 0.550 | 1.00 | 0.00 | AUX | alpha |
| L2-AR1 | 0.460 | 0.460 | 1.00 | 0.00 | 2 | alpha |
| L3-FA | 0.427 | 0.427 | 1.00 | 0.00 | 3 | alpha |
| AUX-A7 | 0.412 | 0.412 | 1.00 | 0.00 | AUX | alpha |
| L2-AR8 | 0.407 | 0.407 | 1.00 | 0.00 | 2 | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | composition_2p2n | 4 | 0.4 | 9.40× | 0.000 |
| relation_type | composition_2p2n | 4 | 0.4 | 9.40× | 0.000 |
| cue_label | neutron_to_proton | 3 | 0.4 | 7.05× | 0.007 |
| relation_type | neutron_to_proton | 3 | 0.4 | 7.05× | 0.007 |
| cue_label | composition_plus_daughter | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | mass_plus_daughter_a | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.715) — features share decisive prompts
- strong abs-cosine coherence (0.983)
- clear orientation bias (Δ=+0.396)
- significant enrichment for cue_label=composition_2p2n (lift=9.40, p=0.000)
- significant enrichment for relation_type=composition_2p2n (lift=9.40, p=0.000)
- significant enrichment for cue_label=neutron_to_proton (lift=7.05, p=0.007)
- significant enrichment for relation_type=neutron_to_proton (lift=7.05, p=0.007)

### Caveats

- Cluster size n=5.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 7 — Multi-Layer Convergence Module

**Tentative semantic label:** output-stage decision · α-oriented · includes 2 circuit feature(s) · global α-discriminator

| Property | Value |
|----------|-------|
| Members (n) | 10 |
| Feature IDs | L24_F18943, L24_F249, L24_F52031, L24_F60777, L24_F88968, L25_F105937, L25_F110282, L25_F126439, L25_F142075, L25_F71226 |
| Layers | 24, 25 (span 1.0) |
| Dominant role | α-discrim (4/10) |
| Role distribution | {'α-discrim': 4, 'β-circuit': 1, 'β-attr': 2, 'α-attr': 2, 'α-circuit': 1} |
| Circuit features | 2 |
| Global α-discriminators | 4 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.822) |
| Depth zone | late |
| Mean abs cosine (within) | 0.8801 |
| Mean co-importance (within) | 0.2532 |
| Mean feature-to-centroid cosine | 0.9445 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-CF3-v01 | 0.900 | 0.900 | 1.00 | AUX | AUX-CF3 | alpha |
| AUX-P3-v01 | 0.838 | 0.838 | 1.00 | AUX | AUX-P3 | alpha |
| L1-A6-v06 | 0.738 | 0.738 | 1.00 | 1 | L1-A6 | alpha |
| L1-A3-v03 | 0.706 | 0.706 | 1.00 | 1 | L1-A3 | alpha |
| AUX-P2-v01 | 0.700 | 0.700 | 1.00 | AUX | AUX-P2 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-04-v02 | -1.163 | 1.163 | 1.00 | 2 | CP-L2-04 | beta |
| AUX-B4-v01 | -1.044 | 1.044 | 1.00 | AUX | AUX-B4 | beta |
| L1-B6-v06 | -0.944 | 0.944 | 1.00 | 1 | L1-B6 | beta |
| L3-B2-v02 | -0.931 | 0.931 | 1.00 | 3 | L3-B2 | beta |
| KW-B2-v02 | -0.887 | 0.887 | 1.00 | 1 | KW-B2 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-CF3 | 0.900 | 0.900 | 1.00 | 0.00 | AUX | alpha |
| AUX-P3 | 0.838 | 0.838 | 1.00 | 0.00 | AUX | alpha |
| AUX-P2 | 0.700 | 0.700 | 1.00 | 0.00 | AUX | alpha |
| KW-A3 | 0.659 | 0.659 | 1.00 | 0.25 | 1 | alpha |
| AUX-P6 | 0.631 | 0.631 | 1.00 | 0.00 | AUX | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | full_beta_process_spec | 2 | 0.3 | 5.88× | 0.041 |
| relation_type | full_beta_process_spec | 2 | 0.3 | 5.88× | 0.041 |
| relation_type | z_plus1_a_unchanged | 2 | 0.4 | 4.70× | 0.063 |
| cue_label | z_plus1_a_unchanged | 2 | 0.4 | 4.70× | 0.063 |
| cue_label | z_plus1_with_antineutrino | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | daughter_n_minus1 | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.253) — features share decisive prompts
- strong abs-cosine coherence (0.880)
- clear orientation bias (Δ=+0.822)
- significant enrichment for cue_label=full_beta_process_spec (lift=5.88, p=0.041)
- significant enrichment for relation_type=full_beta_process_spec (lift=5.88, p=0.041)

### Caveats

- Cluster size n=10.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 8 — Singleton L18 β-Discriminator

**Tentative semantic label:** mid-to-late processing · β-oriented · global β-discriminator

| Property | Value |
|----------|-------|
| Members (n) | 4 |
| Feature IDs | L18_F108180, L18_F145795, L18_F152260, L18_F41804 |
| Layers | 18 (span 0.0) |
| Dominant role | β-attr (3/4) |
| Role distribution | {'β-attr': 3, 'β-discrim': 1} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 1 |
| Orientation | beta (Δ = -0.890) |
| Depth zone | mid-late |
| Mean abs cosine (within) | 0.9926 |
| Mean co-importance (within) | 0.6744 |
| Mean feature-to-centroid cosine | 0.9972 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L1-B1-v03 | 1.203 | 1.203 | 1.00 | 1 | L1-B1 | beta |
| L2-BR2-v06 | 1.109 | 1.109 | 1.00 | 2 | L2-BR2 | beta |
| CP-L2-08-v02 | 1.062 | 1.062 | 1.00 | 2 | CP-L2-08 | beta |
| AUX-P12-v01 | 1.062 | 1.062 | 1.00 | AUX | AUX-P12 | beta |
| L2-BR3-v03 | 1.031 | 1.031 | 1.00 | 2 | L2-BR3 | beta |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L1-A1-v11 | -0.906 | 0.906 | 1.00 | 1 | L1-A1 | alpha |
| L2-AR7-v05 | -0.906 | 0.906 | 1.00 | 2 | L2-AR7 | alpha |
| L1-A3-v10 | -0.828 | 0.828 | 1.00 | 1 | L1-A3 | alpha |
| AUX-P1-v01 | -0.797 | 0.797 | 1.00 | AUX | AUX-P1 | alpha |
| CP-L2-09-v01 | -0.781 | 0.781 | 1.00 | 2 | CP-L2-09 | alpha |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-P12 | 1.062 | 1.062 | 1.00 | 1.00 | AUX | beta |
| L1-B1 | 0.806 | 0.806 | 1.00 | 0.58 | 1 | beta |
| L2-BR2 | 0.793 | 0.793 | 1.00 | 0.00 | 2 | beta |
| KW-B1 | 0.773 | 0.773 | 1.00 | 0.00 | 1 | beta |
| KW-B5 | 0.750 | 0.750 | 1.00 | 0.00 | 1 | beta |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | emitted_charge_minus1 | 5 | 0.5 | 9.79× | 0.000 |
| cue_label | neutron_to_proton | 3 | 0.4 | 7.05× | 0.007 |
| relation_type | neutron_to_proton | 3 | 0.4 | 7.05× | 0.007 |
| cue_label | narrative_ejection | 1 | 0.3 | 3.92× | 0.231 |
| relation_type | narrative_ejection | 1 | 0.3 | 3.92× | 0.231 |
| cue_label | electron_equivalence | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.674) — features share decisive prompts
- strong abs-cosine coherence (0.993)
- clear orientation bias (Δ=-0.890)
- significant enrichment for cue_label=emitted_charge_minus1 (lift=9.79, p=0.000)
- significant enrichment for cue_label=neutron_to_proton (lift=7.05, p=0.007)
- significant enrichment for relation_type=neutron_to_proton (lift=7.05, p=0.007)

### Caveats

- Cluster size n=4.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 9 — Mid-Late α-Attribution Module (L19–L21)

**Tentative semantic label:** mid-layer processing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 7 |
| Feature IDs | L14_F140141, L14_F24749, L17_F111674, L17_F122323, L17_F153539, L17_F63126, L17_F6720 |
| Layers | 14, 17 (span 3.0) |
| Dominant role | β-attr (4/7) |
| Role distribution | {'β-attr': 4, 'α-attr': 3} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.185) |
| Depth zone | mid |
| Mean abs cosine (within) | 0.8802 |
| Mean co-importance (within) | 0.5386 |
| Mean feature-to-centroid cosine | 0.9473 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| AUX-P1-v01 | 0.393 | 0.393 | 1.00 | AUX | AUX-P1 | alpha |
| L1-B3-v04 | 0.330 | 0.330 | 1.00 | 1 | L1-B3 | beta |
| L1-B5-v07 | 0.286 | 0.286 | 1.00 | 1 | L1-B5 | beta |
| L2-AR5-v04 | 0.286 | 0.286 | 1.00 | 2 | L2-AR5 | alpha |
| L2-AR2-v04 | 0.286 | 0.286 | 1.00 | 2 | L2-AR2 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L2-BR2-v06 | -0.955 | 0.955 | 1.00 | 2 | L2-BR2 | beta |
| CP-L2-12-v02 | -0.884 | 0.884 | 1.00 | 2 | CP-L2-12 | beta |
| L2-BR2-v01 | -0.857 | 0.857 | 1.00 | 2 | L2-BR2 | beta |
| L2-BR2-v07 | -0.759 | 0.759 | 1.00 | 2 | L2-BR2 | beta |
| KW-B5-v02 | -0.692 | 0.692 | 1.00 | 1 | KW-B5 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-P1 | 0.393 | 0.393 | 1.00 | 0.00 | AUX | alpha |
| AUX-P3 | 0.277 | 0.277 | 1.00 | 0.00 | AUX | alpha |
| AUX-ISO6 | 0.223 | 0.223 | 1.00 | 0.00 | AUX | alpha |
| AUX-B2 | 0.214 | 0.214 | 1.00 | 0.00 | AUX | beta |
| AUX-ISO4 | 0.179 | 0.179 | 1.00 | 0.00 | AUX | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| relation_type | neutron_to_proton | 6 | 0.4 | 14.10× | 0.000 |
| cue_label | neutron_to_proton | 6 | 0.4 | 14.10× | 0.000 |
| relation_type | n_to_p_with_antineutrino | 2 | 0.3 | 5.88× | 0.041 |
| cue_label | n_to_p_with_antineutrino | 2 | 0.3 | 5.88× | 0.041 |
| cue_label | charge_plus_a_unchanged | 1 | 0.3 | 3.92× | 0.231 |
| relation_type | charge_plus_a_unchanged | 1 | 0.3 | 3.92× | 0.231 |

### Evidence summary

- high co-importance coherence (0.539) — features share decisive prompts
- strong abs-cosine coherence (0.880)
- clear orientation bias (Δ=+0.185)
- significant enrichment for relation_type=neutron_to_proton (lift=14.10, p=0.000)
- significant enrichment for cue_label=neutron_to_proton (lift=14.10, p=0.000)
- significant enrichment for relation_type=n_to_p_with_antineutrino (lift=5.88, p=0.041)
- significant enrichment for cue_label=n_to_p_with_antineutrino (lift=5.88, p=0.041)

### Caveats

- Cluster size n=7.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 10 — Output Decision Module (L24–L25)

**Tentative semantic label:** mid-to-late processing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 5 |
| Feature IDs | L20_F117855, L20_F18705, L20_F72939, L20_F74959, L20_F80071 |
| Layers | 20 (span 0.0) |
| Dominant role | β-attr (3/5) |
| Role distribution | {'β-attr': 3, 'α-attr': 2} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.240) |
| Depth zone | mid-late |
| Mean abs cosine (within) | 0.9629 |
| Mean co-importance (within) | 0.5568 |
| Mean feature-to-centroid cosine | 0.9851 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L3-03-v01 | 0.463 | 0.463 | 1.00 | 3 | CP-L3-03 | alpha |
| L1-A1-v03 | 0.438 | 0.438 | 1.00 | 1 | L1-A1 | alpha |
| AUX-P3-v01 | 0.375 | 0.375 | 1.00 | AUX | AUX-P3 | alpha |
| L3-A2-v03 | 0.350 | 0.350 | 1.00 | 3 | L3-A2 | alpha |
| KW-B1-v02 | 0.338 | 0.338 | 1.00 | 1 | KW-B1 | beta |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-01-v02 | -0.725 | 0.725 | 1.00 | 2 | CP-L2-01 | beta |
| L1-B3-v07 | -0.725 | 0.725 | 1.00 | 1 | L1-B3 | beta |
| L3-B3-v06 | -0.688 | 0.688 | 1.00 | 3 | L3-B3 | beta |
| L3-B2-v02 | -0.662 | 0.662 | 1.00 | 3 | L3-B2 | beta |
| AUX-P9-v01 | -0.650 | 0.650 | 1.00 | AUX | AUX-P9 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-P3 | 0.375 | 0.375 | 1.00 | 0.00 | AUX | alpha |
| AUX-P1 | 0.225 | 0.225 | 1.00 | 0.00 | AUX | alpha |
| AUX-CF9 | 0.212 | 0.212 | 1.00 | 0.00 | AUX | alpha |
| AUX-A7 | 0.188 | 0.188 | 1.00 | 0.00 | AUX | alpha |
| L2-AR2 | 0.184 | 0.186 | 1.00 | 0.00 | 2 | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | z_plus1_with_antineutrino | 3 | 0.3 | 8.81× | 0.003 |
| relation_type | z_plus1_with_antineutrino | 3 | 0.3 | 8.81× | 0.003 |
| cue_label | not_nuclear_fragment | 2 | 0.3 | 7.83× | 0.023 |
| cue_label | daughter_a_unchanged | 3 | 0.5 | 5.88× | 0.011 |
| cue_label | charge_plus_a_unchanged | 1 | 0.3 | 3.92× | 0.231 |
| relation_type | charge_plus_a_unchanged | 1 | 0.3 | 3.92× | 0.231 |

### Evidence summary

- high co-importance coherence (0.557) — features share decisive prompts
- strong abs-cosine coherence (0.963)
- clear orientation bias (Δ=+0.240)
- significant enrichment for cue_label=z_plus1_with_antineutrino (lift=8.81, p=0.003)
- significant enrichment for relation_type=z_plus1_with_antineutrino (lift=8.81, p=0.003)
- significant enrichment for cue_label=not_nuclear_fragment (lift=7.83, p=0.023)
- significant enrichment for cue_label=daughter_a_unchanged (lift=5.88, p=0.011)

### Caveats

- Cluster size n=5.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---

## Cluster 11 — Unnamed

**Tentative semantic label:** mid-to-late processing · α-oriented

| Property | Value |
|----------|-------|
| Members (n) | 10 |
| Feature IDs | L19_F130590, L19_F25394, L19_F41536, L19_F44438, L19_F94584, L21_F12280, L21_F157518, L21_F53078, L21_F56012, L21_F99203 |
| Layers | 19, 21 (span 2.0) |
| Dominant role | β-attr (6/10) |
| Role distribution | {'α-attr': 4, 'β-attr': 6} |
| Circuit features | 0 |
| Global α-discriminators | 0 |
| Global β-discriminators | 0 |
| Orientation | alpha (Δ = +0.173) |
| Depth zone | mid-late |
| Mean abs cosine (within) | 0.8397 |
| Mean co-importance (within) | 0.3244 |
| Mean feature-to-centroid cosine | 0.925 |

### Top prompts by cluster mean signed effect (positive)

| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| L3-A2-v05 | 0.728 | 0.928 | 0.50 | 3 | L3-A2 | alpha |
| L2-AR6-v04 | 0.688 | 0.688 | 1.00 | 2 | L2-AR6 | alpha |
| L1-A6-v07 | 0.597 | 0.859 | 0.50 | 1 | L1-A6 | alpha |
| L2-AR1-v08 | 0.569 | 0.569 | 1.00 | 2 | L2-AR1 | alpha |
| L2-AR1-v02 | 0.569 | 0.569 | 1.00 | 2 | L2-AR1 | alpha |

### Top prompts by cluster mean signed effect (negative)

| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |
|-----------|----------|---------|-----------|-------|-------|--------|
| CP-L2-01-v02 | -0.544 | 0.544 | 1.00 | 2 | CP-L2-01 | beta |
| L3-B3-v05 | -0.537 | 0.537 | 1.00 | 3 | L3-B3 | beta |
| CP-L2-11-v02 | -0.475 | 0.475 | 1.00 | 2 | CP-L2-11 | beta |
| CP-L2-10-v02 | -0.425 | 0.425 | 1.00 | 2 | CP-L2-10 | beta |
| L2-BR2-v06 | -0.425 | 0.425 | 1.00 | 2 | L2-BR2 | beta |

### Top groups by mean cluster effect

| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |
|----------|----------|-------------|-----------|-----|-------|--------|
| AUX-CF3 | 0.472 | 0.472 | 0.90 | 0.00 | AUX | alpha |
| AUX-A1 | 0.325 | 0.375 | 0.50 | 0.00 | AUX | alpha |
| L2-AR6 | 0.324 | 0.364 | 1.00 | 0.00 | 2 | alpha |
| AUX-A8 | 0.300 | 0.588 | 0.50 | 0.00 | AUX | alpha |
| AUX-P5 | 0.291 | 1.016 | 0.50 | 0.00 | AUX | alpha |

### Strongest metadata enrichments (top-20 prompts by |effect|)

| Field | Value | Observed | Expected | Lift | p-value |
|-------|-------|----------|----------|------|---------|
| cue_label | composite_nuclear_object | 4 | 0.3 | 11.75× | 0.000 |
| cue_label | heavy_nuclear_fragment | 2 | 0.3 | 7.83× | 0.023 |
| cue_label | emitted_mass4 | 2 | 0.5 | 3.92× | 0.088 |
| cue_label | emitted_2protons | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | emitted_2neutrons | 1 | 0.3 | 2.94× | 0.296 |
| cue_label | mass_plus_daughter_a | 1 | 0.3 | 2.94× | 0.296 |

### Evidence summary

- high co-importance coherence (0.324) — features share decisive prompts
- clear orientation bias (Δ=+0.173)
- significant enrichment for cue_label=composite_nuclear_object (lift=11.75, p=0.000)
- significant enrichment for cue_label=heavy_nuclear_fragment (lift=7.83, p=0.023)

### Caveats

- Cluster size n=10.
- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.
- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.
- All orientation/enrichment results are observational, not causal proofs.

---
