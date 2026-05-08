# Candidate Trajectory Analysis Report
## physics_internal_candidate_selection_v2 | train

## Key question: does the model activate multiple candidates early and converge late?

| Metric | Early (L10-L14) | Late (L22-L25) | Change |
|---|---|---|---|
| Rank-1 accuracy | 0.895 | 0.380 | -0.515 |
| Mean entropy | 1.354 | 0.384 | -0.970 |
| Mean margin | 0.010 | -1.143 | -1.153 |

### Pattern: **EARLY SELECTION**: model selects correct candidate already at early layers

## First layer at which correct candidate reaches rank-1

| Layer | n prompts | fraction |
|---|---|---|
| L10 | 447 | 1.00 |

## Layer-by-layer trajectory

| Layer | rank_acc | mean_correct_score | mean_comp_score | margin | entropy |
|---|---|---|---|---|---|
| L10 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L11 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L12 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L13 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L14 | 0.474 | 0.489 | 0.441 | 0.048 | 1.224 |
| L15 | 0.313 | 0.791 | 1.631 | -0.840 | 0.773 |
| L16 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L17 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L18 | 0.311 | 0.756 | 1.579 | -0.823 | 0.790 |
| L19 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L20 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L21 | 1.000 | 0.000 | 0.000 | 0.000 | 1.386 |
| L22 | 0.342 | 1.673 | 2.822 | -1.149 | 0.272 |
| L23 | 0.394 | 1.793 | 2.751 | -0.957 | 0.532 |
| L24 | 0.324 | 3.103 | 5.128 | -2.025 | 0.164 |
| L25 | 0.461 | 2.386 | 2.826 | -0.440 | 0.566 |
