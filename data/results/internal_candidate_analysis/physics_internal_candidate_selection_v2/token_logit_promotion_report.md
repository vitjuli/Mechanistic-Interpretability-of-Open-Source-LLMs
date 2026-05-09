# Token-Logit Competitor Promotion Report
## physics_internal_candidate_selection_v2 | k=6

## Method
For each (prompt, cluster) pair: run baseline forward pass and cluster-ablated forward pass.
Record actual log-probabilities for candidate tokens {electron, proton, neutron, photon}.
Promoted competitor = particle with largest positive Δlogp among non-correct particles.

## Sign-Flip Promotion Matrix

| Correct → Promoted | n promotions |
|---|---|
| photon → proton | 57 |
| photon → neutron | 22 |
| neutron → proton | 15 |
| neutron → photon | 15 |
| photon → electron | 11 |
| neutron → electron | 10 |
| proton → photon | 9 |
| electron → photon | 7 |
| electron → proton | 5 |
| proton → electron | 5 |
| proton → neutron | 2 |

## Mean Δlogp per candidate after cluster ablation

| Cluster | Correct | Δlogp_correct | Δlogp_electron | Δlogp_proton | Δlogp_neutron | Δlogp_photon |
|---|---|---|---|---|---|---|
| C0 | electron | -3.197 | -3.197 | +1.439 | +1.885 | +1.714 |
| C0 | neutron | -3.242 | +2.471 | +2.701 | -3.242 | +1.490 |
| C0 | photon | -4.545 | +0.878 | +2.452 | +1.991 | -4.545 |
| C0 | proton | -2.958 | +2.126 | -2.958 | +0.317 | +1.998 |
| C1 | electron | -2.583 | -2.583 | +3.095 | +3.407 | +3.909 |
| C1 | neutron | -2.318 | +3.803 | +4.359 | -2.318 | +3.413 |
| C1 | photon | -3.389 | +2.606 | +5.217 | +4.132 | -3.389 |
| C1 | proton | -1.900 | +3.092 | -1.900 | +1.472 | +3.605 |
| C2 | electron | -2.031 | -2.031 | +1.603 | +2.574 | +2.024 |
| C2 | neutron | -2.429 | +0.511 | +0.788 | -2.429 | +0.467 |
| C2 | photon | -2.841 | +1.720 | +2.163 | +2.750 | -2.841 |
| C2 | proton | -2.693 | +1.538 | -2.693 | +0.309 | +0.957 |
| C3 | electron | -4.717 | -4.717 | +0.715 | +0.692 | +3.097 |
| C3 | neutron | -5.211 | +1.489 | +1.987 | -5.211 | +2.767 |
| C3 | photon | -5.521 | -0.218 | +1.789 | +0.991 | -5.521 |
| C3 | proton | -4.850 | +0.778 | -4.850 | -1.474 | +2.997 |
| C4 | electron | -4.931 | -4.931 | +0.330 | +0.547 | +2.213 |
| C4 | neutron | -4.905 | +1.598 | +2.444 | -4.905 | +2.163 |
| C4 | photon | -5.170 | +0.317 | +2.399 | +1.769 | -5.170 |
| C4 | proton | -4.863 | +0.472 | -4.863 | -1.754 | +1.882 |
| C5 | electron | -3.063 | -3.063 | +1.913 | +1.919 | +3.522 |
| C5 | neutron | -3.102 | +3.430 | +3.774 | -3.102 | +3.729 |
| C5 | photon | -3.692 | +1.544 | +3.366 | +2.832 | -3.692 |
| C5 | proton | -3.441 | +2.251 | -3.441 | +0.004 | +3.466 |