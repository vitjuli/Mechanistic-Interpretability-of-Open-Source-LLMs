# Candidate Clustering Analysis Report
## physics_internal_candidate_selection_v2 | train

## Question: do feature activation clusters align with particle identity?

### k=4 clustering vs particle identity (best → worst by ARI):

| Layer group | n_features | ARI | NMI | Purity | Silhouette |
|---|---|---|---|---|---|
| circuit_layers | 57 | 0.133 | 0.197 | 0.517 | 0.183 |
| late | 20 | 0.127 | 0.194 | 0.515 | 0.207 |
| all | 77 | 0.119 | 0.197 | 0.517 | 0.182 |
| early | 20 | 0.032 | 0.033 | 0.380 | 0.181 |
| mid | 37 | 0.017 | 0.031 | 0.376 | 0.214 |
| retrieval | 15 | 0.014 | 0.037 | 0.385 | 0.188 |

### Comparison: particle vs wording family (k=4, 'all' layers):

- **particle_identity**: ARI=0.119, NMI=0.197, purity=0.517
- **filter_property**: ARI=0.110, NMI=0.193, purity=0.360
- **wording_family**: ARI=0.138, NMI=0.188, purity=0.512

### Interpretation: **Mixed**: clusters align equally with particle and wording family — cannot distinguish content from form
