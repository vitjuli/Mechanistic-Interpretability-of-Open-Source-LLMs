# Clustering results — §3.6 (cluster formation) and Appendix D (geometry of clusters)

Hierarchical agglomerative clustering (HAC, average linkage) of the causally important transcoder
features on the decay task, cut at K = 12, with the causal-impact acceptance criterion applied by joint
ablation. Pipeline: `scripts/22`–`29`, `scripts/27_cluster_joint_ablation.py`,
`scripts/build_hac_k12_labels.py`, `scripts/27c_ir_null_control.py`.

| file | what | thesis |
|---|---|---|
| `hac_k12_table8_renumbered.csv` | the 12 clusters ordered by depth: physical feature, lift, fire α, fire β, mean offset, layer, status, `agglo_id` | **Table 10** (`tab:clusters`) |
| `hac_k12_acceptance.csv` | acceptance criteria c1–c6 (✓/✗/—) per cluster | App D "Cluster acceptance criteria" |
| `hac_k12_fire.csv` | per-cluster fire/offset + layers, `n_in_matrix` | supporting Table 10 |
| `c5_table9_status.csv` | final status after the causal criterion (flip rate, `c5_pass`, `final_status`) | App D → **6 accepted / 3 partial / 3 descriptive** |
| `joint_ablation_physics_decay_type_probe_train.csv` | raw joint-ablation output (5640 rows), source of the c5 flip rates | App D causal criterion |
| `cluster_axis_geometry.csv` | change of Δμ / u directions under cluster ablation: rotation angle, `rot_frac`, `dcos_u` | App D "Cluster ablation" (charge 14.6°, lepton 12.7°, ...) and "Writing direction rotation" |
| `decay_cluster_feature_firing.csv` | **per-feature** firing (fire_alpha/fire_beta/fire_diff) | source of cluster 7's fire 0.49/0.46 (`L18_F145795`) |
| `cluster_stability_k_sweep.csv` | paraphrase-stability sweep over K: `paraphrase_percentile_of_random`, `silhouette` per K (K=12 → 47.5, 0.479) | **Table `tab:kstab`**, App D "Selecting the number of clusters" |
| `cluster_stability_summary.json` | Louvain (2.5th percentile) vs average-linkage agglomerative (47.5th) paraphrase agreement | App D algorithm choice |

Note: the `cluster` column of `decay_cluster_feature_firing.csv` uses a separate (runB) 0–10
indexing, so cluster membership is read from `hac_k12_table8_renumbered.csv`, not from this file.
This file provides the partition-independent **per-feature** firing values (e.g. the L18 singleton
that the HAC activation matrix omitted).
