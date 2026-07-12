"""Build a `hac_k12` column into cluster_labels.csv from the HAC k=12 membership
(the Table 9 partition), so 27_cluster_joint_ablation.py can run c5 on the HAC clusters.

Maps feature_id -> HAC cluster_id (== Table 9 cluster number == agglo_id) from
runB_agglo/cluster_semantics/cluster_feature_summary.csv, adds it as column `hac_k12`
to a copy of cluster_labels.csv under data/results/clustering_hac/.

Run (from repo root):  python scripts/build_hac_k12_labels.py
Then:  python scripts/27_cluster_joint_ablation.py ... \
           --clustering_dir data/results/clustering_hac --cluster_col hac_k12
"""
import os
import pandas as pd

SUMM = "data/analysis/runB_agglo/cluster_semantics/cluster_feature_summary.csv"
SRC = "data/results/clustering/cluster_labels.csv"
OUT_DIR = "data/results/clustering_hac"

summ = pd.read_csv(SUMM)
fmap = dict(zip(summ.feature_id, summ.cluster_id))          # feature -> HAC cluster (Table 9 #)
cl = pd.read_csv(SRC)
cl["hac_k12"] = cl.feature_id.map(fmap).astype("Int64")     # Int64 -> writes "6" not "6.0"; NaN -> ""
os.makedirs(OUT_DIR, exist_ok=True)
cl.to_csv(os.path.join(OUT_DIR, "cluster_labels.csv"), index=False)

n = int(cl.hac_k12.notna().sum())
clusters = sorted(int(x) for x in cl.hac_k12.dropna().unique())
sizes = cl.hac_k12.dropna().astype(int).value_counts().sort_index().to_dict()
print(f"HAC assigned: {n} | clusters: {clusters}")
print(f"cluster sizes: {sizes}")
print(f"wrote {OUT_DIR}/cluster_labels.csv")
