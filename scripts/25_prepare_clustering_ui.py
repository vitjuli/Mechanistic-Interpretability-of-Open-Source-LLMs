#!/usr/bin/env python3
"""
Script 25: Prepare clustering benchmark data for the dashboard Clustering Explorer tab.

Outputs to dashboard_probe/public/data/clustering_explorer.json (~300-400 KB).
"""
import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT    = Path(__file__).parent.parent
CLU_DIR = ROOT / "data/results/clustering"
OUT     = ROOT / "dashboard_probe/public/data/clustering_explorer.json"


def _f(v):
    if v is None: return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 5)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v


def main():
    print("Loading clustering data...")

    def lj(name):
        with open(CLU_DIR / name) as f: return json.load(f)
    def ln(name): return np.load(CLU_DIR / name)

    feat_ids    = lj("feat_ids.json")
    feat_labels = lj("feat_labels.json")
    labels_df   = pd.read_csv(CLU_DIR / "cluster_labels.csv", index_col=0)
    ev_df       = pd.read_csv(CLU_DIR / "evaluation_summary.csv")
    rank_df     = pd.read_csv(CLU_DIR / "method_ranking.csv")
    enrich_df   = pd.read_csv(CLU_DIR / "enrichment_detail.csv")
    boot_df     = pd.read_csv(CLU_DIR / "bootstrap_stability.csv")
    W_abs       = ln("W_abs_cosine.npy")
    W_co        = ln("W_coimportance.npy")
    W_sig       = ln("W_signed_cosine.npy")
    X_abs       = ln("feat_prompt_abs.npy")      # (40, 470)
    X_ga_abs    = ln("feat_group_abs.npy")       # (40, 132)
    delta       = ln("feat_delta_abs.npy")       # (40,)

    with open(CLU_DIR / "method_extra.json") as f:
        extra = json.load(f)

    n_feat = len(feat_ids)

    # ── Feature-level t-SNE (2-D embedding for scatter) ─────────────────────────
    print("Computing feature t-SNE (2-D)...")
    X_feat_norm = normalize(X_abs)
    tsne_f = TSNE(n_components=2, perplexity=min(10, n_feat-1),
                  random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    feat_xy = tsne_f.fit_transform(X_feat_norm)   # (40, 2)
    for i in range(2):
        r = feat_xy[:, i].max() - feat_xy[:, i].min()
        if r > 0:
            feat_xy[:, i] = (feat_xy[:, i] - feat_xy[:, i].mean()) / r * 18

    # ── Feature-group t-SNE ─────────────────────────────────────────────────────
    print("Computing feature t-SNE on group profiles...")
    X_grp_norm = normalize(X_ga_abs)
    tsne_g = TSNE(n_components=2, perplexity=min(10, n_feat-1),
                  random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    feat_xy_grp = tsne_g.fit_transform(X_grp_norm)
    for i in range(2):
        r = feat_xy_grp[:, i].max() - feat_xy_grp[:, i].min()
        if r > 0:
            feat_xy_grp[:, i] = (feat_xy_grp[:, i] - feat_xy_grp[:, i].mean()) / r * 18

    # ── Build feature info ───────────────────────────────────────────────────────
    print("Building feature records...")
    # Map method → cluster label per feature
    method_labels = {}
    for col in labels_df.columns:
        method_labels[col] = labels_df[col].values.tolist()

    features = []
    for i, fid in enumerate(feat_ids):
        fl = feat_labels[fid]
        features.append({
            "id":        fid,
            "layer":     fl["layer"],
            "community": fl["community"],
            "role":      fl["role_label"],
            "is_circuit":fl["is_circuit"],
            "is_alpha_d":fl["is_alpha_d"],
            "is_beta_d": fl["is_beta_d"],
            "grad_sign": fl["grad_sign"],
            "delta_abs": round(float(delta[i]), 4),
            "x_prompt":  round(float(feat_xy[i, 0]), 3),
            "y_prompt":  round(float(feat_xy[i, 1]), 3),
            "x_group":   round(float(feat_xy_grp[i, 0]), 3),
            "y_group":   round(float(feat_xy_grp[i, 1]), 3),
        })

    # ── Method metadata ──────────────────────────────────────────────────────────
    print("Building method metadata...")

    # Merge evaluation + bootstrap
    boot_lu = dict(zip(boot_df["method"], boot_df["bootstrap_sil"]))

    # Define human-readable method families and novelty tags
    METHOD_META = {
        "hac":             {"family":"HAC",             "novel":False,  "short":"HAC",         "color":"#60a5fa"},
        "spectral_abs":    {"family":"Spectral",        "novel":False,  "short":"Spectral-abs", "color":"#a78bfa"},
        "spectral_signed": {"family":"Spectral",        "novel":False,  "short":"Spectral-sgn", "color":"#c084fc"},
        "kmeans_abs":      {"family":"K-Means",         "novel":False,  "short":"KMeans-abs",   "color":"#34d399"},
        "kmeans_group":    {"family":"K-Means",         "novel":False,  "short":"KMeans-grp",   "color":"#10b981"},
        "gmm_group":       {"family":"GMM",             "novel":False,  "short":"GMM",          "color":"#6ee7b7"},
        "nmf":             {"family":"NMF",             "novel":False,  "short":"NMF",          "color":"#f97316"},
        "cocluster":       {"family":"Bicluster",       "novel":False,  "short":"Bicluster",    "color":"#94a3b8"},
        "louvain_abs":     {"family":"Louvain",         "novel":False,  "short":"Louvain",      "color":"#fbbf24"},
        "bipartite_louvain":{"family":"Louvain",        "novel":False,  "short":"Bip-Louvain",  "color":"#f59e0b"},
        "coimp_louvain":   {"family":"Novel",           "novel":True,   "short":"CoImp-Lou",    "color":"#ef4444"},
        "signed_laplacian":{"family":"Novel",           "novel":True,   "short":"SignedLap",    "color":"#f87171"},
        "residual_kmeans": {"family":"Novel",           "novel":True,   "short":"Residual",     "color":"#fb923c"},
        "contrastive":     {"family":"Novel",           "novel":True,   "short":"Contrastive",  "color":"#fca5a5"},
        "multiview_consensus":{"family":"Novel",        "novel":True,   "short":"MultiView",    "color":"#e879f9"},
        "level_stability": {"family":"Novel",           "novel":True,   "short":"LevelStab",    "color":"#c026d3"},
    }

    def get_meta(method_name):
        for prefix, m in METHOD_META.items():
            if method_name.startswith(prefix):
                return m
        return {"family":"Other","novel":False,"short":method_name[:12],"color":"#6b7280"}

    methods = []
    for _, row in rank_df.iterrows():
        mn = row["method"]
        meta = get_meta(mn)
        # Extract k from name if present
        k_part = mn.split("_k")[-1] if "_k" in mn else None
        k_val  = int(k_part) if k_part and k_part.isdigit() else None
        t_part = mn.split("_t")[-1] if "_t" in mn else None
        methods.append({
            "name":       mn,
            "k":          k_val,
            "family":     meta["family"],
            "short":      meta["short"],
            "color":      meta["color"],
            "novel":      meta["novel"],
            "n_clusters": int(row["n_clusters"]),
            "silhouette": _f(row["silhouette"]),
            "entropy":    _f(row["entropy"]),
            "ari_role":   _f(row["ari_role"]),
            "ari_circuit":_f(row["ari_circuit"]),
            "ari_alpha_d":_f(row["ari_alpha_d"]),
            "ari_beta_d": _f(row["ari_beta_d"]),
            "ari_old_comm":_f(row["ari_old_comm"]),
            "nmi_role":   _f(row["nmi_role"]),
            "lsu":        _f(row["lsu"]),
            "lsu_A":      _f(row["lsu_A_circuit_coherence"]),
            "lsu_B":      _f(row["lsu_B_ab_discriminability"]),
            "lsu_C":      _f(row["lsu_C_balance"]),
            "composite":  _f(row["composite"]),
            "bootstrap_sil": _f(boot_lu.get(mn)),
            "labels": method_labels.get(mn, []),
        })

    # ── Similarity matrices (feature × feature) ──────────────────────────────────
    print("Packaging similarity matrices...")
    # Round to 3dp to reduce size
    sim_matrices = {
        "abs_cosine":    [[round(float(v),3) for v in row] for row in W_abs],
        "signed_cosine": [[round(float(v),3) for v in row] for row in W_sig],
        "coimportance":  [[round(float(v),3) for v in row] for row in W_co],
    }

    # ── NMF components ───────────────────────────────────────────────────────────
    nmf_data = {}
    for k_str, nmf_k in extra.get("nmf", {}).items():
        W = np.array(nmf_k["W"])     # (40, k)
        H = np.array(nmf_k["H"])     # (k, 470) — skip, too large
        k_int = int(k_str[1:])
        nmf_data[k_str] = {
            "k": k_int,
            "err": _f(nmf_k["err"]),
            "W": [[round(float(v), 4) for v in row] for row in W],
        }

    # ── HAC linkage matrices ─────────────────────────────────────────────────────
    linkage_data = {}
    for key, Z_list in extra.get("linkage", {}).items():
        Z = np.array(Z_list)
        linkage_data[key] = [[round(float(v), 4) for v in row] for row in Z]

    # ── Level stability ──────────────────────────────────────────────────────────
    ls = extra.get("level_stability", {})
    level_stability = {
        "nmi": ls.get("nmi", {}),
        "labels": ls.get("labels", {}),
        "k": ls.get("k", 4),
    }

    # ── Contrastive extras ───────────────────────────────────────────────────────
    con = extra.get("contrastive", {})
    contrastive = {
        "mu_alpha": [round(float(v), 4) for v in con.get("mu_alpha", [])],
        "mu_beta":  [round(float(v), 4) for v in con.get("mu_beta", [])],
        "delta":    [round(float(v), 4) for v in con.get("delta_signed_per_feat", [])],
    }

    # ── Enrichment detail ────────────────────────────────────────────────────────
    enrichment = {}
    for mn in enrich_df["method"].unique():
        sub = enrich_df[enrich_df["method"] == mn]
        enrichment[mn] = [
            {
                "cluster":      int(r["cluster"]),
                "size":         int(r["size"]),
                "n_circuit":    int(r["n_circuit"]),
                "n_alpha_d":    int(r["n_alpha_d"]),
                "n_beta_d":     int(r["n_beta_d"]),
                "dominant_role":str(r["dominant_role"]),
                "features":     str(r["features"]),
            }
            for _, r in sub.iterrows()
        ]

    # ── Signed laplacian eigvals ─────────────────────────────────────────────────
    eigvals = extra.get("signed_laplacian_eigvals", [])

    # ── Assemble ─────────────────────────────────────────────────────────────────
    out = {
        "meta": {
            "n_features": n_feat,
            "n_methods":  len(methods),
            "feat_ids":   feat_ids,
        },
        "features":        features,
        "methods":         methods,
        "sim_matrices":    sim_matrices,
        "nmf":             nmf_data,
        "linkage":         linkage_data,
        "level_stability": level_stability,
        "contrastive":     contrastive,
        "enrichment":      enrichment,
        "eigvals":         [round(float(v), 4) for v in eigvals],
    }

    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT.stat().st_size / 1024
    print(f"\nSaved {OUT.name}  ({size_kb:.0f} KB)")
    print(f"  features: {n_feat}")
    print(f"  methods:  {len(methods)}")
    print(f"  sim matrices: {list(sim_matrices.keys())}")
    print(f"  nmf configs: {list(nmf_data.keys())}")
    print(f"  linkage configs: {list(linkage_data.keys())}")


if __name__ == "__main__":
    main()
