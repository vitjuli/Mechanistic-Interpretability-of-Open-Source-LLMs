"""
Distributed candidate-state analysis via feature clustering.

Clusters transcoder features by their activation profiles across prompts, then
analyses whether feature clusters correspond to particle identity, wording style,
or mixed representations. Tests the cluster-level T/C/B hypothesis and computes
the layer depth vs alignment transition.

Parts:
  A — Feature activation matrix  (features × prompts)
  B — Feature clustering         (hierarchical + k-means, k=4/6/8)
  C — Cluster semantic analysis  (particle/wording/filter distributions)
  D — Cluster-level T/C/B       (IPR, Mann-Whitney at cluster level)
  E — Layer transition analysis  (per-layer ARI: particle vs wording)
  F — Cluster-to-path overlap   (circuit path / causal edge coverage)

Usage:
  python scripts/45_candidate_feature_clustering.py
  python scripts/45_candidate_feature_clustering.py --k 4 6 8 --layer_subset late circuit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, normalize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BEHAVIOUR  = "physics_internal_candidate_selection_v2"
SPLIT      = "train"
LAYERS     = list(range(10, 26))
PARTICLES  = ["electron", "proton", "neutron", "photon"]
N_ST       = 447   # single-token prompt count (multi-token are indices 447-486)

LAYER_GROUPS = {
    "early":    list(range(10, 14)),
    "mid":      list(range(14, 19)),
    "retrieval": list(range(19, 22)),
    "late":     list(range(22, 26)),
    "all":      list(range(10, 26)),
}

# ─── Paths ────────────────────────────────────────────────────────────────────

def get_paths(behaviour, split):
    base = Path("data")
    return {
        "prompts":       base / "prompts" / f"{behaviour}_{split}.jsonl",
        "graph_json":    base / "results" / "attribution_graphs" / behaviour
                              / f"attribution_graph_{split}_n120_roleaware.json",
        "circuit_json":  base / "results" / f"circuits_{behaviour}_{split}_roleaware.json",
        "causal_json":   base / "results" / "causal_edges" / behaviour
                              / f"causal_edges_{behaviour}_{split}.json",
        "path_val_json": base / "results" / "path_validation" / behaviour
                              / f"path_validation_{behaviour}_{split}.json",
        "feature_dir":   base / "results" / "transcoder_features",
        "feat_table":    base / "results" / "internal_candidate_analysis" / behaviour
                              / "candidate_feature_table.csv",
        "output_dir":    base / "results" / "internal_candidate_analysis" / behaviour,
    }

# ─── Data loading ────────────────────────────────────────────────────────────

def load_prompts(paths):
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct"] = r["correct_answer"].strip()
        r["_pool"]    = set(r["implicit_candidate_pool"])
    return [r for r in rows if not r.get("multi_token_answer", False)]


def load_graph_features(paths):
    with open(paths["graph_json"]) as f:
        g = json.load(f)
    return {(n["layer"], n["feature_idx"]): n
            for n in g["nodes"] if n.get("type") == "feature"}


def load_circuit_features(paths):
    with open(paths["circuit_json"]) as f:
        c = json.load(f)
    result = set()
    for nid in c.get("circuit", {}).get("feature_nodes", []):
        if "_F" in nid and nid.startswith("L"):
            l, f = nid.split("_F")
            result.add((int(l[1:]), int(f)))
    return result


def load_top_k(behaviour, split, layer, feature_dir):
    d   = feature_dir / f"layer_{layer}"
    idx = d / f"{behaviour}_{split}_top_k_indices.npy"
    val = d / f"{behaviour}_{split}_top_k_values.npy"
    if not idx.exists():
        return None
    return np.load(idx)[:N_ST], np.load(val)[:N_ST]


def get_act(indices, values, feat_idx):
    act = np.zeros(indices.shape[0], dtype=np.float32)
    r, c = np.where(indices == feat_idx)
    act[r] = values[r, c]
    return act


# ─── Part A: Feature activation matrix ───────────────────────────────────────

def build_feature_matrix(graph_features, paths, behaviour, split, layers=None):
    """
    Returns:
      X_act  [n_features, n_prompts]  raw activations
      X_attr [n_features, n_prompts]  attribution-weighted activations
      X_z    [n_features, n_prompts]  z-scored activations (across prompts)
      feat_ids  list of (layer, feat_idx)
      feat_meta list of node dicts
    """
    if layers is None:
        layers = LAYERS

    feat_ids  = sorted((l, f) for (l, f) in graph_features if l in layers)
    feat_meta = [graph_features[k] for k in feat_ids]

    X_act  = np.zeros((len(feat_ids), N_ST), dtype=np.float32)
    X_attr = np.zeros((len(feat_ids), N_ST), dtype=np.float32)

    for i, (layer, fidx) in enumerate(feat_ids):
        result = load_top_k(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            continue
        indices, values = result
        act = get_act(indices, values, fidx)
        X_act[i] = act

        node    = graph_features[(layer, fidx)]
        m_act   = node.get("mean_activation_conditional", 1.0)
        m_attr  = node.get("mean_grad_attr_conditional", 0.0)
        scale   = m_attr / m_act if m_act > 0 else 0.0
        X_attr[i] = act * scale

    # Z-score each feature row across prompts
    means = X_act.mean(axis=1, keepdims=True)
    stds  = X_act.std(axis=1, keepdims=True) + 1e-8
    X_z   = (X_act - means) / stds

    print(f"  Feature matrix: {X_act.shape}  "
          f"nonzero={float((X_act > 0).mean()):.3f}")
    return X_act, X_attr, X_z, feat_ids, feat_meta


# ─── Part B: Feature clustering ──────────────────────────────────────────────

def cluster_features(X, feat_ids, feat_meta, k_values=(4, 6, 8)):
    """
    Cluster features (rows of X). Each feature's vector is its activation
    profile across N_ST prompts.

    Returns dict: k → {'labels': array, 'inertia': float}
    Also returns hierarchical linkage matrix.
    """
    # L2-normalise feature vectors for cosine similarity
    X_norm = normalize(X, norm="l2")

    # Hierarchical clustering
    dist_mat = pdist(X_norm, metric="cosine")
    linkage  = sch.linkage(dist_mat, method="ward")

    results = {"hierarchical_linkage": linkage, "feat_ids": feat_ids, "feat_meta": feat_meta}

    for k in k_values:
        # k-means
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels_km = km.fit_predict(X_norm)

        # Hierarchical cut
        labels_hc = sch.fcluster(linkage, k, criterion="maxclust") - 1

        results[k] = {
            "kmeans":       labels_km,
            "hierarchical": labels_hc,
            "inertia_km":   float(km.inertia_),
        }
        print(f"  k={k}: kmeans inertia={km.inertia_:.3f}, "
              f"cluster sizes={np.bincount(labels_km).tolist()}")

    return results


# ─── Part C: Cluster semantic analysis ───────────────────────────────────────

def cluster_semantic_analysis(labels, X_act, feat_ids, feat_meta, prompts, particles=PARTICLES):
    """
    For each cluster: compute prompt-level activation, then analyse
    correct-answer / wording-family / filter-property distributions.
    """
    n_clusters = int(labels.max()) + 1

    # prompt × cluster activation matrix: mean over cluster features
    P_act = np.zeros((N_ST, n_clusters), dtype=np.float32)
    for c in range(n_clusters):
        feat_mask = labels == c
        if feat_mask.sum() == 0:
            continue
        P_act[:, c] = X_act[feat_mask].mean(axis=0)

    rows = []
    for c in range(n_clusters):
        feat_mask   = labels == c
        cluster_feats = [feat_ids[i] for i in np.where(feat_mask)[0]]
        cluster_meta  = [feat_meta[i] for i in np.where(feat_mask)[0]]
        layer_counts  = {}
        for (l, _) in cluster_feats:
            layer_counts[l] = layer_counts.get(l, 0) + 1

        col = P_act[:, c]  # mean cluster activation per prompt

        # Particle-wise mean activation
        particle_means = {}
        for particle in particles:
            mask = np.array([p["_correct"] == particle for p in prompts])
            particle_means[f"mu_{particle}"] = float(col[mask].mean()) if mask.sum() else float("nan")

        # Dominant particle by mean activation
        dominant_particle = max(
            particles, key=lambda p: particle_means.get(f"mu_{p}", -np.inf)
        )

        # Wording family distribution
        wf_counts = {}
        for p in prompts:
            wf = p.get("wording_family", "")
            wf_counts[wf] = wf_counts.get(wf, 0) + 1

        # Entropy of particle activation distribution
        vals  = np.array([particle_means[f"mu_{p}"] for p in particles])
        vals  = np.clip(vals - vals.min() + 1e-8, 0, None)
        probs = vals / vals.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        row = {
            "cluster":          c,
            "n_features":       int(feat_mask.sum()),
            "layer_range":      f"L{min(l for l,_ in cluster_feats)}-L{max(l for l,_ in cluster_feats)}"
                                if cluster_feats else "",
            "dominant_particle": dominant_particle,
            "particle_entropy":  entropy,
            "mean_specific_score": float(np.mean([m.get("specific_score", 0) for m in cluster_meta])),
            "mean_frequency":    float(np.mean([m.get("frequency", 0) for m in cluster_meta])),
            "in_circuit":        int(sum(1 for m in cluster_meta if m.get("causal_status") == "output_attributed")),
        }
        row.update(particle_means)
        row["layer_counts"] = json.dumps(layer_counts)
        rows.append(row)

    return pd.DataFrame(rows), P_act


# ─── Part D: Cluster-level T/C/B analysis ────────────────────────────────────

def cluster_tcb_analysis(P_act, prompts, n_clusters):
    """
    For each cluster × particle: compute T/C/B group mean activations of the
    cluster score (mean activation over cluster features), test T>C>B.

    Pool structure note:
      - electron/proton: background=0 (always in pool) → test T vs C only
      - photon: competitor=0 (never non-selected) → test T vs B only
      - neutron: all three groups ✓ → test T>C>B
    """
    rows = []
    for c in range(n_clusters):
        col = P_act[:, c]
        for particle in PARTICLES:
            t_mask = np.array([p["_correct"] == particle for p in prompts])
            c_mask = np.array([particle in p["_pool"] and p["_correct"] != particle
                               for p in prompts])
            b_mask = np.array([particle not in p["_pool"] for p in prompts])

            mu_T = float(col[t_mask].mean()) if t_mask.sum() >= 3 else float("nan")
            mu_C = float(col[c_mask].mean()) if c_mask.sum() >= 3 else float("nan")
            mu_B = float(col[b_mask].mean()) if b_mask.sum() >= 3 else float("nan")

            # Mann-Whitney T > C (if both groups exist)
            if t_mask.sum() >= 5 and c_mask.sum() >= 5:
                mw_TC = stats.mannwhitneyu(col[t_mask], col[c_mask], alternative="greater")
                p_TC, u_TC = float(mw_TC.pvalue), float(mw_TC.statistic)
            else:
                p_TC, u_TC = float("nan"), float("nan")

            if c_mask.sum() >= 5 and b_mask.sum() >= 5:
                mw_CB = stats.mannwhitneyu(col[c_mask], col[b_mask], alternative="greater")
                p_CB, u_CB = float(mw_CB.pvalue), float(mw_CB.statistic)
            else:
                p_CB, u_CB = float("nan"), float("nan")

            # T > B (for photon)
            if t_mask.sum() >= 5 and b_mask.sum() >= 5:
                mw_TB = stats.mannwhitneyu(col[t_mask], col[b_mask], alternative="greater")
                p_TB = float(mw_TB.pvalue)
            else:
                p_TB = float("nan")

            ipr = (mu_C / mu_T) if (not np.isnan(mu_C) and not np.isnan(mu_T) and mu_T > 0) else float("nan")

            cand_spec = (mu_T - mu_B) if not np.isnan(mu_T + mu_B) else float("nan")
            comp_pres = (mu_C - mu_B) if not np.isnan(mu_C + mu_B) else float("nan")

            ordering = bool(not np.isnan(mu_T + mu_C + mu_B) and mu_T > mu_C > mu_B)

            rows.append({
                "cluster":              c,
                "particle":             particle,
                "n_target":             int(t_mask.sum()),
                "n_competitor":         int(c_mask.sum()),
                "n_background":         int(b_mask.sum()),
                "mu_T":                 mu_T,
                "mu_C":                 mu_C,
                "mu_B":                 mu_B,
                "IPR":                  ipr,
                "candidate_specificity": cand_spec,
                "competitor_presence":  comp_pres,
                "ordering_T_gt_C_gt_B": ordering,
                "mw_p_TC":              p_TC,
                "mw_p_CB":              p_CB,
                "mw_p_TB":              p_TB,
                "sig_TC_005":           bool(not np.isnan(p_TC) and p_TC < 0.05),
                "sig_CB_005":           bool(not np.isnan(p_CB) and p_CB < 0.05),
                "sig_TB_005":           bool(not np.isnan(p_TB) and p_TB < 0.05),
                "strong_candidate":     bool(ordering and not np.isnan(p_TC) and p_TC < 0.05
                                            and not np.isnan(p_CB) and p_CB < 0.05),
            })
    return pd.DataFrame(rows)


# ─── Part E: Per-layer ARI transition ────────────────────────────────────────

def layer_transition_analysis(graph_features, paths, behaviour, split, prompts):
    """
    For each individual layer L10-L25:
      1. Build feature activation matrix at that layer only
      2. Cluster PROMPTS (not features) using k-means k=4
      3. Compute ARI vs particle identity and vs wording family

    Returns DataFrame with one row per layer.
    """
    correct_labels  = np.array([p["_correct"]            for p in prompts])
    wording_labels  = np.array([p.get("wording_family", "") for p in prompts])
    filter_labels   = np.array([p.get("filter_property", "") for p in prompts])

    def encode(arr):
        uniq = sorted(set(arr))
        m    = {v: i for i, v in enumerate(uniq)}
        return np.array([m[v] for v in arr])

    correct_int = encode(correct_labels)
    wording_int = encode(wording_labels)
    filter_int  = encode(filter_labels)

    rows = []
    for layer in LAYERS:
        result = load_top_k(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            continue
        indices, values = result

        # Build prompt-level feature matrix at this layer
        feat_list = [(fidx, graph_features[(layer, fidx)])
                     for (l, fidx) in graph_features if l == layer]
        if not feat_list:
            continue

        X_layer = np.stack([get_act(indices, values, fidx)
                            for (fidx, _) in feat_list], axis=1)  # (N_ST, n_feats)
        X_norm  = normalize(X_layer, norm="l2")

        km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        km4_labels = km4.fit_predict(X_norm)

        ari_particle = adjusted_rand_score(correct_int, km4_labels)
        ari_wording  = adjusted_rand_score(wording_int, km4_labels)
        ari_filter   = adjusted_rand_score(filter_int,  km4_labels)

        nmi_particle = normalized_mutual_info_score(correct_int, km4_labels)
        nmi_wording  = normalized_mutual_info_score(wording_int, km4_labels)

        ratio = (ari_particle / ari_wording) if abs(ari_wording) > 1e-4 else float("nan")

        rows.append({
            "layer":            layer,
            "n_features":       len(feat_list),
            "ari_particle":     ari_particle,
            "ari_wording":      ari_wording,
            "ari_filter":       ari_filter,
            "nmi_particle":     nmi_particle,
            "nmi_wording":      nmi_wording,
            "ratio_p_over_w":   ratio,
        })
        print(f"  L{layer}: ARI_particle={ari_particle:.3f}  ARI_wording={ari_wording:.3f}  ratio={ratio:.3f}")

    return pd.DataFrame(rows)


# ─── Part F: Cluster-to-path overlap ─────────────────────────────────────────

def cluster_path_overlap(labels, feat_ids, paths, circuit_features):
    """
    For each cluster: compute overlap with extracted circuit paths and causal edges.
    """
    feat_id_to_cluster = {fid: int(labels[i]) for i, fid in enumerate(feat_ids)}

    # Load causal edges
    with open(paths["causal_json"]) as f:
        ce = json.load(f)
    causal_edges = ce if isinstance(ce, list) else ce.get("edges", [])

    # Load path validation paths
    with open(paths["path_val_json"]) as f:
        pv = json.load(f)
    circuit_paths = pv.get("paths", [])

    n_clusters = int(labels.max()) + 1
    rows = []
    for c in range(n_clusters):
        cluster_fids = {fid for i, fid in enumerate(feat_ids) if labels[i] == c}
        cluster_set  = set(cluster_fids)

        # Causal edges where BOTH endpoints are in cluster
        intra_causal = sum(
            1 for e in causal_edges
            if (int(e["src_layer"]), int(e["src_feat_idx"])) in cluster_set
            and (int(e["tgt_layer"]), int(e["tgt_feat_idx"])) in cluster_set
        )
        # Causal edges where at least one endpoint is in cluster
        any_causal = sum(
            1 for e in causal_edges
            if (int(e["src_layer"]), int(e["src_feat_idx"])) in cluster_set
            or (int(e["tgt_layer"]), int(e["tgt_feat_idx"])) in cluster_set
        )

        # Circuit path coverage: how many of the top-10 validated paths include cluster features?
        path_coverage = 0
        for p in circuit_paths:
            path_nodes = p.get("path", [])
            for node in path_nodes:
                if "_F" in node and node.startswith("L"):
                    l, f = node.split("_F")
                    if (int(l[1:]), int(f)) in cluster_set:
                        path_coverage += 1
                        break

        # Circuit feature overlap
        circuit_in_cluster = sum(1 for fid in cluster_fids if fid in circuit_features)

        # Mean causal strength for edges involving this cluster
        cluster_causal_strengths = [
            e["mean_delta_abs"] for e in causal_edges
            if (int(e["src_layer"]), int(e["src_feat_idx"])) in cluster_set
            or (int(e["tgt_layer"]), int(e["tgt_feat_idx"])) in cluster_set
        ]

        rows.append({
            "cluster":                  c,
            "n_features":               len(cluster_fids),
            "intra_cluster_causal_edges": intra_causal,
            "any_endpoint_causal_edges": any_causal,
            "circuit_path_coverage":    path_coverage,
            "circuit_features_in_cluster": circuit_in_cluster,
            "frac_circuit":             circuit_in_cluster / len(cluster_fids) if cluster_fids else 0,
            "mean_causal_strength":     float(np.mean(cluster_causal_strengths)) if cluster_causal_strengths else float("nan"),
        })

    return pd.DataFrame(rows)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_layer_transition(layer_df, output_dir):
    if not HAS_MPL or layer_df is None or len(layer_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = layer_df["layer"].tolist()
    ari_p  = layer_df["ari_particle"].tolist()
    ari_w  = layer_df["ari_wording"].tolist()
    ratio  = layer_df["ratio_p_over_w"].tolist()

    ax = axes[0]
    ax.plot(layers, ari_p, "o-", color="#2166ac", lw=2.5, label="Particle identity", zorder=3)
    ax.plot(layers, ari_w, "s--", color="#d6604d", lw=2.5, label="Wording family",   zorder=3)
    # Shade layer groups
    ax.axvspan(9.5,  13.5, alpha=0.07, color="gray",   label="Early (L10–13)")
    ax.axvspan(13.5, 18.5, alpha=0.07, color="orange",  label="Mid (L14–18)")
    ax.axvspan(18.5, 21.5, alpha=0.07, color="green",   label="Retrieval (L19–21)")
    ax.axvspan(21.5, 25.5, alpha=0.07, color="#2166ac", label="Late (L22–25)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Transcoder Layer", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index (k=4 clustering)", fontsize=11)
    ax.set_title("Form → Content Transition:\nClustering Alignment by Layer", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(9, 26)

    ax = axes[1]
    colors = ["#d73027" if r < 1 else "#4575b4" for r in ratio if not np.isnan(r)]
    valid_layers = [l for l, r in zip(layers, ratio) if not np.isnan(r)]
    valid_ratio  = [r for r in ratio if not np.isnan(r)]
    bars = ax.bar(valid_layers, valid_ratio,
                  color=["#4575b4" if r >= 1 else "#d73027" for r in valid_ratio],
                  alpha=0.8, edgecolor="white")
    ax.axhline(1.0, color="black", lw=1.5, linestyle="--", label="ARI particle = ARI wording")
    ax.axvspan(9.5,  13.5, alpha=0.05, color="gray")
    ax.axvspan(13.5, 18.5, alpha=0.05, color="orange")
    ax.axvspan(18.5, 21.5, alpha=0.05, color="green")
    ax.axvspan(21.5, 25.5, alpha=0.05, color="#2166ac")
    ax.set_xlabel("Transcoder Layer", fontsize=12)
    ax.set_ylabel("ARI ratio (particle / wording)", fontsize=11)
    ax.set_title("Particle vs Wording Dominance by Layer\n(>1 = particle-driven, <1 = wording-driven)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(9, 26)

    fig.tight_layout()
    path = output_dir / "layer_transition_ari.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_cluster_tcb(tcb_df, semantic_df, k, method, output_dir):
    if not HAS_MPL:
        return

    particles_with_groups = {
        "neutron": ["T", "C", "B"],
        "electron": ["T", "C"],
        "proton":   ["T", "C"],
        "photon":   ["T", "B"],
    }
    n_clusters  = tcb_df["cluster"].nunique()
    n_particles = len(PARTICLES)

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5), sharey=False)
    if n_clusters == 1:
        axes = [axes]

    colors = {"T": "#2166ac", "C": "#fdae61", "B": "#d73027"}

    for c in range(n_clusters):
        ax   = axes[c]
        dom  = semantic_df[semantic_df["cluster"] == c].iloc[0]["dominant_particle"] if len(semantic_df) else "?"
        n_f  = int(semantic_df[semantic_df["cluster"] == c].iloc[0]["n_features"]) if len(semantic_df) else 0

        ax.set_title(f"C{c}\n({n_f} feats, dom={dom})", fontsize=9)
        ax.axhline(0, color="black", lw=0.5)

        x_pos = 0
        xticks, xlabels = [], []
        for p in PARTICLES:
            groups = particles_with_groups[p]
            row    = tcb_df[(tcb_df["cluster"] == c) & (tcb_df["particle"] == p)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            vals = {"T": row["mu_T"], "C": row["mu_C"], "B": row["mu_B"]}
            for g in groups:
                v = vals[g]
                if not np.isnan(v):
                    bar = ax.bar(x_pos, v, color=colors[g], alpha=0.85, width=0.7, edgecolor="white")
                xticks.append(x_pos)
                xlabels.append(f"{p[0].upper()}-{g}")
                x_pos += 1
            x_pos += 0.3  # gap between particles

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean cluster activation" if c == 0 else "")
        ax.grid(alpha=0.2, axis="y")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in colors.values()]
    labels  = ["Target (T)", "Competitor (C)", "Background (B)"]
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"Cluster-level T/C/B Analysis — k={k} ({method})", fontsize=12, y=1.02)
    fig.tight_layout()
    path = output_dir / f"cluster_tcb_k{k}_{method}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_cluster_heatmap(P_act, labels, prompts, k, method, output_dir):
    """Prompt × cluster activation heatmap, sorted by correct_answer."""
    if not HAS_MPL:
        return

    sort_key = [p["_correct"] for p in prompts]
    sort_idx = np.argsort(sort_key)

    P_sorted = P_act[sort_idx]
    particles_sorted = [sort_key[i] for i in sort_idx]

    n_clusters = int(labels.max()) + 1
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 1.5), 7))

    im = ax.imshow(P_sorted, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean cluster activation")

    # Mark particle boundaries
    prev = None
    for i, par in enumerate(particles_sorted):
        if par != prev:
            ax.axhline(i - 0.5, color="white", lw=1.0, alpha=0.8)
            ax.text(-0.5, i + (particles_sorted.count(par) / 2),
                    par[:4], ha="right", va="center", fontsize=7, color="white")
            prev = par

    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Prompt (sorted by correct answer)", fontsize=11)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{c}" for c in range(n_clusters)])
    ax.set_yticks([])
    ax.set_title(f"Prompt × Cluster Activation — k={k} ({method})", fontsize=11)

    path = output_dir / f"cluster_heatmap_k{k}_{method}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_dendrogram(linkage, feat_ids, feat_meta, labels_k6, output_dir):
    """Feature dendrogram coloured by k=6 cluster assignment."""
    if not HAS_MPL:
        return

    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    leaf_colors = [cluster_colors[labels_k6[i] % len(cluster_colors)]
                   for i in range(len(feat_ids))]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_ids) * 0.25), 5))
    dend = sch.dendrogram(linkage,
                          leaf_label_func=lambda i: f"L{feat_ids[i][0]}_F{feat_ids[i][1]}",
                          leaf_rotation=90, leaf_font_size=6,
                          ax=ax, color_threshold=0)
    ax.set_title("Feature Clustering Dendrogram (Ward linkage, cosine distance)", fontsize=11)
    ax.set_ylabel("Distance")

    path = output_dir / "feature_dendrogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(layer_df, tcb_results, semantic_results, path_results, output_dir):
    lines = [
        "# Distributed Candidate-State Analysis — Feature Cluster Report",
        f"## {BEHAVIOUR} | {SPLIT}",
        "",
        "## Part E: Layer Transition (Form → Content)",
        "",
    ]

    if layer_df is not None and len(layer_df):
        early = layer_df[layer_df["layer"] <= 13]
        late  = layer_df[layer_df["layer"] >= 22]
        e_ari_p = early["ari_particle"].mean()
        e_ari_w = early["ari_wording"].mean()
        l_ari_p = late["ari_particle"].mean()
        l_ari_w = late["ari_wording"].mean()

        lines += [
            "| Layer group | mean ARI particle | mean ARI wording | ratio p/w |",
            "|---|---|---|---|",
            f"| Early (L10–13) | {e_ari_p:.3f} | {e_ari_w:.3f} | {e_ari_p/e_ari_w:.3f} |",
            f"| Late (L22–25)  | {l_ari_p:.3f} | {l_ari_w:.3f} | {l_ari_p/l_ari_w:.3f} |",
            "",
            "### Per-layer values",
            "",
            "| Layer | ARI particle | ARI wording | ratio |",
            "|---|---|---|---|",
        ]
        for _, r in layer_df.sort_values("layer").iterrows():
            ratio_str = f"{r['ratio_p_over_w']:.3f}" if not np.isnan(r["ratio_p_over_w"]) else "N/A"
            lines.append(
                f"| L{int(r['layer'])} | {r['ari_particle']:.3f} | "
                f"{r['ari_wording']:.3f} | {ratio_str} |"
            )
        lines += [""]

        # Evaluate at individual layer level (not group means, which average out extremes)
        final_layers   = layer_df[layer_df["layer"] >= 24]
        early_layers_df = layer_df[layer_df["layer"] <= 13]
        l24_25_p = final_layers["ari_particle"].mean() if len(final_layers) else 0
        l24_25_w = final_layers["ari_wording"].mean()  if len(final_layers) else 0
        early_w  = early_layers_df["ari_wording"].mean() if len(early_layers_df) else 0
        max_ratio = layer_df["ratio_p_over_w"].replace([np.inf, -np.inf], np.nan).max()

        if l24_25_p > l24_25_w and early_w > l24_25_w:
            transition = "**CONFIRMED**: form→content transition at L24–L25. " \
                         "Early/mid layers encode wording; L24–L25 switch to particle identity."
        elif max_ratio is not None and not np.isnan(max_ratio) and max_ratio >= 2.0:
            best_layer = int(layer_df.loc[layer_df["ratio_p_over_w"].idxmax(), "layer"])
            transition = f"**PARTIAL**: particle dominates wording at L{best_layer} " \
                         f"(ratio={max_ratio:.2f}). Weak/noisy overall."
        else:
            transition = "**ABSENT**: no clear form→content transition in graph features."
        lines += [f"**Transition assessment**: {transition}", ""]

    # Cluster-level T/C/B summary
    for k, method, tcb_df, sem_df in tcb_results:
        lines += [
            f"## Part D: Cluster T/C/B — k={k} ({method})",
            "",
        ]
        strong = tcb_df[tcb_df["strong_candidate"] == True]
        lines += [
            f"- Strong candidate clusters (T>C>B + p_TC<0.05 + p_CB<0.05): **{len(strong)}**",
            "",
        ]
        if len(strong):
            lines += [
                "| Cluster | Particle | mu_T | mu_C | mu_B | IPR | p_TC | p_CB |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for _, r in strong.iterrows():
                lines.append(
                    f"| C{int(r['cluster'])} | {r['particle']} | "
                    f"{r['mu_T']:.3f} | {r['mu_C']:.3f} | {r['mu_B']:.3f} | "
                    f"{r['IPR']:.3f} | {r['mw_p_TC']:.4f} | {r['mw_p_CB']:.4f} |"
                )
        # Neutron-specific (most testable)
        neutron_rows = tcb_df[tcb_df["particle"] == "neutron"].sort_values("competitor_presence", ascending=False)
        lines += [
            "",
            "### Neutron (only particle with full T/C/B groups):",
            "",
            "| Cluster | mu_T | mu_C | mu_B | IPR | ordering | p_TC | p_CB |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in neutron_rows.iterrows():
            lines.append(
                f"| C{int(r['cluster'])} | {r['mu_T']:.3f} | {r['mu_C']:.3f} | "
                f"{r['mu_B']:.3f} | {r['IPR']:.3f} | "
                f"{'✓' if r['ordering_T_gt_C_gt_B'] else '✗'} | "
                f"{r['mw_p_TC']:.4f} | {r['mw_p_CB']:.4f} |"
            )
        lines += [""]

    # Path overlap
    if path_results:
        k, method, path_df = path_results[0]
        lines += [
            f"## Part F: Cluster-to-Path Overlap (k={k}, {method})",
            "",
            "| Cluster | n_feats | circuit_feats | frac_circuit | any_causal_edges | path_coverage | mean_causal_δ |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in path_df.sort_values("frac_circuit", ascending=False).iterrows():
            lines.append(
                f"| C{int(r['cluster'])} | {int(r['n_features'])} | "
                f"{int(r['circuit_features_in_cluster'])} | {r['frac_circuit']:.2f} | "
                f"{int(r['any_endpoint_causal_edges'])} | {int(r['circuit_path_coverage'])} | "
                f"{r['mean_causal_strength']:.3f} |"
            )
        lines += [""]

    (output_dir / "report_cluster_analysis.md").write_text("\n".join(lines))
    print(f"  Report: report_cluster_analysis.md")


# ─── Dashboard JSON artefact ──────────────────────────────────────────────────

def write_dashboard_json(layer_df, tcb_results, semantic_results, path_results,
                         labels_best, feat_ids, output_dir):
    """Write a JSON artefact compatible with the offline UI / standalone viewer."""
    artifact = {
        "behaviour":     BEHAVIOUR,
        "split":         SPLIT,
        "generated":     str(pd.Timestamp.now()),
        "layer_transition": [],
        "clusters":      [],
    }

    if layer_df is not None:
        artifact["layer_transition"] = layer_df.to_dict("records")

    # Use the best k result (largest k with most strong candidates, else k=6)
    best_k, best_method, best_tcb, best_sem = tcb_results[0] if tcb_results else (None, None, None, None)
    if best_sem is not None:
        for _, sem_row in best_sem.iterrows():
            c = int(sem_row["cluster"])
            tcb_rows = best_tcb[best_tcb["cluster"] == c].to_dict("records") if best_tcb is not None else []
            cluster_feats = [{"layer": l, "feat_idx": f}
                             for i, (l, f) in enumerate(feat_ids) if labels_best[i] == c]
            artifact["clusters"].append({
                "id":               c,
                "n_features":       int(sem_row["n_features"]),
                "dominant_particle": sem_row["dominant_particle"],
                "layer_range":      sem_row["layer_range"],
                "particle_entropy": float(sem_row["particle_entropy"]),
                "mu_electron":      float(sem_row.get("mu_electron", float("nan"))),
                "mu_proton":        float(sem_row.get("mu_proton", float("nan"))),
                "mu_neutron":       float(sem_row.get("mu_neutron", float("nan"))),
                "mu_photon":        float(sem_row.get("mu_photon", float("nan"))),
                "in_circuit":       int(sem_row["in_circuit"]),
                "tcb_by_particle":  tcb_rows,
                "features":         cluster_feats,
            })

    out = output_dir / "cluster_analysis_dashboard.json"
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2, default=lambda x: None if np.isnan(x) else float(x))
    print(f"  Dashboard JSON: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",    default=BEHAVIOUR)
    ap.add_argument("--split",        default=SPLIT)
    ap.add_argument("--k",            nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--layer_subset", nargs="+",
                    choices=["early", "mid", "retrieval", "late", "all", "circuit"],
                    default=["all"])
    ap.add_argument("--no_plots",     action="store_true")
    ap.add_argument("--skip_transition", action="store_true",
                    help="Skip per-layer ARI computation (slow if features not cached)")
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    out   = paths["output_dir"]

    prompts        = load_prompts(paths)
    graph_features = load_graph_features(paths)
    circuit_feats  = load_circuit_features(paths)
    print(f"Prompts: {len(prompts)} (single-token)")
    print(f"Graph features: {len(graph_features)}, circuit: {len(circuit_feats)}")

    # ── Part A: activation matrix ──────────────────────────────────────────
    print("\n── Part A: Building feature activation matrix ──")
    layers_to_use = LAYERS
    X_act, X_attr, X_z, feat_ids, feat_meta = build_feature_matrix(
        graph_features, paths, args.behaviour, args.split, layers=layers_to_use
    )
    np.save(out / "feature_activation_matrix.npy", X_act)
    np.save(out / "feature_attribution_matrix.npy", X_attr)
    np.save(out / "feature_z_matrix.npy", X_z)
    feat_id_df = pd.DataFrame([
        {"idx": i, "layer": l, "feature_idx": f,
         "feature_id": f"L{l}_F{f}",
         "in_circuit": int((l, f) in circuit_feats),
         "specific_score": graph_features[(l, f)].get("specific_score", float("nan"))}
        for i, (l, f) in enumerate(feat_ids)
    ])
    feat_id_df.to_csv(out / "cluster_feature_index.csv", index=False)

    # ── Part B: feature clustering ─────────────────────────────────────────
    print("\n── Part B: Clustering features ──")
    cluster_results = cluster_features(X_z, feat_ids, feat_meta, k_values=args.k)

    # Save all cluster assignments
    for k in args.k:
        for method in ["kmeans", "hierarchical"]:
            labels = cluster_results[k][method]
            label_df = feat_id_df.copy()
            label_df[f"cluster_k{k}_{method}"] = labels
            label_df.to_csv(out / f"feature_clusters_k{k}_{method}.csv", index=False)

    if not args.no_plots and HAS_MPL:
        plot_dendrogram(
            cluster_results["hierarchical_linkage"],
            feat_ids, feat_meta,
            cluster_results[min(args.k)]["kmeans"],
            out,
        )

    # ── Parts C, D, F per k ───────────────────────────────────────────────
    tcb_results    = []
    semantic_results = []
    path_results   = []
    labels_best    = None

    for k in args.k:
        for method in ["kmeans"]:
            labels = cluster_results[k][method]
            if labels_best is None:
                labels_best = labels

            print(f"\n── Parts C/D/F: k={k} {method} ──")

            # C: semantic
            sem_df, P_act = cluster_semantic_analysis(
                labels, X_act, feat_ids, feat_meta, prompts
            )
            sem_df.to_csv(out / f"cluster_semantic_k{k}_{method}.csv", index=False)
            print("  Semantic summary:")
            print(sem_df[["cluster", "n_features", "dominant_particle",
                           "particle_entropy", "layer_range", "in_circuit"]].to_string(index=False))

            # D: T/C/B
            tcb_df = cluster_tcb_analysis(P_act, prompts, k)
            tcb_df.to_csv(out / f"cluster_tcb_k{k}_{method}.csv", index=False)
            n_strong = tcb_df["strong_candidate"].sum()
            n_order  = tcb_df["ordering_T_gt_C_gt_B"].sum()
            print(f"  T>C>B ordering: {n_order}/{len(tcb_df)}, strong: {n_strong}")

            tcb_results.append((k, method, tcb_df, sem_df))
            semantic_results.append((k, method, sem_df))

            # F: path overlap
            path_df = cluster_path_overlap(labels, feat_ids, paths, circuit_feats)
            path_df.to_csv(out / f"cluster_path_overlap_k{k}_{method}.csv", index=False)
            path_results.append((k, method, path_df))

            # Plots
            if not args.no_plots and HAS_MPL:
                plot_cluster_tcb(tcb_df, sem_df, k, method, out)
                plot_cluster_heatmap(P_act, labels, prompts, k, method, out)

    # ── Part E: layer transition ───────────────────────────────────────────
    layer_df = None
    if not args.skip_transition:
        print("\n── Part E: Per-layer ARI transition ──")
        layer_df = layer_transition_analysis(
            graph_features, paths, args.behaviour, args.split, prompts
        )
        layer_df.to_csv(out / "layer_transition_ari.csv", index=False)
        if not args.no_plots:
            plot_layer_transition(layer_df, out)

    # ── Report + dashboard JSON ────────────────────────────────────────────
    write_report(layer_df, tcb_results, semantic_results, path_results, out)
    write_dashboard_json(
        layer_df, tcb_results, semantic_results, path_results,
        labels_best, feat_ids, out
    )

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
