"""
Cluster prompt-level feature vectors and test whether clusters align with
particle identity rather than wording family or syntax.

Feature vectors are built from transcoder activations at chosen layer groups.
Clustering is evaluated against three label types:
  correct_answer   — particle identity (electron/proton/neutron/photon)
  filter_property  — physical filter (8 values)
  wording_family   — phrasing style (F1/F2/F3/F4)

If clusters align with particle identity but not wording family, this is
strong evidence for content-driven (not form-driven) internal representations.

Usage:
  python scripts/43_candidate_clustering.py
  python scripts/43_candidate_clustering.py --layer_group all circuit_layers late
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

BEHAVIOUR = "physics_internal_candidate_selection_v2"
SPLIT     = "train"
LAYERS    = list(range(10, 26))
PARTICLES = ["electron", "proton", "neutron", "photon"]

LAYER_GROUPS = {
    "all":            list(range(10, 26)),
    "early":          list(range(10, 14)),
    "mid":            list(range(14, 22)),
    "late":           list(range(22, 26)),
    "retrieval":      [19, 20, 21],
    "circuit_layers": [10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def get_paths(behaviour, split):
    base = Path("data")
    return {
        "prompts":    base / "prompts" / f"{behaviour}_{split}.jsonl",
        "graph_json": base / "results" / "attribution_graphs" / behaviour
                           / f"attribution_graph_{split}_n120_roleaware.json",
        "circuit_json": base / "results" / f"circuits_{behaviour}_{split}_roleaware.json",
        "feature_dir": base / "results" / "transcoder_features",
        "output_dir":  base / "results" / "internal_candidate_analysis" / behaviour,
    }


def load_prompts(paths):
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct_stripped"] = r["correct_answer"].strip()
    return [p for p in rows if not p.get("multi_token_answer", False)]


def load_graph_feature_indices(paths):
    with open(paths["graph_json"]) as f:
        g = json.load(f)
    by_layer: dict[int, list[int]] = {}
    for n in g["nodes"]:
        if n.get("type") == "feature":
            by_layer.setdefault(n["layer"], []).append(n["feature_idx"])
    return by_layer


def load_circuit_feature_indices(paths):
    with open(paths["circuit_json"]) as f:
        c = json.load(f)
    by_layer: dict[int, list[int]] = {}
    for node_id in c.get("circuit", {}).get("feature_nodes", []):
        if "_F" in node_id and node_id.startswith("L"):
            l, f = node_id.split("_F")
            by_layer.setdefault(int(l[1:]), []).append(int(f))
    return by_layer


def load_layer_features(behaviour, split, layer, feature_dir):
    d   = feature_dir / f"layer_{layer}"
    idx = d / f"{behaviour}_{split}_top_k_indices.npy"
    val = d / f"{behaviour}_{split}_top_k_values.npy"
    if not idx.exists():
        return None
    return np.load(idx), np.load(val)


def get_activations_for_features(indices, values, feat_idx_list):
    """Returns (n_prompts, n_features) dense matrix."""
    n = indices.shape[0]
    out = np.zeros((n, len(feat_idx_list)), dtype=np.float32)
    for j, fidx in enumerate(feat_idx_list):
        rows, cols = np.where(indices == fidx)
        out[rows, j] = values[rows, cols]
    return out


# ─── Build feature matrix ─────────────────────────────────────────────────────

def build_feature_matrix(behaviour, split, paths, layers, use_circuit_only=False, n_single_token=None):
    """
    Returns (X [n_prompts, n_features], feature_names [list]) where each column
    is the activation of one (layer, feature_idx) pair.
    n_single_token: if set, slices arrays to first n rows (excludes multi-token prompts).
    """
    graph_feats   = load_graph_feature_indices(paths)
    circuit_feats = load_circuit_feature_indices(paths)

    all_cols = []  # list of np.ndarray [n_prompts]
    col_names = []
    missing_layers = []

    for layer in layers:
        result = load_layer_features(behaviour, split, layer, paths["feature_dir"])
        if result is None:
            missing_layers.append(layer)
            continue
        indices, values = result
        # Slice to single-token prompts (always first n_single_token rows by dataset design)
        if n_single_token is not None:
            indices = indices[:n_single_token]
            values  = values[:n_single_token]
        n_prompts = indices.shape[0]

        feat_list = []
        if use_circuit_only:
            feat_list = circuit_feats.get(layer, [])
        else:
            feat_list = graph_feats.get(layer, [])
            # Add circuit features not already in graph
            for f in circuit_feats.get(layer, []):
                if f not in feat_list:
                    feat_list.append(f)

        if not feat_list:
            continue

        acts = get_activations_for_features(indices, values, feat_list)
        for j, fidx in enumerate(feat_list):
            all_cols.append(acts[:, j])
            col_names.append(f"L{layer}_F{fidx}")

    if missing_layers:
        print(f"[WARN] Missing layers: {missing_layers}")

    if not all_cols:
        return None, []

    X = np.stack(all_cols, axis=1)  # (n_prompts, n_features)
    return X, col_names


# ─── Evaluation metrics ───────────────────────────────────────────────────────

def cluster_purity(labels_true, labels_pred):
    """Cluster purity: fraction of samples in the majority class per cluster."""
    total = len(labels_true)
    purity = 0.0
    for cluster_id in np.unique(labels_pred):
        mask = labels_pred == cluster_id
        if not mask.any():
            continue
        cluster_labels = np.array(labels_true)[mask]
        counts = np.bincount(cluster_labels)
        purity += counts.max()
    return purity / total


def encode_labels(labels):
    unique = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[l] for l in labels]), unique


def evaluate_clustering(labels_pred, prompts, label_key):
    labels_str = [p[label_key] for p in prompts]
    labels_int, unique = encode_labels(labels_str)
    ari   = adjusted_rand_score(labels_int, labels_pred)
    nmi   = normalized_mutual_info_score(labels_int, labels_pred)
    pur   = cluster_purity(labels_int, labels_pred)
    return {"ari": ari, "nmi": nmi, "purity": pur, "n_classes": len(unique), "classes": unique}


# ─── Silhouette (lazy import) ─────────────────────────────────────────────────

def silhouette(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


# ─── Main clustering ──────────────────────────────────────────────────────────

def run_clustering(behaviour, split, paths, layer_group_name, use_circuit_only=False):
    prompts = load_prompts(paths)
    n       = len(prompts)

    layers = LAYER_GROUPS.get(layer_group_name, LAYERS)
    print(f"\nLayer group '{layer_group_name}': {layers}")

    X, col_names = build_feature_matrix(behaviour, split, paths, layers, use_circuit_only, n_single_token=n)
    if X is None:
        print(f"  No feature data — skipping")
        return None

    # Sparsity stats
    nonzero_frac = float((X > 0).mean())
    print(f"  Feature matrix: {X.shape}, nonzero={nonzero_frac:.3f}")

    # L2 normalise rows for cosine-based clustering
    X_norm = normalize(X, norm="l2")

    # k-means with k=4 (one per particle)
    km4 = KMeans(n_clusters=4, random_state=42, n_init=20)
    km4_labels = km4.fit_predict(X_norm)

    # Also try k=8 (could align with filter_property if more structure)
    km8 = KMeans(n_clusters=8, random_state=42, n_init=20)
    km8_labels = km8.fit_predict(X_norm)

    # PCA for 2D
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X_norm)
    var_ratio = float(pca.explained_variance_ratio_[:2].sum())
    print(f"  PCA top-2 variance explained: {var_ratio:.3f}")

    rows = []
    for k, labels_pred in [(4, km4_labels), (8, km8_labels)]:
        for label_key, label_name in [
            ("_correct_stripped", "particle_identity"),
            ("filter_property",   "filter_property"),
            ("wording_family",    "wording_family"),
        ]:
            metrics = evaluate_clustering(labels_pred, prompts, label_key)
            sil     = silhouette(X_pca[:, :10], labels_pred)
            rows.append({
                "layer_group":     layer_group_name,
                "use_circuit_only": use_circuit_only,
                "k":               k,
                "label_type":      label_name,
                "ari":             metrics["ari"],
                "nmi":             metrics["nmi"],
                "purity":          metrics["purity"],
                "silhouette":      sil,
                "n_features":      X.shape[1],
                "n_prompts":       n,
            })
            print(f"  k={k} | {label_name}: ARI={metrics['ari']:.3f}, "
                  f"NMI={metrics['nmi']:.3f}, purity={metrics['purity']:.3f}")

    # Cluster assignment per prompt (k=4)
    assignment_df = pd.DataFrame({
        "prompt_idx":      range(n),
        "correct_answer":  [p["_correct_stripped"] for p in prompts],
        "filter_property": [p.get("filter_property", "") for p in prompts],
        "wording_family":  [p.get("wording_family", "") for p in prompts],
        "cluster_k4":      km4_labels,
        "cluster_k8":      km8_labels,
        "pca_x":           X_pca[:, 0],
        "pca_y":           X_pca[:, 1],
    })

    return pd.DataFrame(rows), assignment_df, X_pca, km4_labels, prompts


# ─── Plotting ─────────────────────────────────────────────────────────────────

def make_cluster_plots(X_pca, km4_labels, prompts, layer_group_name, output_dir):
    if not MATPLOTLIB:
        return
    particle_colors = {
        "electron": "#1f77b4", "proton": "#ff7f0e",
        "neutron": "#2ca02c",  "photon": "#d62728"
    }
    family_colors = {
        "F1_direct_implicit": "#9467bd",
        "F2_contextual_implicit": "#8c564b",
        "F3_process_implicit": "#e377c2",
        "F4_contrast_implicit": "#7f7f7f"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = X_pca[:, 0]
    y = X_pca[:, 1]

    # Plot 1: colour by particle identity
    ax = axes[0]
    for particle in PARTICLES:
        mask = np.array([p["_correct_stripped"] == particle for p in prompts])
        ax.scatter(x[mask], y[mask], label=particle, alpha=0.6, s=18,
                   c=particle_colors.get(particle, "gray"))
    ax.set_title(f"PCA by particle identity\n(layer group: {layer_group_name})")
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")

    # Plot 2: colour by k=4 cluster
    ax = axes[1]
    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(4):
        mask = km4_labels == k
        ax.scatter(x[mask], y[mask], label=f"Cluster {k}", alpha=0.6, s=18,
                   c=cluster_colors[k])
    ax.set_title(f"PCA by k-means cluster (k=4)")
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")

    # Plot 3: colour by wording family
    ax = axes[2]
    families = list(set(p.get("wording_family", "other") for p in prompts))
    cmap     = plt.cm.get_cmap("tab10", len(families))
    for i, fam in enumerate(sorted(families)):
        mask = np.array([p.get("wording_family", "") == fam for p in prompts])
        ax.scatter(x[mask], y[mask], label=fam, alpha=0.6, s=18, c=[cmap(i)])
    ax.set_title(f"PCA by wording family")
    ax.legend(fontsize=7)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(output_dir / f"candidate_pca_{layer_group_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved PCA plot: candidate_pca_{layer_group_name}.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",     default=BEHAVIOUR)
    ap.add_argument("--split",         default=SPLIT)
    ap.add_argument("--layer_groups",  nargs="+",
                    default=["all", "early", "mid", "late", "retrieval", "circuit_layers"],
                    choices=list(LAYER_GROUPS.keys()))
    ap.add_argument("--circuit_only",  action="store_true")
    ap.add_argument("--no_plots",      action="store_true")
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split)
    has_any = any(
        (paths["feature_dir"] / f"layer_{l}" / f"{args.behaviour}_{args.split}_top_k_indices.npy").exists()
        for l in LAYERS
    )
    if not has_any:
        print(f"[ERROR] No transcoder feature files for {args.behaviour}_{args.split}")
        sys.exit(1)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for lg in args.layer_groups:
        res = run_clustering(args.behaviour, args.split, paths, lg, args.circuit_only)
        if res is None:
            continue
        metrics_df, assignment_df, X_pca, km4_labels, prompts = res
        all_metrics.append(metrics_df)

        assignment_df.to_csv(
            paths["output_dir"] / f"candidate_cluster_assignments_{lg}.csv", index=False
        )
        if not args.no_plots:
            make_cluster_plots(X_pca, km4_labels, prompts, lg, paths["output_dir"])

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(paths["output_dir"] / "candidate_clustering_metrics.csv", index=False)
        print(f"\nSaved clustering metrics to {paths['output_dir']}/candidate_clustering_metrics.csv")

        # Console summary
        print("\n" + "=" * 60)
        print("CLUSTERING SUMMARY (k=4 vs particle identity)")
        print("=" * 60)
        particle_metrics = combined[
            (combined["k"] == 4) & (combined["label_type"] == "particle_identity")
        ].sort_values("ari", ascending=False)
        print(particle_metrics[["layer_group", "ari", "nmi", "purity", "silhouette"]].to_string(index=False))

        # Write report
        report_lines = [
            "# Candidate Clustering Analysis Report",
            f"## {BEHAVIOUR} | {SPLIT}",
            "",
            "## Question: do feature activation clusters align with particle identity?",
            "",
            "### k=4 clustering vs particle identity (best → worst by ARI):",
            "",
            "| Layer group | n_features | ARI | NMI | Purity | Silhouette |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in particle_metrics.iterrows():
            report_lines.append(
                f"| {r['layer_group']} | {int(r['n_features'])} | "
                f"{r['ari']:.3f} | {r['nmi']:.3f} | {r['purity']:.3f} | "
                f"{r['silhouette']:.3f} |"
            )
        report_lines += [
            "",
            "### Comparison: particle vs wording family (k=4, 'all' layers):",
            "",
        ]
        sub = combined[(combined["k"] == 4) & (combined["layer_group"] == "all")]
        for _, r in sub.iterrows():
            report_lines.append(
                f"- **{r['label_type']}**: ARI={r['ari']:.3f}, "
                f"NMI={r['nmi']:.3f}, purity={r['purity']:.3f}"
            )
        report_lines += [""]
        sub_particle = combined[
            (combined["k"] == 4) & (combined["label_type"] == "particle_identity")
            & (combined["layer_group"] == "all")
        ]
        sub_family = combined[
            (combined["k"] == 4) & (combined["label_type"] == "wording_family")
            & (combined["layer_group"] == "all")
        ]
        if len(sub_particle) and len(sub_family):
            p_ari = sub_particle.iloc[0]["ari"]
            f_ari = sub_family.iloc[0]["ari"]
            if p_ari > f_ari + 0.1:
                interp = ("**Content-driven**: clusters align with particle identity more than "
                          "wording family — evidence for internal candidate representations")
            elif abs(p_ari - f_ari) < 0.05:
                interp = ("**Mixed**: clusters align equally with particle and wording family — "
                          "cannot distinguish content from form")
            else:
                interp = ("**Form-driven**: clusters align with wording family more than "
                          "particle identity — features may be syntactic, not semantic")
            report_lines += [f"### Interpretation: {interp}", ""]

        (paths["output_dir"] / "candidate_clustering_report.md").write_text("\n".join(report_lines))
        print(f"Report written to {paths['output_dir']}/candidate_clustering_report.md")


if __name__ == "__main__":
    main()
