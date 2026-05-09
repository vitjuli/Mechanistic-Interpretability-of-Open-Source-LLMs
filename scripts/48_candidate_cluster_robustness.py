"""
Cluster robustness analysis: k = 4, 5, 6, 8.

Tests whether the key candidate-cluster findings from k=6 (photon identity cluster,
neutron/proton competition cluster) replicate across different clustering resolutions.

For each k:
  1. Re-cluster 77 graph features by activation profiles (77 × 447 matrix)
  2. Compute cluster semantic summary (particle means, entropy)
  3. Compute T/C/B statistics (neutron: full; photon: T/B; electron/proton: T/C)
  4. Identify photon-like and neutron/proton-competition clusters
  5. Compare to k=6 reference clusters using Jaccard overlap

Usage:
  python scripts/48_candidate_cluster_robustness.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BEHAVIOUR  = "physics_internal_candidate_selection_v2"
SPLIT      = "train"
PARTICLES  = ["electron", "proton", "neutron", "photon"]
N_ST       = 447
K_VALUES   = [4, 5, 6, 8]

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE  = Path("data")
ADIR  = BASE / "results" / "internal_candidate_analysis" / BEHAVIOUR
PATHS = {
    "prompts":   BASE / "prompts" / f"{BEHAVIOUR}_{SPLIT}.jsonl",
    "feat_act":  ADIR / "feature_activation_matrix.npy",
    "feat_idx":  ADIR / "cluster_feature_index.csv",
    "ref_clust": ADIR / "feature_clusters_k6_kmeans.csv",  # reference k=6
    "ref_tcb":   ADIR / "cluster_tcb_k6_kmeans.csv",
    "output":    ADIR,
}


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_prompts():
    with open(PATHS["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct"] = r["correct_answer"].strip()
        r["_pool"]    = set(r["implicit_candidate_pool"])
    return [r for r in rows if not r.get("multi_token_answer", False)]


def group_masks(prompts):
    n = len(prompts)
    masks = {}
    for p in PARTICLES:
        t = np.array([r["_correct"] == p for r in prompts])
        c = np.array([p in r["_pool"] and r["_correct"] != p for r in prompts])
        b = np.array([p not in r["_pool"] for r in prompts])
        masks[p] = {"target": t, "competitor": c, "background": b}
    return masks


# ─── Clustering ───────────────────────────────────────────────────────────────

def cluster_features(X_act, k, seed=42):
    """K-means on L2-normalised (n_feats, n_prompts) matrix."""
    X_norm = normalize(X_act, norm="l2")
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(X_norm)
    return labels


# ─── Per-cluster analysis ────────────────────────────────────────────────────

def mw_greater(a, b, min_n=5):
    if len(a) < min_n or len(b) < min_n:
        return float("nan"), float("nan")
    try:
        r = stats.mannwhitneyu(a, b, alternative="greater")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def analyse_clusters(labels, X_act, masks, feat_df, k):
    """
    Returns DataFrame with one row per (cluster, particle).
    Computes cluster score = mean activation of cluster features.
    """
    rows = []
    n_feats, n_prompts = X_act.shape

    for c in range(k):
        feat_mask = labels == c
        if feat_mask.sum() == 0:
            continue
        cluster_act = X_act[feat_mask].mean(axis=0)  # (N_ST,)

        # Feature set info
        feat_df_c = feat_df[feat_mask]
        layer_list = sorted(feat_df_c["layer"].unique())

        # Per-particle mean activations
        mu = {}
        for p in PARTICLES:
            m = masks[p]
            mu_T = float(cluster_act[m["target"]].mean())  if m["target"].sum()  else float("nan")
            mu_C = float(cluster_act[m["competitor"]].mean()) if m["competitor"].sum() else float("nan")
            mu_B = float(cluster_act[m["background"]].mean()) if m["background"].sum() else float("nan")
            mu[p] = {"T": mu_T, "C": mu_C, "B": mu_B}

        # Particle entropy (from target means)
        target_means = np.array([mu[p]["T"] for p in PARTICLES])
        target_means = np.nan_to_num(target_means, nan=0.0)
        shifted = target_means - target_means.min() + 1e-8
        probs   = shifted / shifted.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        dom_idx = int(np.argmax(target_means))
        sec_idx = int(np.argsort(target_means)[-2])

        # T/C/B stats per particle
        for p in PARTICLES:
            m   = masks[p]
            t_a = cluster_act[m["target"]]
            c_a = cluster_act[m["competitor"]]
            b_a = cluster_act[m["background"]]

            _, p_TC = mw_greater(t_a, c_a)
            _, p_CB = mw_greater(c_a, b_a)
            _, p_TB = mw_greater(t_a, b_a)

            mu_T = mu[p]["T"]; mu_C = mu[p]["C"]; mu_B = mu[p]["B"]
            ipr  = (mu_C / mu_T) if (not np.isnan(mu_C) and not np.isnan(mu_T) and mu_T > 0) else float("nan")

            ordering_TCB = bool(not np.isnan(mu_T + mu_C + mu_B) and mu_T > mu_C > mu_B)
            ordering_TB  = bool(not np.isnan(mu_T + mu_B) and mu_T > mu_B)

            rows.append({
                "k":              k,
                "cluster":        c,
                "particle":       p,
                "n_features":     int(feat_mask.sum()),
                "layer_min":      min(layer_list),
                "layer_max":      max(layer_list),
                "dom_particle":   PARTICLES[dom_idx],
                "sec_particle":   PARTICLES[sec_idx],
                "entropy":        entropy,
                "mu_T":           mu_T,
                "mu_C":           mu_C,
                "mu_B":           mu_B,
                "IPR":            ipr,
                "ordering_TCB":   ordering_TCB,
                "ordering_TB":    ordering_TB,
                "mw_p_TC":        p_TC,
                "mw_p_CB":        p_CB,
                "mw_p_TB":        p_TB,
                "sig_TC":         bool(not np.isnan(p_TC) and p_TC < 0.05),
                "sig_TB":         bool(not np.isnan(p_TB) and p_TB < 0.05),
                "sig_CB":         bool(not np.isnan(p_CB) and p_CB < 0.05),
            })

    return pd.DataFrame(rows)


# ─── Identify key cluster types ───────────────────────────────────────────────

def find_photon_cluster(df, k):
    """Find cluster most like C5 (photon): low entropy + photon dominant + T>>B sig."""
    g = df[df["particle"] == "photon"]
    best_score = -np.inf
    best_c = -1
    for c in range(k):
        row = g[g["cluster"] == c]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        # Score = low entropy + significant T>B + high mu_T
        entropy_score = 2.0 - row["entropy"]  # lower entropy = better
        sig_score     = 1.0 if row["sig_TB"] else 0.0
        mu_score      = float(row["mu_T"]) if not np.isnan(row["mu_T"]) else 0.0
        score = entropy_score + 2 * sig_score + 0.1 * mu_score
        if score > best_score:
            best_score = score
            best_c = c
    return best_c, best_score


def find_competition_cluster(df, k):
    """Find cluster most like C4 (neutron/proton competition):
    strong neutron T>C>B + strong proton non-hurt."""
    g_n = df[(df["particle"] == "neutron") & (df["ordering_TCB"] == True)]
    if len(g_n) == 0:
        # Fall back: largest mu_T for neutron
        g_n = df[df["particle"] == "neutron"].sort_values("mu_T", ascending=False)
    if len(g_n) == 0:
        return -1, float("nan")
    # Pick cluster with smallest p_TC (most significant)
    g_n = g_n.sort_values("mw_p_TC")
    return int(g_n.iloc[0]["cluster"]), float(g_n.iloc[0]["mw_p_TC"])


# ─── Jaccard overlap ──────────────────────────────────────────────────────────

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_pairwise_jaccard(labels_k, labels_ref, feat_df, k, k_ref):
    """For each cluster in k, find best-matching cluster in k_ref by Jaccard."""
    rows = []
    feat_indices = np.arange(len(feat_df))
    for c in range(k):
        set_c = set(feat_indices[labels_k == c])
        best_j = 0.0
        best_cr = -1
        for cr in range(k_ref):
            set_cr = set(feat_indices[labels_ref == cr])
            j = jaccard(set_c, set_cr)
            if j > best_j:
                best_j = j
                best_cr = cr
        rows.append({
            "k":         k,
            "cluster":   c,
            "k_ref":     k_ref,
            "best_match_cluster": best_cr,
            "jaccard":   best_j,
            "n_features": int((labels_k == c).sum()),
        })
    return pd.DataFrame(rows)


# ─── Figures ─────────────────────────────────────────────────────────────────

def make_figures(all_tcb, photon_clusters, comp_clusters, pairwise_jacc):
    if not HAS_MPL:
        return

    # ── Fig 1: Photon cluster properties across k ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = K_VALUES

    # Entropy of best photon cluster
    ph_entropy = []
    ph_p_tb    = []
    ph_mu_T    = []
    for k in ks:
        c, _ = photon_clusters[k]
        df_k = all_tcb[all_tcb["k"] == k]
        sub  = df_k[(df_k["cluster"] == c) & (df_k["particle"] == "photon")]
        if len(sub):
            ph_entropy.append(float(sub.iloc[0]["entropy"]))
            p_tb = sub.iloc[0]["mw_p_TB"]
            ph_p_tb.append(-np.log10(max(p_tb, 1e-30)) if not np.isnan(p_tb) else 0)
            ph_mu_T.append(float(sub.iloc[0]["mu_T"]))
        else:
            ph_entropy.append(float("nan"))
            ph_p_tb.append(0)
            ph_mu_T.append(float("nan"))

    axes[0].plot(ks, ph_entropy, "o-", c="#d62728", lw=2)
    axes[0].axhline(0.84, color="gray", ls="--", label="k=6 C5 entropy=0.844")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Cluster entropy")
    axes[0].set_title("Photon cluster entropy\n(lower = more particle-specific)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].bar(ks, ph_p_tb, color="#d62728", alpha=0.8)
    axes[1].axhline(2, color="gray", ls="--", label="p<0.01")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("-log10(p_TB)")
    axes[1].set_title("Photon T>B significance\n(higher = stronger evidence)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")

    axes[2].plot(ks, ph_mu_T, "o-", c="#d62728", lw=2)
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Photon target mean activation")
    axes[2].set_title("Photon cluster target mean\nby k")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Photon Identity Cluster — Robustness Across k", fontsize=12)
    fig.tight_layout()
    fig.savefig(PATHS["output"] / "cluster_robustness_photon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Cluster semantics heatmap (all k) ─────────────────────────────
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4 * len(K_VALUES), 5))
    for ax, k in zip(axes, K_VALUES):
        df_k = all_tcb[all_tcb["k"] == k]
        # Build (k, 4) matrix of target means
        mat = np.full((k, 4), float("nan"))
        for c in range(k):
            for qi, p in enumerate(PARTICLES):
                sub = df_k[(df_k["cluster"] == c) & (df_k["particle"] == p)]
                if len(sub):
                    mat[c, qi] = sub.iloc[0]["mu_T"]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels([p[:3] for p in PARTICLES])
        ax.set_yticks(range(k)); ax.set_yticklabels([f"C{c}" for c in range(k)])
        ax.set_title(f"k={k}")
        plt.colorbar(im, ax=ax, label="Target μ" if k == K_VALUES[-1] else "")
    fig.suptitle("Cluster Target Mean Activation by Particle\n(rows=clusters, cols=particles)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PATHS["output"] / "cluster_semantics_by_k.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Jaccard stability for photon and competition clusters ──────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ref_k = 6

    # For each non-ref k, Jaccard of best-matching photon cluster
    ph_jacc_vals  = []
    comp_jacc_vals = []
    non_ref_ks    = [k for k in K_VALUES if k != ref_k]

    for k in non_ref_ks:
        ref_photon_c = photon_clusters[ref_k][0]   # C5 in k=6
        ref_comp_c   = comp_clusters[ref_k][0]     # best competition cluster in k=6
        test_photon_c = photon_clusters[k][0]
        test_comp_c   = comp_clusters[k][0]

        sub_ph = pairwise_jacc[
            (pairwise_jacc["k"] == k) & (pairwise_jacc["cluster"] == test_photon_c)
        ]
        ph_j = float(sub_ph["jaccard"].iloc[0]) if len(sub_ph) else 0
        ph_jacc_vals.append(ph_j)

        sub_co = pairwise_jacc[
            (pairwise_jacc["k"] == k) & (pairwise_jacc["cluster"] == test_comp_c)
        ]
        co_j = float(sub_co["jaccard"].iloc[0]) if len(sub_co) else 0
        comp_jacc_vals.append(co_j)

    ax = axes[0]
    ax.bar(range(len(non_ref_ks)), ph_jacc_vals, tick_label=[f"k={k}" for k in non_ref_ks],
           color="#d62728", alpha=0.8, width=0.4)
    ax.axhline(0.5, color="gray", ls="--", label="Jaccard=0.5")
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard overlap with k=6 C5")
    ax.set_title("Photon cluster stability\n(Jaccard vs k=6 reference)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(range(len(non_ref_ks)), comp_jacc_vals, tick_label=[f"k={k}" for k in non_ref_ks],
           color="#2166ac", alpha=0.8, width=0.4)
    ax.axhline(0.5, color="gray", ls="--", label="Jaccard=0.5")
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard overlap with k=6 best competition cluster")
    ax.set_title("Competition cluster stability\n(Jaccard vs k=6 reference)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Key Cluster Stability Across k Values", fontsize=12)
    fig.tight_layout()
    fig.savefig(PATHS["output"] / "key_cluster_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("  Saved robustness figures.")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(all_tcb, photon_clusters, comp_clusters, summary_rows, pairwise_jacc):
    lines = [
        "# Cluster Robustness Report",
        f"## {BEHAVIOUR} | k ∈ {{4, 5, 6, 8}}",
        "",
        "## Summary",
        "",
    ]

    # Photon cluster robustness
    ph_sig_count = sum(
        1 for k in K_VALUES
        if photon_clusters[k][0] >= 0 and (
            all_tcb[
                (all_tcb["k"] == k) &
                (all_tcb["cluster"] == photon_clusters[k][0]) &
                (all_tcb["particle"] == "photon")
            ]["sig_TB"].any()
        )
    )
    lines += [
        f"### Photon identity cluster",
        f"Significant photon T>B (p<0.05) present across {ph_sig_count}/{len(K_VALUES)} k values.",
    ]
    if ph_sig_count >= 3:
        verdict_ph = "**ROBUST** — photon identity cluster replicates at most clustering resolutions"
    elif ph_sig_count >= 2:
        verdict_ph = "**PARTIALLY ROBUST** — photon identity cluster present at some k values"
    else:
        verdict_ph = "**UNSTABLE** — photon identity cluster is k=6 artifact"
    lines += [verdict_ph, ""]

    # Competition cluster robustness
    comp_ordering_count = sum(
        1 for k in K_VALUES
        if comp_clusters[k][0] >= 0 and (
            all_tcb[
                (all_tcb["k"] == k) &
                (all_tcb["cluster"] == comp_clusters[k][0]) &
                (all_tcb["particle"] == "neutron")
            ]["ordering_TCB"].any()
        )
    )
    lines += [
        f"### Neutron/proton competition cluster",
        f"T>C>B ordering for neutron present across {comp_ordering_count}/{len(K_VALUES)} k values.",
    ]
    if comp_ordering_count >= 3:
        verdict_comp = "**ROBUST** — neutron/proton competition cluster replicates"
    elif comp_ordering_count >= 2:
        verdict_comp = "**PARTIALLY ROBUST** — competition cluster present at some k values"
    else:
        verdict_comp = "**UNSTABLE** — competition cluster is k=6 artifact"
    lines += [verdict_comp, ""]

    # Per-k table
    lines += [
        "## Per-k Key Cluster Summary",
        "",
        "| k | Photon cluster | Photon entropy | p_TB | T>B sig | Competition cluster | Neutron T>C>B |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in K_VALUES:
        ph_c = photon_clusters[k][0]
        co_c = comp_clusters[k][0]
        ph_sub = all_tcb[(all_tcb["k"]==k) & (all_tcb["cluster"]==ph_c) & (all_tcb["particle"]=="photon")]
        co_sub = all_tcb[(all_tcb["k"]==k) & (all_tcb["cluster"]==co_c) & (all_tcb["particle"]=="neutron")]
        ph_ent = f"{ph_sub.iloc[0]['entropy']:.3f}" if len(ph_sub) else "—"
        ph_ptb = f"{ph_sub.iloc[0]['mw_p_TB']:.2e}" if len(ph_sub) else "—"
        ph_sig = "✓" if (len(ph_sub) and ph_sub.iloc[0]["sig_TB"]) else "✗"
        co_ord = "✓" if (len(co_sub) and co_sub.iloc[0]["ordering_TCB"]) else "✗"
        lines.append(
            f"| {k} | C{ph_c} | {ph_ent} | {ph_ptb} | {ph_sig} | C{co_c} | {co_ord} |"
        )

    # Jaccard table
    lines += [
        "",
        "## Jaccard Overlap with k=6 Reference Clusters",
        "",
        "| k | Best photon cluster Jaccard | Best competition Jaccard |",
        "|---|---|---|",
    ]
    ref_k = 6
    ref_photon_c = photon_clusters[ref_k][0]
    ref_comp_c   = comp_clusters[ref_k][0]
    for k in K_VALUES:
        if k == ref_k:
            lines.append(f"| {k} | — (reference) | — (reference) |")
            continue
        ph_c  = photon_clusters[k][0]
        co_c  = comp_clusters[k][0]
        sub_ph = pairwise_jacc[(pairwise_jacc["k"]==k) & (pairwise_jacc["cluster"]==ph_c)]
        sub_co = pairwise_jacc[(pairwise_jacc["k"]==k) & (pairwise_jacc["cluster"]==co_c)]
        ph_j = f"{sub_ph.iloc[0]['jaccard']:.3f}" if len(sub_ph) else "—"
        co_j = f"{sub_co.iloc[0]['jaccard']:.3f}" if len(sub_co) else "—"
        lines.append(f"| {k} | {ph_j} | {co_j} |")

    (PATHS["output"] / "cluster_robustness_report.md").write_text("\n".join(lines))
    print("  Report: cluster_robustness_report.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    X_act  = np.load(PATHS["feat_act"])     # (77, 447)
    feat_df = pd.read_csv(PATHS["feat_idx"])  # feature index
    prompts = load_prompts()
    masks   = group_masks(prompts)
    print(f"  Features: {X_act.shape}, prompts: {len(prompts)}")

    # Load k=6 reference labels
    ref_df = pd.read_csv(PATHS["ref_clust"])
    ref_labels = ref_df["cluster_k6_kmeans"].values

    all_tcb_dfs   = []
    all_labels    = {}
    photon_clusters = {}
    comp_clusters   = {}
    summary_rows  = []

    for k in K_VALUES:
        print(f"\n── k={k} ──")
        labels = cluster_features(X_act, k)
        all_labels[k] = labels

        tcb_df = analyse_clusters(labels, X_act, masks, feat_df, k)
        all_tcb_dfs.append(tcb_df)

        # Semantic summary (one row per cluster)
        sem_rows = []
        for c in range(k):
            sub = tcb_df[tcb_df["cluster"] == c]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            muts = {p: float(sub[sub["particle"]==p]["mu_T"].iloc[0]) if len(sub[sub["particle"]==p]) else float("nan")
                    for p in PARTICLES}
            sem_rows.append({
                "k": k, "cluster": c,
                "n_features": int(row["n_features"]),
                "layer_min": int(row["layer_min"]),
                "layer_max": int(row["layer_max"]),
                "entropy": float(row["entropy"]),
                "dom_particle": row["dom_particle"],
                **{f"mu_T_{p}": v for p, v in muts.items()},
            })
            print(f"  C{c}: n={row['n_features']}, L{row['layer_min']}-L{row['layer_max']}, "
                  f"dom={row['dom_particle']}, ent={row['entropy']:.3f}")
        summary_rows.extend(sem_rows)

        # Find key clusters
        ph_c, ph_score = find_photon_cluster(tcb_df, k)
        co_c, co_p     = find_competition_cluster(tcb_df, k)
        photon_clusters[k] = (ph_c, ph_score)
        comp_clusters[k]   = (co_c, co_p)

        ph_sub = tcb_df[(tcb_df["cluster"]==ph_c) & (tcb_df["particle"]=="photon")]
        co_sub = tcb_df[(tcb_df["cluster"]==co_c) & (tcb_df["particle"]=="neutron")]
        ph_ptb = ph_sub.iloc[0]["mw_p_TB"] if len(ph_sub) else float("nan")
        co_ord = bool(co_sub.iloc[0]["ordering_TCB"]) if len(co_sub) else False
        print(f"  Photon cluster: C{ph_c} (p_TB={ph_ptb:.2e})")
        print(f"  Competition cluster: C{co_c} (neutron T>C>B={co_ord})")

    all_tcb = pd.concat(all_tcb_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Pairwise Jaccard vs k=6
    ref_k = 6
    pairwise_dfs = []
    for k in K_VALUES:
        if k == ref_k:
            continue
        pj = compute_pairwise_jaccard(all_labels[k], ref_labels, feat_df, k, ref_k)
        pairwise_dfs.append(pj)
    pairwise_jacc = pd.concat(pairwise_dfs, ignore_index=True) if pairwise_dfs else pd.DataFrame()

    # Save
    all_tcb.to_csv(PATHS["output"] / "cluster_robustness_tcb.csv", index=False)
    summary_df.to_csv(PATHS["output"] / "cluster_robustness_summary.csv", index=False)
    if len(pairwise_jacc):
        pairwise_jacc.to_csv(PATHS["output"] / "cluster_robustness_pairwise_jaccard.csv", index=False)
    print("\nSaved CSVs.")

    # Figures
    make_figures(all_tcb, photon_clusters, comp_clusters, pairwise_jacc)

    # Report
    write_report(all_tcb, photon_clusters, comp_clusters, summary_rows, pairwise_jacc)

    # Quick console verdict
    print("\n=== ROBUSTNESS VERDICT ===")
    for k in K_VALUES:
        ph_c = photon_clusters[k][0]
        co_c = comp_clusters[k][0]
        ph_sub = all_tcb[(all_tcb["k"]==k) & (all_tcb["cluster"]==ph_c) & (all_tcb["particle"]=="photon")]
        co_sub = all_tcb[(all_tcb["k"]==k) & (all_tcb["cluster"]==co_c) & (all_tcb["particle"]=="neutron")]
        ph_sig = ph_sub.iloc[0]["sig_TB"] if len(ph_sub) else False
        co_ord = bool(co_sub.iloc[0]["ordering_TCB"]) if len(co_sub) else False
        print(f"  k={k}: photon_sig={ph_sig}, neutron_TCB={co_ord}")


if __name__ == "__main__":
    main()
