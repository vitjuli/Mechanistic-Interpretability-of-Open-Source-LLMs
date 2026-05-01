#!/usr/bin/env python3
"""
Script 23: Run the full clustering benchmark on physics_decay_type_probe features.

Methods implemented
-------------------
STANDARD
  hac_ward_abs / hac_complete_abs / hac_average_abs           HAC on abs cosine
  hac_ward_signed / hac_average_signed                         HAC on signed cosine
  spectral_abs_k{K}                                            Spectral on abs cosine affinity
  spectral_signed_k{K}                                         Spectral on signed cosine affinity
  kmeans_abs_k{K}                                              K-Means on L2-normalised abs prompt vectors
  kmeans_group_k{K}                                            K-Means on L2-normalised group abs vectors
  gmm_group_k{K}                                               GMM on group abs vectors
  nmf_k{K}                                                     NMF decomposition, argmax component
  cocluster_k{K}                                               SpectralCoclustering on feat×group matrix
  louvain_abs_{thr}                                            Louvain on abs cosine graph at threshold

GRAPH / BIPARTITE
  bipartite_louvain                                            Louvain on bipartite feature–group projection

NOVEL (developed specifically for this project)
  coimp_louvain                                                Co-importance Jaccard graph + Louvain
  signed_laplacian_k{K}                                        Signed Laplacian spectral clustering
  residual_kmeans_k{K}                                         KMeans on prompt residuals (after removing group mean)
  contrastive_k{K}                                             Cluster on (α_profile − β_profile) per feature
  multiview_consensus_k{K}                                     Multi-view spectral consensus (3 views, averaged co-cluster matrix)
  level_stability                                              Per-level KMeans(k=4), stored as 4 label vectors (not a single partition)

Outputs
-------
data/results/clustering/
  cluster_labels.csv     feature_id × method — label per method
  nmf_components.npy     NMF W matrices (saved per k)
  dendrogram_data.json   linkage matrices for HAC dendrograms
  run_log.csv            method × internal_score × runtime
"""
import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering, KMeans, SpectralBiclustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
CLU  = ROOT / "data/results/clustering"

K_VALS = [2, 3, 4, 5, 6]
LOUVAIN_THRS = [0.30, 0.50]
N_FEAT = 40
RANDOM_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────────
def load_inputs():
    def ln(name): return np.load(CLU / name)
    def lj(name):
        with open(CLU / name) as f: return json.load(f)

    X_abs    = ln("feat_prompt_abs.npy")       # (40, 470)
    X_signed = ln("feat_prompt_signed.npy")
    X_ga_abs = ln("feat_group_abs.npy")        # (40, 132)
    X_ga     = ln("feat_group_signed.npy")
    X_gsfr   = ln("feat_group_sfr.npy")
    W_abs    = ln("W_abs_cosine.npy")          # (40, 40)
    W_signed = ln("W_signed_cosine.npy")
    W_pearson= ln("W_pearson_abs.npy")
    W_coimp  = ln("W_coimportance.npy")
    delta    = ln("feat_delta_abs.npy")        # (40,)
    residual = ln("feat_residual.npy")         # (40, 470)
    feat_ids = lj("feat_ids.json")
    subsets  = lj("subsets.json")

    return dict(
        X_abs=X_abs, X_signed=X_signed,
        X_ga_abs=X_ga_abs, X_ga=X_ga, X_gsfr=X_gsfr,
        W_abs=W_abs, W_signed=W_signed, W_pearson=W_pearson, W_coimp=W_coimp,
        delta=delta, residual=residual,
        feat_ids=feat_ids, subsets=subsets,
    )


# ── Helper: run Louvain ───────────────────────────────────────────────────────────
def run_louvain(W: np.ndarray, threshold: float = 0.0, weight_attr: str = "weight") -> np.ndarray:
    import community as community_louvain
    import networkx as nx
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            w = float(W[i, j])
            if w > threshold:
                G.add_edge(i, j, weight=w)
    if G.number_of_edges() == 0:
        return np.zeros(n, dtype=int)
    part = community_louvain.best_partition(G, weight=weight_attr, random_state=RANDOM_SEED)
    return np.array([part[i] for i in range(n)], dtype=int)


# ── Silhouette with cosine distance ──────────────────────────────────────────────
def sil(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return float("nan")


# ── Method registry ──────────────────────────────────────────────────────────────
METHODS = {}

def method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


# ═══════════════════════════════════════════════════════════════════════════════════
# STANDARD METHODS
# ═══════════════════════════════════════════════════════════════════════════════════

def _hac(W: np.ndarray, method_str: str, k: int) -> np.ndarray:
    dist = np.clip(1.0 - W, 0, 2)
    np.fill_diagonal(dist, 0.0)
    dist_vec = squareform(dist, checks=False)
    Z = linkage(dist_vec, method=method_str)
    return fcluster(Z, k, criterion="maxclust") - 1, Z


for _link in ["ward", "complete", "average"]:
    for _view, _Wname in [("abs", "W_abs"), ("signed", "W_signed")]:
        if _link == "ward" and _view == "signed":
            continue  # Ward on signed can be problematic; skip
        _key = f"hac_{_link}_{_view}"
        def _make_hac(link=_link, Wname=_Wname, key=_key):
            def fn(d, extra):
                W = d[Wname]
                X_eval = normalize(d["X_abs"])  # features as samples (40, 470)
                results = {}
                for k in K_VALS:
                    labels, Z = _hac(W, link, k)
                    results[f"{key}_k{k}"] = (labels, sil(X_eval, labels))
                    if k == 4:
                        extra["linkage"][key] = Z.tolist()  # store for dendrogram
                return results
            METHODS[key] = fn
        _make_hac()


@method("spectral_abs")
def spectral_abs(d, extra):
    W = np.clip(d["W_abs"], 0, None)   # affinity must be ≥0
    X_eval = normalize(d["X_abs"])
    results = {}
    for k in K_VALS:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=RANDOM_SEED, n_init=10)
        labels = sc.fit_predict(W)
        results[f"spectral_abs_k{k}"] = (labels, sil(X_eval, labels))
    return results


@method("spectral_signed")
def spectral_signed(d, extra):
    # Shift signed cosine to [0,1] for spectral
    W = (d["W_signed"] + 1.0) / 2.0
    X_eval = normalize(d["X_abs"])
    results = {}
    for k in K_VALS:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=RANDOM_SEED, n_init=10)
        labels = sc.fit_predict(W)
        results[f"spectral_signed_k{k}"] = (labels, sil(X_eval, labels))
    return results


@method("kmeans_abs")
def kmeans_abs(d, extra):
    X = normalize(d["X_abs"])     # L2 per feature
    results = {}
    for k in K_VALS:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(X)
        results[f"kmeans_abs_k{k}"] = (labels, sil(X, labels))
    return results


@method("kmeans_group")
def kmeans_group(d, extra):
    X = normalize(d["X_ga_abs"])
    results = {}
    for k in K_VALS:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(X)
        results[f"kmeans_group_k{k}"] = (labels, sil(X, labels))
    return results


@method("gmm_group")
def gmm_group(d, extra):
    X = normalize(d["X_ga_abs"])
    results = {}
    for k in K_VALS[:4]:   # GMM unstable for large k with 40 samples
        try:
            gm = GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_SEED,
                                 covariance_type="diag")
            labels = gm.fit_predict(X)
            results[f"gmm_group_k{k}"] = (labels, sil(X, labels))
        except Exception:
            pass
    return results


@method("nmf")
def nmf_method(d, extra):
    X = d["X_abs"].copy()
    X = np.clip(X, 0, None)   # NMF requires non-negative
    results = {}
    for k in K_VALS[:5]:
        try:
            model = NMF(n_components=k, random_state=RANDOM_SEED,
                        max_iter=500, init="nndsvda")
            W = model.fit_transform(X)   # (40, k) — feature scores per component
            labels = np.argmax(W, axis=1)
            extra.setdefault("nmf", {})[f"k{k}"] = {
                "W": W.tolist(),   # feature loadings
                "H": model.components_.tolist(),  # component profiles
                "err": float(model.reconstruction_err_),
            }
            results[f"nmf_k{k}"] = (labels, sil(normalize(W), labels))
        except Exception:
            pass
    return results


@method("cocluster")
def cocluster(d, extra):
    X = d["X_ga_abs"].copy()
    X = np.clip(X, 0, None)
    results = {}
    for k in K_VALS[:5]:
        try:
            bc = SpectralBiclustering(n_clusters=k, method="log",
                                   random_state=RANDOM_SEED)
            bc.fit(X + 1e-6)  # log needs positive
            labels = bc.row_labels_   # feature cluster labels
            results[f"cocluster_k{k}"] = (labels, sil(normalize(X), labels))
        except Exception:
            pass
    return results


@method("louvain_abs")
def louvain_abs_method(d, extra):
    X_eval = normalize(d["X_abs"])
    results = {}
    for thr in LOUVAIN_THRS:
        try:
            labels = run_louvain(d["W_abs"], threshold=thr)
            results[f"louvain_abs_t{int(thr*100):02d}"] = (
                labels, sil(X_eval, labels))
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# GRAPH / BIPARTITE METHODS
# ═══════════════════════════════════════════════════════════════════════════════════

@method("bipartite_louvain")
def bipartite_louvain(d, extra):
    """
    Build feature–group bipartite graph, project to feature–feature via
    shared groups (weighted), apply Louvain.
    """
    import community as community_louvain
    import networkx as nx
    X = d["X_ga_abs"]
    n_feat, n_group = X.shape
    # Feature–feature co-group similarity: dot product of abs rows
    W_proj = X @ X.T
    np.fill_diagonal(W_proj, 0)
    # Normalise to [0,1]
    mx = W_proj.max()
    if mx > 0:
        W_proj = W_proj / mx
    try:
        labels = run_louvain(W_proj, threshold=0.1)
        return {"bipartite_louvain": (labels, sil(normalize(X), labels))}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════════
# NOVEL METHODS
# ═══════════════════════════════════════════════════════════════════════════════════

@method("coimp_louvain")
def coimp_louvain(d, extra):
    """
    Novel: co-importance Jaccard graph (shared top-k prompt sets) + Louvain.
    Captures 'who matters together' at the level of key prompts, not
    correlation of magnitudes — these are genuinely different structures.
    """
    try:
        labels = run_louvain(d["W_coimp"], threshold=0.15)
        return {"coimp_louvain": (labels, sil(normalize(d["X_abs"]), labels))}
    except Exception:
        return {}


@method("signed_laplacian")
def signed_laplacian(d, extra):
    """
    Novel: Signed Laplacian spectral clustering.

    Build the signed similarity matrix W_s (can contain negative entries).
    Compute the Signed Laplacian:  L_s = |D| - W_s  where D[i,i] = sum_j |W_s[i,j]|.
    The eigenvectors of L_s respect signed relationships:
      positive W → same cluster (cooperation), negative W → different clusters (competition).
    Apply K-Means on first k eigenvectors.

    This is motivated by Harary balance theory and signed graph spectral theory
    (Kunegis et al. 2010) applied to causal feature profiles.
    """
    from scipy.linalg import eigh
    W = d["W_signed_cosine"] if "W_signed_cosine" in d else d["W_signed"]
    D = np.diag(np.abs(W).sum(axis=1))
    L_s = D - W
    # Symmetric → eigh gives real eigenvalues
    try:
        eigvals, eigvecs = eigh(L_s)
    except Exception:
        return {}
    results = {}
    for k in K_VALS:
        V = eigvecs[:, :k]  # first k eigenvectors
        V = normalize(V, norm="l2")
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(V)
        results[f"signed_laplacian_k{k}"] = (labels, sil(normalize(d["X_abs"]), labels))
    extra["signed_laplacian_eigvals"] = eigvals[:10].tolist()
    return results


@method("residual_kmeans")
def residual_kmeans(d, extra):
    """
    Novel: cluster features on their prompt-level residuals after removing
    group-mean effects. This isolates the idiosyncratic component:
    which features respond to the same within-group prompt variation.
    """
    X = d["residual"]   # (40, 470)
    X_norm = normalize(X)
    results = {}
    for k in K_VALS[:5]:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_norm)
        results[f"residual_kmeans_k{k}"] = (labels, sil(X_norm, labels))
    return results


@method("contrastive")
def contrastive_method(d, extra):
    """
    Novel: cluster features on their contrastive α–β signed profiles.

    For each feature, compute the difference vector:
      Δ[f, :] = mean_effect_over_α_prompts[f] − mean_effect_over_β_prompts[f]
    using the signed effect matrix split by answer type.

    This makes the clustering sensitive to how features *differentially* serve
    α vs β routes — finding features that are strategically asymmetric, not just
    universally active.
    """
    X_signed = d["X_signed"]
    subsets  = d["subsets"]
    alpha_j  = subsets["answer"]["alpha"]
    beta_j   = subsets["answer"]["beta"]

    X_alpha = X_signed[:, alpha_j]   # (40, n_alpha)
    X_beta  = X_signed[:, beta_j]    # (40, n_beta)

    # Contrastive profile: mean α effect vs mean β effect per feature
    mu_alpha = X_alpha.mean(axis=1)    # (40,)
    mu_beta  = X_beta.mean(axis=1)
    delta_signed = mu_alpha - mu_beta  # (40,) scalar per feature

    # Also build a richer 2D vector: [mu_alpha, mu_beta]
    contrast_2d = np.stack([mu_alpha, mu_beta], axis=1)   # (40, 2)

    # And a higher-dim version: α-profile vs β-profile in group space
    X_ga = d["X_ga"]
    grp_ans = None  # will build via group annotation
    # For now cluster on [alpha_abs, beta_abs] scalars + layer
    # plus the group-level abs vectors for α and β groups separately
    with open(CLU / "group_ids.json") as f:
        group_ids = json.load(f)
    with open(CLU / "prompt_labels.json") as f:
        prompt_labels = json.load(f)
    # Build group → answer mapping
    grp_ans_map = {}
    for pid, info in prompt_labels.items():
        gid = info["group_id"]
        if gid not in grp_ans_map:
            grp_ans_map[gid] = info["correct_answer"]
    alpha_grp_j = [gi for gi, g in enumerate(group_ids) if grp_ans_map.get(g) == "alpha"]
    beta_grp_j  = [gi for gi, g in enumerate(group_ids) if grp_ans_map.get(g) == "beta"]

    X_ga_alpha = d["X_ga_abs"][:, alpha_grp_j]    # (40, n_alpha_groups)
    X_ga_beta  = d["X_ga_abs"][:, beta_grp_j]

    mu_grp_alpha = X_ga_alpha.mean(axis=1)
    mu_grp_beta  = X_ga_beta.mean(axis=1)
    contrast_grp = np.stack([mu_grp_alpha, mu_grp_beta], axis=1)

    results = {}
    for k in K_VALS:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        # Cluster on group-level contrastive profile
        X_c = normalize(contrast_grp)
        labels = km.fit_predict(X_c)
        results[f"contrastive_k{k}"] = (labels, sil(X_c, labels))

    extra["contrastive"] = {
        "delta_signed_per_feat": delta_signed.tolist(),
        "mu_alpha": mu_grp_alpha.tolist(),
        "mu_beta":  mu_grp_beta.tolist(),
    }
    return results


@method("multiview_consensus")
def multiview_consensus(d, extra):
    """
    Novel: multi-view spectral consensus clustering.

    Run spectral clustering (k=4) on three independent feature-feature views:
      V1: abs cosine similarity (captures magnitude co-variation)
      V2: signed cosine similarity (captures directional alignment)
      V3: co-importance Jaccard (captures shared top-k prompt sets)

    For each view, build a binary co-cluster matrix B_v[i,j] = 1 if same cluster.
    Consensus matrix C[i,j] = mean(B_v[i,j]) ∈ [0,1].
    Apply HAC on (1 − C) to get a consensus partition.

    This removes method-specific artefacts: only clusters stable across all
    three views survive.
    """
    from sklearn.cluster import SpectralClustering
    views = {
        "abs":    np.clip(d["W_abs"],    0, None),
        "signed": (d["W_signed"] + 1)/2,
        "coimp":  d["W_coimp"],
    }
    results = {}
    for k in K_VALS:
        B_sum = np.zeros((N_FEAT, N_FEAT), dtype=np.float32)
        n_views = 0
        for vname, W in views.items():
            try:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                        random_state=RANDOM_SEED, n_init=10)
                lbl = sc.fit_predict(W)
                B = (lbl[:, None] == lbl[None, :]).astype(np.float32)
                B_sum += B
                n_views += 1
            except Exception:
                pass
        if n_views == 0:
            continue
        C = B_sum / n_views   # consensus matrix
        # HAC on consensus
        dist = np.clip(1.0 - C, 0, 1)
        np.fill_diagonal(dist, 0)
        dist_vec = squareform(dist, checks=False)
        Z = linkage(dist_vec, method="average")
        labels = fcluster(Z, k, criterion="maxclust") - 1
        results[f"multiview_consensus_k{k}"] = (labels, sil(normalize(d["X_abs"]), labels))
        if k == 4:
            extra.setdefault("consensus_matrices", {})[f"k{k}"] = C.tolist()
    return results


@method("level_stability")
def level_stability_method(d, extra):
    """
    Novel: Level-conditioned clustering stability analysis.

    For each level L ∈ {1, 2, 3, AUX}, run KMeans(k=4) on the feature vectors
    restricted to only level-L prompts.  This gives 4 independent partitions.

    Compute pairwise NMI between all partition pairs.  Features that cluster
    identically regardless of level are level-agnostic (candidate latent-state
    features).  Features that split differently at L3 vs L1/L2 are route-specific.

    Returns the L1 partition as the canonical label set for comparison;
    stores all four partitions and pairwise NMIs in extra.
    """
    from sklearn.metrics import normalized_mutual_info_score
    X_abs  = d["X_abs"]
    subsets = d["subsets"]
    k = 4
    level_labels = {}
    for lv in ["1","2","3","AUX"]:
        idx = subsets["level"][lv]
        if len(idx) < 20:
            continue
        X_lv = X_abs[:, idx]
        X_lv_norm = normalize(X_lv)
        km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_SEED)
        lbls = km.fit_predict(X_lv_norm)
        level_labels[lv] = lbls.tolist()

    # Pairwise NMI
    nmi_table = {}
    lvs = list(level_labels.keys())
    for i in range(len(lvs)):
        for j in range(i+1, len(lvs)):
            la, lb = np.array(level_labels[lvs[i]]), np.array(level_labels[lvs[j]])
            nmi = float(normalized_mutual_info_score(la, lb))
            nmi_table[f"{lvs[i]}_vs_{lvs[j]}"] = nmi

    extra["level_stability"] = {
        "labels": level_labels,
        "nmi": nmi_table,
        "k": k,
    }
    # Compute feature-level stability score: how often do they land in the same cluster?
    # Stability = fraction of level pairs where features i,j are co-clustered consistently
    # For labelling: use L1 as canonical
    if "1" in level_labels:
        labels_l1 = np.array(level_labels["1"])
        sil_val = sil(normalize(X_abs[:, subsets["level"]["1"]]), labels_l1)
        return {"level_stability_L1": (labels_l1, sil_val)}
    return {}


# ═══════════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════

def run_all():
    print("Loading inputs...")
    d = load_inputs()
    feat_ids = d["feat_ids"]

    all_labels = {fid: {} for fid in feat_ids}
    run_log    = []
    extra      = {"linkage": {}}

    for mname, fn in METHODS.items():
        t0 = time.time()
        print(f"  Running {mname}...")
        try:
            results = fn(d, extra)
        except Exception as ex:
            print(f"    FAILED: {ex}")
            results = {}
        dt = time.time() - t0

        for key, (labels, sil_val) in results.items():
            for i, fid in enumerate(feat_ids):
                all_labels[fid][key] = int(labels[i])
            n_clusters = len(set(labels.tolist()))
            run_log.append({
                "method": key,
                "n_clusters": n_clusters,
                "silhouette": sil_val,
                "runtime_s":  round(dt, 2),
            })
            print(f"    → {key}: k={n_clusters}, sil={sil_val:.3f}" if not np.isnan(sil_val)
                  else f"    → {key}: k={n_clusters}")

    # ── Save cluster labels table ─────────────────────────────────────────────────
    labels_df = pd.DataFrame.from_dict(all_labels, orient="index")
    labels_df.index.name = "feature_id"
    labels_df.to_csv(CLU / "cluster_labels.csv")
    print(f"\nSaved cluster_labels.csv: {labels_df.shape}")

    # ── Save run log ─────────────────────────────────────────────────────────────
    log_df = pd.DataFrame(run_log)
    log_df.to_csv(CLU / "run_log.csv", index=False)

    # ── Save extra data (NMF components, consensus matrices, etc.) ───────────────
    with open(CLU / "method_extra.json", "w") as f:
        # Convert any non-serialisable objects
        def clean(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: clean(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        json.dump(clean(extra), f, indent=2)

    print("Done.\n")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    run_all()
