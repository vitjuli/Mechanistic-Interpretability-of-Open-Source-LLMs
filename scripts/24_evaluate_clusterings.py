#!/usr/bin/env python3
"""
Script 24: Evaluate all clustering methods on physics_decay_type_probe.

Evaluation criteria
-------------------
Internal quality
  silhouette_cosine    silhouette using cosine distance on normalised abs prompt vectors
  cluster_entropy      entropy of cluster-size distribution (high = balanced, 0 = degenerate)

External enrichment (vs known feature labels)
  ari_circuit          ARI vs is_circuit_feature (binary)
  ari_role             ARI vs role_label (6 categories)
  ari_alpha_d          ARI vs is_global_alpha_discrim (binary)
  ari_beta_d           ARI vs is_global_beta_discrim (binary)
  ari_old_community    ARI vs Louvain community from old graph (6 communities)
  ari_grad_sign        ARI vs grad_attr_sign (binary: +1/-1)
  nmi_role             NMI vs role_label

Stability
  seed_stability       Mean ARI across 10 random seeds (for non-deterministic methods)
  bootstrap_sil        Mean silhouette over 50 prompt bootstrap samples

Latent-state usefulness score (composite)
  lsu_score            Custom score combining:
                       - does circuit features land in a coherent cluster together?
                       - does cluster discriminate α from β (group-level effect sign)?
                       - does cluster contain only one latent target?

Outputs
-------
data/results/clustering/
  evaluation_summary.csv    per-method × per-metric table
  method_ranking.csv        ranked list with composite score
  enrichment_detail.csv     per-cluster enrichment for best methods
  bootstrap_stability.csv   bootstrap silhouette per method
  figures/                  plots
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score
)
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
CLU  = ROOT / "data/results/clustering"
FIG  = CLU / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
N_BOOTSTRAP  = 50
N_SEEDS      = 10


def load_all():
    def lj(name):
        with open(CLU / name) as f: return json.load(f)
    def ln(name): return np.load(CLU / name)

    labels_df  = pd.read_csv(CLU / "cluster_labels.csv", index_col=0)
    feat_ids   = lj("feat_ids.json")
    feat_labels= lj("feat_labels.json")
    subsets    = lj("subsets.json")
    X_abs      = ln("feat_prompt_abs.npy")
    X_ga_abs   = ln("feat_group_abs.npy")
    group_ids  = lj("group_ids.json")

    with open(CLU / "prompt_labels.json") as f:
        prompt_labels = json.load(f)
    grp_ans = {}
    for pid, info in prompt_labels.items():
        grp_ans[info["group_id"]] = info["correct_answer"]

    return labels_df, feat_ids, feat_labels, subsets, X_abs, X_ga_abs, group_ids, grp_ans


def cluster_entropy(labels):
    counts = np.bincount(labels - labels.min())
    counts = counts[counts > 0]
    probs  = counts / counts.sum()
    return float(scipy_entropy(probs))


def lsu_score(labels, feat_ids, feat_labels, X_ga_abs, group_ids, grp_ans):
    """
    Latent-State Usefulness score — custom composite for our research question.

    Component A (0–1): circuit features coherence.
      Are all circuit + discriminator features in the same cluster, or at most
      2 clusters?  Score = 1 if ≤2 clusters contain all circuit/discrim features,
      decays linearly to 0 if they span all clusters.

    Component B (0–1): α/β separation at group level.
      For each cluster, compute mean abs effect on α-target groups vs β-target groups.
      Score = mean over clusters of |α_mean − β_mean| / (α_mean + β_mean + ε).

    Component C (0–1): balance (not degenerate).
      Entropy of cluster-size distribution normalised to [0,1] by max entropy.

    LSU = (A + B + C) / 3
    """
    alpha_grp_j = [gi for gi, g in enumerate(group_ids) if grp_ans.get(g) == "alpha"]
    beta_grp_j  = [gi for gi, g in enumerate(group_ids) if grp_ans.get(g) == "beta"]

    key_feats = set(fid for fid, fl in feat_labels.items()
                    if fl["is_circuit"] or fl["is_alpha_d"] or fl["is_beta_d"])
    key_idx   = [i for i, fid in enumerate(feat_ids) if fid in key_feats]

    if len(key_idx) == 0:
        comp_a = 0.0
    else:
        n_clusters = len(set(labels))
        key_clusters = set(labels[i] for i in key_idx)
        comp_a = max(0.0, 1.0 - (len(key_clusters) - 1) / max(1, n_clusters - 1))

    # Component B: α/β discriminability per cluster
    n_clusters = len(set(labels))
    ab_scores  = []
    for c in set(labels):
        cidx = [i for i, l in enumerate(labels) if l == c]
        if len(cidx) == 0:
            continue
        X_c     = X_ga_abs[cidx, :]   # features in cluster c × groups
        mu_alpha = X_c[:, alpha_grp_j].mean()
        mu_beta  = X_c[:, beta_grp_j].mean()
        ab_scores.append(abs(mu_alpha - mu_beta) / (mu_alpha + mu_beta + 1e-6))
    comp_b = float(np.mean(ab_scores)) if ab_scores else 0.0

    # Component C: balance
    counts = np.bincount(labels - labels.min())
    probs  = counts / counts.sum()
    H      = float(scipy_entropy(probs))
    H_max  = np.log(n_clusters) if n_clusters > 1 else 1.0
    comp_c = H / H_max if H_max > 0 else 0.0

    return (comp_a + comp_b + comp_c) / 3.0, comp_a, comp_b, comp_c


def bootstrap_silhouette(method_col: pd.Series, X_abs: np.ndarray,
                          n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED) -> float:
    rng     = np.random.default_rng(seed)
    n_feat, n_prompt = X_abs.shape
    sils = []
    for _ in range(n_bootstrap):
        boot_j = rng.choice(n_prompt, size=n_prompt, replace=True)
        X_boot = X_abs[:, boot_j]
        X_norm = normalize(X_boot)
        labels = method_col.values.astype(int)
        if len(set(labels)) < 2 or len(set(labels)) >= n_feat:
            continue
        try:
            s = silhouette_score(X_norm, labels, metric="cosine")
            sils.append(s)
        except Exception:
            pass
    return float(np.mean(sils)) if sils else float("nan")


def seed_stability_nmf(X_abs: np.ndarray, k: int, n_seeds=N_SEEDS) -> float:
    """Run NMF with different seeds, compute mean pairwise ARI."""
    X = np.clip(X_abs, 0, None)
    from sklearn.decomposition import NMF
    from itertools import combinations
    label_sets = []
    for s in range(n_seeds):
        try:
            model = NMF(n_components=k, random_state=s, max_iter=500, init="nndsvda")
            W     = model.fit_transform(X)
            label_sets.append(np.argmax(W, axis=1))
        except Exception:
            pass
    if len(label_sets) < 2:
        return float("nan")
    aris = [adjusted_rand_score(a, b)
            for a, b in combinations(label_sets, 2)]
    return float(np.mean(aris))


def main():
    print("Loading data...")
    labels_df, feat_ids, feat_labels, subsets, X_abs, X_ga_abs, group_ids, grp_ans = load_all()

    # ── Prepare known label arrays ────────────────────────────────────────────────
    circuit_labels = np.array([int(feat_labels[fid]["is_circuit"])     for fid in feat_ids])
    alpha_d_labels = np.array([int(feat_labels[fid]["is_alpha_d"])     for fid in feat_ids])
    beta_d_labels  = np.array([int(feat_labels[fid]["is_beta_d"])      for fid in feat_ids])
    grad_labels    = np.array([feat_labels[fid]["grad_sign"]           for fid in feat_ids])
    comm_labels    = np.array([feat_labels[fid]["community"]           for fid in feat_ids])
    role_labels_raw= [feat_labels[fid]["role_label"]                   for fid in feat_ids]
    le = LabelEncoder()
    role_labels    = le.fit_transform(role_labels_raw)
    layer_labels   = np.array([feat_labels[fid]["layer"]               for fid in feat_ids])

    X_norm         = normalize(X_abs)

    print(f"Evaluating {len(labels_df.columns)} method × k configurations...")
    records = []

    for col in labels_df.columns:
        labels = labels_df[col].values.astype(int)
        n_k    = len(set(labels))
        if n_k < 2:
            continue

        # Internal
        try:
            sil = float(silhouette_score(X_norm, labels, metric="cosine"))
        except Exception:
            sil = float("nan")
        H = cluster_entropy(labels)

        # External enrichment
        ari_circ  = float(adjusted_rand_score(circuit_labels, labels))
        ari_role  = float(adjusted_rand_score(role_labels,    labels))
        ari_ad    = float(adjusted_rand_score(alpha_d_labels, labels))
        ari_bd    = float(adjusted_rand_score(beta_d_labels,  labels))
        ari_comm  = float(adjusted_rand_score(comm_labels,    labels))
        ari_grad  = float(adjusted_rand_score(grad_labels,    labels))
        nmi_role  = float(normalized_mutual_info_score(role_labels, labels))

        # Latent-state usefulness
        lsu, ca, cb, cc = lsu_score(labels, feat_ids, feat_labels, X_ga_abs, group_ids, grp_ans)

        records.append({
            "method":      col,
            "n_clusters":  n_k,
            "silhouette":  round(sil, 4),
            "entropy":     round(H,   4),
            "ari_circuit": round(ari_circ, 4),
            "ari_role":    round(ari_role, 4),
            "ari_alpha_d": round(ari_ad,   4),
            "ari_beta_d":  round(ari_bd,   4),
            "ari_old_comm":round(ari_comm, 4),
            "ari_grad":    round(ari_grad, 4),
            "nmi_role":    round(nmi_role, 4),
            "lsu":         round(lsu,  4),
            "lsu_A_circuit_coherence": round(ca, 4),
            "lsu_B_ab_discriminability": round(cb, 4),
            "lsu_C_balance": round(cc, 4),
        })

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(CLU / "evaluation_summary.csv", index=False)
    print(f"Saved evaluation_summary.csv ({len(eval_df)} rows)")

    # ── Bootstrap silhouette (expensive, run on key methods) ─────────────────────
    print("Computing bootstrap silhouette (50 prompt resamples per method)...")
    boot_records = []
    priority_methods = [c for c in labels_df.columns if
                        any(k in c for k in ["_k4","_k3","louvain","stability"])]
    for col in priority_methods:
        labels = labels_df[col].values.astype(int)
        if len(set(labels)) < 2:
            continue
        bs = bootstrap_silhouette(labels_df[col], X_abs)
        boot_records.append({"method": col, "bootstrap_sil": round(bs, 4)})

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(CLU / "bootstrap_stability.csv", index=False)
    print(f"Saved bootstrap_stability.csv ({len(boot_df)} rows)")

    # ── NMF seed stability ────────────────────────────────────────────────────────
    print("Computing NMF seed stability...")
    nmf_stab = []
    for k in [2, 3, 4, 5]:
        stab = seed_stability_nmf(X_abs, k)
        nmf_stab.append({"method": f"nmf_k{k}", "seed_stability_ari": round(stab, 4)})
    pd.DataFrame(nmf_stab).to_csv(CLU / "nmf_seed_stability.csv", index=False)

    # ── Composite ranking ─────────────────────────────────────────────────────────
    # Composite = 0.3*silhouette + 0.3*lsu + 0.2*nmi_role + 0.1*ari_old_comm + 0.1*entropy/log(k)
    def composite(row):
        sil  = row["silhouette"] if not np.isnan(row["silhouette"]) else 0.0
        lsu  = row["lsu"]
        nmi  = row["nmi_role"]
        comm = max(0, row["ari_old_comm"])  # old community: partial credit
        k    = max(2, row["n_clusters"])
        bal  = row["entropy"] / np.log(k)   # normalised balance
        return 0.30*sil + 0.30*lsu + 0.20*nmi + 0.10*comm + 0.10*bal

    eval_df["composite"] = eval_df.apply(composite, axis=1)
    ranking = eval_df.sort_values("composite", ascending=False).reset_index(drop=True)
    ranking.to_csv(CLU / "method_ranking.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────────
    print("\n─── Top 15 methods by composite score ──────────────────────────────────")
    cols = ["method","n_clusters","silhouette","lsu","nmi_role","ari_old_comm","composite"]
    print(ranking[cols].head(15).to_string(index=False))

    # ── Enrichment detail for top-5 methods ─────────────────────────────────────
    print("\n─── Cluster-level enrichment detail (top 5 methods) ────────────────────")
    enrich_rows = []
    for col in ranking["method"].head(5):
        if col not in labels_df.columns:
            continue
        labels = labels_df[col].values.astype(int)
        for c in sorted(set(labels)):
            cidx  = [i for i,l in enumerate(labels) if l == c]
            n_c   = len(cidx)
            n_circ= sum(circuit_labels[i] for i in cidx)
            n_ad  = sum(alpha_d_labels[i] for i in cidx)
            n_bd  = sum(beta_d_labels[i]  for i in cidx)
            roles = [role_labels_raw[i] for i in cidx]
            role_dom = max(set(roles), key=roles.count)
            feat_names = ", ".join(feat_ids[i] for i in cidx[:6])
            enrich_rows.append({
                "method": col, "cluster": c, "size": n_c,
                "n_circuit": n_circ, "n_alpha_d": n_ad, "n_beta_d": n_bd,
                "dominant_role": role_dom,
                "features": feat_names,
            })
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(CLU / "enrichment_detail.csv", index=False)
    print(enrich_df.to_string(index=False))

    # ── Level stability NMI summary (from method_extra.json) ─────────────────────
    try:
        with open(CLU / "method_extra.json") as f:
            extra = json.load(f)
        if "level_stability" in extra:
            ls = extra["level_stability"]
            print("\n─── Level-conditioned stability NMI ────────────────────────────────────")
            for pair, nmi in ls["nmi"].items():
                print(f"  {pair}: NMI = {nmi:.4f}")
    except Exception:
        pass

    print("\nAll evaluation outputs saved to data/results/clustering/")


if __name__ == "__main__":
    main()
