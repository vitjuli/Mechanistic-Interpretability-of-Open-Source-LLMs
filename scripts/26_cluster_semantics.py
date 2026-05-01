#!/usr/bin/env python3
"""
Script 26: Cluster-level semantic / latent-state analysis for co-importance Louvain clusters.

Outputs in data/results/cluster_semantics/:
  cluster_feature_summary.csv
  cluster_prompt_scores.csv
  cluster_group_scores.csv
  cluster_metadata_enrichment.csv
  cluster_pairwise_coherence.csv
  cluster_semantic_report.md

Outputs in dashboard_probe/public/data/:
  cluster_semantics.json
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
GROUPING = ROOT / "data/results/grouping"
CLU      = ROOT / "data/results/clustering"
OUT      = ROOT / "data/results/cluster_semantics"
OUT.mkdir(parents=True, exist_ok=True)
DASH_OUT = ROOT / "dashboard_probe/public/data"

TOP_K_SUPPORT = 10   # "top-k" threshold for topk_support_count
TOP_N_ENRICH  = [10, 20, 30]   # enrichment window sizes

# ── Load source data ──────────────────────────────────────────────────────────
print("Loading data...")

contrib  = pd.read_csv(GROUPING / "feature_prompt_contributions.csv")
feat_top = pd.read_csv(GROUPING / "feature_top_prompts.csv")
by_grp   = pd.read_csv(GROUPING / "feature_by_group_effect.csv")
pmeta    = pd.read_csv(GROUPING / "prompt_metadata.csv")
fmeta    = pd.read_csv(GROUPING / "feature_metadata.csv")

# Cluster labels
import csv as csvlib
with open(CLU / "cluster_labels.csv") as f:
    rows = list(csvlib.DictReader(f))
coimp_labels = {r["feature_id"]: int(r["coimp_louvain"]) for r in rows}
feat_ids_mat = json.load(open(CLU / "feat_ids.json"))
feat_idx_map = {fid: i for i, fid in enumerate(feat_ids_mat)}

W_abs = np.load(CLU / "W_abs_cosine.npy")
W_sgn = np.load(CLU / "W_signed_cosine.npy")
W_co  = np.load(CLU / "W_coimportance.npy")

# Build cluster membership
clusters: dict[int, list[str]] = defaultdict(list)
for fid in feat_ids_mat:
    clusters[coimp_labels[fid]].append(fid)
cluster_ids = sorted(clusters.keys())

print(f"  {len(feat_ids_mat)} features, {len(cluster_ids)} clusters, {len(contrib)} contribution rows")

# ── Merge cluster labels into contributions ───────────────────────────────────
contrib["cluster_id"] = contrib["feature_id"].map(coimp_labels)

# Feature-level lookup for top-k support
topk_abs = (
    feat_top[feat_top["ranking_metric"] == "abs_effect_size"]
    .query(f"rank <= {TOP_K_SUPPORT}")
    [["feature_id", "prompt_idx"]]
    .drop_duplicates()
)
topk_set: dict[str, set] = {}
for fid, grp in topk_abs.groupby("feature_id"):
    topk_set[fid] = set(grp["prompt_idx"].tolist())

# Prompt metadata indexed by prompt_idx
pm_idx = pmeta.set_index("prompt_idx")

# ═══════════════════════════════════════════════════════════════════════════════
# A. cluster_feature_summary.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_feature_summary...")

fmeta_ab = fmeta[fmeta["feature_id"].isin(feat_ids_mat)].set_index("feature_id")

# Per-feature: mean_abs_effect and mean_signed_effect across all prompts
feat_overall = (
    contrib.groupby("feature_id")
    .agg(mean_abs_effect=("abs_effect_size","mean"), mean_signed_effect=("effect_size","mean"))
    .reset_index()
)

# Top positive / negative groups per feature (from by_grp)
def top_group(df, fid, ascending=False):
    sub = df[df["feature_id"] == fid].sort_values("mean_effect", ascending=ascending)
    return sub["group_id"].iloc[0] if len(sub) else None

# Top positive / negative prompts per feature (from contributions)
def top_prompt(df, fid, ascending=False):
    sub = (df[df["feature_id"] == fid]
           .sort_values("effect_size", ascending=ascending)
           .iloc[:1])
    return sub["prompt_id"].values[0] if len(sub) else None

rows_feat = []
for cid in cluster_ids:
    for fid in clusters[cid]:
        fm = fmeta_ab.loc[fid] if fid in fmeta_ab.index else None
        ov = feat_overall[feat_overall["feature_id"] == fid].iloc[0] if len(feat_overall[feat_overall["feature_id"]==fid]) else None
        rows_feat.append({
            "cluster_id":           cid,
            "feature_id":           fid,
            "layer":                int(fm["layer"]) if fm is not None else None,
            "role_label":           str(fm["role_label"]) if fm is not None else None,
            "community":            int(fm["community"]) if fm is not None and pd.notna(fm["community"]) else None,
            "is_circuit_feature":   bool(fm["is_circuit_feature"]) if fm is not None else False,
            "is_global_alpha_discrim": bool(fm["is_global_alpha_discrim"]) if fm is not None else False,
            "is_global_beta_discrim":  bool(fm["is_global_beta_discrim"]) if fm is not None else False,
            "mean_abs_effect":      round(float(ov["mean_abs_effect"]),4) if ov is not None else None,
            "mean_signed_effect":   round(float(ov["mean_signed_effect"]),4) if ov is not None else None,
            "top_positive_group":   top_group(by_grp, fid, ascending=False),
            "top_negative_group":   top_group(by_grp, fid, ascending=True),
            "top_positive_prompt":  top_prompt(contrib, fid, ascending=False),
            "top_negative_prompt":  top_prompt(contrib, fid, ascending=True),
        })

feat_summary_df = pd.DataFrame(rows_feat)
feat_summary_df.to_csv(OUT / "cluster_feature_summary.csv", index=False)
print(f"  Saved cluster_feature_summary.csv  ({len(feat_summary_df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# B. cluster_prompt_scores.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_prompt_scores...")

rows_ps = []
for cid in cluster_ids:
    feats = clusters[cid]
    n_feats = len(feats)
    sub = contrib[contrib["cluster_id"] == cid]
    if sub.empty:
        continue
    grp = sub.groupby("prompt_idx").agg(
        cluster_mean_effect    =("effect_size", "mean"),
        cluster_mean_abs_effect=("abs_effect_size", "mean"),
        positive_agreement     =("effect_size", lambda x: (x > 0).mean()),
        negative_agreement     =("effect_size", lambda x: (x < 0).mean()),
    ).reset_index()

    # topk_support_count: how many cluster features have this prompt in their top-k
    for pidx_val in grp["prompt_idx"]:
        topk_count = sum(
            1 for fid in feats
            if fid in topk_set and int(pidx_val) in topk_set[fid]
        )
        grp.loc[grp["prompt_idx"] == pidx_val, "topk_support_count"] = topk_count

    grp["topk_support_fraction"] = grp["topk_support_count"] / n_feats
    grp["cluster_id"] = cid

    # Join prompt metadata
    pm_sub = pm_idx.reindex(grp["prompt_idx"].values)
    for col in ["prompt_id","level","group_id","cue_label","relation_type",
                "latent_state_target","correct_answer","is_anchor","is_kw_variant",
                "is_auxiliary","inference_steps","difficulty"]:
        grp[col] = pm_sub[col].values

    rows_ps.append(grp)

prompt_scores_df = pd.concat(rows_ps, ignore_index=True)
# Round floats
for col in ["cluster_mean_effect","cluster_mean_abs_effect","positive_agreement",
            "negative_agreement","topk_support_fraction"]:
    prompt_scores_df[col] = prompt_scores_df[col].round(4)

col_order = ["cluster_id","prompt_idx","prompt_id","cluster_mean_effect","cluster_mean_abs_effect",
             "positive_agreement","negative_agreement","topk_support_count","topk_support_fraction",
             "level","group_id","cue_label","relation_type","latent_state_target","correct_answer",
             "is_anchor","is_kw_variant","is_auxiliary","inference_steps","difficulty"]
prompt_scores_df = prompt_scores_df[[c for c in col_order if c in prompt_scores_df.columns]]
prompt_scores_df.to_csv(OUT / "cluster_prompt_scores.csv", index=False)
print(f"  Saved cluster_prompt_scores.csv  ({len(prompt_scores_df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# C. cluster_group_scores.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_group_scores...")

rows_gs = []
for cid in cluster_ids:
    feats = clusters[cid]
    sub_bg = by_grp[by_grp["feature_id"].isin(feats)]
    if sub_bg.empty:
        continue
    grp = sub_bg.groupby("group_id").agg(
        mean_effect     =("mean_effect","mean"),
        mean_abs_effect =("mean_abs_effect","mean"),
        positive_agreement=("mean_effect", lambda x: (x > 0).mean()),
        negative_agreement=("mean_effect", lambda x: (x < 0).mean()),
        sfr             =("sfr","mean"),
        n_prompts       =("n_prompts","first"),
    ).reset_index()
    grp["cluster_id"] = cid

    # Dominant level / target / answer from group metadata
    # Join from by_grp first row per group
    grp_meta = sub_bg.drop_duplicates("group_id").set_index("group_id")
    for col in ["level","latent_state_target","correct_answer","cue_label"]:
        grp[col] = grp["group_id"].map(grp_meta.get(col, {}))

    rows_gs.append(grp)

group_scores_df = pd.concat(rows_gs, ignore_index=True)
for col in ["mean_effect","mean_abs_effect","positive_agreement","negative_agreement","sfr"]:
    group_scores_df[col] = group_scores_df[col].round(4)
group_scores_df.to_csv(OUT / "cluster_group_scores.csv", index=False)
print(f"  Saved cluster_group_scores.csv  ({len(group_scores_df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# D. cluster_metadata_enrichment.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_metadata_enrichment...")

ENRICH_FIELDS = ["level","cue_label","relation_type","latent_state_target",
                 "correct_answer","difficulty","inference_steps"]
N_TOTAL = len(pmeta)

enrich_rows = []
for cid in cluster_ids:
    ps = prompt_scores_df[prompt_scores_df["cluster_id"] == cid].copy()
    ps_sorted = ps.sort_values("cluster_mean_abs_effect", ascending=False)

    for top_n in TOP_N_ENRICH:
        top_ps = ps_sorted.head(top_n)
        n_top = len(top_ps)
        if n_top == 0:
            continue

        for field in ENRICH_FIELDS:
            if field not in top_ps.columns:
                continue
            vals = top_ps[field].dropna()
            base = pmeta[field].dropna()
            val_counts = vals.value_counts()
            base_counts = base.value_counts()

            for val, obs_count in val_counts.items():
                base_count = int(base_counts.get(val, 0))
                expected   = n_top * base_count / N_TOTAL if N_TOTAL > 0 else 0
                lift       = obs_count / expected if expected > 0 else None

                # Fisher's exact for this value vs not-this-value
                a = int(obs_count)           # in top, has value
                b = n_top - a               # in top, no value
                c = base_count - a          # not in top, has value (approx)
                d = N_TOTAL - base_count - b # not in top, no value
                c = max(c, 0); d = max(d, 0)
                try:
                    _, pval = fisher_exact([[a,b],[c,d]], alternative="greater")
                except Exception:
                    pval = None

                log_odds = None
                if a > 0 and b > 0 and c > 0 and d > 0:
                    log_odds = round(float(np.log((a * d) / (b * c))), 4)

                enrich_rows.append({
                    "cluster_id":       cid,
                    "top_n":            top_n,
                    "metadata_field":   field,
                    "metadata_value":   str(val),
                    "observed_count":   int(obs_count),
                    "expected_count":   round(float(expected), 3),
                    "lift":             round(float(lift), 3) if lift is not None else None,
                    "log_odds":         log_odds,
                    "pval_fisher":      round(float(pval), 4) if pval is not None else None,
                })

enrich_df = pd.DataFrame(enrich_rows)
enrich_df.to_csv(OUT / "cluster_metadata_enrichment.csv", index=False)
print(f"  Saved cluster_metadata_enrichment.csv  ({len(enrich_df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# E. cluster_pairwise_coherence.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_pairwise_coherence...")

# Load abs effect matrix for centroid computation
X_abs = np.load(CLU / "feat_prompt_abs.npy")   # (40, 470)
from sklearn.preprocessing import normalize as skl_norm
from sklearn.metrics.pairwise import cosine_similarity as cossim

coh_rows = []
for cid in cluster_ids:
    feats = clusters[cid]
    idxs = [feat_idx_map[f] for f in feats if f in feat_idx_map]
    n = len(idxs)

    def within_mean(W, idx_list):
        if len(idx_list) < 2:
            return None
        vals = [W[i,j] for ii,i in enumerate(idx_list)
                         for jj,j in enumerate(idx_list) if jj > ii]
        return round(float(np.mean(vals)),4) if vals else None

    abs_cos  = within_mean(W_abs, idxs)
    sgn_cos  = within_mean(W_sgn, idxs)
    coimp    = within_mean(W_co,  idxs)

    # Centroid cosine similarity
    if n > 1:
        X_c = X_abs[idxs]
        X_norm = skl_norm(X_c)
        centroid = X_norm.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        cent_sims = [float(X_norm[i] @ centroid) for i in range(n)]
        mean_cent = round(float(np.mean(cent_sims)), 4)
    else:
        mean_cent = 1.0

    # Residual similarity (from residual matrix)
    X_res = np.load(CLU / "feat_residual.npy")
    X_res_c = X_res[idxs]
    X_res_norm = skl_norm(X_res_c)
    res_sim = within_mean(
        X_res_norm @ X_res_norm.T, list(range(n))
    ) if n > 1 else None

    # Layer diversity
    layers = [int(f.split('_')[0][1:]) for f in feats]
    layer_span = max(layers) - min(layers) if len(layers) > 1 else 0

    coh_rows.append({
        "cluster_id":                    cid,
        "n_features":                    n,
        "min_layer":                     min(layers),
        "max_layer":                     max(layers),
        "layer_span":                    layer_span,
        "mean_abs_cosine_within":        abs_cos,
        "mean_signed_cosine_within":     sgn_cos,
        "mean_coimportance_within":      coimp,
        "mean_residual_similarity_within": res_sim,
        "mean_feature_to_centroid_cosine": mean_cent,
    })

coh_df = pd.DataFrame(coh_rows)
coh_df.to_csv(OUT / "cluster_pairwise_coherence.csv", index=False)
print(f"  Saved cluster_pairwise_coherence.csv  ({len(coh_df)} rows)")
print(coh_df[["cluster_id","n_features","mean_abs_cosine_within",
              "mean_coimportance_within","mean_feature_to_centroid_cosine"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# Semantic interpretation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_orientation(cid, ps_df):
    """alpha / beta / mixed based on mean signed effect on each answer's prompts."""
    ps = ps_df[ps_df["cluster_id"] == cid]
    a  = ps[ps["correct_answer"]=="alpha"]["cluster_mean_effect"].mean()
    b  = ps[ps["correct_answer"]=="beta" ]["cluster_mean_effect"].mean()
    delta = float(a - b) if pd.notna(a) and pd.notna(b) else 0.0
    if delta > 0.05:  return "alpha",   round(delta,3)
    if delta < -0.05: return "beta",    round(delta,3)
    return "mixed", round(delta,3)

def get_surface_level(min_l, max_l, layer_span):
    """Categorise depth zone."""
    if max_l <= 14:          return "early"
    if min_l >= 21:          return "late"
    if min_l >= 16:          return "mid-late"
    if layer_span <= 4:      return "mid"
    return "multi-layer"

def top_enriched(cid, enrich_df, top_n=20, max_items=3):
    """Return list of (field, value, lift) for most enriched values at top_n."""
    sub = (enrich_df[(enrich_df["cluster_id"]==cid) & (enrich_df["top_n"]==top_n)]
           .dropna(subset=["lift"])
           .sort_values("lift", ascending=False))
    results = []
    for _, row in sub.head(max_items*3).iterrows():
        if len(results) >= max_items: break
        if float(row["lift"]) < 1.2: continue
        results.append((str(row["metadata_field"]), str(row["metadata_value"]),
                        float(row["lift"]), float(row["pval_fisher"]) if pd.notna(row.get("pval_fisher")) else None))
    return results

def cluster_semantic_label(cid, feats, orientation, depth, coh_row, fmeta_ab, enrich_df, n_circuit, n_ad, n_bd):
    """
    Generate tentative semantic label and evidence.
    Conservative: label as 'candidate' unless evidence is very strong.
    """
    roles = Counter(fmeta_ab.loc[f,"role_label"] for f in feats if f in fmeta_ab.index)
    dom_role = roles.most_common(1)[0][0] if roles else "unknown"

    enriched = top_enriched(cid, enrich_df, top_n=20)
    enrich_str = "; ".join(f"{f}={v}(×{lift:.1f})" for f,v,lift,_ in enriched) if enriched else "none"

    # Build tentative label
    parts = []
    if depth == "early":
        parts.append("early-layer routing")
    elif depth == "late":
        parts.append("output-stage decision")
    elif depth == "mid-late":
        parts.append("mid-to-late processing")
    elif depth == "multi-layer":
        parts.append("cross-layer convergence")
    else:
        parts.append(f"{depth}-layer processing")

    if orientation == "alpha":
        parts.append("α-oriented")
    elif orientation == "beta":
        parts.append("β-oriented")

    if n_circuit > 0:
        parts.append(f"includes {n_circuit} circuit feature(s)")
    if n_ad > 0:
        parts.append(f"global α-discriminator")
    if n_bd > 0:
        parts.append(f"global β-discriminator")

    label = " · ".join(parts)
    return label, enrich_str, roles

# ═══════════════════════════════════════════════════════════════════════════════
# F. cluster_semantic_report.md
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_semantic_report.md...")

CLUSTER_NAMES = {
    0:  "Early β-Routing Module",
    1:  "Early α-Attribution Pair (L11)",
    2:  "Singleton L12",
    3:  "Early α-Attribution Pair (L13)",
    4:  "Mid-Layer α-Pair (L20)",
    5:  "Singleton L15",
    6:  "L16 β-Processing Module",
    7:  "Multi-Layer Convergence Module",
    8:  "Singleton L18 β-Discriminator",
    9:  "Mid-Late α-Attribution Module (L19–L21)",
    10: "Output Decision Module (L24–L25)",
}

fmeta_ab = fmeta[fmeta["feature_id"].isin(feat_ids_mat)].set_index("feature_id")

md_lines = [
    "# Cluster Semantic Report — `physics_decay_type_probe`",
    "",
    "**Method:** co-importance Louvain (11 clusters, composite rank #1)",
    "**Date:** 2026-05-01",
    "",
    "All interpretations are **tentative**. Labels reflect the strongest data-driven evidence ",
    "but are not causal proofs. Use 'candidate semantic direction' language in thesis.",
    "",
    "---",
    "",
]

for cid in cluster_ids:
    feats = clusters[cid]
    n = len(feats)
    coh = coh_df[coh_df["cluster_id"]==cid].iloc[0]

    # Feature-level metadata
    n_circuit = sum(bool(fmeta_ab.loc[f,"is_circuit_feature"]) for f in feats if f in fmeta_ab.index)
    n_ad = sum(bool(fmeta_ab.loc[f,"is_global_alpha_discrim"]) for f in feats if f in fmeta_ab.index)
    n_bd = sum(bool(fmeta_ab.loc[f,"is_global_beta_discrim"])  for f in feats if f in fmeta_ab.index)
    roles = Counter(fmeta_ab.loc[f,"role_label"] for f in feats if f in fmeta_ab.index)
    layers = [int(f.split('_')[0][1:]) for f in feats]

    orientation, delta = get_orientation(cid, prompt_scores_df)
    depth = get_surface_level(coh.min_layer, coh.max_layer, coh.layer_span)
    sem_label, enrich_str, _ = cluster_semantic_label(
        cid, feats, orientation, depth, coh, fmeta_ab, enrich_df, n_circuit, n_ad, n_bd)

    # Top prompts
    ps = prompt_scores_df[prompt_scores_df["cluster_id"]==cid].copy()
    top_pos = ps.sort_values("cluster_mean_effect", ascending=False).head(5)
    top_neg = ps.sort_values("cluster_mean_effect", ascending=True).head(5)
    top_abs = ps.sort_values("cluster_mean_abs_effect", ascending=False).head(5)

    # Top groups
    gs = group_scores_df[group_scores_df["cluster_id"]==cid].copy()
    top_grp_pos = gs.sort_values("mean_effect", ascending=False).head(5)
    top_grp_neg = gs.sort_values("mean_effect", ascending=True).head(5)

    # Enrichments at top-20
    enriched = top_enriched(cid, enrich_df, top_n=20, max_items=6)

    md_lines += [
        f"## Cluster {cid} — {CLUSTER_NAMES.get(cid, 'Unnamed')}",
        "",
        f"**Tentative semantic label:** {sem_label}",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Members (n) | {n} |",
        f"| Feature IDs | {', '.join(feats)} |",
        f"| Layers | {', '.join(str(l) for l in sorted(set(layers)))} (span {coh.layer_span}) |",
        f"| Dominant role | {roles.most_common(1)[0][0] if roles else '—'} ({roles.most_common(1)[0][1]}/{n}) |",
        f"| Role distribution | {dict(roles)} |",
        f"| Circuit features | {n_circuit} |",
        f"| Global α-discriminators | {n_ad} |",
        f"| Global β-discriminators | {n_bd} |",
        f"| Orientation | {orientation} (Δ = {delta:+.3f}) |",
        f"| Depth zone | {depth} |",
        f"| Mean abs cosine (within) | {coh.mean_abs_cosine_within if pd.notna(coh.mean_abs_cosine_within) else 'N/A (singleton)'} |",
        f"| Mean co-importance (within) | {coh.mean_coimportance_within if pd.notna(coh.mean_coimportance_within) else 'N/A (singleton)'} |",
        f"| Mean feature-to-centroid cosine | {coh.mean_feature_to_centroid_cosine} |",
        "",
    ]

    md_lines += [
        "### Top prompts by cluster mean signed effect (positive)",
        "",
        "| prompt_id | mean_eff | abs_eff | pos_agree | level | group | answer |",
        "|-----------|----------|---------|-----------|-------|-------|--------|",
    ]
    for _, r in top_pos.iterrows():
        md_lines.append(
            f"| {r.get('prompt_id','?')} | {r.cluster_mean_effect:.3f} | "
            f"{r.cluster_mean_abs_effect:.3f} | {r.positive_agreement:.2f} | "
            f"{r.get('level','?')} | {r.get('group_id','?')} | {r.get('correct_answer','?')} |"
        )
    md_lines.append("")

    md_lines += [
        "### Top prompts by cluster mean signed effect (negative)",
        "",
        "| prompt_id | mean_eff | abs_eff | neg_agree | level | group | answer |",
        "|-----------|----------|---------|-----------|-------|-------|--------|",
    ]
    for _, r in top_neg.iterrows():
        md_lines.append(
            f"| {r.get('prompt_id','?')} | {r.cluster_mean_effect:.3f} | "
            f"{r.cluster_mean_abs_effect:.3f} | {r.negative_agreement:.2f} | "
            f"{r.get('level','?')} | {r.get('group_id','?')} | {r.get('correct_answer','?')} |"
        )
    md_lines.append("")

    md_lines += [
        "### Top groups by mean cluster effect",
        "",
        "| group_id | mean_eff | mean_abs_eff | pos_agree | sfr | level | answer |",
        "|----------|----------|-------------|-----------|-----|-------|--------|",
    ]
    for _, r in top_grp_pos.iterrows():
        md_lines.append(
            f"| {r.group_id} | {r.mean_effect:.3f} | {r.mean_abs_effect:.3f} | "
            f"{r.positive_agreement:.2f} | {r.sfr:.2f} | {r.get('level','?')} | {r.get('correct_answer','?')} |"
        )
    md_lines.append("")

    if enriched:
        md_lines += [
            "### Strongest metadata enrichments (top-20 prompts by |effect|)",
            "",
            "| Field | Value | Observed | Expected | Lift | p-value |",
            "|-------|-------|----------|----------|------|---------|",
        ]
        for field, val, lift, pval in enriched:
            sub = enrich_df[(enrich_df.cluster_id==cid)&(enrich_df.top_n==20)
                            &(enrich_df.metadata_field==field)&(enrich_df.metadata_value==val)]
            if sub.empty: continue
            obs = int(sub.iloc[0].observed_count)
            exp = float(sub.iloc[0].expected_count)
            pstr = f"{pval:.3f}" if pval is not None else "—"
            md_lines.append(f"| {field} | {val} | {obs} | {exp:.1f} | {lift:.2f}× | {pstr} |")
        md_lines.append("")

    # Evidence summary + caveats
    evidence = []
    if n >= 3 and pd.notna(coh.mean_coimportance_within) and coh.mean_coimportance_within > 0.2:
        evidence.append(f"high co-importance coherence ({coh.mean_coimportance_within:.3f}) — features share decisive prompts")
    if pd.notna(coh.mean_abs_cosine_within) and coh.mean_abs_cosine_within is not None and coh.mean_abs_cosine_within > 0.85:
        evidence.append(f"strong abs-cosine coherence ({coh.mean_abs_cosine_within:.3f})")
    if abs(delta) > 0.1:
        evidence.append(f"clear orientation bias (Δ={delta:+.3f})")
    for field, val, lift, pval in enriched:
        if pval is not None and pval < 0.05:
            evidence.append(f"significant enrichment for {field}={val} (lift={lift:.2f}, p={pval:.3f})")

    md_lines += [
        "### Evidence summary",
        "",
        "\n".join(f"- {e}" for e in evidence) if evidence else "- Insufficient evidence (singleton or too few features)",
        "",
        "### Caveats",
        "",
        "- Cluster size n=" + str(n) + ("; single-feature clusters cannot have coherence statistics." if n == 1 else "."),
        "- Co-importance Louvain optimises shared top-10 prompt sets, not functional role.",
        "- Role labels (α-attr, β-discrim, etc.) were assigned by a separate pipeline and may not perfectly map to cluster boundaries.",
        "- All orientation/enrichment results are observational, not causal proofs.",
        "",
        "---",
        "",
    ]

with open(OUT / "cluster_semantic_report.md", "w") as f:
    f.write("\n".join(md_lines))
print(f"  Saved cluster_semantic_report.md")

# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard JSON
# ═══════════════════════════════════════════════════════════════════════════════
print("Building cluster_semantics.json for dashboard...")

def _f(v):
    if v is None: return None
    import math
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    if isinstance(v, (np.bool_,)):    return bool(v)
    return v

clusters_json = []
for cid in cluster_ids:
    feats = clusters[cid]
    n = len(feats)
    coh = coh_df[coh_df["cluster_id"]==cid].iloc[0]
    orientation, delta = get_orientation(cid, prompt_scores_df)
    layers = sorted(set(int(f.split('_')[0][1:]) for f in feats))
    roles = Counter(fmeta_ab.loc[f,"role_label"] for f in feats if f in fmeta_ab.index)
    n_circuit = sum(bool(fmeta_ab.loc[f,"is_circuit_feature"]) for f in feats if f in fmeta_ab.index)
    n_ad = sum(bool(fmeta_ab.loc[f,"is_global_alpha_discrim"]) for f in feats if f in fmeta_ab.index)
    n_bd = sum(bool(fmeta_ab.loc[f,"is_global_beta_discrim"])  for f in feats if f in fmeta_ab.index)
    depth = get_surface_level(layers[0], layers[-1], layers[-1]-layers[0])
    sem_label, enrich_str, _ = cluster_semantic_label(
        cid, feats, orientation, depth, coh, fmeta_ab, enrich_df, n_circuit, n_ad, n_bd)

    # Prompt data: top-30 positive + top-30 negative + top-30 abs
    ps = prompt_scores_df[prompt_scores_df["cluster_id"]==cid].copy()
    def ps_record(row):
        return {
            "idx":   _f(row.get("prompt_idx")),
            "id":    str(row.get("prompt_id","")),
            "me":    _f(row.get("cluster_mean_effect")),
            "mae":   _f(row.get("cluster_mean_abs_effect")),
            "pa":    _f(row.get("positive_agreement")),
            "na":    _f(row.get("negative_agreement")),
            "tks":   _f(row.get("topk_support_count")),
            "tkf":   _f(row.get("topk_support_fraction")),
            "lv":    str(row.get("level","")),
            "gid":   str(row.get("group_id","")),
            "cue":   str(row.get("cue_label","")) if pd.notna(row.get("cue_label")) else None,
            "ans":   str(row.get("correct_answer","")),
            "ok":    bool(pm_idx.loc[row["prompt_idx"],"sign_correct"]) if row["prompt_idx"] in pm_idx.index else None,
            "short": str(pm_idx.loc[row["prompt_idx"],"prompt_short"])[:70] if row["prompt_idx"] in pm_idx.index else "",
        }

    top_pos_j  = [ps_record(r) for _,r in ps.sort_values("cluster_mean_effect",ascending=False).head(30).iterrows()]
    top_neg_j  = [ps_record(r) for _,r in ps.sort_values("cluster_mean_effect",ascending=True).head(30).iterrows()]
    top_abs_j  = [ps_record(r) for _,r in ps.sort_values("cluster_mean_abs_effect",ascending=False).head(30).iterrows()]

    # Group data: top-20 positive + top-20 negative
    gs = group_scores_df[group_scores_df["cluster_id"]==cid].copy()
    def gs_record(row):
        return {
            "gid": str(row.group_id), "me": _f(row.mean_effect),
            "mae": _f(row.mean_abs_effect), "pa": _f(row.positive_agreement),
            "sfr": _f(row.sfr), "lv": str(row.get("level","")),
            "ans": str(row.get("correct_answer","")),
        }
    top_grp_pos = [gs_record(r) for _,r in gs.sort_values("mean_effect",ascending=False).head(20).iterrows()]
    top_grp_neg = [gs_record(r) for _,r in gs.sort_values("mean_effect",ascending=True).head(20).iterrows()]

    # Enrichment (top-20 only, lift > 1.1)
    enrich_j = {}
    sub_en = enrich_df[(enrich_df.cluster_id==cid)&(enrich_df.top_n==20)].copy()
    for field in ENRICH_FIELDS:
        sub_f = sub_en[sub_en.metadata_field==field].dropna(subset=["lift"])
        sub_f = sub_f.sort_values("lift",ascending=False).head(6)
        enrich_j[field] = [
            {"val": str(r.metadata_value), "obs": int(r.observed_count),
             "exp": _f(r.expected_count), "lift": _f(r.lift),
             "log_odds": _f(r.log_odds), "pval": _f(r.pval_fisher)}
            for _,r in sub_f.iterrows()
        ]

    # Feature details
    feat_details = []
    for fid in feats:
        fm = fmeta_ab.loc[fid] if fid in fmeta_ab.index else None
        ov = feat_overall[feat_overall["feature_id"]==fid]
        feat_details.append({
            "id":          fid,
            "layer":       int(fm["layer"]) if fm is not None else None,
            "role":        str(fm["role_label"]) if fm is not None else None,
            "is_circuit":  bool(fm["is_circuit_feature"]) if fm is not None else False,
            "is_alpha_d":  bool(fm["is_global_alpha_discrim"]) if fm is not None else False,
            "is_beta_d":   bool(fm["is_global_beta_discrim"]) if fm is not None else False,
            "mean_abs":    round(float(ov["mean_abs_effect"].iloc[0]),4) if len(ov) else None,
            "mean_eff":    round(float(ov["mean_signed_effect"].iloc[0]),4) if len(ov) else None,
        })

    # Prompt co-occurrence network for top prompts
    top_prompt_idxs = [r["idx"] for r in top_abs_j[:25] if r["idx"] is not None]
    net_edges = []
    for ii, p1 in enumerate(top_prompt_idxs):
        for jj, p2 in enumerate(top_prompt_idxs):
            if jj <= ii: continue
            shared = sum(
                1 for fid in feats
                if fid in topk_set and p1 in topk_set[fid] and p2 in topk_set[fid]
            )
            if shared >= 1:
                net_edges.append({"s": p1, "t": p2, "w": shared})

    clusters_json.append({
        "id":            cid,
        "name":          CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
        "sem_label":     sem_label,
        "n_features":    n,
        "feature_ids":   feats,
        "features":      feat_details,
        "layers":        layers,
        "layer_span":    int(coh.layer_span),
        "roles":         dict(roles),
        "n_circuit":     n_circuit,
        "n_alpha_d":     n_ad,
        "n_beta_d":      n_bd,
        "orientation":   orientation,
        "orient_delta":  _f(delta),
        "depth":         depth,
        "coherence": {
            "abs_cosine":        _f(coh.mean_abs_cosine_within),
            "signed_cosine":     _f(coh.mean_signed_cosine_within),
            "coimportance":      _f(coh.mean_coimportance_within),
            "residual_sim":      _f(coh.mean_residual_similarity_within),
            "centroid_cosine":   _f(coh.mean_feature_to_centroid_cosine),
        },
        "top_pos_prompts": top_pos_j,
        "top_neg_prompts": top_neg_j,
        "top_abs_prompts": top_abs_j,
        "top_pos_groups":  top_grp_pos,
        "top_neg_groups":  top_grp_neg,
        "enrichment":      enrich_j,
        "prompt_network":  {"nodes": top_prompt_idxs, "edges": net_edges},
    })

out_json = {
    "meta": {
        "n_clusters":    len(cluster_ids),
        "n_features":    len(feat_ids_mat),
        "n_prompts":     len(pmeta),
        "method":        "coimp_louvain",
        "top_k_support": TOP_K_SUPPORT,
    },
    "clusters": clusters_json,
}

out_path = DASH_OUT / "cluster_semantics.json"
with open(out_path, "w") as f:
    json.dump(out_json, f, separators=(",",":"))
size_kb = out_path.stat().st_size / 1024
print(f"\nSaved cluster_semantics.json  ({size_kb:.0f} KB)")
print("All outputs complete.")
EOF
