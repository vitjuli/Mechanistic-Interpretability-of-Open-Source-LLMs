"""
28_enrichment_robustness.py

Enrichment stability and weighted enrichment analysis for the 11 co-importance
Louvain clusters in physics_decay_type_probe.

Outputs (all to data/results/cluster_semantics/):
  cluster_enrichment_topk_sweep.csv
  cluster_weighted_enrichment.csv
  cluster_enrichment_stability_summary.csv
  cluster_enrichment_curves.json         (to dashboard_probe/public/data/)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT     = Path(__file__).resolve().parents[1]
CS_DIR   = ROOT / "data/results/cluster_semantics"
DASH_OUT = ROOT / "dashboard_probe/public/data"

# ── Load data ─────────────────────────────────────────────────────────────────
cps = pd.read_csv(CS_DIR / "cluster_prompt_scores.csv")   # 5170 rows
pm  = pd.read_csv(ROOT / "data/results/grouping/prompt_metadata.csv")

N_PROMPTS = pm["prompt_idx"].nunique()   # 470

# Metadata fields to test (categorical, manageable cardinality)
META_FIELDS = [
    "correct_answer",     # alpha / beta
    "level",              # 1 / 2 / 3 / AUX
    "cue_label",          # ~35 unique values
    "difficulty",         # easy / medium / hard
    "inference_steps",    # 1 / 2 / 3
    "cue_type",           # ~15 unique values
    "physics_concept",    # alpha_decay / beta_decay
    "keyword_free",       # True / False
    "evidence_completeness",  # single / partial / full
]

# Merge richer metadata into cluster_prompt_scores
extra_cols = [c for c in ["cue_type", "physics_concept", "keyword_free",
                           "evidence_completeness", "surface_family"]
              if c in pm.columns]
pm_extra = pm[["prompt_idx"] + extra_cols].copy()
cps = cps.merge(pm_extra, on="prompt_idx", how="left")

# Global frequencies per metadata field (over all 470 prompts, each counted once)
global_freq: dict[str, dict] = {}
for field in META_FIELDS:
    if field not in pm.columns:
        continue
    counts = pm[field].value_counts()
    global_freq[field] = (counts / N_PROMPTS).to_dict()

TOP_K_VALUES = [10, 20, 30, 50, 75, 100]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: one-sided Fisher's exact test  (enrichment direction)
# ─────────────────────────────────────────────────────────────────────────────
def fisher_pval(obs: int, topk: int, base_count: int, total: int) -> float:
    """One-sided Fisher: is this metadata value over-represented in top-k?"""
    a = obs
    b = topk - obs
    c = base_count - obs
    d = total - topk - c
    if c < 0 or d < 0:
        return np.nan
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(pval)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Top-k sweep
# ─────────────────────────────────────────────────────────────────────────────
topk_rows = []
total_abs_per_cluster = (
    cps.groupby("cluster_id")["cluster_mean_abs_effect"].sum().to_dict()
)

for cid in sorted(cps["cluster_id"].unique()):
    sub = cps[cps["cluster_id"] == cid].copy()
    sub_sorted = sub.sort_values("cluster_mean_abs_effect", ascending=False).reset_index(drop=True)
    total_abs = total_abs_per_cluster[cid]

    for field in META_FIELDS:
        if field not in sub_sorted.columns:
            continue
        gf = global_freq.get(field, {})

        # rank of each value within field (by lift at top-20)
        for topk in TOP_K_VALUES:
            topk_set = sub_sorted.head(topk)
            mass_covered = topk_set["cluster_mean_abs_effect"].sum() / total_abs

            val_counts = topk_set[field].value_counts()
            for val, obs in val_counts.items():
                exp_frac = gf.get(val, 0.0)
                exp      = exp_frac * topk
                obs_frac = obs / topk
                lift     = obs_frac / exp_frac if exp_frac > 0 else np.nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    lo = np.log(obs / exp) if (exp > 0 and obs > 0) else (
                         np.inf if obs > 0 else (-np.inf if exp > 0 else 0.0))
                # base count in full prompt set
                base_count = int(round(exp_frac * N_PROMPTS))
                pval = fisher_pval(int(obs), topk, base_count, N_PROMPTS)

                topk_rows.append({
                    "cluster_id":             cid,
                    "top_k":                  topk,
                    "metadata_field":         field,
                    "metadata_value":         str(val),
                    "observed_count":         int(obs),
                    "expected_count":         round(exp, 3),
                    "observed_fraction":      round(obs_frac, 4),
                    "expected_fraction":      round(exp_frac, 4),
                    "lift":                   round(lift, 4) if not np.isnan(lift) else None,
                    "log_odds":               round(float(lo), 4) if np.isfinite(lo) else float(lo),
                    "cluster_abs_mass_covered": round(mass_covered, 4),
                    "fisher_pval":            round(pval, 5) if not np.isnan(pval) else None,
                })

df_topk = pd.DataFrame(topk_rows)

# Add rank_within_field at each (cluster, top_k, field) — by lift descending
df_topk["rank_within_field"] = (
    df_topk
    .groupby(["cluster_id", "top_k", "metadata_field"])["lift"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

df_topk.to_csv(CS_DIR / "cluster_enrichment_topk_sweep.csv", index=False)
print(f"Saved cluster_enrichment_topk_sweep.csv  ({len(df_topk)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Weighted enrichment (weight = cluster_mean_abs_effect)
# ─────────────────────────────────────────────────────────────────────────────
weighted_rows = []

for cid in sorted(cps["cluster_id"].unique()):
    sub = cps[cps["cluster_id"] == cid].copy()
    total_weight = sub["cluster_mean_abs_effect"].sum()

    for field in META_FIELDS:
        if field not in sub.columns:
            continue
        gf = global_freq.get(field, {})

        # weighted mass per value
        wm = sub.groupby(field)["cluster_mean_abs_effect"].sum()
        all_vals = set(gf.keys()) | set(wm.index.astype(str))

        # rank by weighted_lift
        val_data = []
        for val in all_vals:
            wt  = wm.get(val, 0.0)
            wfr = wt / total_weight if total_weight > 0 else 0.0
            gfr = gf.get(val, 0.0)
            w_lift = wfr / gfr if gfr > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                w_lo = np.log(wfr / gfr) if (gfr > 0 and wfr > 0) else (
                       np.inf if wfr > 0 else (-np.inf if gfr > 0 else 0.0))
            val_data.append({
                "cluster_id":           cid,
                "metadata_field":       field,
                "metadata_value":       str(val),
                "weighted_mass":        round(float(wt), 5),
                "weighted_fraction":    round(float(wfr), 5),
                "global_fraction":      round(float(gfr), 5),
                "weighted_lift":        round(float(w_lift), 4) if not np.isnan(w_lift) else None,
                "weighted_log_odds":    round(float(w_lo), 4) if np.isfinite(w_lo) else float(w_lo),
            })

        # rank within field by weighted_lift
        val_df = pd.DataFrame(val_data)
        val_df["weighted_rank_within_field"] = (
            val_df["weighted_lift"].rank(method="min", ascending=False).astype("Int64")
        )
        weighted_rows.append(val_df)

df_weighted = pd.concat(weighted_rows, ignore_index=True)
df_weighted.to_csv(CS_DIR / "cluster_weighted_enrichment.csv", index=False)
print(f"Saved cluster_weighted_enrichment.csv  ({len(df_weighted)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stability summary
# ─────────────────────────────────────────────────────────────────────────────
# Classify each (cluster, field, value) enrichment

def classify_enrichment(lifts: dict, w_lift: float | None) -> tuple[str, str, str]:
    """
    lifts: {top_k: lift_value}  (None allowed for missing values)
    w_lift: weighted lift
    Returns (enrichment_type, confidence, note)
    """
    ks = sorted(lifts.keys())
    valid = {k: v for k, v in lifts.items() if v is not None and not np.isnan(v)}
    if not valid:
        return "absent", "low", "no valid lift values"

    max_lift = max(valid.values())
    min_lift = min(valid.values())

    # Count how many top-k have lift > 1.5 (moderate enrichment threshold)
    n_above_1_5 = sum(1 for v in valid.values() if v > 1.5)
    n_above_1_2 = sum(1 for v in valid.values() if v > 1.2)
    n_k = len(ks)

    lift10  = valid.get(10)
    lift20  = valid.get(20)
    lift50  = valid.get(50)
    lift100 = valid.get(100)

    # Trend: is lift monotonically decreasing after top-20?
    later = [valid.get(k) for k in [30, 50, 75, 100] if valid.get(k) is not None]
    early = [v for k, v in valid.items() if k <= 20]

    mean_early = np.mean(early) if early else 0.0
    mean_later = np.mean(later) if later else 0.0
    decay_ratio = mean_later / mean_early if mean_early > 0 else 1.0

    # W_lift support
    w_strong = w_lift is not None and not np.isnan(w_lift) and w_lift > 1.3
    w_moderate = w_lift is not None and not np.isnan(w_lift) and w_lift > 1.1
    w_weak = w_lift is not None and not np.isnan(w_lift) and w_lift <= 1.1

    if max_lift < 1.2 and (w_lift is None or w_lift < 1.1):
        return "absent", "high", "lift < 1.2 at all cutoffs; weighted lift weak"

    if n_above_1_5 >= max(3, n_k // 2) and w_strong:
        return "stable", "high", f"lift>1.5 at {n_above_1_5}/{n_k} cutoffs; weighted lift={w_lift:.2f}"

    if n_above_1_2 >= max(3, n_k // 2) and w_moderate:
        w_str2 = f"{w_lift:.2f}" if (w_lift is not None and not np.isnan(w_lift)) else "N/A"
        return "stable", "medium", f"lift>1.2 at {n_above_1_2}/{n_k} cutoffs; weighted lift={w_str2}"

    if max_lift > 2.0 and (lift10 or lift20) and (lift10 or 0) > 2.0 or (lift20 or 0) > 2.0:
        if decay_ratio < 0.6 or (lift100 is not None and lift100 < 1.2):
            w_str3 = f"{w_lift:.2f}" if (w_lift is not None and not np.isnan(w_lift)) else "N/A"
            note = f"peak lift={max_lift:.2f} at early cutoffs, decays to {mean_later:.2f}; weighted={w_str3}"
            return "narrow_decisive", "medium", note

    if n_above_1_5 >= 2 and not w_strong:
        w_str = f"{w_lift:.2f}" if (w_lift is not None and not np.isnan(w_lift)) else "N/A"
        return "narrow_decisive", "low", f"lift>1.5 at {n_above_1_5}/{n_k} cutoffs but weighted lift weak ({w_str})"

    if max_lift > 1.5 and (lift50 or 0) < 1.0:
        return "unstable", "medium", f"max lift={max_lift:.2f} but drops below 1.0 at top-50"

    if max_lift - min_lift > 2.0:
        return "unstable", "medium", f"lift range [{min_lift:.2f}, {max_lift:.2f}] — highly variable"

    if n_above_1_2 >= 2 or (w_lift and w_lift > 1.2):
        w_str4 = f"{w_lift:.2f}" if (w_lift is not None and not np.isnan(w_lift)) else "N/A"
        return "narrow_decisive", "low", f"moderate lift at limited cutoffs; weighted={w_str4}"

    w_str5 = f"{w_lift:.2f}" if (w_lift is not None and not np.isnan(w_lift)) else "N/A"
    return "absent", "medium", f"max lift={max_lift:.2f}, weighted={w_str5}"


# Build lookup for weighted lifts
w_lookup = {}
for _, row in df_weighted.iterrows():
    key = (int(row["cluster_id"]), row["metadata_field"], str(row["metadata_value"]))
    w_lookup[key] = row["weighted_lift"]

stability_rows = []
for (cid, field, val), grp in df_topk.groupby(["cluster_id", "metadata_field", "metadata_value"]):
    lifts = {}
    best_topk = None
    best_lift = -np.inf
    for _, r in grp.iterrows():
        k     = int(r["top_k"])
        lift  = r["lift"]
        lifts[k] = lift if pd.notna(lift) else None
        if lift is not None and pd.notna(lift) and lift > best_lift:
            best_lift = lift
            best_topk = k

    w_lift = w_lookup.get((cid, field, str(val)))

    def _safe(d, k):
        v = d.get(k)
        return round(float(v), 4) if v is not None and not np.isnan(v) else None

    etype, conf, note = classify_enrichment(lifts, w_lift)

    stability_rows.append({
        "cluster_id":       cid,
        "metadata_field":   field,
        "metadata_value":   str(val),
        "best_topk":        best_topk,
        "best_lift":        round(best_lift, 4) if np.isfinite(best_lift) else None,
        "lift_top10":       _safe(lifts, 10),
        "lift_top20":       _safe(lifts, 20),
        "lift_top30":       _safe(lifts, 30),
        "lift_top50":       _safe(lifts, 50),
        "lift_top75":       _safe(lifts, 75),
        "lift_top100":      _safe(lifts, 100),
        "weighted_lift":    round(float(w_lift), 4) if w_lift is not None and not np.isnan(w_lift) else None,
        "enrichment_type":  etype,
        "confidence":       conf,
        "note":             note,
    })

df_stab = pd.DataFrame(stability_rows)
# Drop rows that are absent with low confidence AND have max lift < 1.1 (uninteresting)
# Keep everything where best_lift >= 1.2 OR enrichment_type != 'absent'
df_stab_full = df_stab.copy()
df_stab_sig  = df_stab[
    (df_stab["best_lift"].notna() & (df_stab["best_lift"] >= 1.2)) |
    (df_stab["enrichment_type"] != "absent")
].copy()

df_stab_sig.to_csv(CS_DIR / "cluster_enrichment_stability_summary.csv", index=False)
print(f"Saved cluster_enrichment_stability_summary.csv  ({len(df_stab_sig)} rows, filtered to lift>=1.2)")

# Print a human-readable summary
print("\n=== STABILITY SUMMARY (significant enrichments) ===")
for cid in sorted(df_stab_sig["cluster_id"].unique()):
    sub = df_stab_sig[df_stab_sig["cluster_id"] == cid]
    # Show top-ranked by best_lift within stable/narrow
    sig = sub[sub["enrichment_type"].isin(["stable", "narrow_decisive"])].sort_values("best_lift", ascending=False)
    print(f"\nC{cid}:")
    for _, r in sig.head(8).iterrows():
        print(f"  [{r['enrichment_type']:18s}] {r['metadata_field']:24s} = {str(r['metadata_value']):30s}"
              f"  lift@10={r['lift_top10']}  @20={r['lift_top20']}  @50={r['lift_top50']}  "
              f"@100={r['lift_top100']}  w={r['weighted_lift']}  ({r['note'][:60]})")
    unstable = sub[sub["enrichment_type"] == "unstable"]
    if len(unstable) > 0:
        print(f"  [unstable: {len(unstable)} motifs]")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dashboard JSON (curves)
# ─────────────────────────────────────────────────────────────────────────────
# Build per-cluster per-field per-value curve
IMPORTANT_FIELDS = ["cue_label", "correct_answer", "level", "difficulty",
                    "inference_steps", "cue_type", "physics_concept", "keyword_free"]

# Select important values: those that ever have lift >= 1.3 at any top-k,
# or that are specifically mentioned in the research memo
curves_out: list[dict] = []

for cid in sorted(df_topk["cluster_id"].unique()):
    sub_topk = df_topk[df_topk["cluster_id"] == cid]
    sub_stab = df_stab_sig[df_stab_sig["cluster_id"] == cid]

    # Keep values with meaningful enrichment
    sig_vals = sub_stab[
        (sub_stab["best_lift"].notna()) &
        (sub_stab["best_lift"] >= 1.3) &
        (sub_stab["metadata_field"].isin(IMPORTANT_FIELDS))
    ][["metadata_field", "metadata_value", "enrichment_type", "weighted_lift"]].copy()

    cluster_curves = []
    for _, sv in sig_vals.iterrows():
        field, val, etype, w_lift = sv["metadata_field"], sv["metadata_value"], sv["enrichment_type"], sv["weighted_lift"]
        row_filt = sub_topk[
            (sub_topk["metadata_field"] == field) &
            (sub_topk["metadata_value"] == str(val))
        ].sort_values("top_k")

        lifts    = []
        log_odds = []
        pvals    = []
        ks       = []
        for _, r in row_filt.iterrows():
            ks.append(int(r["top_k"]))
            lifts.append(round(float(r["lift"]), 4) if pd.notna(r["lift"]) else None)
            log_odds.append(round(float(r["log_odds"]), 4) if pd.notna(r["log_odds"]) else None)
            pvals.append(round(float(r["fisher_pval"]), 5) if pd.notna(r["fisher_pval"]) else None)

        cluster_curves.append({
            "field":          field,
            "value":          str(val),
            "topk":           ks,
            "lift":           lifts,
            "log_odds":       log_odds,
            "fisher_pval":    pvals,
            "weighted_lift":  round(float(w_lift), 4) if w_lift is not None and not np.isnan(w_lift) else None,
            "classification": etype,
        })

    curves_out.append({
        "cluster_id":   int(cid),
        "curves":       cluster_curves,
    })

with open(DASH_OUT / "cluster_enrichment_curves.json", "w") as f:
    json.dump(curves_out, f, indent=2)
print(f"\nSaved cluster_enrichment_curves.json  ({len(curves_out)} clusters)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Print full per-cluster stability table for memo update
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n=== FULL STABILITY TABLE FOR MEMO ===")
for cid in sorted(df_stab_sig["cluster_id"].unique()):
    sub = df_stab_sig[df_stab_sig["cluster_id"] == cid]
    print(f"\n── C{cid} ──")
    # Sort by enrichment_type (stable first), then best_lift
    order = {"stable": 0, "narrow_decisive": 1, "unstable": 2, "absent": 3}
    sub_s = sub.copy()
    sub_s["_sort"] = sub_s["enrichment_type"].map(order).fillna(3)
    sub_s = sub_s.sort_values(["_sort", "best_lift"], ascending=[True, False])
    for _, r in sub_s.iterrows():
        print(f"  {r['enrichment_type']:18s}  {r['metadata_field']:24s}={str(r['metadata_value']):25s}"
              f"  @10={r['lift_top10']}  @20={r['lift_top20']}  @50={r['lift_top50']}"
              f"  @100={r['lift_top100']}  w={r['weighted_lift']}")
