"""
Quick analysis of pair-mode union_cue results.

Questions:
 1. Distribution of overall_flip — насколько вообще пары вмешиваются?
 2. Top pairs by overall_flip — какие 2-кластерные пары вызывают самый сильный сдвиг?
 3. Top pairs by best_family_contrast — какие пары САМЫЕ family-specific?
 4. Per-family: для каждой семьи лучшая пара + её contrast.
 5. Is the default-and-suppress pattern (lepton sink) reproduced in pairs?
     (j81b finding: best_family почти всегда = lepton, contrast ~ inverse baseline margin)
 6. Cluster co-occurrence in top pairs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("data/analysis/runD_v2/union_cue/union_cue_pairs_all.csv")
OUT = Path("data/analysis/runD_v2/union_cue/pairs_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
FAMS = ["charge_Z", "emission", "energy", "lepton", "mass_A", "quark_weak"]
flip_cols = [f"flip_{f}" for f in FAMS]
df["c1"] = df["ids"].str.split("+").str[0].astype(int)
df["c2"] = df["ids"].str.split("+").str[1].astype(int)

print("=" * 78)
print("PAIRS UNION-CUE ANALYSIS — 435 cluster pairs × 6 families")
print("=" * 78)

# 1. distribution of overall_flip
print("\n[1] overall_flip distribution (n=435 pairs):")
desc = df["overall_flip"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
print(desc.to_string())

# 2. top 10 by overall_flip
print("\n[2] TOP 10 pairs by overall_flip (biggest behavioural disruption):")
top_overall = df.nlargest(10, "overall_flip")[["ids", "overall_flip", "best_family",
                                               "best_family_contrast"] + flip_cols]
print(top_overall.to_string(index=False))

# 3. top 10 by best_family_contrast — MOST FAMILY-SPECIFIC pairs
print("\n[3] TOP 10 pairs by best_family_contrast (most family-SPECIFIC):")
top_spec = df.nlargest(10, "best_family_contrast")[["ids", "best_family",
                                                    "best_family_contrast", "overall_flip"] + flip_cols]
print(top_spec.to_string(index=False))

# 4. per-family: best pair per family + its contrast
print("\n[4] BEST PAIR PER FAMILY:")
per_fam_rows = []
for f in FAMS:
    col = f"flip_{f}"
    other_cols = [c for c in flip_cols if c != col]
    df[f"_contrast_{f}"] = df[col] - df[other_cols].max(axis=1)
    best = df.nlargest(1, f"_contrast_{f}").iloc[0]
    per_fam_rows.append({
        "family": f,
        "best_pair": best["ids"],
        f"flip_{f}": best[col],
        "max_other_family_flip": best[other_cols].max(),
        "contrast (this - max_other)": best[f"_contrast_{f}"],
        "overall_flip": best["overall_flip"],
    })
per_fam = pd.DataFrame(per_fam_rows)
print(per_fam.to_string(index=False))

# 5. default-and-suppress check
print("\n[5] DEFAULT-AND-SUPPRESS CHECK:")
bf_counts = df["best_family"].value_counts()
print("best_family frequencies across all 435 pairs:")
print(bf_counts.to_string())
print(f"\nlepton dominance: {bf_counts.get('lepton', 0)}/435 = "
      f"{100*bf_counts.get('lepton', 0)/435:.1f}%")
print(f"(if ~17% per family with no pattern -> 73 pairs each)")

# 6. cluster co-occurrence in top 50 pairs by overall_flip
print("\n[6] CLUSTERS that appear most often in TOP 50 pairs:")
top50 = df.nlargest(50, "overall_flip")
clusters_top = pd.Series(list(top50["c1"]) + list(top50["c2"])).value_counts().head(15)
print(clusters_top.to_string())

# distinct clusters in top 50 — is there a hub cluster?
all_clusters_count = pd.Series(list(df["c1"]) + list(df["c2"])).value_counts()
top50_count = pd.Series(list(top50["c1"]) + list(top50["c2"])).value_counts()
print("\n  Top 5 clusters by enrichment vs baseline (top50 freq / all freq):")
enrich = (top50_count / 29).reindex(top50_count.index)  # each cluster appears in 29 pairs total
enrich = enrich.sort_values(ascending=False).head(5)
print(enrich.to_string())

# 7. write summary
summary = {
    "n_pairs": len(df),
    "overall_flip_median": float(df["overall_flip"].median()),
    "overall_flip_p95": float(df["overall_flip"].quantile(0.95)),
    "best_family_contrast_median": float(df["best_family_contrast"].median()),
    "best_family_contrast_p95": float(df["best_family_contrast"].quantile(0.95)),
    "lepton_dominance_frac": float(bf_counts.get("lepton", 0) / len(df)),
    "top10_overall_pairs": top_overall["ids"].tolist(),
    "top10_specific_pairs": top_spec["ids"].tolist(),
    "best_per_family": per_fam[["family", "best_pair",
                                "contrast (this - max_other)"]].to_dict(orient="records"),
}
(OUT / "pairs_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[saved] {OUT}/pairs_summary.json")

top_overall.to_csv(OUT / "top10_overall_flip.csv", index=False)
top_spec.to_csv(OUT / "top10_specific.csv", index=False)
per_fam.to_csv(OUT / "best_per_family.csv", index=False)
print(f"[saved] {OUT}/top10_overall_flip.csv, top10_specific.csv, best_per_family.csv")
print()
