"""
Script 17: Intervention profile grouping for physics_decay_type.

Implements a 2-stage hierarchical grouping:
  Stage 1 — Coarse α/β causal dominance (asymmetry score + strength)
  Stage 2 — Family disruption annotation (sfr by surface_family)

Outputs:
  data/results/grouping/feature_groups.csv   — full feature table
  data/results/grouping/grouping_heatmap.html — annotated heatmap (Plotly)
  data/results/grouping/effect_matrix.html   — Ward cluster heatmap (40×108)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
ABLATION_CSV = ROOT / "data/ui_offline/20260426-034658_physics_decay_type_train_n108/raw_sources/intervention_ablation_physics_decay_type.csv"
PROMPTS_JSONL = ROOT / "data/prompts/physics_decay_type_train.jsonl"
CIRCUIT_JSON  = ROOT / "dashboard_decay/public/data/circuit.json"
OUT_DIR       = ROOT / "data/results/grouping"

FAMILIES = ["F0", "F1", "F2", "F3", "F4"]
ASYMMETRY_THRESH = 0.4   # |A| threshold for dominant vs general
STRENGTH_PCTILE  = 25    # percentile for background/weak cutoff
FAMILY_CONC_THRESH = 1.5 # f̃ threshold for "concentrated on this family"
EPS = 0.01               # regulariser


def load_data():
    df = pd.read_csv(ABLATION_CSV)
    prompts = [json.loads(l) for l in open(PROMPTS_JSONL)]
    with open(CIRCUIT_JSON) as f:
        circuit = json.load(f)
    circuit_features = {feat["id"] for feat in circuit.get("features", [])}
    return df, prompts, circuit_features


def build_prompt_meta(prompts):
    """Return dict prompt_idx → {physics_concept, surface_family}."""
    return {
        i: {"concept": p["physics_concept"], "family": p["surface_family"]}
        for i, p in enumerate(prompts)
    }


def compute_feature_stats(df, prompt_meta):
    """Compute per-feature Stage 1 + Stage 2 statistics."""
    # Join prompt metadata
    df = df.copy()
    df["concept"] = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("concept"))
    df["family"]  = df["prompt_idx"].map(lambda i: prompt_meta.get(i, {}).get("family"))

    records = []
    for feat_id, grp in df.groupby("feature_id"):
        layer = grp["layer"].iloc[0]

        alpha = grp[grp["concept"] == "alpha_decay"]
        beta  = grp[grp["concept"] == "beta_decay"]

        mu_alpha = alpha["effect_size"].mean() if len(alpha) > 0 else 0.0
        mu_beta  = beta["effect_size"].mean()  if len(beta)  > 0 else 0.0
        sfr_alpha = alpha["sign_flipped"].mean() if len(alpha) > 0 else 0.0
        sfr_beta  = beta["sign_flipped"].mean()  if len(beta)  > 0 else 0.0

        # Stage 1: disruption-based asymmetry.
        # d_X = max(0, -μ_X): how much ablating this feature HURTS concept X.
        # Positive effect_size = ablation improved the score (feature was competing/hurting).
        # Negative effect_size = ablation hurt the score (feature was supporting).
        # A > 0 → feature supports β; A < 0 → feature supports α.
        d_alpha = max(0.0, -mu_alpha)
        d_beta  = max(0.0, -mu_beta)
        A = (d_beta - d_alpha) / (d_beta + d_alpha + EPS)
        strength = max(d_alpha, d_beta)

        # Stage 2: sfr by family
        sfr_by_fam = {}
        for fam in FAMILIES:
            fam_rows = grp[grp["family"] == fam]
            sfr_by_fam[fam] = fam_rows["sign_flipped"].mean() if len(fam_rows) > 0 else 0.0

        mean_sfr = np.mean(list(sfr_by_fam.values()))
        # Normalised family profile f̃
        norm_profile = {
            fam: sfr_by_fam[fam] / (mean_sfr + EPS)
            for fam in FAMILIES
        }
        max_norm = max(norm_profile.values())
        dominant_fam = max(norm_profile, key=norm_profile.get) if max_norm >= FAMILY_CONC_THRESH else "general"

        rec = {
            "feature_id": feat_id,
            "layer": layer,
            "mu_alpha": round(mu_alpha, 5),
            "mu_beta":  round(mu_beta,  5),
            "sfr_alpha": round(sfr_alpha, 4),
            "sfr_beta":  round(sfr_beta,  4),
            "asymmetry_A": round(A, 4),
            "strength":    round(strength, 5),
            "mean_sfr":    round(mean_sfr, 4),
            "family_label": dominant_fam,
            "family_conc_score": round(max_norm, 3),
        }
        for fam in FAMILIES:
            rec[f"sfr_{fam}"] = round(sfr_by_fam[fam], 4)

        records.append(rec)

    feat_df = pd.DataFrame(records).sort_values(["layer", "feature_id"])
    return feat_df


def assign_coarse_groups(feat_df):
    """Assign Stage 1 coarse group labels."""
    strength_threshold = np.percentile(feat_df["strength"], STRENGTH_PCTILE)

    def classify(row):
        if row["strength"] <= strength_threshold:
            return "background"
        if row["asymmetry_A"] > ASYMMETRY_THRESH:
            return "beta_dominant"
        if row["asymmetry_A"] < -ASYMMETRY_THRESH:
            return "alpha_dominant"
        return "concept_general"

    feat_df = feat_df.copy()
    feat_df["coarse_group"] = feat_df.apply(classify, axis=1)
    return feat_df


def build_effect_matrix(df, feat_df, prompt_meta):
    """Build 40×108 signed effect_size matrix for Ward clustering."""
    feature_ids = list(feat_df["feature_id"])
    prompt_indices = sorted(prompt_meta.keys())

    # Pivot: rows = features, cols = prompts
    pivot = df.pivot_table(
        index="feature_id", columns="prompt_idx",
        values="effect_size", aggfunc="mean"
    )
    pivot = pivot.reindex(index=feature_ids, columns=prompt_indices, fill_value=0.0)
    return pivot


def ward_cluster(pivot, n_clusters=4):
    """Run Ward hierarchical clustering on the effect matrix."""
    X = pivot.values  # (40, 108)
    # Cluster features (rows)
    row_link = linkage(pdist(X, metric="euclidean"), method="ward")
    row_order = dendrogram(row_link, no_plot=True)["leaves"]
    # Cluster prompts (cols)
    col_link = linkage(pdist(X.T, metric="euclidean"), method="ward")
    col_order = dendrogram(col_link, no_plot=True)["leaves"]
    row_labels = fcluster(row_link, n_clusters, criterion="maxclust")
    return row_order, col_order, row_labels


def make_grouping_heatmap(feat_df, circuit_features):
    """
    Main annotated heatmap: rows = features sorted by coarse_group × family_label,
    columns = [mu_alpha, mu_beta, sfr_F0..F4] with circuit star annotation.
    """
    GROUP_ORDER = ["beta_dominant", "concept_general", "alpha_dominant", "background"]
    FAM_ORDER   = ["general", "F0", "F1", "F2", "F3", "F4"]
    GROUP_COLORS = {
        "beta_dominant":   "#4a90d9",
        "concept_general": "#7b68ee",
        "alpha_dominant":  "#e07b39",
        "background":      "#555",
    }

    feat_df = feat_df.copy()
    feat_df["_g_rank"] = feat_df["coarse_group"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    )
    feat_df["_f_rank"] = feat_df["family_label"].map(
        {f: i for i, f in enumerate(FAM_ORDER)}
    )
    feat_df = feat_df.sort_values(["_g_rank", "_f_rank", "strength"], ascending=[True, True, False])

    # Build Z matrix: rows = features, cols = [mu_alpha, mu_beta, sfr_F0..F4]
    col_names = ["μ_α (mean effect)", "μ_β (mean effect)"] + [f"sfr_{f}" for f in FAMILIES]
    Z = feat_df[["mu_alpha", "mu_beta"] + [f"sfr_{f}" for f in FAMILIES]].values
    y_labels = []
    for _, row in feat_df.iterrows():
        star = "★ " if row["feature_id"] in circuit_features else ""
        y_labels.append(f"{star}{row['feature_id']}")

    # Group colour bars as shapes on left margin
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Z,
        x=col_names,
        y=y_labels,
        colorscale=[
            [0.0, "#0f1117"],
            [0.25, "#1a365d"],
            [0.5, "#2b6cb0"],
            [0.75, "#ed8936"],
            [1.0, "#f6e05e"],
        ],
        zmid=0,
        colorbar=dict(
            title="Value",
            tickfont=dict(size=9),
            len=0.8,
        ),
        hovertemplate="Feature: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>",
        xgap=2,
        ygap=1,
    ))

    # Add vertical separator between effect and sfr columns
    fig.add_vline(x=1.5, line_color="#888", line_width=1, line_dash="dot")

    # Group boundary horizontal lines
    groups_seen = []
    for i, (_, row) in enumerate(feat_df.iterrows()):
        g = row["coarse_group"]
        if g not in groups_seen:
            groups_seen.append(g)
            if i > 0:
                fig.add_hline(y=i - 0.5, line_color="#ccc", line_width=1, line_dash="dash")

    n_features = len(feat_df)
    fig.update_layout(
        title="Feature grouping: Stage 1 (coarse group) × Stage 2 (family annotation)",
        height=max(500, 24 * n_features),
        width=900,
        margin=dict(l=200, r=80, t=60, b=80),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#ccc"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )

    # Annotation: group label on right side
    group_boundaries = []
    prev_g = None
    start_i = 0
    for i, (_, row) in enumerate(feat_df.iterrows()):
        if row["coarse_group"] != prev_g:
            if prev_g is not None:
                group_boundaries.append((prev_g, start_i, i - 1))
            prev_g = row["coarse_group"]
            start_i = i
    group_boundaries.append((prev_g, start_i, len(feat_df) - 1))

    for g, s, e in group_boundaries:
        mid = (s + e) / 2
        fig.add_annotation(
            x=len(col_names) - 0.4,
            y=mid,
            text=g.replace("_", " ").upper(),
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="left",
            font=dict(size=9, color=GROUP_COLORS.get(g, "#888")),
        )

    return fig


def make_effect_matrix_heatmap(pivot, feat_df, prompt_meta, circuit_features, ward_results):
    """Ward-clustered 40×108 effect matrix heatmap."""
    row_order, col_order, row_clusters = ward_results

    feat_ids_ordered = [pivot.index[i] for i in row_order]
    prompt_ids_ordered = [pivot.columns[i] for i in col_order]

    Z = pivot.loc[feat_ids_ordered, prompt_ids_ordered].values

    # Y axis: feature labels
    y_labels = []
    for fid in feat_ids_ordered:
        row = feat_df[feat_df["feature_id"] == fid].iloc[0]
        star = "★ " if fid in circuit_features else ""
        y_labels.append(f"{star}{fid} [{row['coarse_group'][0].upper()}]")

    # X axis: prompt labels (concept + family)
    x_labels = []
    for pidx in prompt_ids_ordered:
        meta = prompt_meta.get(pidx, {})
        concept_short = "α" if meta.get("concept") == "alpha_decay" else "β"
        x_labels.append(f"{concept_short}/{meta.get('family','?')}")

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#2a0080"],
            [0.35, "#0f1117"],
            [0.5, "#0f1117"],
            [0.65, "#0f1117"],
            [1.0, "#f6e05e"],
        ],
        zmid=0,
        colorbar=dict(title="effect_size", tickfont=dict(size=9), len=0.8),
        hovertemplate="Feature: %{y}<br>Prompt: %{x}<br>Effect: %{z:.4f}<extra></extra>",
        xgap=0,
        ygap=1,
    ))

    fig.update_layout(
        title="Ward-clustered effect matrix (signed effect_size, 40 features × 108 prompts)",
        height=max(600, 20 * len(feat_ids_ordered)),
        width=1400,
        margin=dict(l=220, r=80, t=60, b=120),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#ccc"),
        xaxis=dict(tickangle=-75, tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )

    return fig


def print_summary(feat_df, circuit_features):
    print("\n=== Stage 1: Coarse Group Summary ===")
    for g, grp in feat_df.groupby("coarse_group"):
        print(f"\n{g.upper()} ({len(grp)} features):")
        for _, row in grp.sort_values("strength", ascending=False).iterrows():
            star = "★" if row["feature_id"] in circuit_features else " "
            print(
                f"  {star} {row['feature_id']:18s} "
                f"A={row['asymmetry_A']:+.3f}  str={row['strength']:.4f}  "
                f"μα={row['mu_alpha']:+.4f}  μβ={row['mu_beta']:+.4f}  "
                f"sfrα={row['sfr_alpha']:.2%}  sfrβ={row['sfr_beta']:.2%}  "
                f"fam={row['family_label']}"
            )

    print("\n=== Stage 2: Family Annotation Summary ===")
    fam_col_order = ["feature_id", "coarse_group"] + [f"sfr_{f}" for f in FAMILIES] + ["family_label", "family_conc_score"]
    print(feat_df[fam_col_order].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Script 17: Feature intervention profile grouping")
    parser.add_argument("--n_clusters", type=int, default=4, help="Ward clusters for effect matrix")
    parser.add_argument("--no_plots", action="store_true", help="Skip HTML plot generation")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, prompts, circuit_features = load_data()
    prompt_meta = build_prompt_meta(prompts)

    print("Computing feature statistics...")
    feat_df = compute_feature_stats(df, prompt_meta)
    feat_df = assign_coarse_groups(feat_df)

    print_summary(feat_df, circuit_features)

    # Save feature table
    out_csv = OUT_DIR / "feature_groups.csv"
    feat_df.drop(columns=["_g_rank", "_f_rank"], errors="ignore").to_csv(out_csv, index=False)
    print(f"\nSaved feature table → {out_csv}")

    if not args.no_plots:
        print("Building grouping heatmap...")
        fig1 = make_grouping_heatmap(feat_df, circuit_features)
        out_h1 = OUT_DIR / "grouping_heatmap.html"
        fig1.write_html(str(out_h1))
        print(f"Saved → {out_h1}")

        print("Building effect matrix (Ward)...")
        pivot = build_effect_matrix(df, feat_df, prompt_meta)
        ward_results = ward_cluster(pivot, n_clusters=args.n_clusters)
        feat_df_with_cluster = feat_df.copy()
        cluster_map = {pivot.index[i]: c for i, c in zip(range(len(pivot)), ward_results[2])}
        feat_df_with_cluster["ward_cluster"] = feat_df_with_cluster["feature_id"].map(cluster_map)

        fig2 = make_effect_matrix_heatmap(pivot, feat_df, prompt_meta, circuit_features, ward_results)
        out_h2 = OUT_DIR / "effect_matrix.html"
        fig2.write_html(str(out_h2))
        print(f"Saved → {out_h2}")

        # Cross-tab: Ward cluster vs Stage 1 group
        print("\n=== Ward cluster × Stage 1 coarse group ===")
        print(pd.crosstab(
            feat_df_with_cluster["ward_cluster"],
            feat_df_with_cluster["coarse_group"]
        ))


if __name__ == "__main__":
    main()
