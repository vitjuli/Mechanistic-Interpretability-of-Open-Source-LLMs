#!/usr/bin/env python3
"""
Stage 0 — Intervention Coverage Audit

Computes coverage of the effect matrix (feature × layer × experiment_type)
and distinguishes structural missingness (feature cannot appear in the wrong
layer) from effective missingness (feature was never tested despite being
a graph candidate).

Key insight: in STRICT mode every graph feature is tested for ALL intervention
prompts at its own layer.  The 94-95% "overall missingness" reported by
prepare.py is almost entirely STRUCTURAL (feature L14_Fxxx has NaN in all
columns for layers 10-13 and 15-25, by construction).  The relevant metric
for Stage 3 clustering is EFFECTIVE MISSINGNESS — how many (feature,
experiment_type) cells are actually empty.

Usage:
    # CSD3 antonym:
    python scripts/a_analyze_interventions_coverage.py \
        --interventions_dir data/results/interventions/antonym_operation \
        --behaviour antonym_operation \
        --out_dir data/analysis/antonym_train_n80_coverage

    # Local grammar_agreement test:
    python scripts/a_analyze_interventions_coverage.py \
        --interventions_dir data/results/interventions/grammar_agreement \
        --behaviour grammar_agreement \
        --out_dir data/analysis/grammar_agreement_train_n80_coverage
"""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_interventions(interventions_dir: Path, behaviour: str) -> pd.DataFrame:
    """Load all three intervention CSVs; skip missing files with a warning."""
    patterns = [
        f"intervention_ablation_{behaviour}.csv",
        f"intervention_patching_{behaviour}.csv",
        f"intervention_steering_{behaviour}.csv",
    ]
    dfs = []
    for pat in patterns:
        p = interventions_dir / pat
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  Loaded {pat}: {len(df):,} rows")
        else:
            print(f"  WARNING: {pat} not found — skipping")
    if not dfs:
        raise FileNotFoundError(
            f"No intervention CSVs found in {interventions_dir}"
        )
    return pd.concat(dfs, ignore_index=True)


def parse_feature_indices(series: pd.Series) -> pd.Series:
    def _parse(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return []
        return []
    return series.apply(_parse)


# ---------------------------------------------------------------------------
# Build coverage records
# ---------------------------------------------------------------------------

def explode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode one row per (layer, experiment_type, prompt_idx, feature_idx).
    Returns a tidy DataFrame with column feature_id = 'L{layer}_F{feat_idx}'.
    """
    df = df.copy()
    df["feature_indices"] = parse_feature_indices(df["feature_indices"])
    exploded = df.explode("feature_indices").dropna(subset=["feature_indices"])
    exploded["feature_indices"] = exploded["feature_indices"].astype(int)
    exploded["feature_id"] = exploded.apply(
        lambda r: f"L{int(r['layer'])}_F{int(r['feature_indices'])}", axis=1
    )
    return exploded[["feature_id", "layer", "experiment_type", "prompt_idx", "effect_size"]]


# ---------------------------------------------------------------------------
# Coverage matrices
# ---------------------------------------------------------------------------

def build_full_pivot(records: pd.DataFrame):
    """
    48-dim pivot: feature_id × (experiment_type, layer).
    This is the same matrix that prepare.py uses — structurally sparse.
    """
    pivot = records.groupby(
        ["feature_id", "experiment_type", "layer"]
    )["prompt_idx"].nunique().unstack(["experiment_type", "layer"], fill_value=0)

    # All unique features and all (exp, layer) combos
    all_features = sorted(records["feature_id"].unique())
    all_exp_layer = sorted(
        records[["experiment_type", "layer"]]
        .drop_duplicates()
        .apply(lambda r: (r["experiment_type"], int(r["layer"])), axis=1)
        .tolist()
    )
    idx = pd.MultiIndex.from_tuples(all_exp_layer, names=["experiment_type", "layer"])
    pivot = pivot.reindex(index=all_features, columns=idx, fill_value=0)
    return pivot


def build_effective_pivot(records: pd.DataFrame):
    """
    3-dim effective pivot: feature_id × experiment_type only.
    A feature at layer L can only have effects in its own layer, so we
    aggregate across all layers (already filtered by the feature_id's layer).
    This eliminates structural missingness entirely.
    """
    pivot = records.groupby(
        ["feature_id", "experiment_type"]
    )["prompt_idx"].nunique().unstack("experiment_type", fill_value=0)

    all_features = sorted(records["feature_id"].unique())
    all_expts    = sorted(records["experiment_type"].unique())
    pivot = pivot.reindex(index=all_features, columns=all_expts, fill_value=0)
    return pivot


def missingness_stats(pivot) -> dict:
    n_total   = pivot.shape[0] * pivot.shape[1]
    n_missing = int((pivot == 0).sum().sum())
    return {
        "n_rows": pivot.shape[0],
        "n_cols": pivot.shape[1],
        "n_total_cells": n_total,
        "n_observed": n_total - n_missing,
        "n_missing": n_missing,
        "missingness_rate": n_missing / max(n_total, 1),
    }


# ---------------------------------------------------------------------------
# Subsampling curve (effective space)
# ---------------------------------------------------------------------------

def subsampling_curve(records: pd.DataFrame, seed: int = 42):
    """
    For K = 1 … N prompts, sub-sample K prompts and measure effective
    coverage (fraction of feature × experiment_type cells that have ≥1
    observation).  Returns (ks, coverages).
    """
    rng = np.random.default_rng(seed)
    all_prompts = sorted(records["prompt_idx"].unique())
    N = len(all_prompts)

    # Build lookup: prompt → set of (feature_id, experiment_type) cells
    cell_by_prompt = defaultdict(set)
    for _, row in records.iterrows():
        cell_by_prompt[int(row["prompt_idx"])].add(
            (row["feature_id"], row["experiment_type"])
        )

    all_cells = set()
    for cells in cell_by_prompt.values():
        all_cells |= cells
    n_total = len(all_cells)

    ks, coverages = [], []
    for k in range(1, N + 1):
        selected = rng.choice(all_prompts, size=k, replace=False)
        covered  = set()
        for p in selected:
            covered |= cell_by_prompt[p]
        ks.append(k)
        coverages.append(len(covered) / n_total if n_total > 0 else 0.0)

    return ks, coverages, n_total


def target_n(ks, coverages, targets=(0.80, 0.90)):
    """Smallest K achieving each coverage target (or None if never reached)."""
    result = {}
    for t in targets:
        result[t] = next((k for k, c in zip(ks, coverages) if c >= t), None)
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_coverage_curve(ks, coverages, current_n, targets_n, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [c * 100 for c in coverages], "b-o", markersize=4,
            label="Effective coverage (feature×exp_type)")
    ax.axvline(current_n, color="gray", linestyle="--",
               label=f"Current N = {current_n}")
    colors = ["#e15759", "#f28e2b", "#76b7b2"]
    for (target, k_needed), color in zip(targets_n.items(), colors):
        lbl = f"{target*100:.0f}% coverage → N ≥ {k_needed}" if k_needed else \
              f"{target*100:.0f}% coverage: unreachable"
        ax.axhline(target * 100, color=color, linestyle=":", alpha=0.9, label=lbl)
    ax.set_xlabel("N intervention prompts")
    ax.set_ylabel("Effective coverage (%)")
    ax.set_title("Effective effect-matrix coverage vs N intervention prompts")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_vs_nprompts.png", dpi=120)
    plt.close(fig)


def plot_prompts_per_feature(records: pd.DataFrame, out_dir: Path):
    """Histogram: how many distinct prompts tested each feature (any experiment)."""
    counts = records.groupby("feature_id")["prompt_idx"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts.values, bins=max(1, len(counts) // 5), color="#4e79a7",
            edgecolor="white")
    ax.axvline(counts.mean(), color="red", linestyle="--",
               label=f"Mean = {counts.mean():.1f}")
    ax.set_xlabel("N prompts that tested this feature (any experiment type)")
    ax.set_ylabel("Feature count")
    ax.set_title("Per-feature prompt coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prompts_per_feature_hist.png", dpi=120)
    plt.close(fig)
    return counts


def plot_effective_heatmap(eff_pivot: pd.DataFrame, out_dir: Path):
    """Heatmap: features (rows) × experiment_type (cols), value = n_prompts tested."""
    fig, ax = plt.subplots(figsize=(6, max(4, len(eff_pivot) * 0.12 + 2)))
    presence = (eff_pivot > 0).astype(int)
    im = ax.imshow(presence.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(eff_pivot.columns)))
    ax.set_xticklabels(list(eff_pivot.columns), rotation=30, ha="right", fontsize=8)
    ytick_step = max(1, len(eff_pivot) // 30)
    ax.set_yticks(range(0, len(eff_pivot), ytick_step))
    ax.set_yticklabels(
        [eff_pivot.index[i] for i in range(0, len(eff_pivot), ytick_step)],
        fontsize=6,
    )
    ax.set_title(
        f"Effective coverage: {len(eff_pivot)} features × "
        f"{len(eff_pivot.columns)} experiment types\n"
        f"(blue = at least 1 prompt tested, white = never tested)"
    )
    plt.colorbar(im, ax=ax, label="Observed (0/1)", shrink=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_coverage_heatmap.png", dpi=100,
                bbox_inches="tight")
    plt.close(fig)


def plot_full_heatmap(full_pivot: pd.DataFrame, out_dir: Path, max_rows: int = 80):
    """Full 48-dim heatmap showing structural zeros."""
    disp = full_pivot.iloc[:max_rows, :]
    presence = (disp > 0).astype(int)
    fig, ax = plt.subplots(
        figsize=(max(6, disp.shape[1] * 0.35 + 2),
                 max(4, disp.shape[0] * 0.13 + 2))
    )
    im = ax.imshow(presence.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    col_labels = [f"{str(et)[:3]}L{int(l)}" for et, l in disp.columns]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ytick_step = max(1, len(disp) // 20)
    ax.set_yticks(range(0, len(disp), ytick_step))
    ax.set_yticklabels(
        [disp.index[i] for i in range(0, len(disp), ytick_step)],
        fontsize=6,
    )
    ax.set_title(
        f"Full 48-dim matrix (first {len(disp)} features × {len(disp.columns)} cells)\n"
        f"Most white cells are STRUCTURAL zeros (feature ≠ cell's layer)"
    )
    plt.colorbar(im, ax=ax, label="Observed (0/1)", shrink=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "full_coverage_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: Path, behaviour: str, current_n: int,
                 full_stats: dict, eff_stats: dict,
                 struct_miss: float,
                 ks, coverages, targets_n: dict,
                 prompts_per_feature,
                 records: pd.DataFrame):

    lines = []
    def p(*args): lines.append(" ".join(str(a) for a in args))

    p("=" * 70)
    p("INTERVENTION COVERAGE AUDIT  —  Stage 0")
    p(f"Behaviour: {behaviour}")
    p("=" * 70)
    p()

    p("─── 1. Dataset overview ───")
    p(f"  N intervention prompts (unique)  : {current_n}")
    n_rows = sum(1 for _ in records.iterrows())  # avoid shape confusion
    p(f"  Total exploded feature records   : {len(records):,}")
    for et in sorted(records["experiment_type"].unique()):
        sub = records[records["experiment_type"] == et]
        p(f"    {et}: {sub['feature_id'].nunique()} unique features, "
          f"{sub['prompt_idx'].nunique()} prompts")
    p()

    p("─── 2. FULL 48-dim matrix (as used by prepare.py) ───")
    p(f"  Dimensions  : {full_stats['n_rows']} features × {full_stats['n_cols']} (exp,layer) pairs")
    p(f"  Total cells : {full_stats['n_total_cells']:,}")
    p(f"  Observed    : {full_stats['n_observed']:,}  ({1-full_stats['missingness_rate']:.1%})")
    p(f"  Missing     : {full_stats['n_missing']:,}  ({full_stats['missingness_rate']:.1%})")
    p()

    p("─── 3. Structural vs effective missingness ───")
    p(f"  Structural missingness (wrong-layer NaN, unavoidable) : {struct_miss:.1%}")
    p(f"  Effective missingness  (right-layer cells, unobserved): {eff_stats['missingness_rate']:.1%}")
    p(f"  → The {full_stats['missingness_rate']:.1%} reported by prepare.py is ~{struct_miss:.1%} structural.")
    p(f"  → Only {eff_stats['missingness_rate']:.1%} of cells that SHOULD be observed are missing.")
    p()

    p("─── 4. Effective 3-dim matrix (feature × experiment_type only) ───")
    p(f"  Dimensions  : {eff_stats['n_rows']} features × {eff_stats['n_cols']} experiment types")
    p(f"  Total cells : {eff_stats['n_total_cells']}")
    p(f"  Observed    : {eff_stats['n_observed']}  ({1-eff_stats['missingness_rate']:.1%})")
    p(f"  Missing     : {eff_stats['n_missing']}  ({eff_stats['missingness_rate']:.1%})")
    p()

    p("─── 5. Per-feature prompt coverage ───")
    p(f"  Mean prompts per feature  : {prompts_per_feature.mean():.1f}")
    p(f"  Median                    : {prompts_per_feature.median():.1f}")
    p(f"  Min                       : {prompts_per_feature.min()}")
    p(f"  Max                       : {prompts_per_feature.max()}")
    p(f"  Features tested ≥1 prompt : {(prompts_per_feature >= 1).sum()} / {len(prompts_per_feature)}")
    p()

    p("─── 6. Effective coverage vs N prompts (subsampling curve) ───")
    for k, c in zip(ks, coverages):
        bar = "█" * int(c * 30)
        p(f"  N={k:4d}  {c:.1%}  {bar}")
    p()

    p("─── 7. Recommended N for target effective coverage ───")
    for target, k_needed in targets_n.items():
        miss_target = 1.0 - target
        if k_needed is not None:
            p(f"  Target: effective coverage ≥ {target:.0%}  "
              f"(missingness ≤ {miss_target:.0%})  →  N ≥ {k_needed}")
        else:
            plateau = max(coverages) if coverages else 0
            p(f"  Target: effective coverage ≥ {target:.0%}  →  NOT REACHABLE  "
              f"(curve plateaus at {plateau:.1%})")
    p()

    p("─── 8. Implications for Stage 1 and Stage 3 ───")
    p(f"""  a) STAGE 3 CLUSTERING: use the 3-dim effective matrix (feature × experiment_type),
     NOT the 48-dim full matrix.  Zero-imputing 93%+ structural zeros and then
     normalising/clustering produces artefacts, not real clusters.
     The effective matrix has {eff_stats['missingness_rate']:.1%} missing cells —
     much more tractable.

  b) STAGE 1 PROMPTS: more intervention prompts help by:
       • Reducing variance in mean_effect_size estimates
       • Ensuring rare features (low frequency in top-k) are tested on ≥1 prompt
       • Lowering effective missingness if some features are not always top-k
     See Section 7 for recommended N to achieve target effective coverage.
     Note: if current effective missingness is already low, the primary gain
     from more prompts is lower-variance effect estimates for clustering.

  c) STAGE 2 COMMUNITY DETECTION: does not depend on intervention data.
     Communities should be derived from feature attribute vectors or
     co-activation counts (see topology report).""")
    p()

    p("=" * 70)
    p("END OF REPORT")
    p("=" * 70)

    text = "\n".join(lines)
    path.write_text(text)
    print(text)

    # Machine-readable summary
    summary = {
        "behaviour": behaviour,
        "n_prompts": current_n,
        "full_matrix_missingness": full_stats["missingness_rate"],
        "structural_missingness": struct_miss,
        "effective_missingness": eff_stats["missingness_rate"],
        "n_features": eff_stats["n_rows"],
        "n_experiment_types": eff_stats["n_cols"],
        "subsampling_curve": {"ks": ks, "coverages": coverages},
        "target_n": {str(k): v for k, v in targets_n.items()},
    }
    (path.parent / "coverage_stats.json").write_text(
        json.dumps(summary, indent=2)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Intervention coverage audit"
    )
    parser.add_argument("--interventions_dir", required=True)
    parser.add_argument("--behaviour", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iv_dir  = Path(args.interventions_dir)

    print(f"Loading interventions from: {iv_dir}")
    df = load_interventions(iv_dir, args.behaviour)

    print("Exploding feature indices...")
    records = explode_features(df)
    current_n = records["prompt_idx"].nunique()
    print(f"  {current_n} unique prompts, {len(records):,} exploded records")

    print("Building coverage matrices...")
    full_pivot = build_full_pivot(records)
    eff_pivot  = build_effective_pivot(records)

    full_stats = missingness_stats(full_pivot)
    eff_stats  = missingness_stats(eff_pivot)

    # Structural missingness: fraction of full-matrix cells that can't be
    # observed (feature's layer ≠ cell's layer)
    n_features = len(full_pivot)
    n_full_dims = full_pivot.shape[1]  # number of (exp_type, layer) combos
    n_exp_types = eff_pivot.shape[1]
    # Each feature can only appear in n_exp_types cells (one per experiment at its layer)
    n_possible_structural = n_features * n_exp_types
    n_structural_zeros = n_features * n_full_dims - n_possible_structural
    struct_miss = n_structural_zeros / (n_features * n_full_dims)

    print(f"  Full matrix: {full_stats['n_rows']}×{full_stats['n_cols']} "
          f"missingness={full_stats['missingness_rate']:.1%}")
    print(f"  Effective  : {eff_stats['n_rows']}×{eff_stats['n_cols']} "
          f"missingness={eff_stats['missingness_rate']:.1%}")
    print(f"  Structural missingness (unavoidable): {struct_miss:.1%}")

    print("Computing subsampling curve...")
    ks, coverages, n_eff_cells = subsampling_curve(records)
    targets_n = target_n(ks, coverages, targets=(0.80, 0.90))

    print("Generating plots...")
    plot_coverage_curve(ks, coverages, current_n, targets_n, out_dir)
    prompts_per_feature = plot_prompts_per_feature(records, out_dir)
    plot_effective_heatmap(eff_pivot, out_dir)
    plot_full_heatmap(full_pivot, out_dir)

    # Save pivots
    eff_pivot.to_csv(out_dir / "effective_coverage_matrix.csv")
    full_pivot.to_csv(out_dir / "full_coverage_matrix.csv")

    write_report(
        out_dir / "coverage_report.txt",
        args.behaviour, current_n,
        full_stats, eff_stats, struct_miss,
        ks, coverages, targets_n,
        prompts_per_feature,
        records,
    )

    print(f"\nOutputs written to: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
