#!/usr/bin/env python3
"""
Sanity-check the offline UI dataset produced by 09_prepare_offline_ui.py.

Verifies existence, non-emptiness, required columns, and structural integrity.

Usage:
    python scripts/09_sanity_check_ui_data.py --run_dir data/ui_offline/<run_id>
"""

import argparse
import json
import sys
from pathlib import Path


def check(condition: bool, msg: str, warnings: list, errors: list, is_error: bool = True):
    """Record a pass/fail check."""
    if condition:
        print(f"  PASS  {msg}")
    else:
        marker = "ERROR" if is_error else "WARN "
        print(f"  {marker} {msg}")
        (errors if is_error else warnings).append(msg)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check offline UI dataset")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    errors = []
    warnings = []

    print(f"\nSanity checking: {run_dir}\n")

    # --- Existence checks ---
    print("[1] File existence")
    # Parquet files are optional (need pyarrow); CSV always produced
    required_files = [
        "interventions.csv",
        "audit.json",
        "common_prompt_idx.json",
        "interventions_layer_agg.csv",
        "interventions_prompt_agg.csv",
        "interventions_feature_agg.csv",
        "graph.json",
        "supernodes.json",
        "run_manifest.json",
    ]
    optional_files = [
        "interventions.parquet",
        "interventions_layer_agg.parquet",
        "interventions_prompt_agg.parquet",
        "interventions_feature_agg.parquet",
        "feature_importance.csv",
        "feature_importance.parquet",
        "supernodes_summary.csv",
        "supernodes_summary.parquet",
        "supernodes_effect.json",
        "supernodes_effect_summary.csv",
        "supernodes_effect_summary.parquet",
    ]

    for fname in required_files:
        check((run_dir / fname).exists(), f"{fname} exists", warnings, errors)

    for fname in optional_files:
        check(
            (run_dir / fname).exists(), f"{fname} exists (optional)",
            warnings, errors, is_error=False,
        )

    # --- Content checks ---
    print("\n[2] Content validation")

    try:
        import pandas as pd

        # Load from parquet if available, else CSV
        def _load(stem):
            pq = run_dir / f"{stem}.parquet"
            csv = run_dir / f"{stem}.csv"
            if pq.exists():
                return pd.read_parquet(pq), stem + ".parquet"
            elif csv.exists():
                return pd.read_csv(csv), stem + ".csv"
            return pd.DataFrame(), None

        df, src = _load("interventions")
        if src:
            check(len(df) > 0, f"{src} has {len(df)} rows", warnings, errors)

            required_cols = [
                "experiment_type", "prompt_idx", "layer", "feature_indices",
                "baseline_logit_diff", "intervened_logit_diff", "effect_size",
                "abs_effect_size", "sign_flipped",
                "behaviour", "split", "run_id",
                "in_common_prompt_set", "in_common_layer_set",
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            check(
                len(missing_cols) == 0,
                f"{src} has all required columns (missing: {missing_cols})",
                warnings, errors,
            )

            exp_types = sorted(df["experiment_type"].unique().tolist())
            check(
                len(exp_types) >= 2,
                f"Has {len(exp_types)} experiment types: {exp_types}",
                warnings, errors,
            )

        # Aggregates
        for agg_name in ["interventions_layer_agg", "interventions_prompt_agg", "interventions_feature_agg"]:
            agg_df, agg_src = _load(agg_name)
            if agg_src:
                check(len(agg_df) > 0, f"{agg_src} has {len(agg_df)} rows", warnings, errors)

    except ImportError:
        print("  SKIP  pandas not available — cannot validate data files")

    # graph.json
    graph_path = run_dir / "graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            graph = json.load(f)
        check("nodes" in graph, "graph.json has 'nodes' key", warnings, errors)
        check("links" in graph, "graph.json has 'links' key", warnings, errors)
        n_nodes = len(graph.get("nodes", []))
        n_links = len(graph.get("links", []))
        check(n_nodes > 0, f"graph.json has {n_nodes} nodes", warnings, errors)
        check(n_links > 0, f"graph.json has {n_links} links", warnings, errors)

    # supernodes.json
    sn_path = run_dir / "supernodes.json"
    if sn_path.exists():
        with open(sn_path) as f:
            sn = json.load(f)
        check(len(sn) > 0, f"supernodes.json has {len(sn)} communities", warnings, errors)

    # audit.json
    audit_path = run_dir / "audit.json"
    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)
        missing_prompts = audit.get("missing_prompt_idx", {})
        if missing_prompts:
            for exp, idxs in missing_prompts.items():
                print(f"  INFO  {exp} missing prompt_idx: {idxs}")
        else:
            print("  INFO  No missing prompt_idx across experiment types")

    # run_manifest.json
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        check("parameters" in manifest, "manifest has parameters", warnings, errors)
        check("inputs" in manifest, "manifest has inputs", warnings, errors)

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Results: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    if not errors:
        print("\nAll critical checks passed.")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
