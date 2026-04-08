#!/usr/bin/env python3
"""
Prepare dashboard_b1/public/data/ from multilingual_circuits_b1 B1-v2 canonical outputs.

Generates all files required by the dashboard_b1 React app, including:
  - Standard intervention/graph/supernode files (expected by base loader)
  - B1-specific extras: IoU data, language labels, community summary, circuit, error cases

Usage:
    python scripts/prepare_b1_dashboard.py
    python scripts/prepare_b1_dashboard.py --out_dir dashboard_b1/public/data
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
BEHAVIOUR = "multilingual_circuits_b1"
SPLIT = "train"
N_PROMPTS = 96


def load_intervention_csv(path: Path) -> pd.DataFrame:
    """Load raw intervention CSV from script 07 (has metadata JSON blob)."""
    df = pd.read_csv(path)
    # Parse metadata JSON blob → flat meta.* columns
    meta_cols = {}
    for raw in df["metadata"].astype(str):
        try:
            d = ast.literal_eval(raw)
        except Exception:
            d = {}
        for k, v in d.items():
            key = f"meta.{k}"
            meta_cols.setdefault(key, [])
            meta_cols[key].append(v)
        # Ensure all keys are filled
        for key in list(meta_cols.keys()):
            if len(meta_cols[key]) < len(meta_cols[next(iter(meta_cols))]):
                meta_cols[key].append(None)

    for key, vals in meta_cols.items():
        df[key] = vals

    df.drop(columns=["metadata"], inplace=True, errors="ignore")
    return df


def load_supplement_csv(path: Path) -> pd.DataFrame:
    """Load ablation supplement CSV from script 11 (already has flat meta.*)."""
    return pd.read_csv(path)


def build_interventions(results_dir: Path) -> pd.DataFrame:
    """Merge ablation + patching + supplement into unified interventions.csv."""
    ablation_path = results_dir / "interventions" / BEHAVIOUR / f"intervention_ablation_{BEHAVIOUR}.csv"
    patching_path = results_dir / "interventions" / BEHAVIOUR / f"intervention_patching_C3_{BEHAVIOUR}.csv"
    supplement_path = results_dir / "reasoning_traces" / BEHAVIOUR / f"ablation_supplement_{SPLIT}.csv"

    dfs = []

    for p in [ablation_path, patching_path]:
        if p.exists():
            df = load_intervention_csv(p)
            dfs.append(df)
        else:
            print(f"  WARNING: {p} not found, skipping")

    if supplement_path.exists():
        df_sup = load_supplement_csv(supplement_path)
        dfs.append(df_sup)
    else:
        print(f"  WARNING: supplement not found at {supplement_path}")

    if not dfs:
        raise FileNotFoundError("No intervention data found!")

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Standardise key columns
    merged["behaviour"] = BEHAVIOUR
    merged["split"] = SPLIT
    merged["model_size"] = "4b"
    merged["graph_n_prompts"] = N_PROMPTS
    merged["run_id"] = f"b1_v2_{BEHAVIOUR}_{SPLIT}_n{N_PROMPTS}"
    merged["prep_timestamp"] = pd.Timestamp.now().isoformat()

    # Parse feature_indices if still a string
    def parse_feature_indices(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return [int(v) for v in x.strip("[]").split(",") if v.strip()]
            except Exception:
                return []
        if isinstance(x, (int, float)) and not pd.isna(x):
            return [int(x)]
        return []

    merged["feature_indices"] = merged["feature_indices"].apply(parse_feature_indices)

    # Common prompt / layer flags
    ablation_prompts = set(merged.loc[merged["experiment_type"] == "ablation_zero", "prompt_idx"].dropna().astype(int))
    patching_prompts = set(merged.loc[merged["experiment_type"] == "patching", "prompt_idx"].dropna().astype(int))
    common_prompts = ablation_prompts & patching_prompts
    all_layers = set(merged["layer"].dropna().astype(int))
    ablation_layers = set(merged.loc[merged["experiment_type"] == "ablation_zero", "layer"].dropna().astype(int))
    patching_layers = set(merged.loc[merged["experiment_type"] == "patching", "layer"].dropna().astype(int))
    common_layers = ablation_layers & patching_layers

    merged["in_common_prompt_set"] = merged["prompt_idx"].astype(int).isin(common_prompts)
    merged["in_common_layer_set"] = merged["layer"].astype(int).isin(common_layers)

    # Store common info for later
    merged.attrs["common_prompt_idx"] = sorted(common_prompts)
    merged.attrs["common_layers"] = sorted(common_layers)

    return merged


def compute_feature_agg(interventions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intervention stats per (experiment_type, layer, feature_id)."""
    rows = []
    for _, row in interventions.iterrows():
        exp = row["experiment_type"]
        layer = row["layer"]
        es = row.get("effect_size")
        abs_es = row.get("abs_effect_size")
        rel_es = row.get("relative_effect")
        base_ld = row.get("baseline_logit_diff")
        int_ld = row.get("intervened_logit_diff")
        sf = row.get("sign_flipped", False)
        for fid in row["feature_indices"]:
            rows.append({
                "experiment_type": exp,
                "layer": int(layer),
                "feature_id": int(fid),
                "effect_size": es,
                "abs_effect_size": abs_es,
                "relative_effect": rel_es,
                "baseline_logit_diff": base_ld,
                "intervened_logit_diff": int_ld,
                "sign_flipped": sf,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grp = df.groupby(["experiment_type", "layer", "feature_id"])
    agg = grp.agg(
        mean_abs_effect_size=("abs_effect_size", "mean"),
        median_abs_effect_size=("abs_effect_size", "median"),
        std_abs_effect_size=("abs_effect_size", "std"),
        mean_effect_size=("effect_size", "mean"),
        median_effect_size=("effect_size", "median"),
        mean_relative_effect=("relative_effect", "mean"),
        mean_baseline_logit_diff=("baseline_logit_diff", "mean"),
        mean_intervened_logit_diff=("intervened_logit_diff", "mean"),
        sign_flip_rate=("sign_flipped", "mean"),
        n=("effect_size", "count"),
    ).reset_index()

    return agg


def compute_layer_agg(feature_agg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (experiment_type, layer)."""
    if feature_agg.empty:
        return pd.DataFrame()
    grp = feature_agg.groupby(["experiment_type", "layer"])
    return grp.agg(
        mean_abs_effect_size=("mean_abs_effect_size", "mean"),
        median_abs_effect_size=("mean_abs_effect_size", "median"),
        std_abs_effect_size=("mean_abs_effect_size", "std"),
        mean_effect_size=("mean_effect_size", "mean"),
        median_effect_size=("mean_effect_size", "median"),
        mean_relative_effect=("mean_relative_effect", "mean"),
        mean_baseline_logit_diff=("mean_baseline_logit_diff", "mean"),
        mean_intervened_logit_diff=("mean_intervened_logit_diff", "mean"),
        sign_flip_rate=("sign_flip_rate", "mean"),
        n_features=("feature_id", "count"),
    ).reset_index()


def compute_prompt_agg(interventions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (experiment_type, prompt_idx)."""
    if interventions.empty:
        return pd.DataFrame()
    grp = interventions.groupby(["experiment_type", "prompt_idx"])
    return grp.agg(
        mean_abs_effect_size=("abs_effect_size", "mean"),
        median_abs_effect_size=("abs_effect_size", "median"),
        std_abs_effect_size=("abs_effect_size", "std"),
        mean_effect_size=("effect_size", "mean"),
        median_effect_size=("effect_size", "median"),
        mean_relative_effect=("relative_effect", "mean"),
        mean_baseline_logit_diff=("baseline_logit_diff", "mean"),
        mean_intervened_logit_diff=("intervened_logit_diff", "mean"),
        sign_flip_rate=("sign_flipped", "mean"),
        n_features=("feature_indices", "count"),
    ).reset_index()


def convert_graph(graph_path: Path) -> dict:
    """Convert attribution graph JSON to networkx-style format for dashboard."""
    with open(graph_path) as f:
        g = json.load(f)

    nodes = []
    for n in g["nodes"]:
        nodes.append(n)  # Already has all needed fields

    links = []
    for e in g["edges"]:
        links.append({
            "source": e["source"],
            "target": e["target"],
            "weight": e.get("weight", 1.0),
        })

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }


def build_supernodes(community_summary_path: Path) -> tuple[dict, dict, list]:
    """Build supernodes.json, supernodes_effect.json, supernodes_summary from communities."""
    with open(community_summary_path) as f:
        communities = json.load(f)

    # supernodes.json: {community_id_str: [node_ids]}
    supernodes = {}
    supernodes_effect = {}
    summary_rows = []

    for comm in communities:
        cid = str(comm["community_id"])
        members = comm["members"]
        supernodes[cid] = members
        supernodes_effect[cid] = members

        # Top nodes = first 5
        top_nodes = str(members[:5])
        summary_rows.append({
            "community_id": comm["community_id"],
            "n_features": comm["n_features"],
            "layer_range": comm["layer_range"],
            "dominant_profile": comm["dominant_profile"],
            "top_nodes": top_nodes,
            "n_features": comm["n_features"],
            "representative": members[0] if members else "",
        })

    # supernodes_effect_summary.csv: one row per cluster with stats
    effect_summary_rows = []
    for comm in communities:
        cid = comm["community_id"]
        effect_summary_rows.append({
            "cluster_id": cid,
            "n_features": comm["n_features"],
            "dominant_profile": comm["dominant_profile"],
            "layer_range": comm["layer_range"],
            "representative": comm["members"][0] if comm["members"] else "",
        })

    return supernodes, supernodes_effect, summary_rows, effect_summary_rows


def build_feature_importance(node_labels_path: Path) -> pd.DataFrame:
    """Build feature_importance.csv from node language labels."""
    df = pd.read_csv(node_labels_path)
    result = []
    for _, row in df.iterrows():
        m = __import__("re").match(r"^L(\d+)_F(\d+)$", str(row["node_id"]))
        if not m:
            continue
        layer = int(m.group(1))
        feature_idx = int(m.group(2))
        lang_asym = float(row.get("lang_asym", 0))
        en_freq = float(row.get("en_freq", 0))
        fr_freq = float(row.get("fr_freq", 0))
        lang_profile = str(row.get("lang_profile", "balanced"))
        result.append({
            "layer": layer,
            "feature_idx": feature_idx,
            "mean_activation": en_freq,  # proxy
            "std_activation": abs(en_freq - fr_freq),
            "activation_frequency": (en_freq + fr_freq) / 2,
            "correlation_with_logit_diff": lang_asym,
            "abs_correlation": abs(lang_asym),
            "lang_profile": lang_profile,
            "en_freq": en_freq,
            "fr_freq": fr_freq,
            "lang_asym": lang_asym,
        })
    return pd.DataFrame(result)


def build_iou_data(analysis_dir: Path) -> dict:
    """Combine IoU CSVs into single JSON structure."""
    result = {}
    for mode, fname in [
        ("pooled", "iou_per_layer.csv"),
        ("decision", "iou_per_layer_decision.csv"),
        ("content", "iou_per_layer_content.csv"),
    ]:
        path = analysis_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            result[mode] = df.to_dict(orient="records")
        else:
            print(f"  WARNING: {path} not found, skipping IoU mode '{mode}'")

    # Compute zone summaries
    for mode, rows in list(result.items()):
        if not rows:
            continue
        layers = [r["layer"] for r in rows]
        ious = [r["iou"] for r in rows]
        early = [r["iou"] for r in rows if r["layer"] <= 11]
        mid = [r["iou"] for r in rows if 12 <= r["layer"] <= 20]
        late = [r["iou"] for r in rows if r["layer"] >= 21]
        result[f"{mode}_summary"] = {
            "early_mean": float(np.mean(early)) if early else None,
            "mid_mean": float(np.mean(mid)) if mid else None,
            "late_mean": float(np.mean(late)) if late else None,
            "ratio": float(np.mean(mid) / np.mean(early)) if early and mid and np.mean(early) > 0 else None,
        }

    return result


def build_node_labels_json(node_labels_path: Path) -> dict:
    """Build node_labels.json: {node_id: {lang_profile, en_freq, fr_freq, ...}}."""
    df = pd.read_csv(node_labels_path)
    result = {}
    for _, row in df.iterrows():
        nid = str(row["node_id"])
        result[nid] = {
            "lang_profile": str(row.get("lang_profile", "balanced")),
            "en_freq": float(row.get("en_freq", 0)),
            "fr_freq": float(row.get("fr_freq", 0)),
            "lang_asym": float(row.get("lang_asym", 0)),
            "n_en_active": int(row.get("n_en_active", 0)),
            "n_fr_active": int(row.get("n_fr_active", 0)),
        }
    return result


def build_circuit_json(circuit_path: Path) -> dict:
    """Load circuit JSON and extract useful summary."""
    with open(circuit_path) as f:
        d = json.load(f)
    return {
        "feature_nodes": d["circuit"]["feature_nodes"],
        "edges": d["circuit"]["edges"],
        "n_features": d["circuit"]["n_features"],
        "n_edges": d["circuit"]["n_edges"],
        "n_paths": d["circuit"]["n_paths"],
        "validation": d.get("validation", {}),
        "sufficiency_s1": d.get("sufficiency_s1", {}),
        "sufficiency_s1_5": d.get("sufficiency_s1_5", {}),
        "sufficiency_s2": d.get("sufficiency_s2", {}),
        "metadata": d.get("metadata", {}),
    }


def serialize_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert feature_indices lists back to strings for CSV serialization."""
    out = df.copy()
    out["feature_indices"] = out["feature_indices"].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="dashboard_b1/public/data",
                        help="Output directory (default: dashboard_b1/public/data)")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = ROOT / "data" / "results"
    analysis_dir = ROOT / "data" / "analysis" / BEHAVIOUR

    print("=" * 60)
    print(f"B1 Dashboard Data Preparation")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # --- 1. Interventions ---
    print("\n[1/9] Building interventions...")
    interventions = build_interventions(results_dir)
    common_prompt_idx = sorted(interventions.attrs.get("common_prompt_idx", []))
    common_layers = sorted(interventions.attrs.get("common_layers", []))
    print(f"      {len(interventions)} rows; {len(interventions['experiment_type'].unique())} experiment types")

    interventions_out = serialize_interventions(interventions)
    interventions_out.to_csv(out_dir / "interventions.csv", index=False)
    print(f"      -> interventions.csv")

    # --- 2. Aggregations ---
    print("\n[2/9] Computing aggregations...")
    feature_agg = compute_feature_agg(interventions)
    layer_agg = compute_layer_agg(feature_agg)
    prompt_agg = compute_prompt_agg(interventions)

    feature_agg.to_csv(out_dir / "interventions_feature_agg.csv", index=False)
    layer_agg.to_csv(out_dir / "interventions_layer_agg.csv", index=False)
    prompt_agg.to_csv(out_dir / "interventions_prompt_agg.csv", index=False)
    print(f"      -> {len(feature_agg)} feature_agg rows, {len(layer_agg)} layer_agg rows, {len(prompt_agg)} prompt_agg rows")

    # --- 3. Attribution graph ---
    print("\n[3/9] Converting attribution graph...")
    graph_path = results_dir / "attribution_graphs" / BEHAVIOUR / f"attribution_graph_{SPLIT}_n{N_PROMPTS}_roleaware.json"
    if not graph_path.exists():
        print(f"      WARNING: {graph_path} not found")
    else:
        graph = convert_graph(graph_path)
        with open(out_dir / "graph.json", "w") as f:
            json.dump(graph, f, indent=None)
        print(f"      -> graph.json ({len(graph['nodes'])} nodes, {len(graph['links'])} links)")

    # --- 4. Supernodes from communities ---
    print("\n[4/9] Building supernodes from communities...")
    comm_path = analysis_dir / "community_summary.json"
    if not comm_path.exists():
        print(f"      WARNING: {comm_path} not found")
    else:
        supernodes, supernodes_effect, summary_rows, effect_summary_rows = build_supernodes(comm_path)
        with open(out_dir / "supernodes.json", "w") as f:
            json.dump(supernodes, f, indent=2)
        with open(out_dir / "supernodes_effect.json", "w") as f:
            json.dump(supernodes_effect, f, indent=2)
        pd.DataFrame(summary_rows).to_csv(out_dir / "supernodes_summary.csv", index=False)
        pd.DataFrame(effect_summary_rows).to_csv(out_dir / "supernodes_effect_summary.csv", index=False)
        print(f"      -> {len(supernodes)} communities")

    # --- 5. Feature importance / language labels ---
    print("\n[5/9] Building feature importance from language labels...")
    node_labels_path = analysis_dir / "node_language_labels.csv"
    if not node_labels_path.exists():
        print(f"      WARNING: {node_labels_path} not found")
    else:
        feature_importance = build_feature_importance(node_labels_path)
        feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)
        node_labels = build_node_labels_json(node_labels_path)
        with open(out_dir / "node_labels.json", "w") as f:
            json.dump(node_labels, f, indent=2)
        print(f"      -> {len(feature_importance)} features; {len(node_labels)} label entries")

    # --- 6. Common prompt index ---
    print("\n[6/9] Writing common prompt index...")
    with open(out_dir / "common_prompt_idx.json", "w") as f:
        json.dump({"common_prompt_idx": common_prompt_idx, "common_layers": common_layers}, f, indent=2)
    print(f"      -> {len(common_prompt_idx)} common prompts, {len(common_layers)} common layers")

    # --- 7. Run manifest ---
    print("\n[7/9] Writing run manifest...")
    manifest = {
        "behaviour": BEHAVIOUR,
        "split": SPLIT,
        "n_prompts": N_PROMPTS,
        "model": "Qwen3-4B",
        "model_size": "4b",
        "layers": list(range(10, 26)),
        "run_id": f"b1_v2_{BEHAVIOUR}_{SPLIT}_n{N_PROMPTS}",
        "version": "B1-v2",
        "description": "Multilingual antonym circuits (EN+FR), 8 concepts, 96 train prompts, clean templates",
        "behaviour_type": "abstraction",
        "prep_timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"      -> run_manifest.json")

    # --- 8. B1-specific: IoU data ---
    print("\n[8/9] Building B1-specific extras...")
    iou_data = build_iou_data(analysis_dir)
    with open(out_dir / "iou_data.json", "w") as f:
        json.dump(iou_data, f, indent=2)
    print(f"      -> iou_data.json ({list(iou_data.keys())})")

    # Circuit
    circuit_path = results_dir / "causal_edges" / BEHAVIOUR / f"circuits_{BEHAVIOUR}_{SPLIT}.json"
    if circuit_path.exists():
        circuit = build_circuit_json(circuit_path)
        with open(out_dir / "circuit.json", "w") as f:
            json.dump(circuit, f, indent=2)
        print(f"      -> circuit.json ({circuit['n_features']} features, {circuit['n_edges']} edges)")
    else:
        print(f"      WARNING: circuit not found at {circuit_path}")

    # Error cases
    error_path = results_dir / "reasoning_traces" / BEHAVIOUR / f"error_cases_{SPLIT}.json"
    if error_path.exists():
        import shutil
        shutil.copy(error_path, out_dir / "error_cases.json")
        print(f"      -> error_cases.json")
    else:
        print(f"      WARNING: error_cases not found at {error_path}")

    # Bridge features
    bridge_path = analysis_dir / "bridge_features.csv"
    if bridge_path.exists():
        bridge_df = pd.read_csv(bridge_path)
        bridge_records = bridge_df.to_dict(orient="records")
        with open(out_dir / "bridge_features.json", "w") as f:
            json.dump(bridge_records, f, indent=2)
        n_bridge = int(bridge_df["is_bridge"].sum()) if "is_bridge" in bridge_df.columns else len(bridge_records)
        print(f"      -> bridge_features.json ({n_bridge} bridges)")
    else:
        print(f"      WARNING: bridge_features not found at {bridge_path}")

    # Community summary (copy raw JSON for CommunityCards component)
    if comm_path.exists():
        import shutil
        shutil.copy(comm_path, out_dir / "community_summary.json")
        print(f"      -> community_summary.json")

    # --- 8b. Prompt-level reasoning data ---
    traces_dir = results_dir / "reasoning_traces" / BEHAVIOUR

    # reasoning_traces_train.jsonl → prompt_traces.json (JSON array, one object per prompt)
    traces_jsonl = traces_dir / f"reasoning_traces_{SPLIT}.jsonl"
    if traces_jsonl.exists():
        traces = []
        with open(traces_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        with open(out_dir / "prompt_traces.json", "w") as f:
            json.dump(traces, f)
        print(f"      -> prompt_traces.json ({len(traces)} traces)")
    else:
        print(f"      WARNING: {traces_jsonl} not found")

    # prompt_paths_train.csv → direct copy
    paths_csv = traces_dir / f"prompt_paths_{SPLIT}.csv"
    if paths_csv.exists():
        import shutil
        shutil.copy(paths_csv, out_dir / "prompt_paths.csv")
        print(f"      -> prompt_paths.csv")
    else:
        print(f"      WARNING: {paths_csv} not found")

    # prompt_features_train.csv → direct copy
    feats_csv = traces_dir / f"prompt_features_{SPLIT}.csv"
    if feats_csv.exists():
        import shutil
        shutil.copy(feats_csv, out_dir / "prompt_features.csv")
        print(f"      -> prompt_features.csv")
    else:
        print(f"      WARNING: {feats_csv} not found")

    # layerwise_decision_trace_train.csv → direct copy
    layer_csv = traces_dir / f"layerwise_decision_trace_{SPLIT}.csv"
    if layer_csv.exists():
        import shutil
        shutil.copy(layer_csv, out_dir / "layerwise_traces.csv")
        print(f"      -> layerwise_traces.csv")
    else:
        print(f"      WARNING: {layer_csv} not found")

    # --- 9. Summary ---
    print("\n[9/9] Done.")
    files = sorted(out_dir.glob("*"))
    print(f"      {len(files)} files written to {out_dir}")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"      {f.name:<45} {size_kb:7.1f} KB")


if __name__ == "__main__":
    main()
