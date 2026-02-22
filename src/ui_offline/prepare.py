"""
Core data-preparation logic for the offline UI dataset.

Transforms attribution graphs + intervention CSVs into clean, versioned,
UI-ready artifacts (parquet/csv/json) for Neuronpedia-style dashboards.

No GPU, no model, no torch required.
"""

import ast
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check parquet support once at import time
_HAS_PARQUET = False
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET = True
    except ImportError:
        pass

if not _HAS_PARQUET:
    logger.warning(
        "Neither pyarrow nor fastparquet installed. "
        "Parquet output disabled — will write CSV only."
    )


def _save_df(df: pd.DataFrame, path_stem: Path, output_files: List[Path]):
    """Save DataFrame as parquet (if available) and CSV."""
    csv_path = path_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    output_files.append(csv_path)
    if _HAS_PARQUET:
        pq_path = path_stem.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False)
        output_files.append(pq_path)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_literal_eval(val: str) -> Any:
    """Parse a Python literal string safely (handles single-quoted dicts, lists)."""
    if pd.isna(val) or val == "":
        return {}
    try:
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse literal: {val!r:.120}")
        return {}


def _parse_feature_indices(val) -> List[int]:
    """Normalise feature_indices column to a Python list of ints."""
    if isinstance(val, list):
        return [int(x) for x in val]
    s = str(val).strip()
    if s.startswith("["):
        try:
            return [int(x) for x in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    # Single bare int
    try:
        return [int(float(s))]
    except (ValueError, TypeError):
        return []


def _git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _file_info(path: Path) -> Dict:
    """Return size and mtime for a file (or None if missing)."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


# ---------------------------------------------------------------------------
# A. Unified interventions table
# ---------------------------------------------------------------------------

def load_intervention_csv(path: Path) -> pd.DataFrame:
    """Load a single intervention CSV, tolerating quoting quirks."""
    if not path.exists():
        logger.warning(f"Intervention CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def build_unified_interventions(
    interventions_dir: Path,
    behaviour: str,
    split: str,
    graph_n_prompts: int,
    run_id: str,
    timestamp: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Read ablation/patching/steering CSVs, flatten metadata, add context columns.

    Returns:
        (unified_df, audit_dict)
    """
    exp_types = ["ablation", "patching", "steering"]
    frames = []
    summaries: Dict[str, Dict] = {}

    for exp in exp_types:
        csv_path = interventions_dir / f"intervention_{exp}_{behaviour}.csv"
        df = load_intervention_csv(csv_path)
        if df.empty:
            continue

        # Load companion summary JSON for model_size etc.
        summary_path = interventions_dir / f"intervention_{exp}_{behaviour}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[exp] = json.load(f)

        frames.append(df)

    if not frames:
        logger.error("No intervention CSVs found — nothing to unify.")
        return pd.DataFrame(), {}

    unified = pd.concat(frames, ignore_index=True)

    # --- Flatten metadata column ---
    if "metadata" in unified.columns:
        meta_dicts = unified["metadata"].apply(_safe_literal_eval)
        meta_df = pd.json_normalize(meta_dicts).add_prefix("meta.")
        # Align index
        meta_df.index = unified.index
        unified = pd.concat([unified.drop(columns=["metadata"]), meta_df], axis=1)

    # --- Normalise feature_indices to proper lists ---
    if "feature_indices" in unified.columns:
        unified["feature_indices"] = unified["feature_indices"].apply(_parse_feature_indices)

    # --- Context columns ---
    model_size = None
    for s in summaries.values():
        if "model_size" in s:
            model_size = s["model_size"]
            break

    unified["behaviour"] = behaviour
    unified["split"] = split
    unified["model_size"] = model_size
    unified["graph_n_prompts"] = graph_n_prompts
    unified["run_id"] = run_id
    unified["prep_timestamp"] = timestamp

    # --- Audit ---
    audit = _build_audit(unified, summaries)

    return unified, audit


def _build_audit(df: pd.DataFrame, summaries: Dict[str, Dict]) -> Dict:
    """Build coverage audit dict."""
    audit: Dict[str, Any] = {"summaries": summaries}

    if df.empty:
        audit["warning"] = "No data"
        return audit

    # Per experiment_type coverage
    coverage: Dict[str, Any] = {}
    all_prompt_sets: Dict[str, set] = {}
    all_layer_sets: Dict[str, set] = {}

    for exp_type, grp in df.groupby("experiment_type"):
        prompt_idxs = sorted(grp["prompt_idx"].unique().tolist())
        layers = sorted(grp["layer"].unique().tolist())
        counts = (
            grp.groupby(["layer", "prompt_idx"])
            .size()
            .reset_index(name="count")
        )
        coverage[exp_type] = {
            "n_rows": len(grp),
            "prompt_idx_range": [int(min(prompt_idxs)), int(max(prompt_idxs))],
            "prompt_idx_list": prompt_idxs,
            "n_unique_prompts": len(prompt_idxs),
            "layer_list": layers,
            "n_unique_layers": len(layers),
            "counts_per_layer_prompt": counts.to_dict(orient="records"),
        }
        all_prompt_sets[exp_type] = set(prompt_idxs)
        all_layer_sets[exp_type] = set(layers)

    audit["coverage_per_experiment"] = coverage

    # Missing prompt_idx analysis
    if all_prompt_sets:
        union_prompts = set().union(*all_prompt_sets.values())
        missing: Dict[str, List[int]] = {}
        for exp, s in all_prompt_sets.items():
            diff = sorted(union_prompts - s)
            if diff:
                missing[exp] = diff
        audit["missing_prompt_idx"] = missing

        # Common set
        common_prompts = sorted(set.intersection(*all_prompt_sets.values()))
        audit["common_prompt_idx"] = common_prompts
        audit["n_common_prompts"] = len(common_prompts)

        # Common layers
        common_layers = sorted(set.intersection(*all_layer_sets.values()))
        audit["common_layers"] = common_layers

    # Duplicate prompt text detection (ablation has meta.prompt)
    if "meta.prompt" in df.columns:
        prompt_col = df[df["meta.prompt"].notna()][["prompt_idx", "meta.prompt"]].drop_duplicates()
        dup_texts = prompt_col.groupby("meta.prompt").filter(lambda g: len(g) > 1)
        if not dup_texts.empty:
            dup_list = (
                dup_texts.groupby("meta.prompt")["prompt_idx"]
                .apply(list)
                .to_dict()
            )
            audit["duplicate_prompt_texts"] = {
                k[:80]: v for k, v in dup_list.items()
            }
        else:
            audit["duplicate_prompt_texts"] = {}

    return audit


# ---------------------------------------------------------------------------
# B. Aggregated metrics tables
# ---------------------------------------------------------------------------

_AGG_COLS = {
    "abs_effect_size": ["count", "mean", "median", "std"],
    "effect_size": ["mean", "median"],
    "relative_effect": ["mean"],
    "baseline_logit_diff": ["mean"],
    "intervened_logit_diff": ["mean"],
}


def _agg_group(grp: pd.DataFrame) -> Dict:
    """Compute standard aggregate metrics for a group."""
    out: Dict[str, Any] = {"n": len(grp)}
    for col, fns in _AGG_COLS.items():
        if col not in grp.columns:
            continue
        for fn in fns:
            key = f"{fn}_{col}"
            out[key] = float(getattr(grp[col], fn)())
    if "sign_flipped" in grp.columns:
        out["sign_flip_rate"] = float(grp["sign_flipped"].mean())
        out["n_sign_flips"] = int(grp["sign_flipped"].sum())
    return out


def build_layer_agg(df: pd.DataFrame) -> pd.DataFrame:
    """groupby [experiment_type, layer]."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (exp, layer), grp in df.groupby(["experiment_type", "layer"]):
        row = {"experiment_type": exp, "layer": int(layer)}
        row.update(_agg_group(grp))
        rows.append(row)
    return pd.DataFrame(rows)


def build_prompt_agg(df: pd.DataFrame) -> pd.DataFrame:
    """groupby [experiment_type, prompt_idx]."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (exp, pidx), grp in df.groupby(["experiment_type", "prompt_idx"]):
        row = {"experiment_type": exp, "prompt_idx": int(pidx)}
        row.update(_agg_group(grp))
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode feature_indices -> one row per (experiment_type, layer, feature_id).
    Then aggregate across prompts.
    """
    if df.empty:
        return pd.DataFrame()

    # Explode feature_indices
    exploded = df.explode("feature_indices").rename(
        columns={"feature_indices": "feature_id"}
    )
    exploded["feature_id"] = pd.to_numeric(exploded["feature_id"], errors="coerce")
    exploded = exploded.dropna(subset=["feature_id"])
    exploded["feature_id"] = exploded["feature_id"].astype(int)

    rows = []
    for (exp, layer, fid), grp in exploded.groupby(
        ["experiment_type", "layer", "feature_id"]
    ):
        row = {
            "experiment_type": exp,
            "layer": int(layer),
            "feature_id": int(fid),
        }
        row.update(_agg_group(grp))
        rows.append(row)
    return pd.DataFrame(rows)


def build_common_index(df: pd.DataFrame, audit: Dict) -> Tuple[List[int], List[int]]:
    """Return (common_prompt_idx, common_layers) across all experiment types."""
    common_p = audit.get("common_prompt_idx", [])
    common_l = audit.get("common_layers", [])
    return common_p, common_l


# ---------------------------------------------------------------------------
# C. Supernodes
# ---------------------------------------------------------------------------

def build_supernodes_graph(
    graphml_path: Path,
    method: str = "louvain",
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """
    S1: Community detection on the attribution graph.

    Returns:
        (supernodes_dict, supernodes_summary_df) or (None, None) on failure.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed — skipping graph supernodes.")
        return None, None

    if not graphml_path.exists():
        logger.warning(f"GraphML not found: {graphml_path}")
        return None, None

    G = nx.read_graphml(graphml_path)
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to undirected for community detection
    G_und = G.to_undirected()

    partition = None
    actual_method = method

    if method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G_und)
            actual_method = "louvain"
        except ImportError:
            logger.warning(
                "python-louvain not installed (pip install python-louvain). "
                "Falling back to connected components."
            )
            method = "components"

    if partition is None:
        # Fallback: connected components
        actual_method = "connected_components"
        partition = {}
        for cid, comp in enumerate(nx.connected_components(G_und)):
            for node in comp:
                partition[node] = cid

    # Build supernodes dict: community_id -> [node_ids]
    supernodes: Dict[str, List[str]] = {}
    for node, cid in partition.items():
        supernodes.setdefault(str(cid), []).append(node)

    # Sort node lists for determinism
    for cid in supernodes:
        supernodes[cid] = sorted(supernodes[cid])

    # Summary table
    rows = []
    for cid, members in supernodes.items():
        sub = G_und.subgraph(members)
        deg = dict(sub.degree())
        top_by_degree = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5]
        rows.append({
            "community_id": int(cid),
            "n_nodes": len(members),
            "top_nodes_by_degree": str([n for n, _ in top_by_degree]),
            "method": actual_method,
        })
    summary_df = pd.DataFrame(rows).sort_values("n_nodes", ascending=False)

    logger.info(
        f"Graph supernodes: {len(supernodes)} communities via {actual_method}"
    )
    return supernodes, summary_df


def build_supernodes_effect(
    feature_agg_df: pd.DataFrame,
    n_clusters: Optional[int] = None,
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """
    S2: Effect-similarity clustering from intervention feature aggregates.

    Each feature is represented by a vector of mean_effect_size across layers.
    Missing entries are filled with 0 and a missingness rate is stored.

    Returns:
        (supernodes_effect_dict, supernodes_effect_summary_df) or (None, None).
    """
    if feature_agg_df.empty:
        logger.warning("Feature aggregate table is empty — skipping effect clustering.")
        return None, None

    # Pivot: rows = (feature_id), columns = (experiment_type, layer), values = mean_effect_size
    pivot_col = "mean_effect_size"
    needed = {"experiment_type", "layer", "feature_id", pivot_col}
    if not needed.issubset(feature_agg_df.columns):
        logger.warning(f"Feature agg missing columns {needed - set(feature_agg_df.columns)}")
        return None, None

    # Create unique feature key: (layer, feature_id)
    fa = feature_agg_df.copy()
    fa["feat_key"] = fa.apply(lambda r: f"L{int(r['layer'])}_F{int(r['feature_id'])}", axis=1)

    # Pivot: one row per feat_key, one column per (experiment_type, layer)
    pivot = fa.pivot_table(
        index="feat_key",
        columns=["experiment_type", "layer"],
        values=pivot_col,
        aggfunc="mean",
    )

    n_total_cells = pivot.shape[0] * pivot.shape[1]
    n_missing = int(pivot.isna().sum().sum())
    missingness_rate = n_missing / max(n_total_cells, 1)
    logger.info(
        f"Effect clustering: {pivot.shape[0]} features x {pivot.shape[1]} dims, "
        f"missingness={missingness_rate:.1%}"
    )

    pivot = pivot.fillna(0.0)

    if pivot.shape[0] < 3:
        logger.warning("Too few features for clustering (<3).")
        return None, None

    # Determine n_clusters
    if not n_clusters:  # None or 0 → auto
        n_clusters = min(50, max(2, int(np.sqrt(pivot.shape[0]))))
    n_clusters = min(n_clusters, pivot.shape[0])

    # Normalise rows — drop zero-norm features (no measurable effect)
    # to avoid non-finite distances in Ward linkage.
    raw = pivot.values
    norms = np.linalg.norm(raw, axis=1)
    active_mask = norms > 1e-12
    n_inactive = int((~active_mask).sum())
    if n_inactive > 0:
        logger.info(f"  {n_inactive} features with zero effect vector — assigned to cluster -1")

    active_idx = np.where(active_mask)[0]
    if len(active_idx) < 3:
        logger.warning("Too few active features for clustering (<3 after removing zero-effect).")
        return None, None

    X_active = raw[active_idx] / norms[active_idx, np.newaxis]

    # Replace any remaining non-finite values (safety net)
    X_active = np.nan_to_num(X_active, nan=0.0, posinf=0.0, neginf=0.0)

    n_clusters = min(n_clusters, len(active_idx))

    # Try hierarchical clustering
    active_labels = None
    clustering_method = None

    try:
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
        active_labels = model.fit_predict(X_active)
        clustering_method = "agglomerative"
    except ImportError:
        pass

    if active_labels is None:
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(X_active, method="ward")
            active_labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
            clustering_method = "scipy_ward"
        except ImportError:
            pass

    if active_labels is None:
        logger.warning(
            "Neither sklearn nor scipy available — skipping effect clustering. "
            "Install scikit-learn or scipy for S2 supernodes."
        )
        return None, None

    # Reconstruct full label array: active features get cluster id, inactive get -1
    labels = np.full(pivot.shape[0], -1, dtype=int)
    labels[active_idx] = active_labels

    feat_keys = list(pivot.index)

    # Build mapping
    supernodes: Dict[str, List[str]] = {}
    for fk, cl in zip(feat_keys, labels):
        supernodes.setdefault(str(int(cl)), []).append(fk)

    for cid in supernodes:
        supernodes[cid] = sorted(supernodes[cid])

    # Summary
    rows = []
    for cid, members in supernodes.items():
        # Representative = member with largest norm in original (unnormalised) space
        member_idx = [feat_keys.index(m) for m in members]
        member_norms = norms[member_idx]
        best = members[int(np.argmax(member_norms))]
        rows.append({
            "cluster_id": int(cid),
            "n_features": len(members),
            "representative": best,
            "method": clustering_method,
            "missingness_rate": float(missingness_rate),
        })
    summary_df = pd.DataFrame(rows).sort_values("n_features", ascending=False)

    logger.info(f"Effect supernodes: {len(supernodes)} clusters via {clustering_method}")
    return supernodes, summary_df


# ---------------------------------------------------------------------------
# D. Graph JSON for UI
# ---------------------------------------------------------------------------

def build_graph_json(graphml_path: Path) -> Optional[Dict]:
    """Convert graphml to node-link JSON (networkx.node_link_data)."""
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed — skipping graph JSON export.")
        return None

    if not graphml_path.exists():
        logger.warning(f"GraphML not found: {graphml_path}")
        return None

    G = nx.read_graphml(graphml_path)

    # Enrich nodes with parsed layer info from id (e.g. "L15_F12345")
    for node_id, attrs in G.nodes(data=True):
        if node_id.startswith("L") and "_F" in node_id:
            parts = node_id.split("_")
            try:
                attrs.setdefault("layer", int(parts[0][1:]))
                attrs.setdefault("feature_idx", int(parts[1][1:]))
            except (ValueError, IndexError):
                pass

    data = nx.node_link_data(G)

    # Normalise edge key: nx 2.x uses "links", nx 3.x uses "edges"
    if "edges" in data and "links" not in data:
        data["links"] = data.pop("edges")

    # Ensure stable ordering
    data["nodes"] = sorted(data["nodes"], key=lambda n: str(n.get("id", "")))
    data["links"] = sorted(
        data["links"],
        key=lambda e: (str(e.get("source", "")), str(e.get("target", ""))),
    )

    return data


# ---------------------------------------------------------------------------
# E. Run manifest
# ---------------------------------------------------------------------------

def build_manifest(
    input_files: List[Path],
    output_files: List[Path],
    params: Dict,
) -> Dict:
    """Build a reproducibility manifest."""
    manifest = {
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "inputs": [_file_info(p) for p in input_files],
        "outputs": [_file_info(p) for p in output_files],
    }
    return manifest


# ---------------------------------------------------------------------------
# Feature importance loading
# ---------------------------------------------------------------------------

def load_feature_importance(importance_dir: Path) -> pd.DataFrame:
    """Load all feature_importance_layer_*.csv and concat."""
    if not importance_dir.exists():
        logger.warning(f"Importance directory not found: {importance_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(importance_dir.glob("feature_importance_layer_*.csv")):
        df = pd.read_csv(csv_path)
        frames.append(df)
        logger.info(f"  Loaded {len(df)} rows from {csv_path.name}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def prepare_all(
    results_dir: Path,
    behaviour: str,
    split: str,
    graph_n_prompts: int,
    out_dir: Path,
    run_id: str,
    community_method: str = "louvain",
    effect_clusters: Optional[int] = None,
) -> Path:
    """
    Run the full offline data preparation pipeline.

    Returns:
        Path to the run output directory.
    """
    timestamp = datetime.now().isoformat()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    interventions_dir = results_dir / "interventions" / behaviour
    graphs_dir = results_dir / "attribution_graphs" / behaviour
    graphml_path = graphs_dir / f"attribution_graph_{split}_n{graph_n_prompts}.graphml"

    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Interventions dir: {interventions_dir}")
    logger.info(f"Graph: {graphml_path}")

    # Track input/output files for manifest
    input_files: List[Path] = []
    output_files: List[Path] = []

    # Collect input files
    for exp in ["ablation", "patching", "steering"]:
        p = interventions_dir / f"intervention_{exp}_{behaviour}.csv"
        input_files.append(p)
        p_sum = interventions_dir / f"intervention_{exp}_{behaviour}_summary.json"
        input_files.append(p_sum)
    input_files.append(graphml_path)
    importance_dir = interventions_dir / "importance"
    if importance_dir.exists():
        input_files.extend(sorted(importance_dir.glob("*.csv")))

    # ===== A. Unified interventions =====
    logger.info("=" * 60)
    logger.info("A. Building unified interventions table...")
    unified, audit = build_unified_interventions(
        interventions_dir, behaviour, split, graph_n_prompts, run_id, timestamp,
    )

    if not unified.empty:
        # Determine common prompt_idx
        common_prompts, common_layers = build_common_index(unified, audit)
        unified["in_common_prompt_set"] = unified["prompt_idx"].isin(common_prompts)
        unified["in_common_layer_set"] = unified["layer"].isin(common_layers)

        # Save
        _save_df(unified, run_dir / "interventions", output_files)
        logger.info(f"  Saved unified table: {len(unified)} rows")

        # Save common_prompt_idx
        common_idx_path = run_dir / "common_prompt_idx.json"
        with open(common_idx_path, "w") as f:
            json.dump(
                {"common_prompt_idx": common_prompts, "common_layers": common_layers},
                f, indent=2, sort_keys=True,
            )
        output_files.append(common_idx_path)
    else:
        logger.error("Unified table is empty — downstream artifacts will be empty.")

    # Save audit
    audit_path = run_dir / "audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True, default=str)
    output_files.append(audit_path)
    logger.info(f"  Saved audit: {audit_path}")

    # ===== B. Aggregated metrics =====
    logger.info("=" * 60)
    logger.info("B. Building aggregated metrics tables...")

    layer_agg = build_layer_agg(unified)
    prompt_agg = build_prompt_agg(unified)
    feature_agg = build_feature_agg(unified)

    for name, agg_df in [
        ("interventions_layer_agg", layer_agg),
        ("interventions_prompt_agg", prompt_agg),
        ("interventions_feature_agg", feature_agg),
    ]:
        if not agg_df.empty:
            _save_df(agg_df, run_dir / name, output_files)
            logger.info(f"  {name}: {len(agg_df)} rows")
        else:
            logger.warning(f"  {name}: empty")

    # Feature importance (bonus — just copy/concat into run dir)
    fi_df = load_feature_importance(importance_dir)
    if not fi_df.empty:
        _save_df(fi_df, run_dir / "feature_importance", output_files)
        logger.info(f"  Feature importance: {len(fi_df)} rows")

    # ===== C. Supernodes =====
    logger.info("=" * 60)
    logger.info("C. Building supernodes...")

    # S1: Graph community
    sn_graph, sn_graph_summary = build_supernodes_graph(graphml_path, method=community_method)
    if sn_graph is not None:
        sn_path = run_dir / "supernodes.json"
        with open(sn_path, "w") as f:
            json.dump(sn_graph, f, indent=2, sort_keys=True)
        output_files.append(sn_path)

        _save_df(sn_graph_summary, run_dir / "supernodes_summary", output_files)

    # S2: Effect-similarity clustering
    sn_effect, sn_effect_summary = build_supernodes_effect(feature_agg, n_clusters=effect_clusters)
    if sn_effect is not None:
        sn_eff_path = run_dir / "supernodes_effect.json"
        with open(sn_eff_path, "w") as f:
            json.dump(sn_effect, f, indent=2, sort_keys=True)
        output_files.append(sn_eff_path)

        _save_df(sn_effect_summary, run_dir / "supernodes_effect_summary", output_files)

    # ===== D. Graph JSON =====
    logger.info("=" * 60)
    logger.info("D. Exporting graph JSON...")

    graph_json = build_graph_json(graphml_path)
    if graph_json is not None:
        graph_json_path = run_dir / "graph.json"
        with open(graph_json_path, "w") as f:
            json.dump(graph_json, f, indent=2, sort_keys=True)
        output_files.append(graph_json_path)
        logger.info(
            f"  graph.json: {len(graph_json.get('nodes', []))} nodes, "
            f"{len(graph_json.get('links', []))} edges"
        )

    # ===== E. Manifest =====
    logger.info("=" * 60)
    logger.info("E. Writing run manifest...")

    manifest = build_manifest(
        input_files=input_files,
        output_files=output_files,
        params={
            "behaviour": behaviour,
            "split": split,
            "graph_n_prompts": graph_n_prompts,
            "run_id": run_id,
            "community_method": community_method,
            "effect_clusters": effect_clusters,
            "timestamp": timestamp,
        },
    )
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    logger.info(f"  Manifest: {manifest_path}")

    # Final summary
    logger.info("=" * 60)
    logger.info(f"Done. {len(output_files)} artifacts written to {run_dir}")
    return run_dir
