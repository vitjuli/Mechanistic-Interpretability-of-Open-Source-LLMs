#!/usr/bin/env python3
"""
Validate pipeline outputs for a given behaviour run.

Usage:
    python scripts/validate_run.py --behaviour physics_decay_type [--ui_run <run_id>]

Checks (in order):
  1. Prompt file — count, balance, family breakdown, keyword_free integrity
  2. Attribution graphs — star / roleaware / roleaware_static structure
  3. Ablation CSV — row count, sign_flip_rate, baseline sign_acc, top disrupted features
  4. UI run outputs — graph.json nodes/edges/communities, supernodes, audit
  5. Circuit JSON (optional, on CSD3) — necessity, sufficiency, paths

Exit codes:
  0 = all checks PASS
  1 = one or more checks FAIL (details printed)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ─── Thresholds (from project memory / known results) ─────────────────────────
KNOWN = {
    "physics_decay_type": {
        "n_prompts": 108,
        "n_alpha": 54,
        "n_beta": 54,
        "families": {"F0": 24, "F1": 28, "F2": 20, "F3": 20, "F4": 16},
        "keyword_free_family": "F1",
        "keywords_banned": ["alpha", "beta", "helium", "electron"],
        # ablation CSV baseline uses BASE model (~77%); step-02 baseline uses Instruct (~87%)
        "baseline_sign_acc_min": 0.70,
        "baseline_sign_acc_expected": 0.769,
        "ablation_rows_expected": 4320,  # 40 features × 108 prompts
        "graph_n_prompts": 108,
        "graph_n_nodes_roleaware_static": 69,
        "graph_n_edges_roleaware_static": 472,
        "graph_vw_edges_roleaware_static": 265,
        "graph_n_communities_min": 4,
        "circuit_necessity_min": 0.50,   # 67.6% expected
        "circuit_n_features_expected": 11,
        "circuit_n_edges_expected": 16,
        "circuit_n_paths_expected": 10,
        "_prompts_file": "physics_decay_type_train.jsonl",
    },
    "physics_decay_type_test": {
        "n_prompts": 36,
        "n_alpha": 18,
        "n_beta": 18,
        "families": {"F0": 8, "F1": 10, "F2": 8, "F3": 6, "F4": 4},
        "keyword_free_family": "F1",
        "keywords_banned": ["alpha", "beta", "helium", "electron"],
        # Base model test baseline: 83.3% (30/36); Instruct: 91.7%
        "baseline_sign_acc_min": 0.70,
        "baseline_sign_acc_expected": 0.833,
        "ablation_rows_expected": 1440,  # 40 features × 36 prompts
        "graph_n_prompts": 108,          # uses train graph
        "graph_n_nodes_roleaware_static": 69,
        "graph_n_edges_roleaware_static": 472,
        "graph_vw_edges_roleaware_static": 265,
        "graph_n_communities_min": 4,
        "circuit_necessity_min": None,
        "circuit_n_features_expected": None,
        "circuit_n_edges_expected": None,
        "circuit_n_paths_expected": None,
        "_prompts_file": "physics_decay_type_test.jsonl",
    },
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"

failures: list[str] = []
warnings: list[str] = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = PASS if ok else (WARN if warn_only else FAIL)
    detail_str = f"  → {detail}" if detail else ""
    print(f"  [{status}] {label}{detail_str}")
    if not ok:
        if warn_only:
            warnings.append(label)
        else:
            failures.append(label)
    return ok


# ─── 1. Prompt file ──────────────────────────────────────────────────────────

def validate_prompts(behaviour: str, project_root: Path, cfg: dict) -> None:
    print("\n── 1. Prompt file ─────────────────────────────────────────────")
    prompt_path = cfg.get("_resolved_prompts_path",
                          project_root / "data" / "prompts" / f"{behaviour}_train.jsonl")
    check("prompt file exists", prompt_path.exists(), str(prompt_path))
    if not prompt_path.exists():
        return

    prompts = [json.loads(l) for l in prompt_path.read_text().splitlines() if l.strip()]
    n = len(prompts)
    check("n_prompts", n == cfg["n_prompts"], f"{n} (expected {cfg['n_prompts']})")

    # Balance (field is correct_answer with leading space, e.g. ' alpha')
    answers = [p.get("correct_answer", p.get("correct_token", "")).strip() for p in prompts]
    alpha_n = sum(1 for a in answers if a == "alpha")
    beta_n = sum(1 for a in answers if a == "beta")
    check("alpha count", alpha_n == cfg["n_alpha"], f"{alpha_n}")
    check("beta count", beta_n == cfg["n_beta"], f"{beta_n}")

    # Family breakdown
    families = Counter(p.get("surface_family", "UNKNOWN") for p in prompts)
    for fam, expected_n in cfg["families"].items():
        actual = families.get(fam, 0)
        check(f"family {fam}", actual == expected_n, f"{actual} (expected {expected_n})")

    # Keyword-free integrity
    kw_family = cfg.get("keyword_free_family")
    keywords = cfg.get("keywords_banned", [])
    if kw_family and keywords:
        kw_prompts = [p for p in prompts if p.get("surface_family") == kw_family]
        tail = "is the decay type alpha or beta?"
        violations = []
        for p in kw_prompts:
            text = p.get("prompt", "").lower()
            # Strip the answer-format suffix before checking
            idx = text.rfind(tail)
            body = text[:idx] if idx >= 0 else text
            for kw in keywords:
                if kw in body:
                    violations.append((p.get("prompt", "")[:60], kw))
        check(
            f"{kw_family} keyword_free (no {keywords})",
            len(violations) == 0,
            f"{len(violations)} violations" if violations else "",
        )

    # All prompts have required fields
    required_fields = ["prompt", "correct_answer", "incorrect_answer", "surface_family"]
    for field in required_fields:
        missing = sum(1 for p in prompts if field not in p)
        check(f"field '{field}' present", missing == 0, f"{missing} missing" if missing else "")

    # keyword_free boolean field present
    kw_missing = sum(1 for p in prompts if "keyword_free" not in p)
    check("field 'keyword_free' present", kw_missing == 0,
          f"{kw_missing} missing" if kw_missing else "")


# ─── 2. Attribution graphs ───────────────────────────────────────────────────

def validate_graphs(behaviour: str, project_root: Path, cfg: dict, ui_run_dir: Path) -> None:
    print("\n── 2. Attribution graphs ──────────────────────────────────────")
    raw = ui_run_dir / "raw_sources"
    n = cfg["graph_n_prompts"]

    VW_TYPES = {"virtual_weight", "attribution_approx_v1"}

    def _load_graph(path: Path):
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # --- Star graph (06a) ---
    star_path = raw / f"attribution_graph_train_n{n}.json"
    check("star graph exists", star_path.exists(), str(star_path.name))
    if star_path.exists():
        g = _load_graph(star_path)
        nodes = g.get("nodes", [])
        edges = g.get("edges", g.get("links", []))
        feat_nodes = [nd for nd in nodes if nd.get("type") == "feature"]
        vw_edges = [e for e in edges if e.get("edge_type") in VW_TYPES]
        check("star graph has feature nodes", len(feat_nodes) > 0,
              f"{len(feat_nodes)} feature nodes")
        # Star graph from 06a shouldn't have VW edges (normal)
        check("star graph: no VW edges (expected)", len(vw_edges) == 0,
              f"{len(vw_edges)} VW edges", warn_only=len(vw_edges) > 0)
        # edge_type=None is expected for star graph
        none_types = sum(1 for e in edges if e.get("edge_type") is None)
        print(f"    {INFO} star graph: {len(nodes)} nodes, {len(edges)} edges "
              f"({none_types} edge_type=None — expected for star topology)")

    # --- Roleaware graph (06b) ---
    ra_path = raw / f"attribution_graph_train_n{n}_roleaware.json"
    check("roleaware graph exists", ra_path.exists(), str(ra_path.name))
    if ra_path.exists():
        g = _load_graph(ra_path)
        edges = g.get("edges", g.get("links", []))
        vw_edges = [e for e in edges if e.get("edge_type") in VW_TYPES]
        check("roleaware graph has VW edges", len(vw_edges) > 0, f"{len(vw_edges)} VW edges")
        vw_types_found = set(e.get("edge_type") for e in vw_edges)
        print(f"    {INFO} roleaware: {len(vw_edges)} VW edges, types={vw_types_found}")

    # --- Roleaware static graph (06c) ---
    ras_path = raw / f"attribution_graph_train_n{n}_roleaware_static.json"
    check("roleaware_static graph exists", ras_path.exists(), str(ras_path.name))
    if ras_path.exists():
        g = _load_graph(ras_path)
        nodes = g.get("nodes", [])
        edges = g.get("edges", g.get("links", []))
        feat_nodes = [nd for nd in nodes if nd.get("type") == "feature"]
        vw_edges = [e for e in edges if e.get("edge_type") in VW_TYPES]
        check(
            "roleaware_static: node count",
            len(feat_nodes) == cfg["graph_n_nodes_roleaware_static"],
            f"{len(feat_nodes)} (expected {cfg['graph_n_nodes_roleaware_static']})",
        )
        check(
            "roleaware_static: edge count",
            len(edges) == cfg["graph_n_edges_roleaware_static"],
            f"{len(edges)} (expected {cfg['graph_n_edges_roleaware_static']})",
        )
        check(
            "roleaware_static: VW edge count",
            len(vw_edges) == cfg["graph_vw_edges_roleaware_static"],
            f"{len(vw_edges)} (expected {cfg['graph_vw_edges_roleaware_static']})",
        )
        edge_type_dist = dict(Counter(e.get("edge_type") for e in edges))
        print(f"    {INFO} roleaware_static edge_type dist: {edge_type_dist}")

        # Communities in graph nodes (set in step 08 or UI prep)
        feat_communities = set(nd.get("community") for nd in feat_nodes)
        n_none_communities = sum(1 for nd in feat_nodes if nd.get("community") is None)
        check(
            "roleaware_static: communities present in raw graph",
            n_none_communities == 0,
            f"{n_none_communities}/{len(feat_nodes)} missing community",
            warn_only=True,  # Communities are added by step 09 UI prep, not in raw graph
        )


# ─── 3. Ablation CSV ────────────────────────────────────────────────────────

def validate_ablation(behaviour: str, project_root: Path, cfg: dict, ui_run_dir: Path) -> None:
    print("\n── 3. Ablation CSV ────────────────────────────────────────────")
    # Script 07 saves using the base behaviour name regardless of split;
    # fall back to the base name if the variant-specific file doesn't exist.
    base_behaviour = behaviour.replace("_test", "")
    csv_path = ui_run_dir / "raw_sources" / f"intervention_ablation_{behaviour}.csv"
    if not csv_path.exists():
        csv_path = ui_run_dir / "raw_sources" / f"intervention_ablation_{base_behaviour}.csv"
    check("ablation CSV exists", csv_path.exists(), str(csv_path.name))
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    check("ablation: row count",
          len(df) == cfg["ablation_rows_expected"],
          f"{len(df)} rows (expected {cfg['ablation_rows_expected']})")

    # Required columns
    for col in ["layer", "feature_indices", "baseline_logit_diff", "sign_flipped",
                "effect_size", "feature_source"]:
        check(f"ablation: column '{col}'", col in df.columns)

    # Baseline sign accuracy
    # baseline_logit_diff > 0 means model was correct before intervention
    baseline_correct = (df["baseline_logit_diff"] > 0).mean()
    check(
        f"ablation: baseline sign_acc >= {cfg['baseline_sign_acc_min']:.2f}",
        baseline_correct >= cfg["baseline_sign_acc_min"],
        f"{baseline_correct:.3f}",
    )
    expected_acc = cfg.get("baseline_sign_acc_expected")
    if expected_acc:
        check(
            f"ablation: baseline sign_acc matches expected ({expected_acc:.3f} ± 0.05)",
            abs(baseline_correct - expected_acc) < 0.05,
            f"{baseline_correct:.3f}",
            warn_only=True,
        )

    # Sign flip rate
    sfr = df["sign_flipped"].mean()
    check(
        "ablation: sign_flip_rate < 0.30 (sanity)",
        sfr < 0.30,
        f"{sfr:.3f}",
    )
    print(f"    {INFO} sign_flip_rate = {sfr:.4f} ({df['sign_flipped'].sum():.0f}/{len(df)} flips)")

    # Top disrupted features
    feature_disruption = (
        df[df["sign_flipped"]]
        .groupby("feature_id")
        .size()
        .sort_values(ascending=False)
    )
    print(f"    {INFO} top disrupted features (sign_flip):")
    for fid, count in feature_disruption.head(5).items():
        print(f"      {fid}: {count} flips ({count/cfg['n_prompts']*100:.1f}% of prompts)")

    # Summary JSON (falls back to base name for test variant)
    summary_path = ui_run_dir / "raw_sources" / f"intervention_ablation_{behaviour}_summary.json"
    if not summary_path.exists():
        summary_path = ui_run_dir / "raw_sources" / f"intervention_ablation_{base_behaviour}_summary.json"
    check("ablation summary JSON exists", summary_path.exists())
    if summary_path.exists():
        s = json.load(open(summary_path))
        check("ablation summary: sign_flip_rate matches CSV",
              abs(s.get("sign_flip_rate", -1) - sfr) < 1e-6,
              f"summary={s.get('sign_flip_rate'):.6f}, csv={sfr:.6f}")


# ─── 4. UI run outputs ───────────────────────────────────────────────────────

def validate_ui(behaviour: str, ui_run_dir: Path, cfg: dict) -> None:
    print("\n── 4. UI run outputs ─────────────────────────────────────────")
    check("UI run dir exists", ui_run_dir.exists(), str(ui_run_dir))
    if not ui_run_dir.exists():
        return

    # graph.json
    graph_path = ui_run_dir / "graph.json"
    check("graph.json exists", graph_path.exists())
    if graph_path.exists():
        g = json.load(open(graph_path))
        nodes = g.get("nodes", [])
        links = g.get("links", [])
        feat_nodes = [n for n in nodes if n.get("type") == "feature"]
        check("graph.json: has nodes", len(nodes) > 0, f"{len(nodes)} nodes")
        check("graph.json: has links (edges)", len(links) > 0, f"{len(links)} links")

        # Community check (requires src/ui_offline/prepare.py fix)
        community_vals = set(n.get("community") for n in feat_nodes)
        n_none_comm = sum(1 for n in feat_nodes if n.get("community") is None)
        check(
            "graph.json: communities assigned to nodes",
            n_none_comm == 0,
            f"{n_none_comm}/{len(feat_nodes)} nodes missing community",
        )
        if n_none_comm == 0:
            n_communities = len(community_vals)
            check(
                f"graph.json: >= {cfg['graph_n_communities_min']} communities",
                n_communities >= cfg["graph_n_communities_min"],
                f"{n_communities} communities: {sorted(community_vals)}",
            )

        # edge_type check (requires src/ui_offline/prepare.py fix)
        edge_type_none = sum(1 for l in links if l.get("edge_type") is None)
        check(
            "graph.json: edge_type present in all links",
            edge_type_none == 0,
            f"{edge_type_none}/{len(links)} links missing edge_type",
        )

    # supernodes.json
    sn_path = ui_run_dir / "supernodes.json"
    check("supernodes.json exists", sn_path.exists())
    if sn_path.exists():
        sn = json.load(open(sn_path))
        check("supernodes: non-empty", len(sn) > 0, f"{len(sn)} communities")

    # supernodes_summary.csv
    sns_path = ui_run_dir / "supernodes_summary.csv"
    check("supernodes_summary.csv exists", sns_path.exists())

    # audit.json
    audit_path = ui_run_dir / "audit.json"
    check("audit.json exists", audit_path.exists())
    if audit_path.exists():
        audit = json.load(open(audit_path))
        check("audit has summaries", "summaries" in audit)
        if "summaries" in audit and "ablation" in audit["summaries"]:
            ab_summary = audit["summaries"]["ablation"]
            sfr = ab_summary.get("sign_flip_rate", -1)
            check("audit: sign_flip_rate > 0", sfr > 0, f"{sfr:.4f}")

    # run_manifest.json
    manifest_path = ui_run_dir / "run_manifest.json"
    check("run_manifest.json exists", manifest_path.exists())
    if manifest_path.exists():
        m = json.load(open(manifest_path))
        manifest_behaviour = m.get("parameters", {}).get("behaviour")
        base_behaviour = behaviour.replace("_test", "")
        check("manifest: behaviour matches",
              manifest_behaviour in (behaviour, base_behaviour),
              manifest_behaviour)


# ─── 5. Circuit JSON (optional) ──────────────────────────────────────────────

def validate_circuit(behaviour: str, project_root: Path, cfg: dict) -> None:
    print("\n── 5. Circuit JSON (optional) ────────────────────────────────")
    # Canonical location on CSD3; may or may not be present locally
    circuit_path = (
        project_root
        / "data" / "results" / "causal_edges" / behaviour
        / f"circuits_{behaviour}_train.json"
    )
    if not circuit_path.exists():
        print(f"  [{WARN}] circuit JSON not found locally (CSD3 only): {circuit_path.name}")
        warnings.append("circuit JSON not found locally")
        return

    check("circuit JSON exists", True)
    c = json.load(open(circuit_path))

    # Features
    features = c.get("features", [])
    edges = c.get("edges", [])
    paths = c.get("paths", [])
    check(
        f"circuit: n_features == {cfg['circuit_n_features_expected']}",
        len(features) == cfg["circuit_n_features_expected"],
        f"{len(features)}",
        warn_only=True,
    )
    check(
        f"circuit: n_edges == {cfg['circuit_n_edges_expected']}",
        len(edges) == cfg["circuit_n_edges_expected"],
        f"{len(edges)}",
        warn_only=True,
    )
    check(
        f"circuit: n_paths == {cfg['circuit_n_paths_expected']}",
        len(paths) == cfg["circuit_n_paths_expected"],
        f"{len(paths)}",
        warn_only=True,
    )

    # Necessity
    necessity = c.get("disruption_rate", c.get("necessity", None))
    if necessity is not None:
        check(
            f"circuit: necessity >= {cfg['circuit_necessity_min']:.2f}",
            necessity >= cfg["circuit_necessity_min"],
            f"{necessity:.3f}",
        )
        print(f"    {INFO} necessity (disruption_rate) = {necessity:.3f}")

    # Sufficiency
    s1 = c.get("s1_sufficiency", {})
    if s1:
        sign_acc = s1.get("sign_accuracy", 0)
        retention = s1.get("retention", 0)
        check("circuit: S1 sign_acc > 0.80", sign_acc > 0.80, f"{sign_acc:.3f}", warn_only=True)
        print(f"    {INFO} S1 linear: sign={sign_acc:.3f}, retention={retention:.3f}")

    s15 = c.get("s15_sufficiency", {})
    if s15:
        sign_acc = s15.get("sign_accuracy", 0)
        retention = s15.get("retention", 0)
        print(f"    {INFO} S1.5 layerwise: sign={sign_acc:.3f}, retention={retention:.3f}")

    # Top path
    if paths:
        top_path = paths[0] if isinstance(paths[0], (list, str)) else paths[0].get("nodes", [])
        print(f"    {INFO} top path: {' → '.join(str(n) for n in top_path)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate pipeline run outputs.")
    parser.add_argument("--behaviour", default="physics_decay_type")
    parser.add_argument(
        "--ui_run",
        default=None,
        help="UI run directory name (e.g. 20260426-034658_physics_decay_type_train_n108). "
             "If omitted, uses the most recent run for this behaviour.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ui_offline_root = project_root / "data" / "ui_offline"

    # Auto-detect test variant: if ui_run contains "_test_", use *_test config
    behaviour = args.behaviour
    if args.ui_run and "_test_" in args.ui_run and f"{behaviour}_test" in KNOWN:
        behaviour = f"{behaviour}_test"
    elif args.ui_run is None:
        # For auto-detected run, check if it's test split
        pass

    if behaviour not in KNOWN:
        print(f"No known config for behaviour '{behaviour}'. Add it to KNOWN dict.")
        sys.exit(1)

    cfg = KNOWN[behaviour]
    # Use the correct prompts file for this variant
    prompts_filename = cfg.get("_prompts_file", f"{args.behaviour}_train.jsonl")
    # Monkey-patch PROMPTS_FILE into project_root for validate_prompts
    cfg["_resolved_prompts_path"] = project_root / "data" / "prompts" / prompts_filename

    # Find UI run dir
    base_behaviour = args.behaviour  # without _test suffix, for directory search
    if args.ui_run:
        ui_run_dir = ui_offline_root / args.ui_run
    else:
        candidates = sorted(
            [d for d in ui_offline_root.iterdir()
             if d.is_dir() and f"_{base_behaviour}_" in d.name],
            key=lambda d: d.name,
        )
        if not candidates:
            print(f"No UI run dirs found for '{base_behaviour}' under {ui_offline_root}")
            sys.exit(1)
        ui_run_dir = candidates[-1]

    print(f"\nValidating '{behaviour}' run: {ui_run_dir.name}")
    print(f"Project root: {project_root}")

    validate_prompts(behaviour, project_root, cfg)
    validate_graphs(behaviour, project_root, cfg, ui_run_dir)
    validate_ablation(behaviour, project_root, cfg, ui_run_dir)
    validate_ui(behaviour, ui_run_dir, cfg)
    validate_circuit(behaviour, project_root, cfg)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"  [{FAIL}] {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    • {f}")
    if warnings:
        print(f"  [{WARN}] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    • {w}")
    if not failures:
        print(f"  [{PASS}] All checks passed! ({len(warnings)} warnings)")
    print("=" * 60)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
