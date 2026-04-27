#!/usr/bin/env python3
"""
Structural comparison: physics_decay_type vs multilingual_circuits_b1 circuits.

Compares:
  - Circuit topology (n_features, n_edges, path structure)
  - Causal strength (necessity, sufficiency)
  - Layer distribution of circuit features
  - Edge density and chain-vs-branching topology
  - Community alignment

Usage:
    python scripts/16_circuit_comparison.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── Hardcoded stats (authoritative, from CSD3 runs and memory) ───────────────

CIRCUITS = {
    "physics_decay_type": {
        "behaviour_type": "latent_state",
        "n_prompts_train": 108,
        "n_features": 11,
        "n_edges": 16,
        "n_paths": 10,
        "necessity": 0.676,
        "s1_sign": 0.861,
        "s1_retention": 1.337,
        "s15_sign": 0.935,
        "s15_retention": 1.184,
        "top_path": ["input", "L22_F110496", "L23_F83556", "L24_F60777", "L25_F71226", "output_correct"],
        "peak_layer": 24,
        "graph_n_nodes": 69,
        "graph_n_vw_edges": 265,
        "layers": list(range(10, 26)),
        "baseline_acc_instruct": 0.870,
        "baseline_acc_base": 0.769,
        "circuit_features": [
            "L22_F110496", "L23_F83556", "L23_F71067",
            "L24_F60777", "L24_F52031", "L24_F18943", "L24_F88968", "L24_F249",
            "L25_F71226", "L25_F126439", "L25_F110282",
        ],
        "notes": "Tight 4-hop chain L22→L25; L24 is decisive; β-suppressor features dominate",
    },
    "multilingual_circuits_b1": {
        "behaviour_type": "candidate_set",
        "n_prompts_train": 96,
        "n_features": 23,
        "n_edges": 52,
        "n_paths": 50,
        "necessity": 0.4375,
        "s1_sign": None,
        "s1_retention": None,
        "s15_sign": 0.719,
        "s15_retention": 0.733,
        "top_path": ["input", "L24_F35447", "L25_F43384", "output_correct"],
        "peak_layer": None,
        "graph_n_nodes": 58,
        "graph_n_vw_edges": 84,
        "layers": list(range(10, 26)),
        "baseline_acc_instruct": None,
        "baseline_acc_base": None,
        "circuit_features": None,
        "notes": "Distributed; L22 artifacts (F108295/F32734); bridge 60.4%; 7 communities; "
                 "C2 competitor circuit; EN/FR asymmetry",
    },
}

ANSI = {"bold": "\033[1m", "reset": "\033[0m", "cyan": "\033[36m",
        "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}


def bold(s): return f"{ANSI['bold']}{s}{ANSI['reset']}"
def green(s): return f"{ANSI['green']}{s}{ANSI['reset']}"
def yellow(s): return f"{ANSI['yellow']}{s}{ANSI['reset']}"


def fmt(val, fmt_str=".3f", good_fn=None):
    if val is None:
        return "N/A"
    s = format(val, fmt_str)
    if good_fn:
        return green(s) if good_fn(val) else yellow(s)
    return s


def print_comparison():
    dt = CIRCUITS["physics_decay_type"]
    ml = CIRCUITS["multilingual_circuits_b1"]

    print(bold("\n══════════════════════════════════════════════════════"))
    print(bold("  Circuit Structure Comparison"))
    print(bold("══════════════════════════════════════════════════════"))

    rows = [
        ("Behaviour type",       dt["behaviour_type"],           ml["behaviour_type"]),
        ("N train prompts",      dt["n_prompts_train"],          ml["n_prompts_train"]),
        ("Baseline acc (base)",  f"{dt['baseline_acc_base']:.1%}" if dt["baseline_acc_base"] else "N/A",
                                 "N/A"),
    ]
    print(f"\n  {'Metric':<30} {'physics_decay_type':>22} {'multilingual_b1':>22}")
    print(f"  {'-'*28} {'-'*22} {'-'*22}")
    for label, dv, mv in rows:
        print(f"  {label:<30} {str(dv):>22} {str(mv):>22}")

    print(bold("\n  ── Circuit topology ──"))
    topo = [
        ("n_features",           dt["n_features"],               ml["n_features"]),
        ("n_edges",              dt["n_edges"],                  ml["n_edges"]),
        ("n_paths",              dt["n_paths"],                  ml["n_paths"]),
        ("edges_per_feature",    f"{dt['n_edges']/dt['n_features']:.2f}",
                                 f"{ml['n_edges']/ml['n_features']:.2f}"),
        ("paths_per_feature",    f"{dt['n_paths']/dt['n_features']:.2f}",
                                 f"{ml['n_paths']/ml['n_features']:.2f}"),
        ("graph_nodes",          dt["graph_n_nodes"],            ml["graph_n_nodes"]),
        ("graph_VW_edges",       dt["graph_n_vw_edges"],         ml["graph_n_vw_edges"]),
        ("compression_ratio",    f"{dt['graph_n_nodes']/dt['n_features']:.1f}x",
                                 f"{ml['graph_n_nodes']/ml['n_features']:.1f}x"),
    ]
    for label, dv, mv in topo:
        print(f"  {label:<30} {str(dv):>22} {str(mv):>22}")

    print(bold("\n  ── Causal strength ──"))
    causal = [
        ("necessity",            fmt(dt["necessity"], ".1%", lambda v: v > 0.5),
                                 fmt(ml["necessity"], ".1%", lambda v: v > 0.5)),
        ("S1.5 sign_accuracy",   fmt(dt["s15_sign"], ".3f", lambda v: v > 0.80),
                                 fmt(ml["s15_sign"], ".3f", lambda v: v > 0.80)),
        ("S1.5 logit_retention", fmt(dt["s15_retention"], ".3f"),
                                 fmt(ml["s15_retention"], ".3f")),
        ("S1 sign_accuracy",     fmt(dt["s1_sign"], ".3f", lambda v: v > 0.80),
                                 "N/A"),
        ("S1 logit_retention",   fmt(dt["s1_retention"], ".3f"),
                                 "N/A"),
    ]
    for label, dv, mv in causal:
        print(f"  {label:<30} {dv:>22} {mv:>22}")

    print(bold("\n  ── Top causal path ──"))
    dt_path = " → ".join(dt["top_path"])
    ml_path = " → ".join(ml["top_path"])
    print(f"  physics_decay_type : {dt_path}")
    print(f"  multilingual_b1    : {ml_path}")

    print(bold("\n  ── Layer distribution of circuit features ──"))
    if dt["circuit_features"]:
        from collections import Counter
        dt_layers = Counter(int(f.split("_F")[0][1:]) for f in dt["circuit_features"])
        print(f"  physics_decay_type : {dict(sorted(dt_layers.items()))}")
        print(f"    (peak layer: {dt['peak_layer']})")
    print(f"  multilingual_b1    : distributed L10–L25 (23 features, peak unknown)")

    print(bold("\n  ── Interpretation ──"))
    print("""
  physics_decay_type (latent_state):
    • Compact chain (11 features, 16 edges) — near-linear topology
    • Very high necessity (67.6%) and sufficiency (S1.5=0.935) — tight bottleneck
    • Nearly all computation in L22–L25 (4 late layers)
    • L24 features are β-suppressor class (strongly beta-concept-specific)
    • Retains performance on keyword-free prompts (F1=82%) — not surface-matching
    • F3/F4 failure NOT explained by this circuit — separate mechanism

  multilingual_circuits_b1 (candidate_set):
    • Distributed (23 features, 52 edges) — branching, multi-path topology
    • Lower necessity (43.75%) — circuit is necessary but not monopolising
    • Lower sufficiency (S1.5=0.719) — circuit alone insufficient
    • Known artifacts (L22 FR-competitor circuit, 2 features confirmed not causal)
    • EN/FR asymmetry: transfer=0.125, circuit works better for EN
    • Cross-lingual bridge 60.4% (32/53 features shared across languages)

  Key structural difference:
    physics_decay_type is a CHAIN (chain-like, high bottleneck)
    multilingual_b1 is a FOREST (distributed, lower bottleneck)
    This matches behaviour type: latent_state → direct lookup;
    candidate_set → multi-path disambiguation
""")


def try_load_local_circuits():
    """Try to load actual circuit JSONs if present locally (CSD3 rsync)."""
    found = {}
    paths = {
        "physics_decay_type": PROJECT_ROOT / "data" / "results" / "causal_edges" /
                               "physics_decay_type" / "circuits_physics_decay_type_train.json",
        "multilingual_circuits_b1": PROJECT_ROOT / "data" / "results" / "causal_edges" /
                                    "multilingual_circuits_b1" /
                                    "circuits_multilingual_circuits_b1_train_roleaware_k3_t05.json",
    }
    for name, p in paths.items():
        if p.exists():
            found[name] = json.load(open(p))
            print(f"  Loaded local circuit: {p.name}")
    return found


def print_local_verification(local):
    """If local circuit JSONs are available, verify hardcoded stats."""
    if not local:
        return
    print(bold("\n  ── Local circuit verification ──"))
    for name, c in local.items():
        n_feat = len(c.get("features", []))
        n_edge = len(c.get("edges", []))
        n_path = len(c.get("paths", []))
        nec = c.get("disruption_rate", c.get("necessity", "?"))
        print(f"  {name}: {n_feat} features, {n_edge} edges, {n_path} paths, necessity={nec}")


def main():
    local = try_load_local_circuits()
    print_comparison()
    print_local_verification(local)
    print()


if __name__ == "__main__":
    main()
