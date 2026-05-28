"""
diag_extract_circuit_features.py

Extract top-K features from an attribution graph, ranked by:
  - max |edge weight| to/from output
  - total |edge weight| across all edges

For H1 experiment in scripts/53_iia_diagnosis.py.

Usage (local, after syncing n538 graph from CSD3):
  python scripts/diag_extract_circuit_features.py \
      --graph data/results/attribution_graphs/physics_decay_type_probe_v2/attribution_graph_train_n538_roleaware_static_k20.json

  # Fallback to whatever exists:
  python scripts/diag_extract_circuit_features.py --behaviour physics_decay_type_probe_v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None,
                    help="Explicit path to attribution graph JSON")
    ap.add_argument("--behaviour", default="physics_decay_type_probe_v2",
                    help="Used when --graph not specified")
    ap.add_argument("--n_graph",   default=None, type=int,
                    help="n_prompts in graph filename, if not specifying --graph")
    ap.add_argument("--suffix",    default="_roleaware_static_k20",
                    help="Graph file suffix")
    ap.add_argument("--top_k",     type=int, default=30)
    ap.add_argument("--out",
                    default="data/analysis/iia_failure_diagnosis/circuit_features_for_h1.json")
    args = ap.parse_args()

    root = Path(__file__).parent.parent

    # Resolve graph path
    if args.graph:
        gpath = Path(args.graph)
        if not gpath.is_absolute():
            gpath = root / gpath
    else:
        gdir = root / "data/results/attribution_graphs" / args.behaviour
        if args.n_graph:
            gpath = gdir / f"attribution_graph_train_n{args.n_graph}{args.suffix}.json"
        else:
            # Auto: prefer largest n
            candidates = sorted(gdir.glob("attribution_graph_train_n*.json"),
                                key=lambda p: int(p.stem.split("_n")[1].split("_")[0]),
                                reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No graphs in {gdir}")
            gpath = candidates[0]
            print(f"Auto-selected largest graph: {gpath.name}")

    assert gpath.exists(), f"Graph not found: {gpath}"
    g = json.load(open(gpath))
    print(f"Loaded {gpath.name}: {len(g['nodes'])} nodes, {len(g['edges'])} edges")

    # Score features
    out_edge_score = defaultdict(float)
    total_w        = defaultdict(float)
    for e in g["edges"]:
        s, t = e["source"], e["target"]
        w    = abs(e.get("weight", 0))
        if s.startswith("L") and ("output" in t or "logit" in t):
            out_edge_score[s] = max(out_edge_score[s], w)
        if t.startswith("L") and ("output" in s or "logit" in s):
            out_edge_score[t] = max(out_edge_score[t], w)
        if s.startswith("L"):
            total_w[s] += w
        if t.startswith("L"):
            total_w[t] += w

    top_out  = sorted(out_edge_score.items(), key=lambda x: -x[1])[:args.top_k]
    top_tot  = sorted(total_w.items(),        key=lambda x: -x[1])[:args.top_k]

    out = {
        "source_graph": str(gpath.relative_to(root)),
        "n_nodes":      len(g["nodes"]),
        "n_edges":      len(g["edges"]),
        "top_k":        args.top_k,
        "top_by_output_edge": [
            {"feature_id": fid, "layer": int(fid.split("_F")[0].lstrip("L")),
             "weight": float(s)} for fid, s in top_out],
        "top_by_total_attribution": [
            {"feature_id": fid, "layer": int(fid.split("_F")[0].lstrip("L")),
             "weight": float(s)} for fid, s in top_tot],
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    print(f"\nTop 10 by output-edge weight:")
    for f in out["top_by_output_edge"][:10]:
        print(f"  {f['feature_id']:>20s}  L{f['layer']:2d}  |w|={f['weight']:.4f}")


if __name__ == "__main__":
    main()
