import json
import networkx as nx

JSON_PATH = "data/results/attribution_graphs/grammar_agreement/attribution_graph_train_n80.json"
GRAPHML_PATH = "data/results/attribution_graphs/grammar_agreement/attribution_graph_train_n80.graphml"

def node_key(n: dict, fallback: str) -> str:
    # try common id fields in priority order
    for k in ("id", "node_id", "name", "key"):
        v = n.get(k)
        if v is not None:
            return str(v)
    return fallback

def resolve_endpoint(x, nodes, idx_to_key):
    """
    x can be:
      - int: index into nodes list
      - str: already a node key
      - dict: may contain 'id'/'node_id'/'name' or sometimes 'index'
    """
    if isinstance(x, int):
        return idx_to_key.get(x)
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # sometimes D3 puts an object with an index
        if "index" in x and isinstance(x["index"], int):
            return idx_to_key.get(x["index"])
        # or it embeds the id
        for k in ("id", "node_id", "name", "key"):
            if k in x and x[k] is not None:
                return str(x[k])
    return None

with open(JSON_PATH, "r") as f:
    g = json.load(f)

nodes = g.get("nodes", [])
links = g.get("links", [])

G = nx.DiGraph()

# build stable node keys
idx_to_key = {}
for i, n in enumerate(nodes):
    k = node_key(n, fallback=f"idx_{i}")
    idx_to_key[i] = k
    attrs = {kk: vv for kk, vv in n.items() if kk not in ("id", "node_id", "name", "key")}
    G.add_node(k, **attrs)

# add edges
n_added = 0
n_skipped = 0
for e in links:
    src = resolve_endpoint(e.get("source"), nodes, idx_to_key)
    tgt = resolve_endpoint(e.get("target"), nodes, idx_to_key)

    if src is None or tgt is None:
        n_skipped += 1
        continue
    if src not in G.nodes or tgt not in G.nodes:
        n_skipped += 1
        continue

    attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
    G.add_edge(src, tgt, **attrs)
    n_added += 1

nx.write_graphml(G, GRAPHML_PATH)

print("Wrote:", GRAPHML_PATH)
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
print("Edges added:", n_added, "skipped:", n_skipped, "links_in_json:", len(links))
