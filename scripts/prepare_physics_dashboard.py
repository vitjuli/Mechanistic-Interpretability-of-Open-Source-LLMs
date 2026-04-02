"""
prepare_physics_dashboard.py

Prepare scalar/vector dashboard data files.

Generates:
  dashboard_physics/public/data/prompts.json        — per-prompt metadata + labels
  dashboard_physics/public/data/label_stats.json     — scalar vs vector effect stats
  dashboard_physics/public/data/cluster_label_stats.json — per-cluster label breakdown

Usage:
    python scripts/prepare_physics_dashboard.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
PROMPTS_JSONL = PROJECT / "data/prompts/physics_scalar_vector_operator_train.jsonl"
UI_RUN_DIR   = PROJECT / "data/ui_offline/20260224-122225_physics_scalar_vector_operator_train_n80"
OUTPUT_DIR   = PROJECT / "dashboard_physics/public/data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Physics concept → difficulty category ─────────────────────────────────────
CONCEPT_CATEGORY = {
    "dot_product":          "easy_lexical",    # "scalar product" / "dot product" contain the answer
    "cross_product":        "easy_lexical",    # "vector product" / "cross product" contain the answer
    "curl":                 "operator",        # curl/rot → vector; requires operator knowledge
    "divergence":           "operator",        # div → scalar; requires operator knowledge
    "gradient":             "operator",        # grad → vector; requires operator knowledge (★ tricky: scalar input → vector output)
    "laplacian_scalar":     "operator",        # Laplacian → scalar; requires operator knowledge
    "mechanics_vector":     "named_quantity",  # velocity, force, angular momentum, torque
    "electric_potential":   "contrast",        # potential (scalar) vs field (vector) — key demo contrast
    "energy_quantity":      "named_quantity",  # KE, PE, work → scalar
    "field_vector":         "named_quantity",  # gravitational field, force field → vector
    "thermodynamic_scalar": "named_quantity",  # mass, temperature, pressure, density
    "vector_potential":     "contrast",        # vector potential → vector (tricky naming)
}

CATEGORY_LABEL = {
    "easy_lexical":   "Easy (lexical)",
    "named_quantity": "Named quantity",
    "operator":       "Operator (harder)",
    "contrast":       "Key contrast",
}

# ── Step 1: Build prompts.json ─────────────────────────────────────────────────
print("Building prompts.json...")
prompts_meta = {}

with open(PROMPTS_JSONL) as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        concept = p.get("physics_concept", "unknown")
        category = CONCEPT_CATEGORY.get(concept, "other")
        label = p["field_type"]  # "scalar" or "vector"
        prompts_meta[str(idx)] = {
            "prompt":         p["prompt"],
            "correct_answer": p["correct_answer"].strip(),
            "incorrect_answer": p["incorrect_answer"].strip(),
            "label":          label,
            "physics_concept": concept,
            "operator_name":  p.get("operator_name", ""),
            "context_prefix": p.get("context_prefix", ""),
            "category":       category,
            "category_label": CATEGORY_LABEL.get(category, category),
        }

print(f"  {len(prompts_meta)} prompts loaded")
with open(OUTPUT_DIR / "prompts.json", "w") as f:
    json.dump(prompts_meta, f, indent=2)
print("  → prompts.json written")

# ── Step 2: Compute label_stats.json from interventions ────────────────────────
# For each layer: average effect size broken down by scalar vs vector prompts
# Uses ablation_zero experiment only (cleanest signal)
print("\nComputing label_stats.json...")

interventions_path = UI_RUN_DIR / "interventions.csv"
with open(interventions_path) as f:
    iv_rows = list(csv.DictReader(f))

# Build prompt → label map from meta.correct_token
prompt_label = {}
for r in iv_rows:
    pid = str(r["prompt_idx"])
    if pid not in prompt_label:
        ct = (r.get("meta.correct_token") or "").strip()
        prompt_label[pid] = ct  # "scalar" or "vector"

# Compute per-layer stats broken down by label
# Structure: {layer: {label: {n, sum_effect, sum_abs_effect}}}
layer_label_stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "sum_abs": 0.0, "sum_eff": 0.0, "sign_flips": 0}))

for r in iv_rows:
    if r.get("experiment_type") != "ablation_zero":
        continue
    pid = str(r["prompt_idx"])
    label = prompt_label.get(pid, "unknown")
    layer = int(r["layer"])
    try:
        abs_eff = float(r["abs_effect_size"])
        eff     = float(r["effect_size"])
        sf      = 1 if str(r.get("sign_flipped","")).lower() == "true" else 0
    except (ValueError, TypeError):
        continue
    s = layer_label_stats[layer][label]
    s["n"] += 1
    s["sum_abs"] += abs_eff
    s["sum_eff"] += eff
    s["sign_flips"] += sf

label_stats = {}
for layer in sorted(layer_label_stats.keys()):
    label_stats[layer] = {}
    for label, s in layer_label_stats[layer].items():
        n = s["n"]
        label_stats[layer][label] = {
            "n": n,
            "mean_abs_effect": round(s["sum_abs"] / n, 4) if n > 0 else 0,
            "mean_effect":     round(s["sum_eff"] / n, 4) if n > 0 else 0,
            "sign_flip_rate":  round(s["sign_flips"] / n, 4) if n > 0 else 0,
        }

with open(OUTPUT_DIR / "label_stats.json", "w") as f:
    json.dump(label_stats, f, indent=2)
print(f"  → label_stats.json written ({len(label_stats)} layers)")

# ── Step 3: cluster_label_stats.json ──────────────────────────────────────────
# For each cluster: how do scalar vs vector prompts perform (ablation effect)
print("\nComputing cluster_label_stats.json...")

supernodes_effect_path = UI_RUN_DIR / "supernodes_effect.json"
with open(supernodes_effect_path) as f:
    supernodes_effect = json.load(f)  # {cluster_id: [node_id, ...]}

# feature_agg: {(layer, feature_id): rows}
feature_agg_path = UI_RUN_DIR / "interventions_feature_agg.csv"
with open(feature_agg_path) as f:
    fa_rows = list(csv.DictReader(f))

# Per-cluster, per-label ablation stats
# We need to match ablation rows to cluster features
# Feature rank (feature_id 0-4) is used in intervention data; cluster members are node IDs
# Strategy: for cluster members, their layers tell us which layer's ablation data to use
# We can compute per-layer stats and map to cluster layer ranges

# Build cluster metadata (layers)
cluster_meta = {}
for cid, members in supernodes_effect.items():
    layers = []
    for nid in members:
        m = __import__("re").match(r"^L(\d+)_F(\d+)$", nid)
        if m:
            layers.append(int(m.group(1)))
    if layers:
        cluster_meta[cid] = {
            "n_features": len(members),
            "layers": sorted(set(layers)),
            "layer_min": min(layers),
            "layer_max": max(layers),
        }

# Per-cluster, per-label: aggregate layer_label_stats for that cluster's layers
cluster_label_stats = {}
for cid, meta in cluster_meta.items():
    cluster_label_stats[cid] = {"meta": meta, "by_label": {}}
    for label in ["scalar", "vector"]:
        agg = {"n": 0, "sum_abs": 0.0, "sum_eff": 0.0}
        for layer in meta["layers"]:
            s = layer_label_stats.get(layer, {}).get(label, {})
            n = s.get("n", 0)
            agg["n"] += n
            agg["sum_abs"] += s.get("sum_abs", 0.0)
            agg["sum_eff"] += s.get("sum_eff", 0.0)
        n = agg["n"]
        cluster_label_stats[cid]["by_label"][label] = {
            "n": n,
            "mean_abs_effect": round(agg["sum_abs"] / n, 4) if n > 0 else 0,
            "mean_effect":     round(agg["sum_eff"] / n, 4) if n > 0 else 0,
        }
    # Compute "label bias" — which label has stronger effect
    sv = cluster_label_stats[cid]["by_label"]
    scalar_abs = sv.get("scalar", {}).get("mean_abs_effect", 0)
    vector_abs = sv.get("vector", {}).get("mean_abs_effect", 0)
    total = scalar_abs + vector_abs
    if total > 0:
        cluster_label_stats[cid]["scalar_fraction"] = round(scalar_abs / total, 4)
        cluster_label_stats[cid]["vector_fraction"] = round(vector_abs / total, 4)
        diff = scalar_abs - vector_abs
        if abs(diff) < 0.05 * total:
            cluster_label_stats[cid]["dominant_label"] = "balanced"
        elif diff > 0:
            cluster_label_stats[cid]["dominant_label"] = "scalar-leaning"
        else:
            cluster_label_stats[cid]["dominant_label"] = "vector-leaning"
    else:
        cluster_label_stats[cid]["dominant_label"] = "unknown"
        cluster_label_stats[cid]["scalar_fraction"] = 0.5
        cluster_label_stats[cid]["vector_fraction"] = 0.5

with open(OUTPUT_DIR / "cluster_label_stats.json", "w") as f:
    json.dump(cluster_label_stats, f, indent=2)
print(f"  → cluster_label_stats.json written ({len(cluster_label_stats)} clusters)")

# ── Step 4: per_prompt_stats.json ─────────────────────────────────────────────
# For each prompt: baseline_logit_diff, mean_abs_effect across layers, top layer
print("\nComputing per_prompt_stats.json...")

prompt_agg_path = UI_RUN_DIR / "interventions_prompt_agg.csv"
with open(prompt_agg_path) as f:
    pa_rows = list(csv.DictReader(f))

per_prompt = defaultdict(lambda: {"ablation": {}, "patching": {}, "steering": {}})
for r in pa_rows:
    pid = str(r["prompt_idx"])
    exp = r.get("experiment_type", "ablation_zero")
    key = exp.replace("ablation_zero", "ablation")
    try:
        per_prompt[pid][key] = {
            "mean_abs_effect": float(r["mean_abs_effect_size"]),
            "mean_effect":     float(r["mean_effect_size"]),
            "sign_flip_rate":  float(r["sign_flip_rate"]),
            "mean_baseline":   float(r["mean_baseline_logit_diff"]),
        }
    except (ValueError, TypeError):
        pass

# Merge with prompt meta
per_prompt_stats = {}
for pid, meta in prompts_meta.items():
    per_prompt_stats[pid] = {
        **meta,
        "interventions": dict(per_prompt.get(pid, {})),
    }

with open(OUTPUT_DIR / "per_prompt_stats.json", "w") as f:
    json.dump(per_prompt_stats, f, indent=2)
print(f"  → per_prompt_stats.json written ({len(per_prompt_stats)} prompts)")

print("\n✓ Dashboard data preparation complete.")
print(f"  Output: {OUTPUT_DIR}")
print(f"\n  Files written:")
for p in sorted(OUTPUT_DIR.glob("*.json")):
    print(f"    {p.name}  ({p.stat().st_size:,} bytes)")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"    {p.name}  ({p.stat().st_size:,} bytes)")
