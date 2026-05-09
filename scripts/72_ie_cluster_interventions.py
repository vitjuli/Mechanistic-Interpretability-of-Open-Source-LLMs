"""
Cluster group ablation for physics_intensive_extensive_v1.

For each feature cluster (from k-means on feature activation matrix):
  - Ablate all cluster features simultaneously (set activations to 0 in transcoder)
  - Measure ΔND = ND_ablated - ND_baseline separately for intensive and extensive prompts
  - Normalise by |baseline ND| (relative effect)
  - Compute sign-flip rate: fraction where correct answer changes after ablation
  - Classify cluster as:
      intensive-supporting: ablation hurts intensive more (SFR_int >> SFR_ext)
      extensive-supporting: ablation hurts extensive more (SFR_ext >> SFR_int)
      general scaling:      ablation hurts both equally
  - Visualise as heatmap

Requires:
  - scripts/04 output (feature matrix, cluster labels)
  - scripts/71 output (cluster label assignment per feature)
  - transcoders loaded via src/transcoder.py

Usage:
    python scripts/72_ie_cluster_interventions.py --behaviour physics_intensive_extensive_v1 --device cuda --k 6
"""

import argparse
import json
import sys
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

PROMPT_DIR  = Path("data/prompts")
OUT_BASE    = Path("data/results/abstraction_ie")
MODEL_NAME  = "Qwen/Qwen3-4B"
MODEL_SIZE  = "4b"
INT_TOKEN   = 36195
EXT_TOKEN   = 16376
LABEL_MAP   = {"intensive": 0, "extensive": 1}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_prompts(behaviour, split, n=None):
    path = PROMPT_DIR / f"{behaviour}_{split}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    if n:
        from collections import defaultdict
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["abstraction_class"]].append(r)
        half = n // 2
        rows = by_cls["intensive"][:half] + by_cls["extensive"][:half]
    return rows

def load_cluster_labels(behaviour, k, out_dir):
    path = out_dir / f"ie_cluster_labels_k{k}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Cluster labels not found: {path}. Run script 71 first.")
    return np.load(path)

def load_feature_meta(behaviour, split):
    feat_dir = Path("data/features") / f"{behaviour}_{split}"
    meta_path = feat_dir / "feature_meta.json"
    if not meta_path.exists():
        meta_path = Path(f"data/results/internal_candidate_analysis/{behaviour}/feature_meta.json")
    with open(meta_path) as f:
        return json.load(f)


# ── Ablation utilities ────────────────────────────────────────────────────────

def _get_norm(model_inner, layer_idx):
    return model_inner.model.layers[layer_idx].post_attention_layernorm

@torch.no_grad()
def baseline_nd(model, tok, prompt, device):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out    = model.model(**inputs, use_cache=False)
    lp     = torch.log_softmax(out.logits[0, -1], dim=0)
    return float(lp[INT_TOKEN] - lp[EXT_TOKEN])  # >0 = model says intensive

@torch.no_grad()
def capture_all_layers(model, inputs, layers):
    captured, handles = {}, []
    for l in layers:
        norm = _get_norm(model.model, l)
        def hook(m, inp, out, _l=l):
            x = out[0] if isinstance(out, tuple) else out
            captured[_l] = x[:, -1, :].detach()
        handles.append(norm.register_forward_hook(hook))
    out = model.model(**inputs, use_cache=False)
    for h in handles: h.remove()
    return captured, out.logits[0, -1, :]

def ablate_cluster(model, transcoder_set, inputs, by_layer, device):
    all_layers = sorted(by_layer.keys())
    base_acts, base_logits = capture_all_layers(model, inputs, all_layers)
    ablated_inputs = {}
    for l, fidxs in by_layer.items():
        tc   = transcoder_set[l]
        act  = base_acts[l]
        feats = tc.encode(act.to(tc.dtype))
        feats[:, fidxs] = 0.0
        ablated_inputs[l] = tc.decode(feats).to(act.dtype)
    with ExitStack() as stack:
        for l, new_mlp in sorted(ablated_inputs.items()):
            norm = _get_norm(model.model, l)
            def hook(m, inp, out, _l=l, _new=new_mlp):
                if isinstance(out, tuple):
                    lst = list(out); lst[0][:, -1] = _new; return tuple(lst)
                out[:, -1] = _new; return out
            stack.enter_context(ExitStack())   # placeholder; hook registered below
        # Proper ExitStack with hooks
        for l, new_mlp in sorted(ablated_inputs.items()):
            norm = _get_norm(model.model, l)
            def _make_hook(new_mlp_=new_mlp):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        lst = list(out); lst[0][:, -1] = new_mlp_; return tuple(lst)
                    out[:, -1] = new_mlp_; return out
                return hook
            h = norm.register_forward_hook(_make_hook())
        abl_out = model.model(**inputs, use_cache=False)
        # Clean up (ExitStack manages this better; simplified here)
    abl_logits = abl_out.logits[0, -1, :]
    base_lp = float(torch.log_softmax(base_logits, dim=0)[INT_TOKEN]
                    - torch.log_softmax(base_logits, dim=0)[EXT_TOKEN])
    abl_lp  = float(torch.log_softmax(abl_logits,  dim=0)[INT_TOKEN]
                    - torch.log_softmax(abl_logits,  dim=0)[EXT_TOKEN])
    return base_lp, abl_lp


# Better ablation using ExitStack properly (from existing pattern in script 46/50):
@torch.no_grad()
def run_cluster_ablation(model, transcoder_set, inputs, by_layer):
    all_layers = sorted(by_layer.keys())
    base_acts, base_logits = capture_all_layers(model, inputs, all_layers)

    ablated_inputs = {}
    for l, fidxs in by_layer.items():
        tc    = transcoder_set[l]
        act   = base_acts[l]
        feats = tc.encode(act.to(tc.dtype))
        feats[:, fidxs] = 0.0
        ablated_inputs[l] = tc.decode(feats).to(act.dtype)

    handles = []
    for l, new_mlp in sorted(ablated_inputs.items()):
        norm = _get_norm(model.model, l)
        def _hook(m, inp, out, _new=new_mlp):
            if isinstance(out, tuple):
                lst = list(out); lst[0][:, -1] = _new; return tuple(lst)
            out[:, -1] = _new; return out
        handles.append(norm.register_forward_hook(_hook))

    abl_out = model.model(**inputs, use_cache=False)
    for h in handles:
        h.remove()

    abl_logits = abl_out.logits[0, -1, :]

    def nd(logits):
        lp = torch.log_softmax(logits, dim=0)
        return float(lp[INT_TOKEN] - lp[EXT_TOKEN])

    return nd(base_logits), nd(abl_logits)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_ablation_analysis(model, tok, transcoder_set, prompts, cluster_labels,
                          feat_meta, device, k):
    rows = []
    clusters = {}
    for fi, c in enumerate(cluster_labels):
        if fi >= len(feat_meta):
            break
        m = feat_meta[fi]
        clusters.setdefault(int(c), []).append((m["layer"], m["feature_idx"]))

    for cluster_id, feat_list in sorted(clusters.items()):
        by_layer = {}
        for (l, fidx) in feat_list:
            by_layer.setdefault(l, []).append(fidx)

        print(f"  C{cluster_id}: {len(feat_list)} feats across layers {sorted(by_layer.keys())}")

        for i, row in enumerate(prompts):
            inputs  = tok(row["prompt"], return_tensors="pt")
            inputs  = {k_: v.to(device) for k_, v in inputs.items()}
            cls     = row["abstraction_class"]

            try:
                nd_base, nd_abl = run_cluster_ablation(model, transcoder_set, inputs, by_layer)
            except Exception as e:
                print(f"    [WARN] {e}"); continue

            delta_nd   = nd_abl - nd_base
            sign_flip  = bool(nd_base > 0 and nd_abl <= 0) or bool(nd_base < 0 and nd_abl >= 0)
            rel_effect = delta_nd / (abs(nd_base) + 1e-6)

            rows.append({
                "cluster":          cluster_id,
                "prompt_idx":       i,
                "abstraction_class":cls,
                "property":         row.get("property",""),
                "wording_family":   row.get("wording_family",""),
                "nd_baseline":      round(nd_base, 4),
                "nd_ablated":       round(nd_abl, 4),
                "delta_nd":         round(delta_nd, 4),
                "rel_effect":       round(rel_effect, 4),
                "sign_flip":        sign_flip,
            })

    return pd.DataFrame(rows), clusters


def compute_cluster_selectivity(df):
    rows = []
    for c, grp in df.groupby("cluster"):
        int_mask = grp["abstraction_class"] == "intensive"
        ext_mask = grp["abstraction_class"] == "extensive"
        sfr_int  = float(grp[int_mask]["sign_flip"].mean()) if int_mask.sum() else float("nan")
        sfr_ext  = float(grp[ext_mask]["sign_flip"].mean()) if ext_mask.sum() else float("nan")
        dnd_int  = float(grp[int_mask]["delta_nd"].mean())  if int_mask.sum() else float("nan")
        dnd_ext  = float(grp[ext_mask]["delta_nd"].mean())  if ext_mask.sum() else float("nan")
        rel_int  = float(grp[int_mask]["rel_effect"].mean()) if int_mask.sum() else float("nan")
        rel_ext  = float(grp[ext_mask]["rel_effect"].mean()) if ext_mask.sum() else float("nan")

        # Classify cluster
        if not (np.isnan(sfr_int) or np.isnan(sfr_ext)):
            if sfr_int > sfr_ext + 0.10:
                role = "intensive-supporting"
            elif sfr_ext > sfr_int + 0.10:
                role = "extensive-supporting"
            else:
                role = "general-scaling"
        else:
            role = "unknown"

        rows.append({
            "cluster":    int(c),
            "sfr_intensive": round(sfr_int, 4),
            "sfr_extensive": round(sfr_ext, 4),
            "delta_nd_intensive": round(dnd_int, 4),
            "delta_nd_extensive": round(dnd_ext, 4),
            "rel_effect_intensive": round(rel_int, 4),
            "rel_effect_extensive": round(rel_ext, 4),
            "role": role,
        })
    return pd.DataFrame(rows)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_heatmap(sel_df, out_dir):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    clusters = sel_df["cluster"].tolist()
    sfr_int  = sel_df["sfr_intensive"].tolist()
    sfr_ext  = sel_df["sfr_extensive"].tolist()
    x  = np.arange(len(clusters))
    w  = 0.35

    ax = axes[0]
    ax.bar(x - w/2, sfr_int, w, color="#60a5fa", label="Intensive prompts")
    ax.bar(x + w/2, sfr_ext, w, color="#f97316", label="Extensive prompts")
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_ylabel("Sign-flip rate"); ax.set_title("Sign-flip rate by cluster × class")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    for i, role in enumerate(sel_df["role"]):
        color = "#2563eb" if "intensive" in role else "#dc2626" if "extensive" in role else "#6b7280"
        ax.annotate(role.replace("-supporting","").replace("general-","gen."),
                    (i, max(sfr_int[i], sfr_ext[i]) + 0.01), ha="center", fontsize=7.5, color=color)

    ax = axes[1]
    ax.bar(x - w/2, sel_df["rel_effect_intensive"].tolist(), w, color="#60a5fa", label="Intensive")
    ax.bar(x + w/2, sel_df["rel_effect_extensive"].tolist(), w, color="#f97316", label="Extensive")
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_ylabel("Relative ΔND / |baseline|")
    ax.set_title("Normalised ablation effect by cluster × class")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("IE Cluster Interventions — Causal Selectivity", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "ie_cluster_interventions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_dir}/ie_cluster_interventions.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",  default="physics_intensive_extensive_v1")
    ap.add_argument("--split",      default="train")
    ap.add_argument("--k",          type=int, default=6)
    ap.add_argument("--n_prompts",  type=int, default=100)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",      default="bfloat16", choices=["float32","bfloat16","float16"])
    args = ap.parse_args()

    device  = torch.device(args.device)
    dtype   = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[args.dtype]
    out_dir = OUT_BASE / args.behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.behaviour, args.split, n=args.n_prompts)
    print(f"Prompts: {len(prompts)} ({sum(1 for r in prompts if r['abstraction_class']=='intensive')} int,"
          f" {sum(1 for r in prompts if r['abstraction_class']=='extensive')} ext)")

    cluster_labels = load_cluster_labels(args.behaviour, args.k, out_dir)
    feat_meta      = load_feature_meta(args.behaviour, args.split)
    print(f"Features: {len(feat_meta)} | Cluster labels: {len(cluster_labels)} | k={args.k}")

    # Get unique layers from feature meta
    all_layers = sorted(set(m["layer"] for m in feat_meta))
    print(f"Loading transcoders for layers: {all_layers}")
    model = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=dtype)
    model.model.eval()
    transcoder_set = load_transcoder_set(
        model_size=MODEL_SIZE, device=device, dtype=dtype,
        lazy_load=True, layers=all_layers,
    )

    print("\nRunning cluster ablations…")
    df, clusters = run_ablation_analysis(
        model, model.tokenizer, transcoder_set,
        prompts, cluster_labels, feat_meta, device, args.k
    )

    df.to_csv(out_dir / f"ie_cluster_ablation_k{args.k}.csv", index=False)

    sel_df = compute_cluster_selectivity(df)
    sel_df.to_csv(out_dir / f"ie_cluster_selectivity_k{args.k}.csv", index=False)

    print("\n=== CLUSTER SELECTIVITY ===")
    print(sel_df.to_string(index=False))

    make_heatmap(sel_df, out_dir)

    # Report
    lines = [
        "# IE Cluster Intervention Report",
        f"## {args.behaviour} | k={args.k} | n={args.n_prompts} prompts",
        "",
        "## Cluster Roles (ablation-based)",
        "",
        sel_df.to_string(index=False),
        "",
        "## Interpretation",
        "- intensive-supporting: ablating this cluster disrupts intensive prompts more (SFR_int >> SFR_ext)",
        "- extensive-supporting: ablating disrupts extensive prompts more",
        "- general-scaling: ablating affects both classes equally",
    ]
    (out_dir / f"ie_cluster_intervention_report_k{args.k}.md").write_text("\n".join(lines))
    print(f"\nReport + CSVs in: {out_dir}")


if __name__ == "__main__":
    main()
