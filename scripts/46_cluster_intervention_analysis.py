"""
Cluster-level causal intervention analysis.

For each feature cluster identified by script 45, tests causal selectivity by:
  A — Group ablation: zero all cluster features, measure ΔND per particle class
  B — Group steering: add decoder directions, measure logit shifts per particle
  C — Causal selectivity: ΔND_target − ΔND_non_target
  D — Early vs late cluster comparison

Reads cluster assignments from script 45 outputs.
Requires GPU — run via: sbatch jobs/run_cluster_intervention.sbatch

Usage:
  python scripts/46_cluster_intervention_analysis.py --behaviour physics_internal_candidate_selection_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BEHAVIOUR  = "physics_internal_candidate_selection_v2"
SPLIT      = "train"
PARTICLES  = ["electron", "proton", "neutron", "photon"]
N_ST       = 447
DEFAULT_K  = 6
DEFAULT_METHOD = "kmeans"
STEERING_COEFF = 5.0


def get_paths(behaviour, split, k=DEFAULT_K, method=DEFAULT_METHOD):
    base  = Path("data")
    adir  = base / "results" / "internal_candidate_analysis" / behaviour
    return {
        "prompts":      base / "prompts" / f"{behaviour}_{split}.jsonl",
        "graph_json":   base / "results" / "attribution_graphs" / behaviour
                             / f"attribution_graph_{split}_n120_roleaware.json",
        "cluster_csv":  adir / f"feature_clusters_k{k}_{method}.csv",
        "cluster_idx":  adir / "cluster_feature_index.csv",
        "output_dir":   adir,
    }


# ─── Data loading ────────────────────────────────────────────────────────────

def load_prompts(paths):
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct"] = r["correct_answer"].strip()
        r["_pool"]    = set(r["implicit_candidate_pool"])
    return [r for r in rows if not r.get("multi_token_answer", False)]


def load_clusters(paths, k, method):
    """Returns {cluster_id: [(layer, feat_idx), ...]}."""
    csv_path = paths["output_dir"] / f"feature_clusters_k{k}_{method}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster CSV not found: {csv_path}\n"
            f"Run script 45 first: python scripts/45_candidate_feature_clustering.py"
        )
    df = pd.read_csv(csv_path)
    col = f"cluster_k{k}_{method}"
    clusters = {}
    for _, row in df.iterrows():
        c = int(row[col])
        clusters.setdefault(c, []).append((int(row["layer"]), int(row["feature_idx"])))
    return clusters


def load_transcoder(model_size="4b"):
    from src.transcoder.load import load_transcoders
    return load_transcoders(model_size=model_size)


def load_model(model_size="4b"):
    from src.model_utils import load_model as _load_model
    return _load_model(model_size=model_size, instruct=False)


# ─── Prompt utilities ─────────────────────────────────────────────────────────

def get_group_masks(prompts):
    """Returns {particle: {'target':mask, 'competitor':mask, 'background':mask}}."""
    n = len(prompts)
    masks = {}
    for p in PARTICLES:
        t = np.array([r["_correct"] == p for r in prompts])
        c = np.array([p in r["_pool"] and r["_correct"] != p for r in prompts])
        b = np.array([p not in r["_pool"] for r in prompts])
        masks[p] = {"target": t, "competitor": c, "background": b}
    return masks


# ─── Inference with hooks ─────────────────────────────────────────────────────

@torch.no_grad()
def run_logprob_diff(model, tokenizer, prompt, correct_token, incorrect_token,
                     ablation_hooks=None, device="cuda"):
    """Returns (ND_baseline, ND_ablated) given ablation hook functions."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def forward(hooks_to_register):
        handles = []
        if hooks_to_register:
            for layer_idx, hook_fn in hooks_to_register:
                handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
                handles.append(handle)
        out = model(**inputs)
        for h in handles:
            h.remove()
        return out.logits[0, -1, :]

    # Baseline
    logits = forward([])
    correct_id   = tokenizer(correct_token,   add_special_tokens=False)["input_ids"][0]
    incorrect_id = tokenizer(incorrect_token, add_special_tokens=False)["input_ids"][0]
    nd_base = float(logits[correct_id] - logits[incorrect_id])

    # Ablated
    if ablation_hooks:
        logits_abl = forward(ablation_hooks)
        nd_abl = float(logits_abl[correct_id] - logits_abl[incorrect_id])
    else:
        nd_abl = nd_base

    return nd_base, nd_abl


def make_ablation_hook(transcoder, feat_indices, layer_idx):
    """Returns a hook that zeros the transcoder activations for feat_indices."""
    def hook(module, input, output):
        # Transcoder operates on MLP input → get features
        x = input[0]
        with torch.no_grad():
            acts = transcoder[layer_idx].encode(x)
            acts[:, :, feat_indices] = 0.0
            patched_output = transcoder[layer_idx].decode(acts)
        return patched_output
    return hook


def make_steering_hook(transcoder, feat_indices, layer_idx, coeff=STEERING_COEFF):
    """Returns a hook that adds scaled decoder vectors for feat_indices."""
    def hook(module, input, output):
        x = input[0]
        with torch.no_grad():
            # Get decoder vectors for these features and add them
            W_dec = transcoder[layer_idx].W_dec[:, feat_indices]  # [d_model, n_feats]
            steering_vec = W_dec.sum(dim=1) * coeff
            patched_output = output + steering_vec.unsqueeze(0).unsqueeze(0)
        return patched_output
    return hook


# ─── Part A: Group ablation ───────────────────────────────────────────────────

def run_group_ablation(model, tokenizer, transcoder, clusters, prompts,
                       group_masks, n_prompts=80, device="cuda"):
    """
    For each cluster: zero all its features, measure ΔND per prompt.
    Returns DataFrame with one row per (cluster, prompt_idx).
    """
    rows = []
    sample_idx = np.random.choice(N_ST, min(n_prompts, N_ST), replace=False)

    for cluster_id, feat_list in sorted(clusters.items()):
        # Group features by layer
        by_layer = {}
        for (layer, fidx) in feat_list:
            by_layer.setdefault(layer, []).append(fidx)

        print(f"  Cluster {cluster_id}: {len(feat_list)} features across {len(by_layer)} layers")

        # Build hook functions
        ablation_hooks = []
        for layer_idx, feat_indices in by_layer.items():
            fidx_tensor = torch.tensor(feat_indices, dtype=torch.long, device=device)
            hook_fn = make_ablation_hook(transcoder, fidx_tensor, layer_idx)
            # Map transcoder layer to model layer (transcoder covers L10-L25)
            model_layer = layer_idx  # adjust if needed
            ablation_hooks.append((model_layer, hook_fn))

        for i in sample_idx:
            prompt   = prompts[i]
            nd_base, nd_abl = run_logprob_diff(
                model, tokenizer,
                prompt["prompt"], prompt["correct_answer"], prompt["incorrect_answer"],
                ablation_hooks=ablation_hooks, device=device
            )
            delta_nd   = nd_abl - nd_base
            sign_flip  = bool(nd_base > 0 and nd_abl <= 0)
            rows.append({
                "cluster":        cluster_id,
                "prompt_idx":     int(i),
                "correct_answer": prompt["_correct"],
                "filter_property": prompt.get("filter_property", ""),
                "wording_family": prompt.get("wording_family", ""),
                "nd_baseline":    float(nd_base),
                "nd_ablated":     float(nd_abl),
                "delta_nd":       float(delta_nd),
                "abs_delta_nd":   float(abs(delta_nd)),
                "sign_flip":      sign_flip,
            })

    return pd.DataFrame(rows)


# ─── Part B: Group steering ───────────────────────────────────────────────────

def run_group_steering(model, tokenizer, transcoder, clusters, prompts,
                       n_prompts=80, device="cuda", coeff=STEERING_COEFF):
    rows = []
    sample_idx = np.random.choice(N_ST, min(n_prompts, N_ST), replace=False)

    for cluster_id, feat_list in sorted(clusters.items()):
        by_layer = {}
        for (layer, fidx) in feat_list:
            by_layer.setdefault(layer, []).append(fidx)

        steering_hooks = []
        for layer_idx, feat_indices in by_layer.items():
            fidx_tensor = torch.tensor(feat_indices, dtype=torch.long, device=device)
            hook_fn = make_steering_hook(transcoder, fidx_tensor, layer_idx, coeff=coeff)
            steering_hooks.append((layer_idx, hook_fn))

        for i in sample_idx:
            prompt = prompts[i]
            nd_base, nd_steered = run_logprob_diff(
                model, tokenizer,
                prompt["prompt"], prompt["correct_answer"], prompt["incorrect_answer"],
                ablation_hooks=steering_hooks, device=device
            )
            rows.append({
                "cluster":        cluster_id,
                "prompt_idx":     int(i),
                "correct_answer": prompt["_correct"],
                "filter_property": prompt.get("filter_property", ""),
                "nd_baseline":    float(nd_base),
                "nd_steered":     float(nd_steered),
                "delta_nd":       float(nd_steered - nd_base),
                "steering_coeff": coeff,
            })

    return pd.DataFrame(rows)


# ─── Part C: Causal selectivity ──────────────────────────────────────────────

def compute_selectivity(ablation_df):
    """
    selectivity(cluster, particle) = mean_ΔND over target prompts
                                   - mean_ΔND over non-target prompts

    Negative selectivity = ablating this cluster hurts this particle more than others.
    """
    rows = []
    for (cluster, particle), grp in ablation_df.groupby(["cluster", "correct_answer"]):
        target_delta  = grp["delta_nd"].mean()
        non_target    = ablation_df[
            (ablation_df["cluster"] == cluster) &
            (ablation_df["correct_answer"] != particle)
        ]["delta_nd"].mean()
        selectivity   = float(target_delta - non_target)
        sign_flip_rate = float(grp["sign_flip"].mean())
        rows.append({
            "cluster":          cluster,
            "particle":         particle,
            "mean_delta_nd_target": float(target_delta),
            "mean_delta_nd_other":  float(non_target),
            "selectivity":      selectivity,
            "sign_flip_rate":   sign_flip_rate,
            "n_target_prompts": len(grp),
        })
    return pd.DataFrame(rows)


# ─── Report ───────────────────────────────────────────────────────────────────

def write_intervention_report(ablation_df, selectivity_df, steering_df, output_dir):
    lines = [
        "# Cluster Intervention Analysis Report",
        f"## {BEHAVIOUR} | {SPLIT}",
        "",
        "## Part A: Group Ablation Summary",
        "",
    ]

    if ablation_df is not None and len(ablation_df):
        agg = ablation_df.groupby("cluster").agg(
            mean_delta_nd=("delta_nd", "mean"),
            mean_abs_delta=("abs_delta_nd", "mean"),
            sign_flip_rate=("sign_flip", "mean"),
            n_prompts=("prompt_idx", "count"),
        ).reset_index()
        lines += [
            "| Cluster | mean ΔND | mean |ΔND| | sign_flip_rate | n_prompts |",
            "|---|---|---|---|---|",
        ]
        for _, r in agg.sort_values("mean_abs_delta", ascending=False).iterrows():
            lines.append(
                f"| C{int(r['cluster'])} | {r['mean_delta_nd']:.3f} | "
                f"{r['mean_abs_delta']:.3f} | {r['sign_flip_rate']:.3f} | "
                f"{int(r['n_prompts'])} |"
            )

    if selectivity_df is not None and len(selectivity_df):
        lines += [
            "",
            "## Part C: Causal Selectivity",
            "",
            "Selectivity = mean_ΔND(target) − mean_ΔND(other).",
            "Negative = ablation hurts target particle more than others (particle-selective).",
            "",
            "| Cluster | Particle | ΔND_target | ΔND_other | Selectivity | sign_flip% |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in selectivity_df.sort_values("selectivity").iterrows():
            flag = " ◀" if r["selectivity"] < -0.2 else ""
            lines.append(
                f"| C{int(r['cluster'])} | {r['particle']} | "
                f"{r['mean_delta_nd_target']:.3f} | {r['mean_delta_nd_other']:.3f} | "
                f"**{r['selectivity']:.3f}**{flag} | {r['sign_flip_rate']:.1%} |"
            )

    (output_dir / "report_cluster_intervention.md").write_text("\n".join(lines))
    print(f"  Report: report_cluster_intervention.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",   default=BEHAVIOUR)
    ap.add_argument("--split",       default=SPLIT)
    ap.add_argument("--k",           type=int, default=DEFAULT_K)
    ap.add_argument("--method",      default=DEFAULT_METHOD)
    ap.add_argument("--n_prompts",   type=int, default=80)
    ap.add_argument("--steering_coeff", type=float, default=STEERING_COEFF)
    ap.add_argument("--skip_steering",  action="store_true")
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split, args.k, args.method)
    out   = paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    prompts  = load_prompts(paths)
    clusters = load_clusters(paths, args.k, args.method)
    masks    = get_group_masks(prompts)

    print(f"Clusters loaded: {len(clusters)}")
    for c, feats in sorted(clusters.items()):
        print(f"  C{c}: {len(feats)} features")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model and transcoders...")
    model, tokenizer = load_model()
    model = model.to(device)
    model.eval()
    transcoder = load_transcoder()

    # Part A: ablation
    print("\n── Part A: Group ablation ──")
    ablation_df = run_group_ablation(
        model, tokenizer, transcoder, clusters, prompts,
        masks, n_prompts=args.n_prompts, device=str(device)
    )
    ablation_df.to_csv(out / f"cluster_ablation_k{args.k}_{args.method}.csv", index=False)
    print(f"  Saved ablation results: {len(ablation_df)} rows")

    # Part C: selectivity
    print("\n── Part C: Causal selectivity ──")
    selectivity_df = compute_selectivity(ablation_df)
    selectivity_df.to_csv(out / f"cluster_selectivity_k{args.k}_{args.method}.csv", index=False)
    print(selectivity_df[["cluster", "particle", "selectivity", "sign_flip_rate"]].to_string(index=False))

    # Part B: steering
    steering_df = None
    if not args.skip_steering:
        print("\n── Part B: Group steering ──")
        steering_df = run_group_steering(
            model, tokenizer, transcoder, clusters, prompts,
            n_prompts=args.n_prompts, device=str(device), coeff=args.steering_coeff
        )
        steering_df.to_csv(out / f"cluster_steering_k{args.k}_{args.method}.csv", index=False)
        print(f"  Saved steering results: {len(steering_df)} rows")

    write_intervention_report(ablation_df, selectivity_df, steering_df, out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
