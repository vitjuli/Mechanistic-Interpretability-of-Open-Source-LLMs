"""
Cluster-level causal intervention analysis.

For each feature cluster identified by script 45, tests causal selectivity by:
  A — Group ablation: zero all cluster features, measure ΔND per particle class
  B — Group steering: add decoder directions, measure logit shifts per particle
  C — Causal selectivity: ΔND_target − ΔND_non_target

Reads cluster assignments from script 45 (feature_clusters_k{k}_{method}.csv).
Uses same model loading / hook infrastructure as script 07.

Usage:
  python scripts/46_cluster_intervention_analysis.py
  python scripts/46_cluster_intervention_analysis.py --k 6 --n_prompts 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project path (same pattern as all other scripts) ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set
from scripts.script07_utils import get_mlp_input_activation, patch_mlp_input, ensure_single_token

BEHAVIOUR   = "physics_internal_candidate_selection_v2"
SPLIT       = "train"
PARTICLES   = ["electron", "proton", "neutron", "photon"]
N_ST        = 447
MODEL_NAME  = "Qwen/Qwen3-4B"
MODEL_SIZE  = "4b"
STEERING_COEFF = 5.0


# ─── Locate helper functions from script 07 ───────────────────────────────────
# Rather than re-implementing hooks, import the tested functions directly.

def _import_script07_utils():
    """Import hook utilities from script 07 by exec-ing the relevant functions."""
    import importlib.util, ast

    script07 = Path(__file__).parent / "07_run_interventions.py"
    # We only need: patch_mlp_input, get_mlp_input_activation, ensure_single_token
    # These are pure functions; import them by loading the module.
    spec = importlib.util.spec_from_file_location("script07", script07)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.patch_mlp_input, mod.get_mlp_input_activation, mod.ensure_single_token


try:
    patch_mlp_input, get_mlp_input_activation, ensure_single_token = _import_script07_utils()
except Exception as e:
    print(f"[WARN] Could not import from script 07: {e}")
    print("       Falling back to inline implementations.")
    # Inline fallbacks (simpler, less robust than script 07's version)
    from contextlib import contextmanager

    def ensure_single_token(model, tok):
        ids = model.tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"Token '{tok}' is multi-token: {ids}"
        return ids[0]

    def get_mlp_input_activation(model, inputs, layer_idx, token_pos=-1):
        captured = {}
        blocks = model.model.model.layers
        # Try post_attention_layernorm (Qwen3 architecture)
        norm = blocks[layer_idx].post_attention_layernorm
        def hook(module, inp, out):
            captured["x"] = (out[0] if isinstance(out, tuple) else out).detach()
        h = norm.register_forward_hook(hook)
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
        h.remove()
        assert "x" in captured, f"Hook didn't fire at layer {layer_idx}"
        return captured["x"][:, token_pos, :]  # (batch=1, d_model)

    @contextmanager
    def patch_mlp_input(model_inner, layer_idx, token_pos, new_mlp_input):
        blocks = model_inner.model.layers
        norm   = blocks[layer_idx].post_attention_layernorm
        def hook(module, inp, out):
            if isinstance(out, tuple):
                lst = list(out); lst[0][:, token_pos] = new_mlp_input; return tuple(lst)
            out[:, token_pos] = new_mlp_input; return out
        h = norm.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()


# ─── Paths ────────────────────────────────────────────────────────────────────

def get_paths(behaviour, split, k, method):
    base = Path("data")
    adir = base / "results" / "internal_candidate_analysis" / behaviour
    return {
        "prompts":     base / "prompts" / f"{behaviour}_{split}.jsonl",
        "cluster_csv": adir / f"feature_clusters_k{k}_{method}.csv",
        "output_dir":  adir,
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_prompts(paths):
    with open(paths["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct"] = r["correct_answer"].strip()
        r["_pool"]    = set(r["implicit_candidate_pool"])
    return [r for r in rows if not r.get("multi_token_answer", False)]


def load_clusters(paths, k, method):
    csv_path = paths["cluster_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster CSV not found: {csv_path}\n"
            f"Run script 45 first: python scripts/45_candidate_feature_clustering.py"
        )
    df  = pd.read_csv(csv_path)
    col = f"cluster_k{k}_{method}"
    clusters = {}
    for _, row in df.iterrows():
        c = int(row[col])
        clusters.setdefault(c, []).append((int(row["layer"]), int(row["feature_idx"])))
    return clusters


# ─── Logit diff baseline ──────────────────────────────────────────────────────

def compute_logit_diff(model, inputs, device, correct_token, incorrect_token):
    with torch.no_grad():
        out    = model.model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :]
    log_p = torch.log_softmax(logits, dim=0)
    cid   = ensure_single_token(model, correct_token)
    iid   = ensure_single_token(model, incorrect_token)
    return (log_p[cid] - log_p[iid]).item()


# ─── Part A: Group ablation ────────────────────────────────────────────────────

def run_group_ablation(model, transcoder_set, device, clusters, prompts,
                       n_prompts=80, rng_seed=42):
    from contextlib import ExitStack

    rng  = np.random.default_rng(rng_seed)
    idxs = rng.choice(N_ST, min(n_prompts, N_ST), replace=False)
    rows = []

    for cluster_id, feat_list in sorted(clusters.items()):
        by_layer = {}
        for (layer, fidx) in feat_list:
            by_layer.setdefault(layer, []).append(fidx)

        print(f"  C{cluster_id}: {len(feat_list)} feats, layers {sorted(by_layer)}")

        for i in idxs:
            prompt = prompts[i]
            inputs = model.tokenize([prompt["prompt"]])
            inputs = {k: v.to(device) for k, v in inputs.items()}

            nd_base = compute_logit_diff(model, inputs, device,
                                         prompt["correct_answer"], prompt["incorrect_answer"])
            nd_abl  = nd_base

            try:
                # Step 1: capture baseline MLP inputs at all cluster layers
                baseline_acts = {}
                for layer_idx in sorted(by_layer):
                    baseline_acts[layer_idx] = get_mlp_input_activation(
                        model, inputs, layer_idx
                    )

                # Step 2: compute ablated MLP inputs (zero cluster features)
                ablated_inputs = {}
                for layer_idx, feat_indices in sorted(by_layer.items()):
                    tc   = transcoder_set[layer_idx]
                    act  = baseline_acts[layer_idx]
                    feats = tc.encode(act.to(tc.dtype))
                    feats[:, feat_indices] = 0.0
                    ablated_inputs[layer_idx] = tc.decode(feats).to(act.dtype)

                # Step 3: single forward pass with all patches active simultaneously
                with ExitStack() as stack:
                    for layer_idx, new_mlp in sorted(ablated_inputs.items()):
                        stack.enter_context(
                            patch_mlp_input(model.model, layer_idx, -1, new_mlp)
                        )
                    with torch.no_grad():
                        out_abl = model.model(**inputs, use_cache=False)

                log_p_abl = torch.log_softmax(out_abl.logits[0, -1, :], dim=0)
                cid   = ensure_single_token(model, prompt["correct_answer"])
                iid   = ensure_single_token(model, prompt["incorrect_answer"])
                nd_abl = (log_p_abl[cid] - log_p_abl[iid]).item()

            except Exception as exc:
                print(f"    [WARN] prompt {i} C{cluster_id}: {exc}")

            rows.append({
                "cluster":         cluster_id,
                "prompt_idx":      int(i),
                "correct_answer":  prompt["_correct"],
                "filter_property": prompt.get("filter_property", ""),
                "wording_family":  prompt.get("wording_family", ""),
                "nd_baseline":     float(nd_base),
                "nd_ablated":      float(nd_abl),
                "delta_nd":        float(nd_abl - nd_base),
                "abs_delta_nd":    float(abs(nd_abl - nd_base)),
                "sign_flip":       bool(nd_base > 0 and nd_abl <= 0),
            })

        abl_rows = [r for r in rows if r["cluster"] == cluster_id]
        print(f"    mean_ΔND={np.mean([r['delta_nd'] for r in abl_rows]):.3f}  "
              f"sign_flip={np.mean([r['sign_flip'] for r in abl_rows]):.3f}")

    return pd.DataFrame(rows)


# ─── Part B: Group steering ────────────────────────────────────────────────────

def run_group_steering(model, transcoder_set, device, clusters, prompts,
                       n_prompts=80, rng_seed=42, coeff=STEERING_COEFF):
    rng  = np.random.default_rng(rng_seed + 1)
    idxs = rng.choice(N_ST, min(n_prompts, N_ST), replace=False)
    rows = []

    for cluster_id, feat_list in sorted(clusters.items()):
        by_layer = {}
        for (layer, fidx) in feat_list:
            by_layer.setdefault(layer, []).append(fidx)

        for i in idxs:
            prompt = prompts[i]
            inputs = model.tokenize([prompt["prompt"]])
            inputs = {k: v.to(device) for k, v in inputs.items()}

            nd_base = compute_logit_diff(model, inputs, device,
                                          prompt["correct_answer"], prompt["incorrect_answer"])
            nd_steered = nd_base

            try:
                from contextlib import ExitStack
                steered_inputs = {}
                for layer_idx, feat_indices in sorted(by_layer.items()):
                    tc      = transcoder_set[layer_idx]
                    mlp_act = get_mlp_input_activation(model, inputs, layer_idx)
                    feats   = tc.encode(mlp_act.to(tc.dtype))
                    feats[:, feat_indices] += coeff
                    steered_inputs[layer_idx] = tc.decode(feats).to(mlp_act.dtype)

                with ExitStack() as stack:
                    for layer_idx, new_mlp in sorted(steered_inputs.items()):
                        stack.enter_context(
                            patch_mlp_input(model.model, layer_idx, -1, new_mlp)
                        )
                    with torch.no_grad():
                        out_s = model.model(**inputs, use_cache=False)
                log_p_s    = torch.log_softmax(out_s.logits[0, -1, :], dim=0)
                cid        = ensure_single_token(model, prompt["correct_answer"])
                iid        = ensure_single_token(model, prompt["incorrect_answer"])
                nd_steered = (log_p_s[cid] - log_p_s[iid]).item()

            except Exception as exc:
                print(f"    [WARN] Steering failed prompt {i} cluster {cluster_id}: {exc}")

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


# ─── Part C: Causal selectivity ───────────────────────────────────────────────

def compute_selectivity(ablation_df):
    """
    selectivity(cluster, particle) = mean_ΔND(target) − mean_ΔND(other particles)
    Negative = ablation hurts this particle more than others (particle-selective).
    """
    rows = []
    for cluster in ablation_df["cluster"].unique():
        sub = ablation_df[ablation_df["cluster"] == cluster]
        for particle in PARTICLES:
            t_mask = sub["correct_answer"] == particle
            o_mask = sub["correct_answer"] != particle
            if t_mask.sum() < 3:
                continue
            target_delta = sub[t_mask]["delta_nd"].mean()
            other_delta  = sub[o_mask]["delta_nd"].mean() if o_mask.sum() >= 3 else float("nan")
            selectivity  = float(target_delta - other_delta) if not np.isnan(other_delta) else float("nan")
            rows.append({
                "cluster":              int(cluster),
                "particle":             particle,
                "mean_delta_nd_target": float(target_delta),
                "mean_delta_nd_other":  float(other_delta),
                "selectivity":          selectivity,
                "sign_flip_rate":       float(sub[t_mask]["sign_flip"].mean()),
                "n_target_prompts":     int(t_mask.sum()),
            })
    return pd.DataFrame(rows)


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(ablation_df, selectivity_df, steering_df, output_dir):
    lines = [
        "# Cluster Intervention Analysis",
        f"## {BEHAVIOUR} | {SPLIT}",
        "",
        "## Part A: Group Ablation",
        "",
    ]

    if ablation_df is not None and len(ablation_df):
        agg = ablation_df.groupby("cluster").agg(
            mean_delta_nd  =("delta_nd",     "mean"),
            mean_abs_delta =("abs_delta_nd", "mean"),
            sign_flip_rate =("sign_flip",    "mean"),
            n              =("prompt_idx",   "count"),
        ).reset_index()
        lines += [
            "| Cluster | mean ΔND | mean |ΔND| | sign_flip% | n |",
            "|---|---|---|---|---|",
        ]
        for _, r in agg.sort_values("mean_abs_delta", ascending=False).iterrows():
            lines.append(
                f"| C{int(r['cluster'])} | {r['mean_delta_nd']:+.3f} | "
                f"{r['mean_abs_delta']:.3f} | {r['sign_flip_rate']:.1%} | {int(r['n'])} |"
            )

    if selectivity_df is not None and len(selectivity_df):
        lines += [
            "",
            "## Part C: Causal Selectivity",
            "",
            "Negative selectivity = ablation hurts this particle more than others.",
            "",
            "| Cluster | Particle | ΔND_target | ΔND_other | Selectivity | sign_flip% |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in selectivity_df.sort_values("selectivity").iterrows():
            flag = " ◀" if not np.isnan(r["selectivity"]) and r["selectivity"] < -0.15 else ""
            lines.append(
                f"| C{int(r['cluster'])} | {r['particle']} | "
                f"{r['mean_delta_nd_target']:+.3f} | "
                f"{r['mean_delta_nd_other']:+.3f} | "
                f"**{r['selectivity']:+.3f}**{flag} | "
                f"{r['sign_flip_rate']:.1%} |"
            )

    (output_dir / "report_cluster_intervention.md").write_text("\n".join(lines))
    print(f"  Saved: report_cluster_intervention.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",      default=BEHAVIOUR)
    ap.add_argument("--split",          default=SPLIT)
    ap.add_argument("--k",              type=int, default=6)
    ap.add_argument("--method",         default="kmeans")
    ap.add_argument("--n_prompts",      type=int, default=80)
    ap.add_argument("--steering_coeff", type=float, default=STEERING_COEFF)
    ap.add_argument("--skip_steering",  action="store_true")
    ap.add_argument("--device",         default="cuda")
    args = ap.parse_args()

    paths = get_paths(args.behaviour, args.split, args.k, args.method)
    out   = paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    prompts  = load_prompts(paths)
    clusters = load_clusters(paths, args.k, args.method)

    print(f"Clusters loaded: {len(clusters)}")
    for c, feats in sorted(clusters.items()):
        layers = sorted(set(l for l, _ in feats))
        print(f"  C{c}: {len(feats)} features, layers {layers}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    model = ModelWrapper(
        model_name=MODEL_NAME,
        device=str(device),
        dtype=torch.bfloat16,
    )
    model.model.eval()

    # Collect all layers used across clusters
    all_layers = sorted(set(l for feats in clusters.values() for l, _ in feats))
    print(f"Loading transcoders for layers {all_layers}...")
    transcoder_set = load_transcoder_set(
        model_size=MODEL_SIZE,
        device=device,
        dtype=torch.bfloat16,
        lazy_load=True,
        layers=all_layers,
    )
    print("Ready.\n")

    # Part A
    print("── Part A: Group ablation ──")
    ablation_df = run_group_ablation(
        model, transcoder_set, device, clusters, prompts,
        n_prompts=args.n_prompts,
    )
    ablation_df.to_csv(out / f"cluster_ablation_k{args.k}_{args.method}.csv", index=False)
    print(f"  Saved {len(ablation_df)} rows → cluster_ablation_k{args.k}_{args.method}.csv")

    # Part C
    print("\n── Part C: Causal selectivity ──")
    sel_df = compute_selectivity(ablation_df)
    sel_df.to_csv(out / f"cluster_selectivity_k{args.k}_{args.method}.csv", index=False)
    print(sel_df[["cluster", "particle", "selectivity", "sign_flip_rate"]].to_string(index=False))

    # Part B
    steering_df = None
    if not args.skip_steering:
        print("\n── Part B: Group steering ──")
        steering_df = run_group_steering(
            model, transcoder_set, device, clusters, prompts,
            n_prompts=args.n_prompts, coeff=args.steering_coeff,
        )
        steering_df.to_csv(out / f"cluster_steering_k{args.k}_{args.method}.csv", index=False)
        print(f"  Saved {len(steering_df)} rows → cluster_steering_k{args.k}_{args.method}.csv")

    write_report(ablation_df, sel_df, steering_df, out)

    print(f"\nAll outputs in: {out}")
    print("\n  Rsync to local Mac:")
    BASE = "iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
    ADIR = f"data/results/internal_candidate_analysis/{args.behaviour}"
    print(f"  rsync -av \"{BASE}/{ADIR}/\" /path/to/local/{ADIR}/")


if __name__ == "__main__":
    main()
