"""
Token-logit competitor promotion analysis.

For each prompt × cluster ablation, records the actual logits/logprobs for
candidate tokens {electron, proton, neutron, photon} before and after ablation.
Directly identifies which competitor rises when a cluster is ablated — replacing
the heuristic pool-ordering proxy used in script 47.

Requires GPU — run via: sbatch jobs/run_token_logit_promotion.sbatch

Usage:
  python scripts/50_token_logit_competitor_promotion.py
  python scripts/50_token_logit_competitor_promotion.py --n_prompts 60 --k 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import ModelWrapper
from src.transcoder import load_transcoder_set

BEHAVIOUR   = "physics_internal_candidate_selection_v2"
SPLIT       = "train"
MODEL_NAME  = "Qwen/Qwen3-4B"
MODEL_SIZE  = "4b"
PARTICLES   = ["electron", "proton", "neutron", "photon"]
CAND_TOKENS = [" electron", " proton", " neutron", " photon"]  # with leading space
N_ST        = 447

ADIR  = Path("data/results/internal_candidate_analysis") / BEHAVIOUR
PATHS = {
    "prompts":    Path("data/prompts") / f"{BEHAVIOUR}_{SPLIT}.jsonl",
    "ablation":   ADIR / "cluster_ablation_k6_kmeans.csv",
    "clusters":   ADIR / "feature_clusters_k6_kmeans.csv",
    "output":     ADIR,
}


# ─── Model utils ─────────────────────────────────────────────────────────────

def _import_script07():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "script07", Path(__file__).parent / "07_run_interventions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.patch_mlp_input, mod.get_mlp_input_activation


try:
    patch_mlp_input, get_mlp_input_activation = _import_script07()
except Exception:
    from contextlib import contextmanager

    def get_mlp_input_activation(model, inputs, layer_idx, token_pos=-1):
        captured = {}
        block = model.model.model.layers[layer_idx]
        norm  = block.post_attention_layernorm
        def hook(m, inp, out):
            captured["x"] = (out[0] if isinstance(out, tuple) else out).detach()
        h = norm.register_forward_hook(hook)
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
        h.remove()
        return captured["x"][:, token_pos, :]

    @contextmanager
    def patch_mlp_input(model_inner, layer_idx, token_pos, new_mlp_input):
        block = model_inner.model.layers[layer_idx]
        norm  = block.post_attention_layernorm
        def hook(m, inp, out):
            if isinstance(out, tuple):
                lst = list(out); lst[0][:, token_pos] = new_mlp_input; return tuple(lst)
            out[:, token_pos] = new_mlp_input; return out
        h = norm.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_prompts():
    with open(PATHS["prompts"]) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["_correct"] = r["correct_answer"].strip()
        r["_pool"]    = set(r["implicit_candidate_pool"])
    return [r for r in rows if not r.get("multi_token_answer", False)]


def load_clusters(k, method="kmeans"):
    df  = pd.read_csv(PATHS["clusters"])
    col = f"cluster_k{k}_{method}"
    clusters = {}
    for _, row in df.iterrows():
        c = int(row[col])
        clusters.setdefault(c, []).append((int(row["layer"]), int(row["feature_idx"])))
    return clusters


def get_candidate_token_ids(model):
    """Get token IDs for candidate tokens with leading space."""
    ids = {}
    for particle, token in zip(PARTICLES, CAND_TOKENS):
        tok_ids = model.tokenizer.encode(token, add_special_tokens=False)
        assert len(tok_ids) == 1, f"Token '{token}' is multi-token: {tok_ids}"
        ids[particle] = tok_ids[0]
    return ids


# ─── Single-pass: capture activations at all cluster layers ──────────────────

def capture_all_layers(model, inputs, layers):
    """Single forward pass capturing MLP inputs at all specified layers."""
    captured = {}
    handles  = []
    for layer_idx in layers:
        block = model.model.model.layers[layer_idx]
        norm  = block.post_attention_layernorm
        def hook(m, inp, out, _l=layer_idx):
            x = out[0] if isinstance(out, tuple) else out
            captured[_l] = x[:, -1, :].detach()
        handles.append(norm.register_forward_hook(hook))
    with torch.no_grad():
        out = model.model(**inputs, use_cache=False)
    for h in handles:
        h.remove()
    return captured, out.logits[0, -1, :]  # also return logits


# ─── Ablation with full logit capture ────────────────────────────────────────

@torch.no_grad()
def run_ablation_with_logits(
    model, transcoder_set, inputs, token_ids,
    by_layer, device
):
    """
    Ablates cluster features and returns logits for candidate tokens.
    Returns (logits_baseline [n_cand], logits_ablated [n_cand]).
    """
    from contextlib import ExitStack

    # Step 1: capture baseline activations at all cluster layers + baseline logits
    all_layers = sorted(by_layer.keys())
    baseline_acts, baseline_logits = capture_all_layers(model, inputs, all_layers)

    # Step 2: compute ablated MLP inputs
    ablated_inputs = {}
    for layer_idx, feat_indices in by_layer.items():
        tc   = transcoder_set[layer_idx]
        act  = baseline_acts[layer_idx]
        feats = tc.encode(act.to(tc.dtype))
        feats[:, feat_indices] = 0.0
        ablated_inputs[layer_idx] = tc.decode(feats).to(act.dtype)

    # Step 3: ablated forward pass
    with ExitStack() as stack:
        for layer_idx, new_mlp in sorted(ablated_inputs.items()):
            stack.enter_context(
                patch_mlp_input(model.model, layer_idx, -1, new_mlp)
            )
        ablated_out = model.model(**inputs, use_cache=False)
    ablated_logits = ablated_out.logits[0, -1, :]

    # Extract candidate token logits
    def extract_cand(logits):
        log_p = torch.log_softmax(logits, dim=0)
        return {p: float(log_p[tid]) for p, tid in token_ids.items()}

    return extract_cand(baseline_logits), extract_cand(ablated_logits)


# ─── Main analysis loop ───────────────────────────────────────────────────────

def run_promotion_analysis(model, transcoder_set, device, clusters, prompts,
                           token_ids, n_prompts, k, rng_seed=42):
    rng  = np.random.default_rng(rng_seed)
    idxs = rng.choice(N_ST, min(n_prompts, N_ST), replace=False)

    # Also include top neutron prompts from existing ablation (if available)
    if PATHS["ablation"].exists():
        abl = pd.read_csv(PATHS["ablation"])
        top_neutron = abl[abl["correct_answer"] == "neutron"].sort_values("delta_nd").head(20)
        extra_idxs  = top_neutron["prompt_idx"].unique()
        idxs        = np.unique(np.concatenate([idxs, extra_idxs]))[:n_prompts]

    rows = []
    for cluster_id, feat_list in sorted(clusters.items()):
        by_layer = {}
        for (layer, fidx) in feat_list:
            by_layer.setdefault(layer, []).append(fidx)

        print(f"  C{cluster_id}: {len(feat_list)} feats")

        for i in idxs:
            prompt = prompts[i]
            inputs = model.tokenize([prompt["prompt"]])
            inputs = {kk: v.to(device) for kk, v in inputs.items()}

            try:
                logp_base, logp_abl = run_ablation_with_logits(
                    model, transcoder_set, inputs, token_ids, by_layer, device
                )
            except Exception as exc:
                print(f"    [WARN] prompt {i} C{cluster_id}: {exc}")
                continue

            correct = prompt["_correct"]
            baseline_nd = logp_base[correct] - max(
                logp_base[p] for p in PARTICLES if p != correct
            )
            ablated_nd = logp_abl[correct] - max(
                logp_abl[p] for p in PARTICLES if p != correct
            )
            delta_nd   = ablated_nd - baseline_nd
            sign_flip  = bool(baseline_nd > 0 and ablated_nd <= 0)

            # Which competitor gained most (positive delta = increased logp)?
            delta_by_particle = {p: logp_abl[p] - logp_base[p] for p in PARTICLES}
            competitors = [p for p in PARTICLES if p != correct]
            promoted    = max(competitors, key=lambda p: delta_by_particle[p])

            # New top candidate
            new_top = max(PARTICLES, key=lambda p: logp_abl[p])

            row = {
                "cluster":          cluster_id,
                "prompt_idx":       int(i),
                "correct_answer":   correct,
                "wording_family":   prompt.get("wording_family", ""),
                "filter_property":  prompt.get("filter_property", ""),
                "baseline_nd":      float(baseline_nd),
                "ablated_nd":       float(ablated_nd),
                "delta_nd":         float(delta_nd),
                "sign_flip":        sign_flip,
                "promoted_competitor": promoted,
                "new_top_candidate":   new_top,
                "correct_changed":  new_top != correct,
            }
            for p in PARTICLES:
                row[f"logp_base_{p}"]  = float(logp_base[p])
                row[f"logp_abl_{p}"]   = float(logp_abl[p])
                row[f"delta_logp_{p}"] = float(delta_by_particle[p])

            rows.append(row)

    return pd.DataFrame(rows)


# ─── Promotion matrix ─────────────────────────────────────────────────────────

def build_promotion_matrix(df):
    """Count promotions: correct → promoted competitor after sign flip."""
    sign_flips = df[df["sign_flip"] == True]
    from collections import defaultdict
    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in sign_flips.iterrows():
        matrix[row["correct_answer"]][row["promoted_competitor"]] += 1

    mat_rows = []
    for correct in PARTICLES:
        for promoted in PARTICLES:
            if correct == promoted:
                continue
            mat_rows.append({
                "correct":      correct,
                "promoted":     promoted,
                "n_promotions": matrix[correct][promoted],
            })
    return pd.DataFrame(mat_rows)


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(df, matrix_df, k):
    lines = [
        "# Token-Logit Competitor Promotion Report",
        f"## {BEHAVIOUR} | k={k}",
        "",
        "## Method",
        "For each (prompt, cluster) pair: run baseline forward pass and cluster-ablated forward pass.",
        "Record actual log-probabilities for candidate tokens {electron, proton, neutron, photon}.",
        "Promoted competitor = particle with largest positive Δlogp among non-correct particles.",
        "",
        "## Sign-Flip Promotion Matrix",
        "",
        "| Correct → Promoted | n promotions |",
        "|---|---|",
    ]
    for _, r in matrix_df[matrix_df["n_promotions"] > 0].sort_values(
        "n_promotions", ascending=False
    ).iterrows():
        lines.append(f"| {r['correct']} → {r['promoted']} | {int(r['n_promotions'])} |")

    lines += [
        "",
        "## Mean Δlogp per candidate after cluster ablation",
        "",
        "| Cluster | Correct | Δlogp_correct | Δlogp_electron | Δlogp_proton | Δlogp_neutron | Δlogp_photon |",
        "|---|---|---|---|---|---|---|",
    ]
    for (cluster, correct), grp in df.groupby(["cluster", "correct_answer"]):
        d_e  = grp["delta_logp_electron"].mean()
        d_p  = grp["delta_logp_proton"].mean()
        d_n  = grp["delta_logp_neutron"].mean()
        d_ph = grp["delta_logp_photon"].mean()
        d_correct = grp[f"delta_logp_{correct}"].mean()
        lines.append(
            f"| C{cluster} | {correct} | {d_correct:+.3f} | {d_e:+.3f} | "
            f"{d_p:+.3f} | {d_n:+.3f} | {d_ph:+.3f} |"
        )

    (PATHS["output"] / "token_logit_promotion_report.md").write_text("\n".join(lines))
    print("  Report: token_logit_promotion_report.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour",  default=BEHAVIOUR)
    ap.add_argument("--split",      default=SPLIT)
    ap.add_argument("--k",          type=int, default=6)
    ap.add_argument("--n_prompts",  type=int, default=80)
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prompts  = load_prompts()
    clusters = load_clusters(args.k)

    print("Loading model...")
    model = ModelWrapper(model_name=MODEL_NAME, device=str(device), dtype=torch.bfloat16)
    model.model.eval()

    token_ids = get_candidate_token_ids(model)
    print(f"Candidate token IDs: {token_ids}")

    all_layers = sorted(set(l for feats in clusters.values() for l, _ in feats))
    print(f"Loading transcoders for {len(all_layers)} layers...")
    transcoder_set = load_transcoder_set(
        model_size=MODEL_SIZE, device=device, dtype=torch.bfloat16,
        lazy_load=True, layers=all_layers,
    )

    print("\n── Promotion analysis ──")
    df = run_promotion_analysis(
        model, transcoder_set, device, clusters, prompts,
        token_ids, args.n_prompts, args.k
    )

    out = PATHS["output"]
    df.to_csv(out / f"token_logit_promotion_by_prompt_k{args.k}.csv", index=False)
    matrix_df = build_promotion_matrix(df)
    matrix_df.to_csv(out / f"token_logit_promotion_matrix_k{args.k}.csv", index=False)

    summary = {
        "n_pairs":          len(df),
        "sign_flip_rate":   float(df["sign_flip"].mean()),
        "n_sign_flips":     int(df["sign_flip"].sum()),
        "top_promoted":     matrix_df.sort_values("n_promotions", ascending=False).iloc[0].to_dict()
                            if len(matrix_df) else {},
    }
    with open(out / f"token_logit_promotion_summary_k{args.k}.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_report(df, matrix_df, args.k)

    print(f"\nAll outputs in: {out}")
    print("\n=== PROMOTION MATRIX ===")
    print(matrix_df[matrix_df["n_promotions"] > 0].sort_values("n_promotions", ascending=False).to_string(index=False))

    print("\n  Rsync to local Mac:")
    BASE = "iv294@login.hpc.cam.ac.uk:/rds/user/iv294/hpc-work/thesis/project"
    print(f"  git add -f {out}/token_logit_promotion_*")


if __name__ == "__main__":
    main()
