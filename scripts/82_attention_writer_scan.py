"""
82_attention_writer_scan.py   [WHERE is the concept WRITTEN? — all 36 layers, top-50]
=========================================================================================
Exp 80 scanned only L14/18/21/24 (probes) and found L21 heads write the readable axis.
That is not enough to claim "this is WHERE the concept is written": we must scan ALL
layers and take a wider top-N. This script does the exhaustive writer-localisation.

It also fixes a subtlety exp 80 glossed over: the reading axis w_res ROTATES with depth
(adjacent-layer causal cos 0.61 -> 0.09). So "alignment with w_res" is measured against a
DIFFERENT axis at each layer, which makes cross-layer |d| comparison unfair. We therefore
score every head by TWO metrics in one pass:

  (A) d_along_wres : |Cohen's d| of (head contribution . w_res_of_that_layer) separating
                     classes -- alignment with the per-layer reading axis (comparable to 80).
  (B) intrinsic_auc: held-out AUC of the head's OWN diff-of-means direction on its
                     contribution vector -- "does this head's output carry alpha/beta at
                     all", AXIS-INDEPENDENT and therefore honestly comparable across layers.

Pipeline:
  1. One forward per prompt; capture residual at every layer (for per-layer w_res) AND the
     o_proj input (concatenated head outputs) at every layer, at the answer position.
  2. Per head (all n_layers x n_heads of them), compute metrics (A) and (B). Save the full
     table + per-layer aggregates (max |d|, #heads above threshold) -> shows WHERE writer
     heads concentrate, across ALL layers, robust to axis rotation.
  3. INTERVENTION on the top-N (default 50) heads by the chosen ranking: ablation (factor 0,
     necessity) and negation (factor -1), vs a random-head null, reporting BOTH metrics:
       margin-flip  (sign of logit_beta - logit_alpha; relative, NOT the answer)
       intact-flip  (top-1 token actually became the toward-class answer; BEHAVIOURAL)
  4. CUMULATIVE ablation: remove top-1, top-2, ... top-k heads and watch whether the answer
     ever changes (intact-flip) -- a direct test of how many heads carry NECESSARY signal.

Honest expectation (from exp 80 + j75): margin-flip may be large under negation, but
behavioural intact-flip ~0 -> heads WRITE a readable axis but are not a behavioural lever.
A clean positive (intact-flip rising with top-N or cumulative k) would overturn that and
would be reported as such.

SELF-TEST (no torch / no repo):  python 82_attention_writer_scan.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("writer_scan")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def auc_scalar(s, y):
    s = np.asarray(s, float)
    o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def cohens_d(s, y):
    a, b = s[y == 0], s[y == 1]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) + 1e-12
    return float((b.mean() - a.mean()) / pooled)


def intrinsic_auc(V, y, tr):
    """Held-out AUC of the head's OWN diff-of-means direction (axis-independent)."""
    mtr = V[tr & (y == 1)].mean(0) - V[tr & (y == 0)].mean(0)
    s = V @ mtr
    return auc_scalar(s[~tr], y[~tr])


def is_flip_margin(base, after, toward):
    return int(base < 0 and after > 0) if toward == "beta" else int(base > 0 and after < 0)


def flip_rate(base, after, toward):
    return float(np.mean([is_flip_margin(b, a, t) for b, a, t in zip(base, after, toward)])) if len(base) else float("nan")


def percentile_of(value, null):
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


def per_layer_aggregate(scores_by_layer: Dict[int, np.ndarray], thresh: float):
    """scores_by_layer[L] = array over heads. Returns per-layer (max, mean, n_above)."""
    return {L: {"max": float(s.max()), "mean": float(s.mean()), "n_above": int((s >= thresh).sum())}
            for L, s in scores_by_layer.items()}


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n_heads, n_layers, n = 12, 4, 5, 80
    y = np.array([0, 1] * (n // 2))
    tr = np.zeros(n, bool); tr[: int(0.6 * n)] = True; rng.shuffle(tr)
    w_true = unit_raw(rng.standard_normal(d))
    # only layer 3 has writer heads (0 and 2 carry the concept); others noise
    contrib = {}
    for L in range(n_layers):
        V = 0.3 * rng.standard_normal((n, n_heads, d))
        if L == 3:
            for p in range(n):
                sgn = 1.0 if y[p] == 1 else -1.0
                V[p, 0] += sgn * 2.0 * w_true; V[p, 2] += sgn * 1.6 * w_true
        contrib[L] = V

    # metric B (intrinsic) should flag layer-3 heads 0,2 and rank them top globally
    rows = []
    for L in range(n_layers):
        for h in range(n_heads):
            rows.append((L, h, intrinsic_auc(contrib[L][:, h, :], y, tr)))
    rows.sort(key=lambda t: -t[2])
    top2 = {(r[0], r[1]) for r in rows[:2]}
    assert top2 == {(3, 0), (3, 2)}, f"intrinsic should rank L3 writers top, got {top2}"

    # metric A (along axis) with w_res ~ w_true on layer 3
    projL3 = contrib[3] @ w_true  # (n, n_heads)
    dvals = np.array([abs(cohens_d(projL3[:, h], y)) for h in range(n_heads)])
    assert set(np.argsort(dvals)[-2:]) == {0, 2}, "along-axis |d| should rank heads 0,2 top on L3"

    # per-layer aggregate: layer 3 has the writers
    agg = per_layer_aggregate({L: np.array([intrinsic_auc(contrib[L][:, h, :], y, tr) for h in range(n_heads)])
                               for L in range(n_layers)}, thresh=0.8)
    assert agg[3]["n_above"] >= 2 and agg[3]["max"] >= 0.9, "layer 3 should concentrate writer heads"
    assert all(agg[L]["n_above"] == 0 for L in range(n_layers) if L != 3), "noise layers have no writers"

    # cumulative monotonic intuition + flip/percentile helpers
    assert flip_rate(np.array([-1.0]), np.array([1.0]), np.array(["beta"])) == 1.0
    assert percentile_of(0.9, np.array([0.1, 0.5])) == 100.0
    print("[self_test] OK — intrinsic ranking, along-axis |d|, per-layer aggregate, helpers pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for a in p.split("."):
        o = getattr(o, a)
    return o


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks)
    norm = _chain(model, "model.norm")
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]

    layers = args.layers if args.layers else list(range(n_layers))
    layers = [L for L in layers if 0 <= L < n_layers]
    o_proj = {L: _chain(blocks[L], "self_attn.o_proj") for L in layers}
    W_O = {L: o_proj[L].weight.detach().float().cpu().numpy() for L in layers}
    for L in layers:
        assert W_O[L].shape[1] == n_heads * head_dim, f"L{L}: o_proj in {W_O[L].shape[1]} != {n_heads*head_dim}"
    logger.info("model: %d layers, %d heads, head_dim %d; scanning %d layers", n_layers, n_heads, head_dim, len(layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    # ---------- capture residual + o_proj input at ALL layers (answer pos), one forward/prompt ----------
    logger.info("Capturing residual + per-head outputs at %d layers for %d prompts...", len(layers), len(prompts))
    res = {L: np.zeros((len(prompts), cfg.hidden_size), np.float32) for L in layers}
    zin = {L: np.zeros((len(prompts), n_heads * head_dim), np.float32) for L in layers}
    y = np.zeros(len(prompts), int); tr_mask = np.zeros(len(prompts), bool); clean_margin = np.zeros(len(prompts))
    last = n_layers - 1
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in layers:
            tgt = blocks[L + 1] if L < last else norm
            def mkres(L=L):
                def pre(m, a): g[f"r{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(tgt.register_forward_pre_hook(mkres(), with_kwargs=False))
            def mkz(L=L):
                def pre(m, a): g[f"z{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(o_proj[L].register_forward_pre_hook(mkz(), with_kwargs=False))
        try:
            with torch.no_grad():
                o = model(**inp, use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1, :].float(), 0)
                clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            res[L][i] = g[f"r{L}"]; zin[L][i] = g[f"z{L}"]
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        tr_mask[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    # ---------- score every head: (A) along per-layer w_res, (B) intrinsic held-out AUC ----------
    logger.info("Scoring %d heads (%d layers x %d) by two metrics...", len(layers) * n_heads, len(layers), n_heads)
    rows = []
    d_along = {}; auc_intr = {}
    for L in layers:
        wL = fisher_axis(res[L][tr_mask].astype(np.float64), y[tr_mask], args.shrink)
        layer_auc = auc_scalar((res[L][~tr_mask].astype(np.float64)) @ wL, y[~tr_mask])
        Z = zin[L].astype(np.float64).reshape(len(prompts), n_heads, head_dim)
        Wt = W_O[L].T.reshape(n_heads, head_dim, -1)
        proj = np.einsum("phk,hkd,d->ph", Z, Wt, wL)                 # (n, H) along-axis projection
        dA = np.array([abs(cohens_d(proj[:, h], y)) for h in range(n_heads)])
        V = np.einsum("phk,hkd->phd", Z, Wt)                          # (n, H, d) full head contributions
        dB = np.array([intrinsic_auc(V[:, h, :], y, tr_mask) for h in range(n_heads)])
        d_along[L] = dA; auc_intr[L] = dB
        for h in range(n_heads):
            rows.append({"layer": int(L), "head": int(h), "d_along_wres": float(dA[h]),
                         "intrinsic_auc": float(dB[h]), "layer_wres_auc": float(layer_auc)})
        if (L % 4 == 0) or (L == layers[-1]):
            logger.info("  L%d: layer-AUC=%.3f  max d_along=%.2f  max intrinsic_auc=%.3f",
                        L, layer_auc, dA.max(), dB.max())

    # save full table + per-layer aggregates
    import csv as _csv
    with open(out / "head_writer_scores.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer", "head", "d_along_wres", "intrinsic_auc", "layer_wres_auc"])
        w.writeheader(); [w.writerow(r) for r in rows]
    agg_along = per_layer_aggregate(d_along, args.d_thresh)
    agg_intr = per_layer_aggregate(auc_intr, args.auc_thresh)
    json.dump({"d_along": agg_along, "intrinsic_auc": agg_intr},
              open(out / "per_layer_writer_aggregate.json", "w"), indent=2)

    # ranking
    key = "d_along_wres" if args.rank_metric == "along_wres" else "intrinsic_auc"
    rows_sorted = sorted(rows, key=lambda r: -r[key])
    top_heads = [(r["layer"], r["head"]) for r in rows_sorted[: args.top_k_heads]]
    logger.info("WHERE writers concentrate (top-%d by %s):", args.top_k_heads, key)
    from collections import Counter
    layer_counts = Counter(L for L, _ in top_heads)
    logger.info("  layer histogram of top-%d: %s", args.top_k_heads,
                ", ".join(f"L{L}:{c}" for L, c in sorted(layer_counts.items(), key=lambda t: -t[1])))
    logger.info("  top-15: %s", ", ".join(f"L{r['layer']}.H{r['head']}({r[key]:.2f})" for r in rows_sorted[:15]))

    # ---------- targets ----------
    held = [i for i in range(len(prompts)) if not tr_mask[i]]
    held_a = [i for i in held if y[i] == 0][: args.max_targets] if args.max_targets else [i for i in held if y[i] == 0]
    held_b = [i for i in held if y[i] == 1][: args.max_targets] if args.max_targets else [i for i in held if y[i] == 1]
    targets = [(i, "beta") for i in held_a] + [(i, "alpha") for i in held_b]

    def run_scaling(ptext, head_set, factor):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        by_layer: Dict[int, List[int]] = {}
        for (L, h) in head_set:
            by_layer.setdefault(L, []).append(h)
        handles = []
        for L, hs in by_layer.items():
            def mk(hs=hs, L=L):
                def pre(m, a):
                    x = a[0].clone()
                    for h in hs:
                        x[0, -1, h * head_dim:(h + 1) * head_dim] *= factor
                    return (x,)
                return pre
            handles.append(o_proj[L].register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0)
                return float(lp[beta_id] - lp[alpha_id]), int(row.argmax().item())
        finally:
            for h in handles:
                h.remove()

    mc = np.array([clean_margin[i] for i, _ in targets])
    tw = np.array([t for _, t in targets])
    toward_tok = np.array([beta_id if t == "beta" else alpha_id for t in tw])
    flat = [(L, h) for L in layers for h in range(n_heads)]
    results = {"n_layers": n_layers, "n_heads": n_heads, "top_heads": [[int(L), int(h)] for L, h in top_heads],
               "layer_histogram_top": {str(L): int(c) for L, c in layer_counts.items()},
               "per_layer_aggregate_along": agg_along, "per_layer_aggregate_intrinsic": agg_intr}

    # ---------- (A) intervention on top-N vs random-head null, BOTH metrics ----------
    logger.info("(A) intervention on top-%d heads vs random-head null (margin + intact)...", args.top_k_heads)
    head_rows = []
    for factor in args.head_factors:
        rtop = [run_scaling(prompts[i]["prompt"], top_heads, factor) for i, _ in targets]
        m_top = np.array([r[0] for r in rtop]); t1 = np.array([r[1] for r in rtop])
        fr = flip_rate(mc, m_top, tw); intact = float(np.mean(t1 == toward_tok))
        intact_ab = float(np.mean([(t in (alpha_id, beta_id)) for t in t1]))
        nf, ni = [], []
        for _ in range(args.n_random_head):
            rset = [flat[j] for j in rng.choice(len(flat), size=len(top_heads), replace=False)]
            rr = [run_scaling(prompts[i]["prompt"], rset, factor) for i, _ in targets]
            nf.append(flip_rate(mc, np.array([r[0] for r in rr]), tw))
            ni.append(float(np.mean(np.array([r[1] for r in rr]) == toward_tok)))
        nf, ni = np.array(nf), np.array(ni)
        row = {"factor": float(factor), "top_margin_flip": fr, "top_intact_flip": intact, "top_intact_ab_rate": intact_ab,
               "rand_margin_p95": float(np.percentile(nf, 95)), "rand_intact_p95": float(np.percentile(ni, 95)),
               "margin_pct_vs_null": percentile_of(fr, nf), "intact_pct_vs_null": percentile_of(intact, ni)}
        head_rows.append(row)
        logger.info("  factor=%+.1f: MARGIN top=%.3f (rand p95 %.3f) | INTACT top=%.3f (rand p95 %.3f) | top-1 a/b %.3f",
                    factor, fr, np.percentile(nf, 95), intact, np.percentile(ni, 95), intact_ab)
    results["intervention"] = head_rows

    # ---------- cumulative ablation: remove top-1..top-k, watch intact-flip ----------
    logger.info("Cumulative ablation (remove top-k, factor 0): does the answer ever change?")
    cum = []
    grid = sorted({k for k in args.cumulative_grid if k <= args.top_k_heads} | {args.top_k_heads})
    for k in grid:
        hs = top_heads[:k]
        rr = [run_scaling(prompts[i]["prompt"], hs, 0.0) for i, _ in targets]
        m = np.array([r[0] for r in rr]); t1 = np.array([r[1] for r in rr])
        cum.append({"k": int(k), "margin_flip": flip_rate(mc, m, tw),
                    "intact_flip": float(np.mean(t1 == toward_tok))})
        logger.info("  remove top-%d: margin-flip=%.3f  intact-flip=%.3f", k, cum[-1]["margin_flip"], cum[-1]["intact_flip"])
    results["cumulative_ablation"] = cum

    (out / "writer_scan.json").write_text(json.dumps(results, indent=2))

    # ---------- verdict ----------
    print("\n" + "=" * 90)
    print("ATTENTION WRITER SCAN (all layers) -- where is alpha/beta written, and is it a lever?")
    print("=" * 90)
    top_layers = ", ".join(f"L{L}({c})" for L, c in sorted(layer_counts.items(), key=lambda t: -t[1])[:5])
    print(f"WRITERS: top-{args.top_k_heads} concept-aligned heads concentrate on layers: {top_layers}")
    best_b = max(head_rows, key=lambda r: r["top_intact_flip"]) if head_rows else None
    best_m = max(head_rows, key=lambda r: r["top_margin_flip"]) if head_rows else None
    cum_max = max((c["intact_flip"] for c in cum), default=float("nan"))
    if best_b and best_b["top_intact_flip"] >= args.tau_flip and best_b["intact_pct_vs_null"] >= 95:
        print(f"BEHAVIOUR: OUTCOME 1 -- intervening on the top heads makes the model's TOP-1 the answer on "
              f"{best_b['top_intact_flip']:.2f} of targets (rand p95 {best_b['rand_intact_p95']:.2f}). Attention IS a lever.")
    else:
        bi = best_b["top_intact_flip"] if best_b else float("nan")
        bm = best_m["top_margin_flip"] if best_m else float("nan")
        print(f"BEHAVIOUR: NOT A LEVER -- best behavioural INTACT-flip = {bi:.2f} (margin-flip reaches {bm:.2f}), "
              f"and cumulative ablation up to top-{args.top_k_heads} never exceeds intact-flip {cum_max:.2f}. "
              f"Writer heads write a readable axis but do not behaviourally carry alpha/beta (H3-geometry + H2-behaviour).")
    print("=" * 90 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/writer_scan")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--rank_metric", choices=["along_wres", "intrinsic"], default="along_wres")
    p.add_argument("--top_k_heads", type=int, default=50)
    p.add_argument("--head_factors", type=float, nargs="*", default=[0.0, -1.0])
    p.add_argument("--n_random_head", type=int, default=30)
    p.add_argument("--cumulative_grid", type=int, nargs="*", default=[1, 2, 5, 10, 20, 30, 50])
    p.add_argument("--d_thresh", type=float, default=1.0, help="|d| threshold for per-layer writer count")
    p.add_argument("--auc_thresh", type=float, default=0.8, help="intrinsic-AUC threshold for per-layer writer count")
    p.add_argument("--max_targets", type=int, default=None)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--tau_flip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
