"""
83_wres_causality_all_positions.py   [is w_res causal at ANY layer / ANY position?]
=========================================================================================
Exp 80's CIE used only 4 layers (L14/18/21/24) and ONLY the answer position. That leaves
two coverage holes: (i) layers L0-L13, L25-L35 untested; (ii) the concept might be USED at
an intermediate token position, which an answer-position-only patch cannot see (2602.07794
patch across positions). This script closes both: it sweeps the w_res-component
intervention over ALL layers and ALL positions.

Three interventions per layer (w_res = the per-layer Fisher reading axis, unit):
  (1) DONOR patch @ answer position  -- broadens 80's CIE to every layer. Replace the
      w_res-component of the answer-position residual with an opposite-class donor's.
  (2) ABLATE @ all positions         -- remove the w_res-component at EVERY token:
      h_p <- h_p - <h_p, w> w  for all positions p. If the answer never changes, the
      concept direction carries no causal signal anywhere in the sequence.
  (3) NEGATE @ all positions         -- flip it: h_p <- h_p - 2<h_p, w> w.
Plus a POSITION-OFFSET sweep (ablate at offset -1,-2,...,-K per layer) to localise whether
any single position matters.

Two metrics, kept strictly separate (the lesson from exp 80):
  margin-flip = sign of logit_beta - logit_alpha changed (relative; NOT the answer).
  intact-flip = the model's TOP-1 token became the OPPOSITE-class answer token (behavioural).
Decisive metric = intact-flip. Every effect is compared to a RANDOM-DIRECTION null of the
same operation (ablate/negate/patch a random unit direction), to rule out "any large push
moves things" artifacts. A clean positive (intact-flip rises at some layer/position above
null) would overturn the section-6 negative and is reported as such.

Honest expectation (from 80 + j75): intact-flip ~0 across all layers and positions ->
direction-level interventions on w_res are not a behavioural lever anywhere, confirming the
negative on the FULL setup and contrasting with 2602.07794 (concept-forced, large effect).

SELF-TEST (no torch / no repo):  python 83_wres_causality_all_positions.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("wres_causal")


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
    s = np.asarray(s, float); o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def remove_component(H, w):
    """Remove the unit-w component from each row of H (numpy mirror of the ablation hook)."""
    w = unit_raw(w)
    return H - np.outer(H @ w, w)


def margin_flip(base, after, toward):
    out = []
    for b, a, t in zip(base, after, toward):
        out.append(int(b < 0 and a > 0) if t == "beta" else int(b > 0 and a < 0))
    return float(np.mean(out)) if len(out) else float("nan")


def percentile_of(value, null):
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    return float(100.0 * np.mean(null <= value)) if null.size else float("nan")


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 10, 60
    y = np.array([0, 1] * (n // 2))
    w = unit_raw(rng.standard_normal(d))
    H = 0.5 * rng.standard_normal((n, d)) + np.outer(np.where(y == 1, 1.5, -1.5), w)
    # ablation removes the w-component -> projection onto w becomes ~0
    Ha = remove_component(H, w)
    assert abs(float((Ha @ w).mean())) < 1e-9 and np.allclose(Ha @ w, 0, atol=1e-9), "ablation must zero the w-projection"
    # AUC along w collapses after ablation
    assert auc_scalar(H @ w, y) > 0.95 and abs(auc_scalar(Ha @ w, y) - 0.5) < 0.15, "AUC along w should collapse"
    # margin-flip / percentile helpers
    assert margin_flip(np.array([-1.0, 1.0]), np.array([1.0, -1.0]), np.array(["beta", "alpha"])) == 1.0
    assert percentile_of(0.9, np.array([0.1, 0.2])) == 100.0
    # fisher recovers planted direction
    assert abs(np.dot(fisher_axis(H, y), w)) > 0.8
    print("[self_test] OK — ablation removes w-component, AUC collapse, flip/percentile, fisher recovery pass.")


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
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm = _chain(model, "model.norm")
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    last = n_layers - 1
    layers = args.layers if args.layers else list(range(n_layers))
    layers = [L for L in layers if 0 <= L < n_layers]
    logger.info("model: %d layers; sweeping %d layers, all positions", n_layers, len(layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    def tap(L):
        return blocks[L + 1] if L < last else norm

    # ---------- capture answer-position residual per layer (for w_res + donor scalars) ----------
    logger.info("Capturing answer-position residual at %d layers for %d prompts...", len(layers), len(prompts))
    res = {L: np.zeros((len(prompts), model.config.hidden_size), np.float32) for L in layers}
    y = np.zeros(len(prompts), int); tr = np.zeros(len(prompts), bool); clean_margin = np.zeros(len(prompts))
    clean_top1 = np.zeros(len(prompts), int)
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a): g[L] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0); clean_margin[i] = float(lp[beta_id] - lp[alpha_id])
                clean_top1[i] = int(row.argmax().item())
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            res[L][i] = g[L]
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        tr[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    w_res = {}; donor = {}
    for L in layers:
        wL = fisher_axis(res[L][tr].astype(np.float64), y[tr], args.shrink)
        w_res[L] = wL
        proj = res[L].astype(np.float64) @ wL
        donor[L] = {"alpha": float(proj[y == 0].mean()), "beta": float(proj[y == 1].mean())}

    # ---------- targets ----------
    held = [i for i in range(len(prompts)) if not tr[i]]
    ha = [i for i in held if y[i] == 0]; hb = [i for i in held if y[i] == 1]
    if args.max_targets:
        ha, hb = ha[: args.max_targets], hb[: args.max_targets]
    targets = [(i, "beta") for i in ha] + [(i, "alpha") for i in hb]   # toward = OPPOSITE class
    mc = np.array([clean_margin[i] for i, _ in targets]); tw = np.array([t for _, t in targets])
    toward_tok = np.array([beta_id if t == "beta" else alpha_id for t in tw])

    def run(ptext, L, w_vec, mode, donor_scalar=None):
        """mode in {donor_answer, ablate_all, negate_all, ablate_at}. w_vec unit (np)."""
        inp = tok([ptext], return_tensors="pt").to(args.device)
        w = torch.tensor(unit_raw(w_vec), dtype=torch.float32, device=args.device)
        ds = float(donor_scalar) if donor_scalar is not None else 0.0
        off = args.ablate_offset
        def pre(m, a):
            x = a[0].clone()
            if mode == "donor_answer":
                cur = float(x[0, -1, :] @ w); x[0, -1, :] = x[0, -1, :] + (ds - cur) * w
            elif mode == "ablate_all":
                proj = x[0] @ w; x[0] = x[0] - torch.outer(proj, w)
            elif mode == "negate_all":
                proj = x[0] @ w; x[0] = x[0] - 2.0 * torch.outer(proj, w)
            elif mode == "ablate_at":
                if x.shape[1] >= abs(off):
                    cur = float(x[0, off, :] @ w); x[0, off, :] = x[0, off, :] - cur * w
            return (x,)
        h = tap(L).register_forward_pre_hook(pre, with_kwargs=False)
        try:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
                lp = torch.log_softmax(row, 0)
                return float(lp[beta_id] - lp[alpha_id]), int(row.argmax().item())
        finally:
            h.remove()

    def eval_mode(L, mode):
        """Returns (margin_flip, intact_flip, top1_ab_rate) for w_res, + null arrays."""
        rr = []
        for i, toward in targets:
            ds = donor[L][toward] if mode == "donor_answer" else None
            rr.append(run(prompts[i]["prompt"], L, w_res[L], mode, ds))
        m = np.array([r[0] for r in rr]); t1 = np.array([r[1] for r in rr])
        fr = margin_flip(mc, m, tw); intact = float(np.mean(t1 == toward_tok))
        ab = float(np.mean([(t in (alpha_id, beta_id)) for t in t1]))
        nf, ni = [], []
        for _ in range(args.n_random_dir):
            rv = unit_raw(rng.standard_normal(model.config.hidden_size))
            # for donor mode the random-dir donor scalar = opposite-class mean along rv
            projr = res[L].astype(np.float64) @ rv
            dmap = {"alpha": float(projr[y == 0].mean()), "beta": float(projr[y == 1].mean())}
            rrn = []
            for i, toward in targets:
                ds = dmap[toward] if mode == "donor_answer" else None
                rrn.append(run(prompts[i]["prompt"], L, rv, mode, ds))
            mn = np.array([r[0] for r in rrn]); t1n = np.array([r[1] for r in rrn])
            nf.append(margin_flip(mc, mn, tw)); ni.append(float(np.mean(t1n == toward_tok)))
        return fr, intact, ab, np.array(nf), np.array(ni)

    results = {"layers": layers, "modes": ["donor_answer", "ablate_all", "negate_all"]}
    per_layer = []
    clean_intact_ab = float(np.mean([(clean_top1[i] in (alpha_id, beta_id)) for i, _ in targets]))
    logger.info("clean top-1 is alpha/beta on %.3f of targets (format-quirk baseline)", clean_intact_ab)

    for mode in ["donor_answer", "ablate_all", "negate_all"]:
        logger.info("=== mode: %s — sweeping %d layers ===", mode, len(layers))
        for L in layers:
            fr, intact, ab, nf, ni = eval_mode(L, mode)
            rec = {"layer": int(L), "mode": mode, "margin_flip": fr, "intact_flip": intact, "top1_ab_rate": ab,
                   "rand_margin_p95": float(np.percentile(nf, 95)), "rand_intact_p95": float(np.percentile(ni, 95)),
                   "margin_pct_vs_null": percentile_of(fr, nf), "intact_pct_vs_null": percentile_of(intact, ni)}
            per_layer.append(rec)
            if intact > 0 or (L % 6 == 0) or (L == layers[-1]):
                logger.info("  L%d %s: MARGIN=%.3f (rand p95 %.3f) | INTACT=%.3f (rand p95 %.3f) | top-1 a/b %.3f",
                            L, mode, fr, np.percentile(nf, 95), intact, np.percentile(ni, 95), ab)
    results["per_layer"] = per_layer

    # ---------- position-offset sweep (ablate at offset, coarse layer grid) ----------
    if args.offset_layers:
        logger.info("=== position-offset ablation sweep ===")
        off_rows = []
        save = args.ablate_offset
        for L in [L for L in args.offset_layers if L in w_res]:
            for off in args.offsets:
                args.ablate_offset = off
                rr = [run(prompts[i]["prompt"], L, w_res[L], "ablate_at") for i, _ in targets]
                t1 = np.array([r[1] for r in rr]); m = np.array([r[0] for r in rr])
                off_rows.append({"layer": int(L), "offset": int(off),
                                 "margin_flip": margin_flip(mc, m, tw), "intact_flip": float(np.mean(t1 == toward_tok))})
                logger.info("  L%d offset %d: margin=%.3f intact=%.3f", L, off, off_rows[-1]["margin_flip"], off_rows[-1]["intact_flip"])
        args.ablate_offset = save
        results["offset_sweep"] = off_rows

    (out / "wres_causality_all_positions.json").write_text(json.dumps(results, indent=2))

    # ---------- verdict ----------
    print("\n" + "=" * 92)
    print("w_res CAUSALITY across ALL layers x ALL positions -- is the reading axis a lever anywhere?")
    print("=" * 92)
    max_intact = max((r["intact_flip"] for r in per_layer), default=float("nan"))
    max_margin = max((r["margin_flip"] for r in per_layer), default=float("nan"))
    sig = [r for r in per_layer if r["intact_flip"] >= args.tau_flip and r["intact_pct_vs_null"] >= 95]
    if sig:
        s = max(sig, key=lambda r: r["intact_flip"])
        print(f"OUTCOME 1 -- w_res IS causal: at L{s['layer']} ({s['mode']}) the answer flips behaviourally on "
              f"{s['intact_flip']:.2f} of targets (rand p95 {s['rand_intact_p95']:.2f}). Hole was real.")
    else:
        print(f"NOT A LEVER ANYWHERE -- across all {len(layers)} layers and all positions, behavioural "
              f"intact-flip never exceeds {max_intact:.2f} above null (margin-flip up to {max_margin:.2f}). "
              f"The w_res direction is causally inert on the FULL setup -> the section-6 negative holds, "
              f"and CONTRASTS with 2602.07794 (concept-forced, behaviourally large).")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/wres_causality")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--n_random_dir", type=int, default=15, help="random-direction null draws per layer per mode")
    p.add_argument("--offset_layers", type=int, nargs="*", default=None, help="layers for position-offset sweep (e.g. 18 21 24)")
    p.add_argument("--offsets", type=int, nargs="*", default=[-1, -2, -3, -5, -8])
    p.add_argument("--ablate_offset", type=int, default=-1)
    p.add_argument("--max_targets", type=int, default=80, help="cap targets per class (broad sweep is expensive)")
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
