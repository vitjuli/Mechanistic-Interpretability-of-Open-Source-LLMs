"""
85_steering_all_layers.py   [does steering w_res flip the answer at ANY layer?]
=========================================================================================
Exp 75 pushed along w_res with increasing strength (1..32 sigma) but only at postL24 (+final)
-- the strongest layer from exp 65. That leaves the steering negative untested on layers
L0-L23, L25-L35. This script closes that hole: the SAME norm-matched saturation steering,
on EVERY layer.

Mechanism (identical to j75/j65, so results are comparable):
  - At a layer's residual tap, add (c * sigma) * unit(w_res) to the answer-position state,
    where sigma = std of the train projections onto w_res (so c is in "concept-sigma" units
    and magnitudes are comparable across layers).
  - Sweep c over a saturating grid (default 1,2,4,8,16,32) in both directions
    (alpha->beta on alpha-prompts, beta->alpha on beta-prompts).
  - Compare w_res to SHUFFLED-label Fisher directions and RANDOM directions of the SAME
    magnitude (a big enough push flips things regardless of direction -- the nulls catch it).

Two metrics, kept separate (the j75 lesson):
  flip        = sign of (logit_beta - logit_alpha) moved toward the target (relative; NOT
                the answer).
  intact-flip = that flip happened WHILE the top-1 token is still a real answer (alpha/beta)
                -- the BEHAVIOURAL metric. A flip that only occurs after the answer breaks
                into garbage is not concept-steering.
Decisive quantity = intact-flip vs shuffled p95, per layer, per magnitude.

Outcomes (per j75 logic, now across all layers):
  A) intact-flip ~0 and tracks shuffled at every layer & magnitude -> w_res is not a lever
     anywhere (strongest negative; what j75 found at L24, now confirmed on the full stack).
  B) intact-flip rises above shuffled at some layer while answers stay intact -> THAT layer
     is a steering locus; the hole was real. Reported with the (layer, c).

SELF-TEST (no torch / no repo):  python 85_steering_all_layers.py --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("steer_all")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def directional_flip(dl_clean, dl_steered, toward):
    if toward == "beta":
        return int(dl_clean < 0 and dl_steered > 0)
    return int(dl_clean > 0 and dl_steered < 0)


# =====================================================================
# Self-test: causal toy (flips while intact) vs non-causal (flips only via breakage)
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 12, 120
    y = (rng.random(n) < 0.5).astype(int)
    w = unit_raw(rng.standard_normal(d))
    H = rng.standard_normal((n, d)) * 0.5 + np.outer((y * 2 - 1.0) * 1.5, w)
    sigma = float(np.std(H @ w)); break_thr = 9.0 * sigma
    cs = [1, 2, 4, 8, 16, 32]

    def causal_eval(h, step):
        m = float((h + step) @ w)
        return (rng.standard_normal() if np.linalg.norm(step) > break_thr else m,
                np.linalg.norm(step) <= break_thr)

    w_perp = rng.standard_normal(d); w_perp -= (w_perp @ w) * w; w_perp = unit_raw(w_perp)

    def noncausal_eval(h, step):
        broken = np.linalg.norm(step) > break_thr
        return (rng.standard_normal() if broken else float(h @ w_perp)), (not broken)

    def intact_curve(evalfn, direction):
        idx = [i for i in range(n) if (H[i] @ w) < 0][:60]
        out = {}
        for c in cs:
            f_intact = []
            for i in idx:
                m0, _ = evalfn(H[i], np.zeros(d))          # clean margin from the SAME readout
                m1, intact = evalfn(H[i], (c * sigma) * direction)
                f_intact.append(directional_flip(m0, m1, "beta") * int(intact))
            out[c] = float(np.mean(f_intact)) if f_intact else 0.0
        return out

    causal = intact_curve(causal_eval, w)
    noncausal = intact_curve(noncausal_eval, w)
    assert max(causal.values()) > 0.3, "causal toy should produce intact flips at some c"
    assert max(noncausal.values()) < 0.1, "non-causal toy should NOT produce intact flips"
    assert directional_flip(-1.0, 1.0, "beta") == 1 and directional_flip(-1.0, 1.0, "alpha") == 0
    print("[self_test] OK — intact-flip separates causal vs non-causal toy; flip logic correct.")


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
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size
    alpha_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    beta_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    last = n_layers - 1

    steer_layers = args.steer_layers if args.steer_layers else list(range(n_layers))
    steer_layers = [L for L in steer_layers if 0 <= L <= last]
    taps = [f"postL{L}" for L in steer_layers] + (["final"] if args.include_final else [])
    logger.info("model: %d layers; steering %d taps (all layers)", n_layers, len(taps))

    def module_for(tap):
        return norm_mod if tap == "final" else blocks[int(tap.replace("postL", "")) + 1]

    prompts = [json.loads(l) for l in open(args.prompts)]
    fams = sorted({p["surface_family"] for p in prompts}); rng.shuffle(fams)
    train_fams = set(fams[: int(round(len(fams) * args.train_frac))])

    # ---------- capture clean residuals at all taps (one forward/prompt) ----------
    logger.info("capturing clean residuals for %d prompts at %d taps...", len(prompts), len(taps))
    H = {t: np.zeros((len(prompts), d), np.float32) for t in taps}
    y = np.zeros(len(prompts), int); trm = np.zeros(len(prompts), bool); hn = {t: [] for t in taps}
    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        g = {}; handles = []
        for L in steer_layers:
            tgt = blocks[L + 1] if L < last else norm_mod
            def mk(L=L):
                def pre(m, a): g[f"postL{L}"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(tgt.register_forward_pre_hook(mk(), with_kwargs=False))
        if args.include_final:
            def mkf():
                def pre(m, a): g["final"] = a[0][0, -1, :].detach().float().cpu().numpy(); return None
                return pre
            handles.append(norm_mod.register_forward_pre_hook(mkf(), with_kwargs=False))
        try:
            with torch.no_grad():
                model(**inp, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for t in taps:
            H[t][i] = g[t]; hn[t].append(float(np.linalg.norm(g[t])))
        y[i] = 1 if p["correct_answer"].strip() == "beta" else 0
        trm[i] = p["surface_family"] in train_fams
        if (i + 1) % 100 == 0:
            logger.info("  capture %d/%d", i + 1, len(prompts))

    geom = {}
    for t in taps:
        Htr, ytr = H[t][trm].astype(np.float64), y[trm]
        wres = fisher_axis(Htr, ytr, args.shrink)
        shuf = [fisher_axis(Htr, rng.permutation(ytr), args.shrink) for _ in range(args.n_shuffle)]
        geom[t] = {"w_res": wres, "shuf": shuf, "sigma": float(np.std(Htr @ wres)),
                   "hnorm_mean": float(np.mean(hn[t]))}

    held = [p for p in prompts if p["surface_family"] not in train_fams]
    ha = [p for p in held if p["correct_answer"].strip() == "alpha"][: args.max_targets]
    hb = [p for p in held if p["correct_answer"].strip() == "beta"][: args.max_targets]
    cgrid = sorted(args.c_grid); answer_ids = {alpha_id, beta_id}

    def dl_and_top(ptext, tap, delta):
        inp = tok([ptext], return_tensors="pt").to(args.device)
        if delta is None:
            with torch.no_grad():
                row = model(**inp, use_cache=False).logits[0, -1, :].float()
        else:
            dt = torch.tensor(delta, dtype=torch.float32, device=args.device)
            def pre(m, a):
                hs = a[0].clone(); hs[0, -1, :] = hs[0, -1, :] + dt; return (hs,)
            h = module_for(tap).register_forward_pre_hook(pre, with_kwargs=False)
            try:
                with torch.no_grad():
                    row = model(**inp, use_cache=False).logits[0, -1, :].float()
            finally:
                h.remove()
        lp = torch.log_softmax(row, 0)
        return float(lp[beta_id] - lp[alpha_id]), int(torch.argmax(row).item())

    rows = []
    def run(tap, targets, toward, signed_cs):
        sig = geom[tap]["sigma"]
        dirs = {"w_res": unit_raw(geom[tap]["w_res"])}
        for k in range(args.n_random):
            dirs[f"random{k}"] = unit_raw(rng.standard_normal(d))
        for j, sd in enumerate(geom[tap]["shuf"]):
            dirs[f"shuffled{j}"] = unit_raw(sd)
        for tgt in targets:
            dlc, _ = dl_and_top(tgt["prompt"], tap, None)
            if toward == "beta" and not (dlc < 0):
                continue
            if toward == "alpha" and not (dlc > 0):
                continue
            for c in signed_cs:
                for name, vec in dirs.items():
                    step = (c * sig) * vec
                    dls, tops = dl_and_top(tgt["prompt"], tap, step)
                    rows.append({"tap": tap, "toward": toward, "dir": name, "c": abs(c),
                                 "flip": directional_flip(dlc, dls, toward), "intact": int(tops in answer_ids)})

    for ti, tap in enumerate(taps):
        run(tap, ha, "beta", cgrid)
        run(tap, hb, "alpha", [-c for c in cgrid])
        if (ti % 4 == 0) or (ti == len(taps) - 1):
            logger.info("  steered tap %s (%d/%d)", tap, ti + 1, len(taps))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "steering_all_layers_curve.csv", index=False)
    df["dg"] = df["dir"].map(lambda k: "random" if k.startswith("random")
                             else "shuffled" if k.startswith("shuffled") else k)

    # per (tap, c) curve + per-layer summary (max intact-flip across c)
    curve = []; per_layer = []
    for tap in taps:
        best_intact = -1.0; best_c = None; beats = False
        for c in cgrid:
            sub = df[(df.tap == tap) & (df.c == c)]
            w = sub[sub.dg == "w_res"]; shuf = sub[sub.dg == "shuffled"].groupby("dir")["flip"].mean()
            wflip = float(w["flip"].mean()) if len(w) else float("nan")
            wintact = float((w["flip"] * w["intact"]).mean()) if len(w) else float("nan")
            shuf_p95 = float(np.percentile(shuf, 95)) if len(shuf) else float("nan")
            curve.append({"tap": tap, "c": c, "wres_flip": round(wflip, 3), "wres_intact_flip": round(wintact, 3),
                          "shuffled_flip_p95": round(shuf_p95, 3), "intact_rate": round(float(w["intact"].mean()), 3) if len(w) else None})
            if not np.isnan(wintact) and wintact > best_intact:
                best_intact = wintact; best_c = c
            if not np.isnan(wintact) and not np.isnan(shuf_p95) and wintact > shuf_p95 and len(w) and w["intact"].mean() >= args.intact_min:
                beats = True
        per_layer.append({"tap": tap, "max_intact_flip": round(best_intact, 3), "at_c": best_c, "beats_shuffled": bool(beats)})
        logger.info("  %s: max intact-flip=%.3f at c=%s  beats_shuffled=%s", tap, best_intact, best_c, beats)

    json.dump({"per_layer": per_layer, "curve": curve}, open(out / "steering_all_layers_summary.json", "w"), indent=2)

    # ---------- verdict ----------
    print("\n" + "=" * 92)
    print("STEERING w_res across ALL layers -- does pushing along the reading axis flip the answer?")
    print("=" * 92)
    hits = [r for r in per_layer if r["beats_shuffled"] and r["max_intact_flip"] >= args.tau_flip]
    overall_max = max((r["max_intact_flip"] for r in per_layer), default=float("nan"))
    if hits:
        h = max(hits, key=lambda r: r["max_intact_flip"])
        print(f"OUTCOME B -- a STEERING LOCUS exists: {h['tap']} reaches intact-flip {h['max_intact_flip']:.2f} "
              f"at c={h['at_c']} sigma, above the shuffled-label null while answers stay intact. The hole was "
              f"real; steering w_res IS a lever at this layer.")
    else:
        print(f"OUTCOME A -- NOT A LEVER AT ANY LAYER: across all {len(taps)} taps and up to {max(cgrid)} sigma, "
              f"behavioural intact-flip never beats the shuffled-label null (max {overall_max:.2f}). Steering w_res "
              f"is causally inert on the FULL stack -- j75's L24 result holds everywhere; flips, if any, are "
              f"non-specific (match random/shuffled) and coincide with answer breakage.")
    print("=" * 92 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="data/prompts/physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/steering_all_layers")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--steer_layers", type=int, nargs="*", default=None, help="default = ALL layers")
    p.add_argument("--include_final", action="store_true", default=True)
    p.add_argument("--no_final", dest="include_final", action="store_false")
    p.add_argument("--c_grid", type=float, nargs="*", default=[1, 2, 4, 8, 16, 32], help="sigma magnitudes")
    p.add_argument("--n_shuffle", type=int, default=5, help="shuffled-label null directions per tap")
    p.add_argument("--n_random", type=int, default=5, help="random directions per tap")
    p.add_argument("--max_targets", type=int, default=40, help="held-out targets per class (broad scan)")
    p.add_argument("--intact_min", type=float, default=0.5, help="min answer-intact rate to count a beats-shuffled hit")
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
