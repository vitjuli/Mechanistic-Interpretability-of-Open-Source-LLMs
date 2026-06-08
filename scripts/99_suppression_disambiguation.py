"""
99_suppression_disambiguation.py   [is d_error suppression concept-awakening, or decision-axis erasure?]
========================================================================================================
Exp 98 found: projecting out d_error = mean(failed-beta) - mean(succeeded-beta) at L20/L24 makes the
model output beta (intact-flip 0.5-0.88, null 0, cos(d_error,w_res)=0.05). That LOOKS like waking the
unused concept -- but a near-tautology gives the same log: d_error is, by construction, the axis that
separates alpha-leaning (failed) from beta-leaning (succeeded) beta-prompts, i.e. ~the decision axis at
layer L. Projecting it out may simply erase the current (alpha) decision, after which beta surfaces for
mechanical reasons, NOT because the concept re-drove the answer. Low cos with w_res (the READABLE axis)
does not rule this out, because the decision lives along the USED axis u, not w_res.

This script disambiguates on the BASE format (n=149 failed-beta -- large sample; projection is rank-1 so
it is NOT subject to the norm-blowup artifact that broke injection):

  (1) cos(d_error, u) and cos(d_error, w_res), where u = mean gradient of (logit_beta - logit_alpha) on
      the failed-beta set (the used/decision direction). High cos with u => erasing the lever (tautology).
  (2) CONTROL on correct-alpha prompts: project d_error out of prompts the model gets RIGHT as alpha.
      If they ALSO flip toward beta, the projection is generic alpha-erasure, not concept-specific healing.
  (3) PLACEBO: a direction from a RANDOM split of the beta prompts (shuffled failed/succeeded labels,
      averaged over draws). If it heals as much as d_error, the effect is non-specific.

Decision:
  REAL (concept-specific) if: recovery(failed-beta) >> placebo AND >> disturbance(correct-alpha),
      and cos(d_error,u) is not ~1.
  TAUTOLOGY (decision erasure) if: correct-alpha also flips to beta, OR placebo heals, OR cos(d_error,u)~1.

SELF-TEST (no torch):  python 99_suppression_disambiguation.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("supp_disambig")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def self_test():
    rng = np.random.default_rng(0)
    d, n = 16, 240
    y = np.array([0, 1] * (n // 2))                          # beta=1
    u = unit_raw(rng.standard_normal(d))                     # used/decision axis
    concept = unit_raw(rng.standard_normal(d) - (rng.standard_normal(d) @ u) * u)
    H = 0.2 * rng.standard_normal((n, d))
    # beta prompts: half "failed" (low u => alpha-leaning), half "succeeded" (high u)
    beta = y == 1
    failed = beta & (rng.random(n) > 0.5); succ = beta & (~failed)
    H[failed] -= u * 1.0; H[succ] += u * 1.0                 # decision axis = u
    d_err = unit_raw(H[failed].mean(0) - H[succ].mean(0))
    # in this toy d_error SHOULD align with u (it's the decision axis) -> tautology signature
    assert abs(d_err @ u) > 0.7, f"toy: d_error should align with the decision axis u: {d_err@u:.2f}"
    # placebo from random split of beta
    bidx = np.where(beta)[0]; rng.shuffle(bidx); A, B = bidx[:len(bidx)//2], bidx[len(bidx)//2:]
    d_pl = unit_raw(H[A].mean(0) - H[B].mean(0))
    assert abs(d_pl @ u) < abs(d_err @ u), "placebo should align with u less than the true error axis"
    print("[self_test] OK — toy reproduces the tautology signature (d_error ~ decision axis u; placebo weaker).")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    bm = model.model; blocks = bm.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = sorted({p["surface_family"] for p in prompts})
    rng.shuffle(fams); train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p["surface_family"] in train_fams for p in prompts])

    def tap_module(L):
        return blocks[L + 1] if L < last else bm.norm

    # ---- capture residual + gradient (u) at answer position, base format ----
    H = {L: np.zeros((nP, d), np.float32) for L in layers}
    G = {L: np.zeros((nP, d), np.float32) for L in layers}
    margin = np.zeros(nP)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    logger.info("capturing residual + gradient (base format) over %d prompts...", nP)
    for i in range(nP):
        enc = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
        keep = {}; hs = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]
                H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
            margin[i] = float(lo[b_id].detach() - lo[a_id].detach())
        finally:
            for h in hs:
                h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0:
            logger.info("  %d/%d", i + 1, nP)

    pred_alpha = margin < 0
    failed_beta = (y == 1) & pred_alpha
    succ_beta = (y == 1) & (~pred_alpha)
    correct_alpha = (y == 0) & pred_alpha
    logger.info("failed_beta=%d succ_beta=%d correct_alpha=%d", failed_beta.sum(), succ_beta.sum(), correct_alpha.sum())

    # ---- directions per layer ----
    info = {}
    beta_idx = np.where(y == 1)[0]
    for L in layers:
        u = unit_raw(G[L][failed_beta].astype(np.float64).mean(0))         # used/decision dir on the errors
        w_res = fisher_axis(H[L][is_train].astype(np.float64), y[is_train], args.shrink)
        d_err = unit_raw(H[L][failed_beta].astype(np.float64).mean(0) - H[L][succ_beta].astype(np.float64).mean(0))
        # placebo directions: random splits of beta prompts (matched sizes to failed/succ)
        nf = int(failed_beta.sum())
        placebos = []
        for _ in range(args.n_placebo):
            perm = beta_idx.copy(); rng.shuffle(perm)
            A, B = perm[:nf], perm[nf:]
            placebos.append(unit_raw(H[L][A].astype(np.float64).mean(0) - H[L][B].astype(np.float64).mean(0)))
        info[L] = dict(u=u, w_res=w_res, d_err=d_err, placebos=placebos,
                       cos_err_u=float(abs(d_err @ u)), cos_err_wres=float(abs(d_err @ w_res)),
                       cos_pl_u=float(np.mean([abs(pl @ u) for pl in placebos])))
        logger.info("  L%d: |cos(d_err,u)|=%.3f  |cos(d_err,w_res)|=%.3f  |cos(placebo,u)|=%.3f",
                    L, info[L]["cos_err_u"], info[L]["cos_err_wres"], info[L]["cos_pl_u"])

    # ---- projection eval (rank-1 project-out at layer L, base format, margin metric) ----
    def proj_hook(dhat):
        v = torch.tensor(dhat, dtype=torch.float32, device=args.device)
        def pre(m, a):
            h = a[0]; a[0][:, :, :] = h - (h @ v).unsqueeze(-1) * v
            return (a[0],) + tuple(a[1:])
        return pre

    def beta_rate(idxs, L, dhat):
        """fraction whose margin (logit_beta - logit_alpha) is >0 after projecting dhat out."""
        if len(idxs) == 0:
            return float("nan")
        c = 0
        for i in idxs:
            enc = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
            h = tap_module(L).register_forward_pre_hook(proj_hook(dhat), with_kwargs=False)
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                h.remove()
            c += int(float(lo[b_id] - lo[a_id]) > 0)
        return c / len(idxs)

    grp = {"failed_beta": np.where(failed_beta)[0].tolist(),
           "correct_alpha_CONTROL": np.where(correct_alpha)[0].tolist(),
           "succ_beta": np.where(succ_beta)[0].tolist()}
    # baseline beta-rate (no intervention)
    base_rate = {g: float(np.mean(margin[idx] > 0)) for g, idx in grp.items()}

    rows = []
    for L in layers:
        for g, idx in grp.items():
            r_err = beta_rate(idx, L, info[L]["d_err"])
            r_pl = float(np.mean([beta_rate(idx, L, pl) for pl in info[L]["placebos"]]))
            r_rand = float(np.mean([beta_rate(idx, L, unit_raw(rng.standard_normal(d))) for _ in range(args.n_random)]))
            rows.append(dict(layer=int(L), group=g, n=len(idx),
                             base_beta_rate=base_rate[g], proj_d_error=r_err,
                             proj_placebo=r_pl, proj_random=r_rand,
                             cos_err_u=info[L]["cos_err_u"], cos_err_wres=info[L]["cos_err_wres"]))
            logger.info("[L%d %s n=%d] base β-rate=%.2f | d_error->%.2f  placebo->%.2f  random->%.2f",
                        L, g, len(idx), base_rate[g], r_err, r_pl, r_rand)

    with open(out / "suppression_disambiguation.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

    print("\n" + "=" * 104)
    print("SUPPRESSION DISAMBIGUATION — is d_error projection concept-awakening, or decision-axis erasure?")
    print("=" * 104)
    for L in layers:
        fr = next(r for r in rows if r["layer"] == L and r["group"] == "failed_beta")
        ca = next(r for r in rows if r["layer"] == L and r["group"] == "correct_alpha_CONTROL")
        print(f"\nL{L}:  |cos(d_error,u)|={fr['cos_err_u']:.2f}  |cos(d_error,w_res)|={fr['cos_err_wres']:.2f}")
        print(f"   failed_beta  : base β-rate {fr['base_beta_rate']:.2f} -> d_error {fr['proj_d_error']:.2f} "
              f"(placebo {fr['proj_placebo']:.2f}, random {fr['proj_random']:.2f})")
        print(f"   ALPHA-CONTROL: base β-rate {ca['base_beta_rate']:.2f} -> d_error {ca['proj_d_error']:.2f} "
              f"(if this jumps to beta, projection is generic alpha-erasure)")
        heal = fr["proj_d_error"] - max(fr["proj_placebo"], fr["proj_random"])
        leak = ca["proj_d_error"] - ca["base_beta_rate"]
        taut_u = fr["cos_err_u"] > 0.6
        if heal > 0.2 and leak < 0.2 and not taut_u:
            print(f"   => SPECIFIC & REAL at L{L}: failed-beta recovers (+{heal:.2f} over placebo/random), alpha-control "
                  f"barely moves (+{leak:.2f}), cos(d_error,u)={fr['cos_err_u']:.2f} not ~1 -> not mere lever-erasure.")
        elif taut_u or leak > 0.3 or fr["proj_placebo"] > fr["proj_d_error"] - 0.1:
            print(f"   => TAUTOLOGY signature at L{L}: " +
                  ("d_error ~ used axis u (erasing the lever); " if taut_u else "") +
                  (f"alpha-control also flips to beta (+{leak:.2f}, generic alpha-erasure); " if leak > 0.3 else "") +
                  ("placebo heals as much (non-specific). " if fr["proj_placebo"] > fr["proj_d_error"] - 0.1 else ""))
        else:
            print(f"   => MIXED/inconclusive at L{L}: heal+{heal:.2f}, alpha-leak+{leak:.2f}, cos(d_error,u)={fr['cos_err_u']:.2f}.")
    print("\nReading: a real awakening needs failed-beta to recover specifically (beyond placebo/random) WITHOUT the "
          "alpha-control flipping and WITHOUT d_error being the used axis u. Otherwise removing d_error is just erasing "
          "the (alpha-leaning) decision, and beta surfacing is mechanical, not the concept re-driving the answer.")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/supp_disambig")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[17, 20, 24])
    p.add_argument("--n_placebo", type=int, default=6)
    p.add_argument("--n_random", type=int, default=4)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
