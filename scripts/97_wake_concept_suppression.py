"""
97_wake_concept_suppression.py   [is the decodable-but-unused concept causally recoverable?]
=============================================================================================
Exp 2 of the four. The model defaults to alpha (content-driven, NOT positional -- see answer-order
control 95) and outputs beta only when surface process-framing routes it there; the concept decodes
beta throughout but does not drive the output. Here we REMOVE the competitor that drives the wrong
answer and let the network RECOMPUTE downstream, asking whether the (present) concept then surfaces.

Competitor direction at layer L:
    d_error[L] = unit( mean(resid | failed-beta) - mean(resid | succeeded-beta) )
This is the axis along which failing vs succeeding beta-prompts differ. Crucially the CONCEPT does
NOT separate them (exp 92: AUC_wres(failed vs succ) ~0.40), so d_error is ~orthogonal to w_res:
projecting it out removes the decision/surface signal WITHOUT removing the concept (we verify
cos(d_error, w_res) and that the w_res projection is preserved after suppression).

Intervention: at a MID layer (the divergence band L15-L17, which has downstream room per exp 92),
project d_error out of the residual at all positions  h <- h - (h.d_hat) d_hat , then let the model
finish -> the output is RECOMPUTED, not hand-pushed. Measured on the failed-beta prompts:
  * recomputed intact-flip to beta (top-1 == ' beta') and margin-sign flip (logit_beta > logit_alpha)
  * concept-preservation sanity: mean w_res projection before vs after (should be ~unchanged)
  * vs NULL: project out a norm/rank-matched random direction
Late-layer (e.g. L30) suppression is included as a contrast to expose any readout tautology.

Two outcomes, both informative:
  - beta surfaces above null AND concept preserved -> the concept is causally connected downstream and
    was being GATED by the competitor -> decodable-but-unused knowledge is RECOVERABLE.
  - no recovery -> the concept has no downstream causal path -> it is DEAD (surface is the only route).

SELF-TEST (no torch):  python 97_wake_concept_suppression.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("wake_concept")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def project_out(H, dhat):
    """Remove the component along dhat (rank-1). H: (...,d), dhat: (d,) unit."""
    return H - np.outer(H @ dhat, dhat) if H.ndim == 2 else H - (H @ dhat) * dhat


def self_test():
    rng = np.random.default_rng(0)
    d, n = 16, 200
    c = unit_raw(rng.standard_normal(d))                  # concept axis
    e = unit_raw(rng.standard_normal(d) - (rng.standard_normal(d) @ c) * c)  # decision axis ~orth to c
    e = unit_raw(e - (e @ c) * c)
    y = np.array([0, 1] * (n // 2))
    H = 0.2 * rng.standard_normal((n, d)) + y[:, None] * c[None, :] * 1.0      # concept decodes beta
    failed = (y == 1) & (rng.random(n) > 0.5)
    succ = (y == 1) & (~failed)
    H[failed] -= e * 1.0; H[succ] += e * 1.0              # failing vs succeeding differ along e
    d_err = unit_raw(H[failed].mean(0) - H[succ].mean(0))
    assert abs(d_err @ c) < 0.3, f"d_error should be ~orthogonal to concept: {d_err@c:.2f}"
    Hp = project_out(H, d_err)
    # projection of d_err removed; concept projection preserved
    assert abs(float((Hp @ d_err).mean())) < abs(float((H @ d_err).mean())) * 0.2 + 1e-6
    assert abs(float((Hp @ c).mean()) - float((H @ c).mean())) < 0.05, "concept must be preserved"
    w = fisher_axis(H, y); assert (H @ w).std() > 0
    print("[self_test] OK — d_error ~orthogonal to concept, projection removes it and preserves the concept.")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    base = model.model; blocks = base.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])  # beta=1
    fams = sorted({p.get("surface_family", str(i)) for i, p in enumerate(prompts)})
    rng.shuffle(fams); train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p.get("surface_family", "") in train_fams for p in prompts])

    def S(x):
        return str(x) if x is not None else "NA"

    def tap_module(L):
        return blocks[L + 1] if L < last else base.norm

    H = {L: np.zeros((nP, d), np.float32) for L in layers}
    clean_margin = np.zeros(nP); clean_top1 = np.zeros(nP, dtype=np.int64)
    logger.info("capturing residuals (single no-pad) over %d prompts...", nP)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            enc = tok([p["prompt"]], return_tensors="pt").to(args.device)
            store = {}; hs = []
            for L in layers:
                def mk(L=L):
                    def pre(m, a):
                        store[L] = a[0][:, -1, :].detach(); return None
                    return pre
                hs.append(tap_module(L).register_forward_pre_hook(mk(), with_kwargs=False))
            try:
                lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                for h in hs:
                    h.remove()
            for L in layers:
                H[L][i] = store[L].float().cpu().numpy()
            clean_margin[i] = float(lo[b_id] - lo[a_id])     # beta - alpha (>0 => beta)
            clean_top1[i] = int(lo.argmax())
            if (i + 1) % 150 == 0:
                logger.info("  %d/%d", i + 1, nP)

    pred_alpha = clean_margin < 0
    failed_beta = (y == 1) & pred_alpha
    succ_beta = (y == 1) & (~pred_alpha)
    bucket_prop = np.array([S(p.get("relation_type")) in
                            {"weak_force_mechanism", "charge_plus_a_unchanged", "z_plus1_a_unchanged", "charge_plus_z_change"}
                            or S(p.get("concept_route")) in {"electron_equivalence", "muon_family", "lepton_family", "not_nuclear_fragment"}
                            or S(p.get("prompt_format")) in {"property", "equivalence"} for p in prompts])
    prop_failed = failed_beta & bucket_prop
    logger.info("failed_beta=%d succ_beta=%d property-failed=%d", failed_beta.sum(), succ_beta.sum(), prop_failed.sum())

    # directions per layer
    info = {}
    for L in layers:
        d_err = unit_raw(H[L][failed_beta].astype(np.float64).mean(0) - H[L][succ_beta].astype(np.float64).mean(0))
        w_res = fisher_axis(H[L][is_train].astype(np.float64), y[is_train], args.shrink)
        info[L] = dict(d_err=d_err, w_res=w_res, cos=float(abs(d_err @ w_res)),
                       rms=float(np.sqrt((H[L] ** 2).sum(1).mean())))
        logger.info("  L%d: |cos(d_error, w_res)|=%.3f (want small) rms=%.1f", L, info[L]["cos"], info[L]["rms"])

    # suppression hook (project out a unit direction at all positions)
    def supp_hook(dhat):
        v = torch.tensor(dhat, dtype=torch.float32, device=args.device)
        def pre(m, a):
            h = a[0]
            coef = (h @ v).unsqueeze(-1)                  # (1, seq, 1)
            a[0][:, :, :] = h - coef * v
            return (a[0],) + tuple(a[1:])
        return pre

    def eval_supp(idxs, L, dhat, wres):
        if len(idxs) == 0:
            return dict(intact_flip=float("nan"), margin_flip=float("nan"),
                        wres_proj_before=float("nan"), wres_proj_after=float("nan"))
        intact = mflip = 0; wb = []; wa = []
        wv = torch.tensor(wres, dtype=torch.float32, device=args.device)
        for i in idxs:
            enc = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
            # before: concept projection at L (no intervention)
            wb.append(float(H[L][i] @ wres))
            h = tap_module(L).register_forward_pre_hook(supp_hook(dhat), with_kwargs=False)
            store = {}
            def capw(m, a):
                store["h"] = a[0][:, -1, :].detach(); return None
            hw = tap_module(L).register_forward_pre_hook(capw, with_kwargs=False)
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                h.remove(); hw.remove()
            wa.append(float((store["h"][0].float().cpu().numpy()) @ wres))
            m = float(lo[b_id] - lo[a_id]); t1 = int(lo.argmax())
            mflip += int(m > 0); intact += int(t1 == b_id)
        return dict(intact_flip=intact / len(idxs), margin_flip=mflip / len(idxs),
                    wres_proj_before=float(np.mean(wb)), wres_proj_after=float(np.mean(wa)))

    sets = {"all_failed_beta": np.where(failed_beta)[0].tolist(),
            "property_failed_beta": np.where(prop_failed)[0].tolist()}
    rows = []
    for setname, idxs in sets.items():
        for L in layers:
            r = eval_supp(idxs, L, info[L]["d_err"], info[L]["w_res"])
            rows.append(dict(prompt_set=setname, layer=int(L), suppress="d_error",
                             cos_derr_wres=info[L]["cos"], **r, n=len(idxs)))
            # null: random direction suppression
            rfl, rmf = [], []
            for _ in range(args.n_random):
                rv = unit_raw(rng.standard_normal(d))
                rr = eval_supp(idxs, L, rv, info[L]["w_res"])
                rfl.append(rr["margin_flip"]); rmf.append(rr["intact_flip"])
            rows.append(dict(prompt_set=setname, layer=int(L), suppress="null_random",
                             cos_derr_wres=float("nan"),
                             margin_flip=float(np.mean(rfl)), intact_flip=float(np.mean(rmf)),
                             null_margin_p95=float(np.percentile(rfl, 95)),
                             null_intact_p95=float(np.percentile(rmf, 95)),
                             wres_proj_before=float("nan"), wres_proj_after=float("nan"), n=len(idxs)))
            logger.info("[%s L%d] suppress d_error: margin-flip=%.2f intact-flip=%.2f (w_res proj %.2f->%.2f) | null margin-flip p95=%.2f",
                        setname, L, rows[-2]["margin_flip"], rows[-2]["intact_flip"],
                        rows[-2]["wres_proj_before"], rows[-2]["wres_proj_after"], rows[-1]["null_margin_p95"])

    with open(out / "wake_concept.csv", "w", newline="") as f:
        flds = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(f, fieldnames=flds); w.writeheader(); [w.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("WAKE-THE-CONCEPT — does removing the competitor let the (present) concept surface?")
    print("=" * 100)
    for setname in sets:
        print(f"\n[{setname}]")
        for L in layers:
            d_rows = [r for r in rows if r["prompt_set"] == setname and r["layer"] == L]
            de = next(r for r in d_rows if r["suppress"] == "d_error")
            nu = next(r for r in d_rows if r["suppress"] == "null_random")
            recov = de["margin_flip"] > nu.get("null_margin_p95", 0) + 0.1
            conc_ok = abs(de["wres_proj_after"] - de["wres_proj_before"]) < 0.25 * (abs(de["wres_proj_before"]) + 1e-6) + 0.5
            print(f"  L{L}: suppress d_error -> recovered-to-beta margin={de['margin_flip']:.2f} intact={de['intact_flip']:.2f} "
                  f"(null p95 {nu.get('null_margin_p95',float('nan')):.2f}) | cos(d_err,w_res)={de['cos_derr_wres']:.2f} | "
                  f"concept {'PRESERVED' if conc_ok else 'DISTURBED'} ({de['wres_proj_before']:.2f}->{de['wres_proj_after']:.2f})"
                  f"{'  <= RECOVERS' if recov and conc_ok else ''}")
    print("\nReading: recovery ABOVE null with the concept PRESERVED => the unused concept is causally recoverable "
          "(it was gated by the competitor). No recovery => the concept is not downstream-causal (dead; surface-only route). "
          "Mid layers (L15-17) are the meaningful test; a late layer would be a readout tautology.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/wake_concept")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[14, 17, 20, 30], help="mid band + one late as a readout-tautology contrast")
    p.add_argument("--n_random", type=int, default=6)
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
