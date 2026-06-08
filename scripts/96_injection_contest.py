"""
96_injection_contest.py    [does concept, surface, or noise heal the failed-beta prompts?]
============================================================================================
On prompts the model gets WRONG (true beta, predicted alpha), we ADD a direction toward beta at a
mid layer (all positions) and ask whether the model now answers beta. Three directions, head to head,
norm-matched, same prompts:

  (1) CONCEPT   w_res  = Fisher(train residuals, TRUE labels), signed +beta. The decodable axis.
                Prediction: does NOT heal (decodable != causal; cf. ITI / the medical-LLM result).
  (3) SURFACE   d_surf = mean(resid | process-framed beta) - mean(resid | property-framed beta),
                within true-beta, signed +beta. The surface phrasing the failed prompts lack.
                Diagnostic: if this heals and concept does not -> the model listens to FORM, not meaning.
  (4) NULL      random unit direction, norm-matched. Floor: any real effect must beat this.

Two prompt sets, reported separately: ALL failed-beta; PROPERTY-framed failed-beta (cleaner).
Two heal metrics: intact-flip to beta (top-1 becomes ' beta'); margin-sign flip (logit_beta>logit_alpha).
Swept over --layers and --alphas (strength in units of the layer's residual RMS).

NOTE on direction definitions (post-correction): on the REAL run the model recognises beta from
process/transformation phrasing (neutron->proton, antineutrino) and defaults to alpha on
property/lepton phrasing -- so d_surf points from property toward process. w_res is defined from the
TRUE alpha/beta labels and is framing-invariant.

SELF-TEST (no torch):  python 96_injection_contest.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("inj_contest")


# ---------------- shared numpy core ----------------
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    """LDA axis signed so higher projection => class 1 (beta)."""
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def diff_means(H, mask_pos, mask_neg):
    """Signed difference-in-means direction: mean(pos) - mean(neg), unit-normalised (points +pos)."""
    return unit_raw(H[mask_pos].mean(0) - H[mask_neg].mean(0))


def framing_bucket(rt, cr, pf):
    process = {"neutron_to_proton", "n_to_p_with_antineutrino", "n_to_p_plus_z_plus1",
               "full_beta_process_spec", "quark_level_process", "quark_level_consequence",
               "z_plus1_with_antineutrino"}
    prop_rel = {"weak_force_mechanism", "charge_plus_a_unchanged", "z_plus1_a_unchanged", "charge_plus_z_change"}
    prop_route = {"electron_equivalence", "muon_family", "lepton_family", "not_nuclear_fragment"}
    if rt in process or pf == "process":
        return "process"
    if rt in prop_rel or cr in prop_route or pf in ("property", "equivalence"):
        return "property"
    return "other"


def self_test():
    rng = np.random.default_rng(0)
    d, n = 16, 200
    y = np.array([0, 1] * (n // 2))                       # beta=1
    beta_dir = unit_raw(rng.standard_normal(d))           # the *causal/output* direction (what moves the answer)
    surf_dir = unit_raw(rng.standard_normal(d))           # surface axis, partially aligned with beta_dir
    surf_dir = unit_raw(0.6 * beta_dir + 0.4 * surf_dir)
    concept_dir = unit_raw(rng.standard_normal(d))        # decodable but ~orthogonal to beta_dir
    H = 0.3 * rng.standard_normal((n, d))
    H += y[:, None] * beta_dir[None, :] * 1.0             # true beta sits along beta_dir (decodes & drives)
    H += y[:, None] * concept_dir[None, :] * 1.0          # concept also decodes beta (extra readable axis)
    # framing labels (within beta): process vs property; process beta-prompts carry +surf
    proc = (y == 1) & (rng.random(n) > 0.5)
    H[proc] += surf_dir * 1.2
    # margin model: answer = sign(h . beta_dir); fisher recovers a decodable axis
    w = fisher_axis(H, y); assert (H @ w).std() > 0
    ds = diff_means(H, proc, (y == 1) & (~proc))
    # injecting along beta_dir should raise the margin; along an orthogonal concept axis should not
    base = H[(y == 1)].mean(0) @ beta_dir
    bumped = (H[(y == 1)].mean(0) + 0.5 * beta_dir) @ beta_dir
    assert bumped > base
    # d_surf should have larger overlap with beta_dir than a random dir, by construction
    assert abs(ds @ beta_dir) > 0.2
    assert framing_bucket("neutron_to_proton", "NA", "process") == "process"
    assert framing_bucket("NA", "lepton_family", "property") == "property"
    print("[self_test] OK — fisher, diff-means, framing bucket, injection-raises-margin all pass.")


# ---------------- real run ----------------
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
    logger.info("model %d layers d=%d; inject at layers %s; alphas %s", n_layers, d, layers, args.alphas)

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])  # beta=1
    fams = sorted({p.get("surface_family", str(i)) for i, p in enumerate(prompts)})
    rng.shuffle(fams); train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p.get("surface_family", "") in train_fams for p in prompts])

    def S(x):
        return str(x) if x is not None else "NA"

    bucket = np.array([framing_bucket(S(p.get("relation_type")), S(p.get("concept_route")), S(p.get("prompt_format")))
                       for p in prompts])

    # tap = INPUT to block L+1 (residual after block L); for last layer, the final norm input
    def tap_module(L):
        return blocks[L + 1] if L < last else base.norm

    # ---------- capture residuals at the answer position for all prompts (single, no pad) ----------
    H = {L: np.zeros((nP, d), np.float32) for L in layers}
    clean_margin = np.zeros(nP); clean_top1 = np.zeros(nP, dtype=np.int64)
    logger.info("capturing residuals at answer position (single no-pad)...")
    with torch.no_grad():
        for i, p in enumerate(prompts):
            enc = tok([p["prompt"]], return_tensors="pt").to(args.device)
            store = {}
            hs = []
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
            clean_margin[i] = float(lo[b_id] - lo[a_id])     # NOTE: beta - alpha (>0 => beta = correct for beta prompts)
            clean_top1[i] = int(lo.argmax())
            if (i + 1) % 150 == 0:
                logger.info("  %d/%d", i + 1, nP)

    pred_alpha = clean_margin < 0                            # predicted alpha
    failed_beta = (y == 1) & pred_alpha
    prop_failed = failed_beta & (bucket == "property")
    logger.info("failed_beta=%d  property-framed failed_beta=%d", failed_beta.sum(), prop_failed.sum())

    # ---------- build directions per layer ----------
    dirs = {}
    for L in layers:
        w_res = fisher_axis(H[L][is_train].astype(np.float64), y[is_train], args.shrink)   # +beta
        proc_mask = (y == 1) & (bucket == "process")
        prop_mask = (y == 1) & (bucket == "property")
        d_surf = diff_means(H[L].astype(np.float64), proc_mask, prop_mask)                  # process - property (+beta-ish)
        rms = float(np.sqrt((H[L] ** 2).sum(1).mean()))                                     # residual RMS scale
        dirs[L] = dict(w_res=w_res, d_surf=d_surf, rms=rms,
                       cos_surf_wres=float(abs(w_res @ d_surf)))
        logger.info("  L%d: |cos(w_res,d_surf)|=%.3f  rms=%.1f", L, dirs[L]["cos_surf_wres"], rms)

    # ---------- injection eval ----------
    def add_hook(L, vec):
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        def pre(m, a):
            a[0][:, :, :] = a[0][:, :, :] + v            # add to all positions
            return (a[0],) + tuple(a[1:])
        return pre

    def eval_inject(idxs, L, vec):
        if len(idxs) == 0:
            return dict(intact_flip=float("nan"), margin_flip=float("nan"))
        intact = 0; mflip = 0
        for i in idxs:
            enc = tok([prompts[i]["prompt"]], return_tensors="pt").to(args.device)
            h = tap_module(L).register_forward_pre_hook(add_hook(L, vec), with_kwargs=False)
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                h.remove()
            m = float(lo[b_id] - lo[a_id]); t1 = int(lo.argmax())
            mflip += int(m > 0)                          # flipped to beta (margin)
            intact += int(t1 == b_id)                    # top-1 is ' beta'
        return dict(intact_flip=intact / len(idxs), margin_flip=mflip / len(idxs))

    sets = {"all_failed_beta": np.where(failed_beta)[0].tolist(),
            "property_failed_beta": np.where(prop_failed)[0].tolist()}
    n_rand = args.n_random
    rows = []
    for setname, idxs in sets.items():
        for L in layers:
            rms = dirs[L]["rms"]
            for alpha in args.alphas:
                scale = alpha * rms
                # concept and surface
                for dname, vec in [("concept_wres", dirs[L]["w_res"] * scale),
                                   ("surface_dsurf", dirs[L]["d_surf"] * scale)]:
                    r = eval_inject(idxs, L, vec)
                    rows.append(dict(prompt_set=setname, layer=int(L), alpha=alpha, direction=dname,
                                     **r, n=len(idxs)))
                # null: average over random directions
                rfl, rmf = [], []
                for _ in range(n_rand):
                    rv = rng.standard_normal(d); rv = unit_raw(rv) * scale
                    rr = eval_inject(idxs, L, rv); rfl.append(rr["intact_flip"]); rmf.append(rr["margin_flip"])
                rows.append(dict(prompt_set=setname, layer=int(L), alpha=alpha, direction="null_random",
                                 intact_flip=float(np.mean(rfl)), margin_flip=float(np.mean(rmf)),
                                 null_intact_p95=float(np.percentile(rfl, 95)),
                                 null_margin_p95=float(np.percentile(rmf, 95)), n=len(idxs)))
                logger.info("[%s L%d a=%.1f] concept m=%.2f/i=%.2f | surface m=%.2f/i=%.2f | null m=%.2f(p95 %.2f)",
                            setname, L, alpha,
                            rows[-3]["margin_flip"], rows[-3]["intact_flip"],
                            rows[-2]["margin_flip"], rows[-2]["intact_flip"],
                            rows[-1]["margin_flip"], rows[-1]["null_margin_p95"])

    with open(out / "injection_contest.csv", "w", newline="") as f:
        flds = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(f, fieldnames=flds); w.writeheader(); [w.writerow(r) for r in rows]

    # ---------- verdict ----------
    def best(setname, dname, metric):
        vals = [r[metric] for r in rows if r["prompt_set"] == setname and r["direction"] == dname and not np.isnan(r[metric])]
        return max(vals) if vals else float("nan")
    print("\n" + "=" * 100)
    print("INJECTION CONTEST — does CONCEPT, SURFACE, or NOISE heal the failed-beta prompts?")
    print("=" * 100)
    for setname in sets:
        nullm = max([r["null_margin_p95"] for r in rows if r["prompt_set"] == setname and r["direction"] == "null_random"] or [float("nan")])
        print(f"\n[{setname}]  (best over layers/strengths; null p95 margin-flip ~{nullm:.2f})")
        for dname in ["concept_wres", "surface_dsurf", "null_random"]:
            print(f"   {dname:14s}: max margin-flip={best(setname,dname,'margin_flip'):.2f}  max intact-flip={best(setname,dname,'intact_flip'):.2f}")
        cs = best(setname, "concept_wres", "margin_flip"); su = best(setname, "surface_dsurf", "margin_flip")
        if su > nullm + 0.1 and cs <= nullm + 0.1:
            print(f"   => SURFACE heals (>{nullm:.2f}), CONCEPT does not -> the model listens to FORM, not meaning.")
        elif cs > nullm + 0.1 and su > nullm + 0.1:
            print(f"   => both move the answer; compare against the geometry note (cos(w_res,d_surf)) and the null carefully.")
        elif cs <= nullm + 0.1 and su <= nullm + 0.1:
            print(f"   => neither beats null at these strengths -> injection does not heal here.")
    print(f"\nGeometry: |cos(w_res, d_surf)| per layer = " +
          ", ".join(f"L{L}:{dirs[L]['cos_surf_wres']:.2f}" for L in layers))
    print("Caveat: margin-flip is the relative metric, intact-flip the behavioural one; null = norm-matched random "
          "directions; injection is added at all positions at the given layer. Suppression of the competitor (exp 2) "
          "is a SEPARATE run pending the answer-order control (95).")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/injection_contest")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[14, 17, 20, 24])
    p.add_argument("--alphas", type=float, nargs="*", default=[2.0, 4.0, 8.0], help="strengths in units of residual RMS")
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
