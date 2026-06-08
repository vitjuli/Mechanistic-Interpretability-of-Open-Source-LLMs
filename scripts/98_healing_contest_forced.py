"""
98_healing_contest_forced.py   [clean four-way healing contest: forced format, null-calibrated, intact-first]
=============================================================================================================
Scripts 96/97 hit an artifact: at large injection norms even a RANDOM direction flips the margin (the
null reached margin-flip ~1.0), and intact-flip was 0 everywhere because the BASE format almost never
emits alpha/beta as top-1. This script fixes both:

  (1) FORCED format (the exp-90 balanced 2-shot template) so top-1 is actually ' alpha'/' beta' and
      intact-flip is a valid behavioural metric.
  (2) SMALL strength sweep with the null as a yardstick: report the largest alpha at which a random
      direction is still inert (null intact-flip <= thr); only within that window is a direction's
      effect meaningful.
  (3) intact-flip (top-1 becomes ' beta') is PRIMARY; margin-flip is secondary, always shown beside the
      null p95.

Contest on forced-format failed-beta prompts (true beta, forced top-1 = alpha), two sets reported
separately (all failed-beta; property-framed failed-beta):
  INJECTION  add c*dhat at a mid layer (all positions), c in --alphas * residual RMS:
     concept_wres  : Fisher(train, TRUE labels), +beta            -> prediction: does NOT heal
     surface_dsurf : mean(process-beta) - mean(property-beta)     -> diagnostic: heals => form, not meaning
     null_random   : norm-matched random direction                -> the yardstick
  SUPPRESSION  project out the competitor d_error = mean(failed-beta) - mean(succeeded-beta) (rank-1),
               vs random-direction projection; concept-preservation checked (cos(d_error,w_res), w_res proj).

SELF-TEST (no torch):  python 98_healing_contest_forced.py --self_test
"""

from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("heal_forced")


# ---------------- shared numpy core ----------------
def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def diff_means(H, pos, neg):
    return unit_raw(H[pos].mean(0) - H[neg].mean(0))


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


def build_forced_prompt(target, ex_alpha, ex_beta, suffix, flip_order=False):
    a_block = f"{ex_alpha}{suffix} alpha"
    b_block = f"{ex_beta}{suffix} beta"
    first, second = (b_block, a_block) if flip_order else (a_block, b_block)
    return f"{first}\n\n{second}\n\n{target}{suffix}"


def calib_window(alphas, null_by_alpha, thr):
    """Largest alpha whose null intact-flip <= thr (the valid window). None if even smallest exceeds."""
    ok = [a for a in alphas if null_by_alpha.get(a, 1.0) <= thr]
    return max(ok) if ok else None


def self_test():
    rng = np.random.default_rng(0)
    # forced builder
    fp = build_forced_prompt("TGT", "EA", "EB", "\nAnswer (alpha or beta):")
    assert fp.count("Answer (alpha or beta):") == 3 and fp.rstrip().endswith("Answer (alpha or beta):")
    assert " alpha" in fp and " beta" in fp
    fp2 = build_forced_prompt("TGT", "EA", "EB", "\nA:", flip_order=True)
    assert fp2.index(" beta") < fp2.index(" alpha")
    # directions
    d, n = 16, 200
    y = np.array([0, 1] * (n // 2))
    H = 0.2 * rng.standard_normal((n, d)) + y[:, None] * unit_raw(rng.standard_normal(d))[None, :]
    w = fisher_axis(H, y); assert (H @ w).std() > 0
    proc = (y == 1) & (rng.random(n) > 0.5); prop = (y == 1) & (~proc)
    ds = diff_means(H, proc, prop); assert abs(np.linalg.norm(ds) - 1) < 1e-6
    # calibration window
    alphas = [0.25, 0.5, 1.0, 2.0]
    nulls = {0.25: 0.0, 0.5: 0.02, 1.0: 0.2, 2.0: 0.9}
    assert calib_window(alphas, nulls, 0.05) == 0.5
    assert calib_window(alphas, {0.25: 0.5}, 0.05) is None
    print("[self_test] OK — forced builder, directions, calibration-window logic pass.")


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
    bm = model.model; blocks = bm.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers})
    logger.info("forced healing contest; layers %s; alphas %s; suffix %r", layers, args.alphas, args.suffix)

    prompts = [json.loads(l) for l in open(args.prompts)]
    nP = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = sorted({p["surface_family"] for p in prompts})
    rng.shuffle(fams); train_fams = set(fams[: int(round(len(fams) * args.train_frac))])
    is_train = np.array([p["surface_family"] in train_fams for p in prompts])

    def S(x):
        return str(x) if x is not None else "NA"

    bucket = np.array([framing_bucket(S(p.get("relation_type")), S(p.get("concept_route")), S(p.get("prompt_format")))
                       for p in prompts])

    # forced prompts (exemplars from OTHER families, balanced, order alternated)
    pool_a = [p for p in prompts if p["surface_family"] in train_fams and p["correct_answer"].strip() == "alpha"]
    pool_b = [p for p in prompts if p["surface_family"] in train_fams and p["correct_answer"].strip() == "beta"]
    forced = []
    for i, p in enumerate(prompts):
        ea = next((q for q in (pool_a[int(rng.integers(len(pool_a)))] for _ in range(8))
                   if q["surface_family"] != p["surface_family"]), pool_a[0])["prompt"]
        eb = next((q for q in (pool_b[int(rng.integers(len(pool_b)))] for _ in range(8))
                   if q["surface_family"] != p["surface_family"]), pool_b[0])["prompt"]
        forced.append(build_forced_prompt(p["prompt"], ea, eb, args.suffix, flip_order=bool(i % 2)))

    def tap_module(L):
        return blocks[L + 1] if L < last else bm.norm

    # capture forced residuals + forced prediction
    H = {L: np.zeros((nP, d), np.float32) for L in layers}
    margin = np.zeros(nP); top1 = np.zeros(nP, dtype=np.int64)
    logger.info("capturing forced residuals + predictions over %d prompts...", nP)
    with torch.no_grad():
        for i in range(nP):
            enc = tok([forced[i]], return_tensors="pt").to(args.device)
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
            margin[i] = float(lo[b_id] - lo[a_id]); top1[i] = int(lo.argmax())
            if (i + 1) % 150 == 0:
                logger.info("  %d/%d", i + 1, nP)

    intact_clean = np.isin(top1, [a_id, b_id])
    acc = float(np.mean((top1 == b_id) == (y == 1)))
    logger.info("forced clean: intact-rate=%.3f  acc=%.3f", intact_clean.mean(), acc)
    if intact_clean.mean() < 0.5:
        logger.warning("forced intact-rate < 0.5 -> format not forcing answers; intact metric weak")

    failed_beta = (y == 1) & (top1 == a_id)                 # forced: true beta, said alpha
    succ_beta = (y == 1) & (top1 == b_id)
    prop_failed = failed_beta & (bucket == "property")
    logger.info("forced failed_beta=%d  succ_beta=%d  property-failed=%d", failed_beta.sum(), succ_beta.sum(), prop_failed.sum())

    # directions per layer
    info = {}
    for L in layers:
        w_res = fisher_axis(H[L][is_train].astype(np.float64), y[is_train], args.shrink)
        d_surf = diff_means(H[L].astype(np.float64), (y == 1) & (bucket == "process"), (y == 1) & (bucket == "property"))
        d_err = unit_raw(H[L][failed_beta].astype(np.float64).mean(0) - H[L][succ_beta].astype(np.float64).mean(0))
        info[L] = dict(w_res=w_res, d_surf=d_surf, d_err=d_err,
                       rms=float(np.sqrt((H[L] ** 2).sum(1).mean())),
                       cos_surf=float(abs(w_res @ d_surf)), cos_err=float(abs(w_res @ d_err)))
        logger.info("  L%d rms=%.1f |cos(w_res,d_surf)|=%.3f |cos(w_res,d_err)|=%.3f",
                    L, info[L]["rms"], info[L]["cos_surf"], info[L]["cos_err"])

    # ---- hooks ----
    def add_hook(vec):
        v = torch.tensor(vec, dtype=torch.float32, device=args.device)
        def pre(m, a):
            a[0][:, :, :] = a[0] + v
            return (a[0],) + tuple(a[1:])
        return pre

    def proj_hook(dhat):
        v = torch.tensor(dhat, dtype=torch.float32, device=args.device)
        def pre(m, a):
            h = a[0]; a[0][:, :, :] = h - (h @ v).unsqueeze(-1) * v
            return (a[0],) + tuple(a[1:])
        return pre

    def run_set(idxs, L, hookfn):
        if len(idxs) == 0:
            return dict(intact_flip=float("nan"), margin_flip=float("nan"))
        intact = mflip = 0
        for i in idxs:
            enc = tok([forced[i]], return_tensors="pt").to(args.device)
            h = tap_module(L).register_forward_pre_hook(hookfn, with_kwargs=False)
            try:
                with torch.no_grad():
                    lo = model(**enc, use_cache=False).logits[0, -1, :]
            finally:
                h.remove()
            intact += int(int(lo.argmax()) == b_id)
            mflip += int(float(lo[b_id] - lo[a_id]) > 0)
        return dict(intact_flip=intact / len(idxs), margin_flip=mflip / len(idxs))

    sets = {"all_failed_beta": np.where(failed_beta)[0].tolist(),
            "property_failed_beta": np.where(prop_failed)[0].tolist()}
    rows = []

    # ---------- INJECTION (small-alpha sweep, null-calibrated) ----------
    for setname, idxs in sets.items():
        for L in layers:
            rms = info[L]["rms"]
            null_intact = {}
            for al in args.alphas:
                sc = al * rms
                for dname, vec in [("concept_wres", info[L]["w_res"] * sc), ("surface_dsurf", info[L]["d_surf"] * sc)]:
                    r = run_set(idxs, L, add_hook(vec))
                    rows.append(dict(mode="inject", prompt_set=setname, layer=int(L), alpha=al, direction=dname,
                                     n=len(idxs), **r))
                ni, nm = [], []
                for _ in range(args.n_random):
                    rv = unit_raw(rng.standard_normal(d)) * sc
                    rr = run_set(idxs, L, add_hook(rv)); ni.append(rr["intact_flip"]); nm.append(rr["margin_flip"])
                null_intact[al] = float(np.percentile(ni, 95))
                rows.append(dict(mode="inject", prompt_set=setname, layer=int(L), alpha=al, direction="null_random",
                                 n=len(idxs), intact_flip=float(np.mean(ni)), margin_flip=float(np.mean(nm)),
                                 null_intact_p95=float(np.percentile(ni, 95)), null_margin_p95=float(np.percentile(nm, 95))))
            win = calib_window(args.alphas, null_intact, args.null_thr)
            for al in args.alphas:
                tag = " *" if al == win else ""
                cc = next(r for r in rows if r["mode"]=="inject" and r["prompt_set"]==setname and r["layer"]==L and r["alpha"]==al and r["direction"]=="concept_wres")
                su = next(r for r in rows if r["mode"]=="inject" and r["prompt_set"]==setname and r["layer"]==L and r["alpha"]==al and r["direction"]=="surface_dsurf")
                logger.info("[inj %s L%d a=%.2f] concept i=%.2f surface i=%.2f null_i_p95=%.2f%s",
                            setname, L, al, cc["intact_flip"], su["intact_flip"], null_intact[al], tag)

    # ---------- SUPPRESSION (project out competitor; rank-1, no alpha) ----------
    for setname, idxs in sets.items():
        for L in layers:
            de = run_set(idxs, L, proj_hook(info[L]["d_err"]))
            ni, nm = [], []
            for _ in range(args.n_random):
                rr = run_set(idxs, L, proj_hook(unit_raw(rng.standard_normal(d))))
                ni.append(rr["intact_flip"]); nm.append(rr["margin_flip"])
            rows.append(dict(mode="suppress", prompt_set=setname, layer=int(L), alpha=float("nan"), direction="d_error",
                             n=len(idxs), cos_err=info[L]["cos_err"], **de))
            rows.append(dict(mode="suppress", prompt_set=setname, layer=int(L), alpha=float("nan"), direction="null_random",
                             n=len(idxs), intact_flip=float(np.mean(ni)), margin_flip=float(np.mean(nm)),
                             null_intact_p95=float(np.percentile(ni, 95)), null_margin_p95=float(np.percentile(nm, 95))))
            logger.info("[supp %s L%d] d_error i=%.2f m=%.2f | null i_p95=%.2f (cos(d_err,w_res)=%.2f)",
                        setname, L, de["intact_flip"], de["margin_flip"], float(np.percentile(ni, 95)), info[L]["cos_err"])

    with open(out / "healing_contest_forced.csv", "w", newline="") as f:
        flds = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(f, fieldnames=flds); w.writeheader(); [w.writerow(r) for r in rows]

    # ---------- verdict (intact-flip, within the null-inert window) ----------
    print("\n" + "=" * 104)
    print("HEALING CONTEST (FORCED, null-calibrated, intact-first) — does anything actually make the model say beta?")
    print("=" * 104)
    print(f"forced clean intact-rate {intact_clean.mean():.2f}, acc {acc:.2f}")
    for setname in sets:
        print(f"\n[{setname}]  (n={len(sets[setname])})")
        # injection within the valid window
        for L in layers:
            nullmap = {r["alpha"]: r["null_intact_p95"] for r in rows
                       if r["mode"]=="inject" and r["prompt_set"]==setname and r["layer"]==L and r["direction"]=="null_random"}
            win = calib_window(args.alphas, nullmap, args.null_thr)
            if win is None:
                print(f"  INJECT L{L}: no strength keeps the null inert (null intact-flip > {args.null_thr} even at alpha={min(args.alphas)}) -> uninformative here")
                continue
            cc = next(r for r in rows if r["mode"]=="inject" and r["prompt_set"]==setname and r["layer"]==L and r["alpha"]==win and r["direction"]=="concept_wres")
            su = next(r for r in rows if r["mode"]=="inject" and r["prompt_set"]==setname and r["layer"]==L and r["alpha"]==win and r["direction"]=="surface_dsurf")
            verdict = ("SURFACE heals, concept doesn't" if su["intact_flip"] > nullmap[win] + 0.1 and cc["intact_flip"] <= nullmap[win] + 0.1 else
                       "CONCEPT heals" if cc["intact_flip"] > nullmap[win] + 0.1 and su["intact_flip"] <= nullmap[win] + 0.1 else
                       "both heal" if cc["intact_flip"] > nullmap[win] + 0.1 and su["intact_flip"] > nullmap[win] + 0.1 else
                       "neither heals")
            print(f"  INJECT L{L} @valid alpha={win}: concept intact={cc['intact_flip']:.2f}  surface intact={su['intact_flip']:.2f}  (null {nullmap[win]:.2f}) -> {verdict}")
        # suppression
        for L in layers:
            de = next(r for r in rows if r["mode"]=="suppress" and r["prompt_set"]==setname and r["layer"]==L and r["direction"]=="d_error")
            nu = next(r for r in rows if r["mode"]=="suppress" and r["prompt_set"]==setname and r["layer"]==L and r["direction"]=="null_random")
            rec = de["intact_flip"] > nu["null_intact_p95"] + 0.1
            print(f"  SUPPRESS L{L}: d_error intact={de['intact_flip']:.2f} (null {nu['null_intact_p95']:.2f}) cos(d_err,w_res)={de.get('cos_err',float('nan')):.2f}"
                  f"{'  <= concept surfaces' if rec else ''}")
    print("\nReading: intact-flip is the behavioural metric; only strengths where the RANDOM null stays ~0 are valid. "
          "If nothing beats null on intact-flip in the valid window, nothing heals behaviourally -- a clean negative "
          "(surface-only route, concept causally inert even to repair).")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/healing_forced")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--suffix", default="\nAnswer (alpha or beta):")
    p.add_argument("--layers", type=int, nargs="*", default=[17, 20, 24])
    p.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 1.0, 1.5, 2.0], help="injection strengths in residual-RMS units")
    p.add_argument("--n_random", type=int, default=4)
    p.add_argument("--null_thr", type=float, default=0.05, help="max null intact-flip for a strength to count as valid")
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
