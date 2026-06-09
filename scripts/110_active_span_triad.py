"""
110_active_span_triad.py   [do decoded/written/used live in the span of the ACTIVE features? same sub-dict?]
========================================================================================================
Single features are ~orthogonal to w_res/u (exp 109), but a direction can be DISTRIBUTED over many features.
The FULL decoder span is overcomplete (163840 >> 2560) so it captures everything trivially -- useless. The
non-trivial object is the span of features that ACTUALLY FIRE. For each layer we:
  - encode the MLP input over prompts -> active feature set A (fires on >= tau fraction of prompts)
  - build decoder sub-matrix D_A (active rows of W_dec) and project each of {w_res, u, delta} onto span(D_A):
        captured(v) = ||P_{span(D_A)} v||^2 / ||v||^2
  - sparse reconstruction size: how many active features (greedy by |cos|) to reach 90% of v
  - principal angles between span(read-features) / span(use-features) sub-spaces
  - NULL: random unit directions projected onto span(D_A) -> baseline capture (since |A| can be large)
If w_res is well captured by the active span but u is captured from a DIFFERENT (high principal-angle)
sub-span, the dictionary encodes 'readable' and 'used' in different active sub-dictionaries.

Needs transcoders (encode + decoder). Heavier. SELF-TEST (no torch): python 110_active_span_triad.py --self_test
"""
from __future__ import annotations
import argparse, csv as _csv, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("active_span")


def unit_raw(v):
    n = np.linalg.norm(v); return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def captured_fraction(V, vhat):
    """fraction of unit vhat's norm captured by the column space of V (d x k) via least squares."""
    # orthonormal basis of span(V)
    Q, _ = np.linalg.qr(V)                       # d x r (r<=k)
    proj = Q @ (Q.T @ vhat)
    return float(proj @ proj) / (float(vhat @ vhat) + 1e-30), Q


def greedy_recon_size(D, vhat, target=0.9, max_k=400):
    """min #atoms (greedy by correlation with residual) to reach target captured fraction."""
    r = vhat.copy(); chosen = []; cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-9)
    for _ in range(min(max_k, cols.shape[1])):
        c = np.abs(cols.T @ r); j = int(np.argmax(c)); chosen.append(j)
        Q, _ = np.linalg.qr(D[:, chosen])
        r = vhat - Q @ (Q.T @ vhat)
        cap = 1.0 - float(r @ r) / (float(vhat @ vhat) + 1e-30)
        if cap >= target:
            return len(chosen), cap
    cap = 1.0 - float(r @ r) / (float(vhat @ vhat) + 1e-30)
    return len(chosen), cap


def principal_angles(Qa, Qb):
    """principal angles (deg) between two orthonormal bases; return min, mean."""
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False); s = np.clip(s, -1, 1)
    ang = np.degrees(np.arccos(s)); return float(ang.min()), float(ang.mean())


def self_test():
    rng = np.random.default_rng(0); d, k = 64, 20
    D = rng.standard_normal((d, k))
    v_in = unit_raw(D @ rng.standard_normal(k))           # lies in span(D)
    v_out = unit_raw(rng.standard_normal(d))              # generic, mostly outside a 20-dim span
    cap_in, _ = captured_fraction(D, v_in); cap_out, _ = captured_fraction(D, v_out)
    assert cap_in > 0.99 and cap_out < 0.6, f"in-span captured ~1 ({cap_in:.2f}), generic less ({cap_out:.2f})"
    nsz, cap = greedy_recon_size(D, v_in, 0.9); assert cap >= 0.9
    print(f"[self_test] OK — span capture ({cap_in:.2f} vs {cap_out:.2f}); greedy reached {cap:.2f} in {nsz} atoms.")


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True, trust_remote_code=True).to(args.device).eval()
    bm = model.model; blocks = bm.layers; n_layers = len(blocks); last = n_layers - 1
    d = model.config.hidden_size
    a_id = tok.encode(args.alpha_answer, add_special_tokens=False)[0]
    b_id = tok.encode(args.beta_answer, add_special_tokens=False)[0]
    layers = sorted({L for L in args.layers if 0 <= L < n_layers}) if args.layers != [-1] else list(range(n_layers))

    prompts = [json.loads(l) for l in open(args.prompts)]
    Pn = len(prompts)
    y = np.array([1 if p["correct_answer"].strip() == "beta" else 0 for p in prompts])
    fams = [p["surface_family"] for p in prompts]
    ufam = sorted(set(fams)); rng.shuffle(ufam)
    tr = np.array([f in set(ufam[:int(round(len(ufam) * args.train_frac))]) for f in fams])

    by = {0: {}, 1: {}}
    for i, p in enumerate(prompts):
        by[int(y[i])].setdefault(p["surface_family"], []).append(i)
    def exemplar(cls, avoid):
        fl = [f for f in by[cls] if f != avoid] or list(by[cls]); f = fl[rng.integers(len(fl))]
        j = by[cls][f][rng.integers(len(by[cls][f]))]
        return f"{prompts[j]['prompt']}\nAnswer (alpha or beta):{args.beta_answer if cls==1 else args.alpha_answer}"
    def forced(i):
        fam = prompts[i]["surface_family"]; ea, eb = exemplar(0, fam), exemplar(1, fam)
        s = [ea, eb] if i % 2 == 0 else [eb, ea]
        return s[0] + "\n\n" + s[1] + "\n\n" + prompts[i]["prompt"] + "\nAnswer (alpha or beta):"
    def tap(L): return blocks[L + 1] if L < last else bm.norm
    def mlp_in_module(L): return blocks[L].mlp           # transcoder input = MLP input (pre-hook on mlp)

    # capture: residual + grad (for directions) AND mlp-input (for encode), answer position
    H = {L: np.zeros((Pn, d), np.float32) for L in layers}
    G = {L: np.zeros((Pn, d), np.float32) for L in layers}
    MIN = {L: np.zeros((Pn, d), np.float32) for L in layers}
    for p_ in model.parameters(): p_.requires_grad_(True)
    logger.info("capturing residual+grad+mlp_input over %d prompts; layers=%s ...", Pn, layers)
    for i in range(Pn):
        enc = tok([forced(i)], return_tensors="pt").to(args.device); keep, kmin, hs = {}, {}, []
        for L in layers:
            def mk(L=L):
                def pre(m, a): a[0].retain_grad(); keep[L] = a[0]; return None
                return pre
            def mkm(L=L):
                def pre(m, a): kmin[L] = a[0].detach()[0, -1, :].float().cpu().numpy(); return None
                return pre
            hs.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
            hs.append(mlp_in_module(L).register_forward_pre_hook(mkm(), with_kwargs=False))
        try:
            lo = model(**enc, use_cache=False).logits[0, -1, :]
            (lo[b_id] - lo[a_id]).backward()
            for L in layers:
                t = keep[L]; H[L][i] = t.detach()[0, -1, :].float().cpu().numpy()
                G[L][i] = t.grad[0, -1, :].float().cpu().numpy() if t.grad is not None else 0.0
                MIN[L][i] = kmin[L]
        finally:
            for h in hs: h.remove()
        model.zero_grad(set_to_none=True)
        if (i + 1) % 150 == 0: logger.info("  %d/%d", i + 1, Pn)
    for p_ in model.parameters(): p_.requires_grad_(False)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcoder"))
    from transcoder import load_transcoder_set
    ts = load_transcoder_set(args.transcoder_set, device=args.device, dtype=torch.bfloat16, lazy_load=True)

    rows = []
    for L in layers:
        # encode active features over prompts
        acts = np.zeros((Pn, 0))
        mlp_in = torch.tensor(MIN[L], dtype=torch.bfloat16, device=args.device)
        with torch.no_grad():
            a = ts[L].encode(mlp_in)                       # (Pn, F) jumprelu acts
        a = a.float().cpu().numpy()
        fire_rate = (a > 0).mean(0)                        # per-feature fraction of prompts active
        active = np.where(fire_rate >= args.tau)[0]
        if len(active) < 3:
            logger.info("L%02d: only %d active features (tau=%.2f) -> skipping span", L, len(active), args.tau)
            continue
        Wdec = ts[L].W_dec.detach().float().cpu().numpy()  # (F, d)
        D = Wdec[active].T                                  # (d, |A|)
        wres = fisher_axis(H[L][tr].astype(np.float64), y[tr], args.shrink)
        u = unit_raw(G[L][tr].astype(np.float64).mean(0))
        delta = unit_raw(H[L][y == 1].astype(np.float64).mean(0) - H[L][y == 0].astype(np.float64).mean(0))
        cap_w, Qw_full = captured_fraction(D, wres)
        cap_u, _ = captured_fraction(D, u)
        cap_d, _ = captured_fraction(D, delta)
        # null: random directions
        caps_rand = []
        for _ in range(args.n_null):
            cr, _ = captured_fraction(D, unit_raw(rng.standard_normal(d))); caps_rand.append(cr)
        cap_rand = float(np.mean(caps_rand))
        # sparse reconstruction sizes
        nz_w, capk_w = greedy_recon_size(D, wres, args.recon_target, args.max_atoms)
        nz_u, capk_u = greedy_recon_size(D, u, args.recon_target, args.max_atoms)
        # principal angles between the greedy read- and use- sub-spans (top-r atoms each)
        def topatoms(vhat, r):
            cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-9)
            idx = np.argsort(np.abs(cols.T @ vhat))[::-1][:r]
            Q, _ = np.linalg.qr(D[:, idx]); return Q
        r = args.angle_rank
        Qw = topatoms(wres, r); Qu = topatoms(u, r)
        pa_min, pa_mean = principal_angles(Qw, Qu)
        rows.append(dict(layer=int(L), n_active=int(len(active)),
                         cap_wres=cap_w, cap_u=cap_u, cap_delta=cap_d, cap_random_null=cap_rand,
                         recon_atoms_wres=nz_w, recon_cap_wres=capk_w, recon_atoms_u=nz_u, recon_cap_u=capk_u,
                         principal_angle_min_read_use=pa_min, principal_angle_mean_read_use=pa_mean))
        logger.info("L%02d |A|=%4d | cap: wres=%.2f u=%.2f δ=%.2f (null=%.2f) | recon90: wres=%d u=%d | angle(read,use)=%.0f°min %.0f°mean",
                    L, len(active), cap_w, cap_u, cap_d, cap_rand, nz_w, nz_u, pa_min, pa_mean)
        del Wdec, D; torch.cuda.empty_cache()

    with open(out / "active_span_triad.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]

    print("\n" + "=" * 100)
    print("ACTIVE-FEATURE SPAN vs TRIAD — are decoded/written/used in the active sub-dictionary? same sub-span?")
    print("=" * 100)
    print("layer | |A| | cap(w_res) | cap(u) | cap(delta) | NULL | read-vs-use angle (min/mean deg)")
    for r in rows:
        print(f"  L{r['layer']:02d} | {r['n_active']:4d} |   {r['cap_wres']:.2f}    |  {r['cap_u']:.2f}  |   {r['cap_delta']:.2f}    "
              f"| {r['cap_random_null']:.2f} |  {r['principal_angle_min_read_use']:.0f} / {r['principal_angle_mean_read_use']:.0f}")
    print("\nReading: compare cap(w_res)/cap(u)/cap(delta) to the random NULL (an overcomplete-ish active span captures")
    print("a lot by default). A direction is 'in the active dictionary' only if its capture EXCEEDS the null. Equal")
    print("read/use principal angle near 90 deg => readable and used live in different active sub-spans. recon90 atoms")
    print("= how distributed each axis is (large => not localized). Saved active_span_triad.csv")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--prompts", default="physics_decay_type_probe_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/active_span")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha_answer", default=" alpha")
    p.add_argument("--beta_answer", default=" beta")
    p.add_argument("--layers", type=int, nargs="*", default=[13, 17, 21, 24, 28, 32, 35])
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--recon_target", type=float, default=0.9)
    p.add_argument("--max_atoms", type=int, default=400)
    p.add_argument("--angle_rank", type=int, default=20)
    p.add_argument("--n_null", type=int, default=8)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test: self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
