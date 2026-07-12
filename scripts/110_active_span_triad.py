"""
110_active_span_triad.py   [active-span analysis + per-feature triad map (read/write/used)]
========================================================================================================
Two outputs from one capture pass over the prompts:

  (A) AGGREGATE per-layer span analysis  →  active_span_triad.csv
      - encode MLP input over prompts → active feature set A (fires on ≥ tau fraction of prompts)
      - build decoder sub-matrix D_A and project each of {w_res, u, delta} onto span(D_A):
            captured(v) = ||P_{span(D_A)} v||^2 / ||v||^2
      - sparse reconstruction size: how many active features (greedy by |cos|) to reach 90% of v
      - principal angles between span(read-features) / span(use-features) sub-spaces
      - NULL: random unit directions projected onto span(D_A) → baseline capture
      → 1 row per layer.

  (B) PER-FEATURE triad map  →  feature_triad_alignment.csv  +  triad_type_summary.csv
      For each ACTIVE feature f at each layer ℓ, record cos(W_dec[f], v) for v ∈ {w_res, u, δ}.
      Classify into 8 'triad' buckets at threshold T = --triad_threshold (default 0.05):
            none | wres_only | u_only | delta_only |
            wres+u | wres+delta | u+delta | all_three
      Lets you ask: do features write to ONE axis or several? Are there 'shared writers'?
      → |A_ℓ| rows per layer (feature CSV) + 1 row per layer (summary CSV).

If w_res is well captured by the active span but u is captured from a DIFFERENT (high principal-angle)
sub-span, AND the per-feature map shows mostly single-axis loaders, the dictionary encodes 'readable',
'written', and 'used' in three structurally distinct active sub-dictionaries.

Needs transcoders (encode + decoder). SELF-TEST (no torch): python 110_active_span_triad.py --self_test
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


def classify_triad(loads_w, loads_u, loads_d):
    """8-bucket classification by which axes a feature loads above threshold."""
    code = (int(loads_w) << 2) | (int(loads_u) << 1) | int(loads_d)
    return {
        0b000: "none", 0b100: "wres_only", 0b010: "u_only", 0b001: "delta_only",
        0b110: "wres+u", 0b101: "wres+delta", 0b011: "u+delta", 0b111: "all_three",
    }[code]


def self_test():
    rng = np.random.default_rng(0); d, k = 64, 20
    D = rng.standard_normal((d, k))
    v_in = unit_raw(D @ rng.standard_normal(k))           # lies in span(D)
    v_out = unit_raw(rng.standard_normal(d))              # generic, mostly outside a 20-dim span
    cap_in, _ = captured_fraction(D, v_in); cap_out, _ = captured_fraction(D, v_out)
    assert cap_in > 0.99 and cap_out < 0.6, f"in-span captured ~1 ({cap_in:.2f}), generic less ({cap_out:.2f})"
    nsz, cap = greedy_recon_size(D, v_in, 0.9); assert cap >= 0.9
    # triad classifier: plant 3 distinct best-loaders, rest should be 'none'
    F = 80
    Wdec = rng.standard_normal((F, d))
    Wdec /= np.linalg.norm(Wdec, axis=1, keepdims=True) + 1e-9
    wres = Wdec[0].copy(); u_ = Wdec[1].copy(); delta = Wdec[2].copy()
    T = 0.5
    types = [classify_triad(abs(Wdec[i] @ wres) > T,
                             abs(Wdec[i] @ u_) > T,
                             abs(Wdec[i] @ delta) > T) for i in range(F)]
    counts = {t: types.count(t) for t in set(types)}
    assert "wres_only" in counts and "u_only" in counts and "delta_only" in counts and counts.get("none", 0) >= F // 2, counts
    print(f"[self_test] OK — span ({cap_in:.2f} vs {cap_out:.2f}, greedy {cap:.2f}/{nsz}); triad {counts}")


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

    rows = []           # aggregate per-layer
    feat_rows = []      # per-feature (long format)
    triad_summary = []  # per-layer triad-type counts
    for L in layers:
        # encode active features over prompts
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

        # ── Per-feature triad map (cheap: ~ms) ────────────────────────────────
        D_unit = Wdec[active] / (np.linalg.norm(Wdec[active], axis=1, keepdims=True) + 1e-12)
        cos_w = D_unit @ wres
        cos_u_ = D_unit @ u
        cos_d_ = D_unit @ delta
        abs_sum = np.abs(cos_w) + np.abs(cos_u_) + np.abs(cos_d_)
        max_cos = np.maximum.reduce([np.abs(cos_w), np.abs(cos_u_), np.abs(cos_d_)])
        # per-feature stats
        a_act_cols = a[:, active]                          # (Pn, |A|)
        mean_act_all = a_act_cols.mean(0)
        fire_count = (a_act_cols > 0).sum(0).astype(float)
        sum_when_fired = np.where(a_act_cols > 0, a_act_cols, 0.0).sum(0)
        mean_act_fired = np.where(fire_count > 0, sum_when_fired / np.maximum(fire_count, 1), 0.0)
        # attribution proxy: mean_act × <d, g>
        g_mean = G[L][tr].astype(np.float64).mean(0)
        proj_d_g = Wdec[active] @ g_mean                   # raw, not unit-normalised
        attr_proxy = mean_act_all * proj_d_g
        # per-feature ranks (smaller = stronger loader)
        rank_w = np.argsort(-np.abs(cos_w)).argsort()
        rank_u = np.argsort(-np.abs(cos_u_)).argsort()
        rank_d = np.argsort(-np.abs(cos_d_)).argsort()
        T = args.triad_threshold
        loads_w = np.abs(cos_w) > T
        loads_u = np.abs(cos_u_) > T
        loads_d = np.abs(cos_d_) > T
        for k, f in enumerate(active):
            feat_rows.append({
                "layer": int(L), "feature": int(f),
                "fire_rate": float(fire_rate[f]),
                "mean_act": float(mean_act_all[k]),
                "mean_act_fired": float(mean_act_fired[k]),
                "cos_d_wres": float(cos_w[k]),
                "cos_d_u": float(cos_u_[k]),
                "cos_d_delta": float(cos_d_[k]),
                "abs_sum_cos": float(abs_sum[k]),
                "max_cos": float(max_cos[k]),
                "attribution_proxy": float(attr_proxy[k]),
                "rank_wres": int(rank_w[k]),
                "rank_u": int(rank_u[k]),
                "rank_delta": int(rank_d[k]),
                "loads_wres": bool(loads_w[k]),
                "loads_u": bool(loads_u[k]),
                "loads_delta": bool(loads_d[k]),
                "triad_type": classify_triad(loads_w[k], loads_u[k], loads_d[k]),
            })
        # per-layer summary
        types = [classify_triad(loads_w[k], loads_u[k], loads_d[k]) for k in range(len(active))]
        cnt = {t: types.count(t) for t in
               ["none", "wres_only", "u_only", "delta_only",
                "wres+u", "wres+delta", "u+delta", "all_three"]}
        n_single = cnt["wres_only"] + cnt["u_only"] + cnt["delta_only"]
        n_multi = cnt["wres+u"] + cnt["wres+delta"] + cnt["u+delta"] + cnt["all_three"]
        n_any = n_single + n_multi
        triad_summary.append({
            "layer": int(L), "n_active": int(len(active)),
            **{f"n_{k}": int(v) for k, v in cnt.items()},
            "single_axis_frac": float(n_single / max(n_any, 1)),
            "multi_axis_frac": float(n_multi / max(n_any, 1)),
            "max_cos_wres": float(np.abs(cos_w).max()),
            "max_cos_u": float(np.abs(cos_u_).max()),
            "max_cos_delta": float(np.abs(cos_d_).max()),
            "n_feat_above_T_wres": int(loads_w.sum()),
            "n_feat_above_T_u": int(loads_u.sum()),
            "n_feat_above_T_delta": int(loads_d.sum()),
        })
        logger.info("  triad L%02d (T=%.2f): loads w=%d u=%d δ=%d | single=%d multi=%d all3=%d",
                    L, T, int(loads_w.sum()), int(loads_u.sum()), int(loads_d.sum()),
                    n_single, n_multi, cnt["all_three"])
        del Wdec, D, D_unit; torch.cuda.empty_cache()

    with open(out / "active_span_triad.csv", "w", newline="") as f:
        w_ = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows]
    if feat_rows:
        with open(out / "feature_triad_alignment.csv", "w", newline="") as f:
            w_ = _csv.DictWriter(f, fieldnames=list(feat_rows[0].keys()))
            w_.writeheader(); [w_.writerow(r) for r in feat_rows]
        logger.info("saved per-feature triad CSV: %d rows", len(feat_rows))
    if triad_summary:
        with open(out / "triad_type_summary.csv", "w", newline="") as f:
            w_ = _csv.DictWriter(f, fieldnames=list(triad_summary[0].keys()))
            w_.writeheader(); [w_.writerow(r) for r in triad_summary]
        logger.info("saved triad-type summary: %d rows", len(triad_summary))

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
    p.add_argument("--triad_threshold", type=float, default=0.05,
                    help="loading threshold T: a feature 'loads' an axis if |cos(d_f, axis)| > T")
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
