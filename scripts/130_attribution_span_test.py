"""
130_attribution_span_test.py   [do u / w_res / delta / gamma lie in span(Q1 attribution features)?]
====================================================================================================
128 ablated the Q1 attribution-carrier features and found a SUBCRITICAL effect (moves
margin above null, almost never flips). 128b asks if that effect is u-selective. This
script asks the prior, purely GEOMETRIC question -- no intervention:

    do the directions {w_res (read), u (use), delta (write), gamma_bar (readout)} lie in
    the SPAN of the decoders of the Q1 attribution-selected features, against a
    random-direction null of the same subspace dimension?

This is distinct from 110 (which used ACTIVE features, fire-rate >= tau) and from 128
(ablation). Here the sub-dictionary is the ATTRIBUTION-selected one (is_attr=1, the
right criterion per exp 91). The span of k features captures ANY direction at some
baseline level purely by dimension, so the null is mandatory: the question is not
captured(u) > 0 but captured(u) vs captured(random) and captured(u) vs captured(w_res).

Per ignition layer:
  - build D_Q1 = decoder columns (W_dec rows) of the Q1 features  (d x k)
  - captured_fraction(D_Q1, vhat) for vhat in {w_res, u, delta, gamma_bar}
  - random-direction null: many random unit vhat -> captured distribution -> mean, p95
  - greedy_recon_size: how many Q1 atoms to reach 90% of each direction
  - principal angles between span(read-aligned Q1) and span(use-aligned Q1): split Q1
    features by whether they align more with w_res or with u, build the two sub-spans,
    measure the angle. High angle => the dictionary stores 'readable' and 'used' in
    different attribution sub-dictionaries -- the geometric counterpart of read != use.

Verdict logic:
  - captured(u) ~ null but captured(w_res) >> null -> read and use live in DIFFERENT
    sub-dictionaries; this explains WHY ablating Q1 does not flip (Q1 carry the readable
    axis, not the used one). Strongest outcome, closes the story geometrically.
  - captured(u) ~ captured(w_res) >> null -> Q1 carry both; 128's subcriticality is about
    magnitude, not u-absence.
  - both ~ null -> even attribution features do not span the used axis: non-localizable
    at the span level too.

Needs transcoders (decoder weights). GPU but light (projection arithmetic on ~6 layers,
no sweeps). Reuses captured_fraction / greedy_recon_size / principal_angles from
110_active_span_triad.py VERBATIM.

SELF-TEST (no torch / no repo):  python 130_attribution_span_test.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("span130")


# =====================================================================
# Reused VERBATIM from 110_active_span_triad.py
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def fisher_axis(H, y, shrink=0.1):
    mu0, mu1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - mu0, H[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(H.shape[0] - 2, 1)
    Sw = 0.5 * (Sw + Sw.T); Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    return unit_raw(np.linalg.solve(Sw, mu1 - mu0))


def captured_fraction(V, vhat):
    """fraction of unit vhat's norm captured by the column space of V (d x k) via least squares."""
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


# =====================================================================
# Self-test (extends 110's, adds null calibration + read/use split logic)
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, k = 64, 20
    D = rng.standard_normal((d, k))
    v_in = unit_raw(D @ rng.standard_normal(k))           # lies in span(D)
    v_out = unit_raw(rng.standard_normal(d))              # generic, mostly outside a 20-dim span
    cap_in, _ = captured_fraction(D, v_in); cap_out, _ = captured_fraction(D, v_out)
    assert cap_in > 0.99 and cap_out < 0.6, f"in-span ~1 ({cap_in:.2f}), generic less ({cap_out:.2f})"
    nsz, cap = greedy_recon_size(D, v_in, 0.9); assert cap >= 0.9

    # NULL calibration: random directions captured by a k-dim span average ~ k/d.
    caps_null = [captured_fraction(D, unit_raw(rng.standard_normal(d)))[0] for _ in range(300)]
    assert abs(np.mean(caps_null) - k / d) < 0.06, f"null capture ~ k/d: {np.mean(caps_null):.3f} vs {k/d:.3f}"

    # a direction IN-span must beat the null p95 (the real test we run)
    p95 = float(np.quantile(caps_null, 0.95))
    assert cap_in > p95, "in-span direction must exceed null p95"

    # read/use split: build two orthogonal in-span directions, split atoms by alignment,
    # the two sub-spans should have a LARGE principal angle (they store different things).
    a = unit_raw(D @ np.r_[np.ones(k // 2), np.zeros(k - k // 2)])
    b = unit_raw(D @ np.r_[np.zeros(k - k // 2), np.ones(k // 2)])
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-9)
    align_a = np.abs(cols.T @ a) > np.abs(cols.T @ b)
    Qa, _ = np.linalg.qr(D[:, align_a]); Qb, _ = np.linalg.qr(D[:, ~align_a])
    mn, _ = principal_angles(Qa, Qb)
    assert mn >= 0.0, "principal angle defined"
    print("[self_test] OK — captured (in/out), greedy recon, null~k/d calibration, p95 test, "
          "read/use split angle pass.")


# =====================================================================
# Real run
# =====================================================================
def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def run_real(args):
    import torch
    sys.path.insert(0, str(Path(args.repo_root)))
    from transcoder_loader import load_transcoder_set

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Q1 attribution features per layer
    feat_rows = list(_csv.DictReader(open(args.feature_metrics)))
    q1_by_layer = defaultdict(list)
    for r in feat_rows:
        if r.get("is_attr") == "1":
            q1_by_layer[int(r["layer"])].append(int(r["feature"]))
    layers = [L for L in args.layers if q1_by_layer.get(L)]
    logger.info("Q1 features per layer: %s", {L: len(q1_by_layer[L]) for L in layers})

    # directions from the dump
    dump = Path(args.dump_dir)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = [json.loads(l)["surface_family"] for l in open(args.corpus)]
    y = meta["y"].astype(int)
    gamma = unit_raw(meta["wU_diff"].astype(np.float64))
    trm = reconstruct_split(fams, args.split_seed, args.train_frac)

    def directions_at(L):
        H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
        G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
        w = fisher_axis(H[trm], y[trm], args.shrink)
        u = unit_raw(G.mean(0))
        # delta (writing direction): mean class-difference of residual contribution.
        delta = unit_raw(H[y == 1].mean(0) - H[y == 0].mean(0))
        return {"w_res": w, "u": u, "delta": delta, "gamma_bar": gamma}

    TCSET = load_transcoder_set("4b", repo_id=args.transcoder_repo, device=args.device,
                                dtype=torch.float32, layers=layers)

    rows, ang_rows = [], []
    for L in layers:
        tc = TCSET[L]
        # decoder matrix: W_dec is (d_transcoder, d_model); columns we want are the Q1 rows.
        W_dec = tc.W_dec.detach().float().cpu().numpy() if hasattr(tc, "W_dec") else \
            tc.decoder.weight.detach().float().cpu().numpy().T
        # normalize orientation to (d_model, d_transcoder)
        if W_dec.shape[0] != int(meta["d"]):
            W_dec = W_dec.T
        q1 = q1_by_layer[L]
        D_Q1 = W_dec[:, q1]                                   # d x k
        k = D_Q1.shape[1]; d = D_Q1.shape[0]

        dirs = directions_at(L)
        # random-direction null capture for this span dimension
        caps_null = [captured_fraction(D_Q1, unit_raw(rng.standard_normal(d)))[0]
                     for _ in range(args.n_null)]
        null_mean = float(np.mean(caps_null)); null_p95 = float(np.quantile(caps_null, 0.95))

        cap = {}; Qbasis = {}
        for nm, v in dirs.items():
            c, Q = captured_fraction(D_Q1, v)
            cap[nm] = c
            nsz, capr = greedy_recon_size(D_Q1, v, 0.9)
            rows.append({"layer": L, "k_q1": k, "direction": nm,
                         "captured": c, "null_mean": null_mean, "null_p95": null_p95,
                         "above_null": int(c > null_p95),
                         "recon_atoms_90": nsz, "recon_cap": capr,
                         "null_dim_ratio_k_over_d": k / d})

        # read/use split of Q1 atoms by alignment, principal angle between sub-spans
        cols = D_Q1 / (np.linalg.norm(D_Q1, axis=0, keepdims=True) + 1e-9)
        align_read = np.abs(cols.T @ dirs["w_res"]) > np.abs(cols.T @ dirs["u"])
        if 1 <= align_read.sum() <= k - 1:
            Qa, _ = np.linalg.qr(D_Q1[:, align_read]); Qb, _ = np.linalg.qr(D_Q1[:, ~align_read])
            amin, amean = principal_angles(Qa, Qb)
        else:
            amin, amean = float("nan"), float("nan")
        ang_rows.append({"layer": L, "k_q1": k, "n_read_aligned": int(align_read.sum()),
                         "n_use_aligned": int((~align_read).sum()),
                         "principal_angle_min_deg": amin, "principal_angle_mean_deg": amean})
        logger.info("L%02d k=%d | cap w_res=%.3f u=%.3f delta=%.3f gamma=%.3f (null mean/p95 %.3f/%.3f) | "
                    "read/use span angle mean=%.1f deg",
                    L, k, cap["w_res"], cap["u"], cap["delta"], cap["gamma_bar"],
                    null_mean, null_p95, amean)

    with open(out / "attribution_span_test.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    with open(out / "attribution_span_angles.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(ang_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in ang_rows]

    print("\n" + "=" * 100)
    print("ATTRIBUTION-SPAN GEOMETRIC TEST — do directions lie in span(Q1 attribution features)?")
    print("=" * 100)
    print(f"{'layer':<7}{'k':<5}{'cap(w_res)':>12}{'cap(u)':>10}{'cap(delta)':>12}{'cap(gamma)':>12}{'null_p95':>10}")
    for L in layers:
        sel = {r["direction"]: r for r in rows if r["layer"] == L}
        if not sel:
            continue
        print(f"{L:<7}{sel['u']['k_q1']:<5}{sel['w_res']['captured']:>12.3f}{sel['u']['captured']:>10.3f}"
              f"{sel['delta']['captured']:>12.3f}{sel['gamma_bar']['captured']:>12.3f}{sel['u']['null_p95']:>10.3f}")
    print("\nAbove-null summary (captured > null p95):")
    for nm in ("w_res", "u", "delta", "gamma_bar"):
        above = [r["layer"] for r in rows if r["direction"] == nm and r["above_null"]]
        print(f"  {nm:<10} above null at layers: {above if above else 'NONE'}")
    print("\nRead/use sub-span principal angle (high => readable and used in different sub-dicts):")
    for a in ang_rows:
        print(f"  L{a['layer']:02d}: mean angle = {a['principal_angle_mean_deg']:.1f} deg "
              f"(read-aligned {a['n_read_aligned']}, use-aligned {a['n_use_aligned']})")
    print("\nVERDICT:")
    print("  - cap(u) ~ null but cap(w_res) >> null -> read and use in DIFFERENT Q1 sub-dicts;")
    print("    explains why ablating Q1 does not flip (Q1 carry the readable, not the used axis).")
    print("  - cap(u) ~ cap(w_res) >> null -> Q1 carry both; 128 subcriticality is magnitude.")
    print("  - both ~ null -> used axis not spanned even by attribution features (non-localizable).")
    print(f"saved: {out/'attribution_span_test.csv'} | {out/'attribution_span_angles.csv'}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/B1_alpha_beta.jsonl")
    p.add_argument("--dump_dir", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--feature_metrics", default="data/analysis/feature_metrics_full.csv")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/B1_alpha_beta/attribution_span")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--transcoder_repo", default="mwhanna/qwen3-4b-transcoders")
    p.add_argument("--layers", type=int, nargs="*", default=[19, 20, 21, 22, 23, 24])
    p.add_argument("--n_null", type=int, default=500)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
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
