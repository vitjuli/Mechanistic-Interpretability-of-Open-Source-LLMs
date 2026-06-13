"""
126_usage_field_strategy.py   [does u encode the computational STRATEGY, not the concept?]
===========================================================================================
Two dumps of the SAME concept (alpha_beta) in two regimes:
  raw       — base model, no scaffold; 123 showed u is BLIND mid-stack (AUC~0.40)
  scaffold  — 2-shot; u DECODES mid-stack (AUC~0.97)
Same concept, same labels, same prompts (modulo scaffold). If u were "the concept
axis" it should be the SAME direction in both. If u encodes HOW the model solves
the task, u should DIFFER between regimes mid-stack and converge only at the readout
(where both -> gamma_bar). This script measures that, per layer:

(A) ANGLE between u_raw and u_scaffold (calibrated against the within-span null —
    in d~2560 even "different" directions are near-orthogonal, so the question is
    whether they are MORE aligned than chance, and where).

(B) USAGE-FIELD STRUCTURE per regime: participation ratio of the per-prompt usage
    gradients (PR of the usage-Gram). Low PR = coherent single-direction usage;
    high PR = the model uses many per-prompt directions (diffuse strategy).

(C) CROSS-REGIME SUBSPACE TEST (the decisive one): does u_raw lie in the SCAFFOLD
    class subspace, and vice versa? Build each regime's Fisher/LDA axis w and the
    class-mean-shift delta; measure cos(u_regimeX, w_regimeY) and the fraction of
    u captured by the other regime's top class-subspace. If u_raw is captured by
    the scaffold concept-subspace but u_scaffold is NOT captured by the raw
    shortcut-subspace (or vice versa), the two regimes are reading DIFFERENT
    structure -> u is strategy-dependent, not concept-fixed.

(D) READOUT CONVERGENCE: cos(u_raw, gamma_bar) and cos(u_scaffold, gamma_bar) per
    layer — both should rise to ~1 at the readout regardless of mid-stack divergence.

All CPU, both dumps already on disk. Aligns the two corpora by canonical class
(y) since prompt sets differ (scaffold corpus is a re-render; raw is the original
538). Directions are computed on each dump's own train split; comparison is
between the resulting axes, which live in the same residual basis.

SELF-TEST (no torch / no repo):  python 126_usage_field_strategy.py --self_test
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("usage126")


# =====================================================================
# Pure-numpy core (exercised by --self_test)
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


def class_delta(H, y):
    return H[y == 1].mean(0) - H[y == 0].mean(0)


def participation_ratio(M):
    """PR of the Gram of rows of M (per-prompt directions): (sum s^2)^2 / sum s^4
    where s are singular values of centered M. High PR = diffuse usage field."""
    Mc = M - M.mean(0)
    s = np.linalg.svd(Mc, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-30))


def span_basis(Hc, rank=None, tol=1e-10):
    _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    r = int((s > s.max() * tol).sum()) if rank is None else min(rank, Vt.shape[0])
    return Vt[:r].T  # (d, r)


def captured_fraction(v, basis):
    """fraction of unit v's energy inside the column span of basis (orthonormal-ish)."""
    Q, _ = np.linalg.qr(basis)
    proj = Q @ (Q.T @ v)
    return float((proj @ proj) / (v @ v + 1e-30))


def abs_cos(a, b):
    return float(abs(unit_raw(a) @ unit_raw(b)))


def within_span_null_p95(Hc, anchor, rng, n=2000):
    V = span_basis(Hc)
    R = rng.standard_normal((n, V.shape[1])) @ V.T
    cc = np.abs(R @ unit_raw(anchor)) / (np.linalg.norm(R, axis=1) + 1e-30)
    return float(np.quantile(cc, 0.95))


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)
    d, n = 60, 400
    y = (np.arange(n) % 2).astype(int)
    # Construct a "concept axis" and two DISTINCT strategy axes.
    concept = unit_raw(rng.standard_normal(d))
    short = unit_raw(rng.standard_normal(d)); short -= (short @ concept) * concept; short = unit_raw(short)

    # raw regime: classes separated along the SHORTCUT axis (u should ~ shortcut)
    H_raw = rng.standard_normal((n, d)) * 0.3 + np.outer(2 * y - 1.0, 1.2 * short)
    # scaffold regime: classes separated along the CONCEPT axis (u should ~ concept)
    H_sca = rng.standard_normal((n, d)) * 0.3 + np.outer(2 * y - 1.0, 1.2 * concept)

    # usage proxy = class delta direction (stand-in for the gradient mean)
    u_raw = unit_raw(class_delta(H_raw, y))
    u_sca = unit_raw(class_delta(H_sca, y))
    # (1) the two regimes' usage directions are NEARLY ORTHOGONAL (different strategy)
    assert abs_cos(u_raw, u_sca) < 0.2, f"distinct-strategy toy: u_raw/u_sca should differ: {abs_cos(u_raw, u_sca)}"

    # (2) cross-subspace: u_raw aligns with raw-shortcut, u_sca with concept
    assert abs_cos(u_raw, short) > 0.9 and abs_cos(u_sca, concept) > 0.9

    # (3) capture: u_sca is captured by the CONCEPT 1-D subspace, not the shortcut one
    cap_concept = captured_fraction(u_sca, concept[:, None])
    cap_short = captured_fraction(u_sca, short[:, None])
    assert cap_concept > 0.8 and cap_short < 0.2, f"capture must localize: {cap_concept}, {cap_short}"

    # (4) PR: a coherent single-direction field has PR ~ 1; an isotropic field has PR ~ rank
    M_coh = np.outer(rng.standard_normal(n), concept) + 0.01 * rng.standard_normal((n, d))
    M_dif = rng.standard_normal((n, d))
    assert participation_ratio(M_coh) < 3 and participation_ratio(M_dif) > 20, \
        f"PR must separate coherent from diffuse: {participation_ratio(M_coh)}, {participation_ratio(M_dif)}"

    # (5) null calibration sane
    Hc = H_sca - H_sca.mean(0)
    assert within_span_null_p95(Hc, u_sca, rng) > 0.05

    # (6) split-half stability: a strong shared gradient direction is split-half stable;
    #     pure per-row noise is split-half near-orthogonal.
    shared = unit_raw(rng.standard_normal(d))
    G_strong = shared[None, :] + 0.15 * rng.standard_normal((n, d))  # strong shared mean direction
    half = n // 2
    ua = unit_raw(G_strong[:half].mean(0)); ub = unit_raw(G_strong[half:].mean(0))
    assert abs_cos(ua, ub) > 0.9, f"strong shared signal must be split-half stable: {abs_cos(ua, ub)}"
    G_noise = rng.standard_normal((n, d))                                          # no shared direction
    na = unit_raw(G_noise[:half].mean(0)); nb = unit_raw(G_noise[half:].mean(0))
    assert abs_cos(na, nb) < 0.2, f"pure noise must be split-half unstable: {abs_cos(na, nb)}"
    print("[self_test] OK — strategy-distinct angle, cross-subspace alignment, capture localization, "
          "PR coherent/diffuse, span null, split-half stability pass.")


# =====================================================================
# Real run
# =====================================================================
def load_dump(dpath):
    dump = Path(dpath)
    meta = np.load(dump / "meta.npz", allow_pickle=True)
    fams = json.load(open(dump / "families.json"))
    return dump, meta, fams, int(meta["n_layers"])


def reconstruct_split(fams, seed, train_frac):
    rng = np.random.default_rng(seed)
    fl = sorted(set(fams)); rng.shuffle(fl)
    train = set(fl[: int(round(len(fl) * train_frac))])
    return np.array([f in train for f in fams], bool)


def regime_objects(dump, L, y, trm, shrink, top_rank):
    H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
    G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
    Htr = H[trm]
    w = fisher_axis(Htr, y[trm], shrink)
    delta = unit_raw(class_delta(Htr, y[trm]))
    u = unit_raw(G.mean(0))
    pr_usage = participation_ratio(G)               # diffuseness of the usage field
    # class subspace = span of {delta} augmented by top within-class-removed PCs that
    # separate classes; here we use the top_rank LDA-whitened class directions proxy:
    # simply the 1-D delta plus the leading between-class PCA axis.
    cls_basis = np.stack([delta]).T                  # (d, 1) minimal class subspace
    Hc = Htr - Htr.mean(0)
    return {"H": H, "G": G, "w": w, "delta": delta, "u": u,
            "pr_usage": pr_usage, "cls_basis": cls_basis, "Hc": Hc}


def split_half_u(dump, L, y, trm, shrink, rng):
    """Estimate u on two disjoint halves of the TRAIN set; return both unit u's
    and the centered train residual (for null calibration). Tests whether u is a
    stable direction within a regime or just noise at this layer."""
    G = np.load(dump / f"grad_L{L:02d}.npy").astype(np.float64)
    H = np.load(dump / f"res_L{L:02d}.npy").astype(np.float64)
    idx = np.where(trm)[0]
    perm = rng.permutation(len(idx))
    a, b = idx[perm[:len(idx) // 2]], idx[perm[len(idx) // 2:]]
    ua = unit_raw(G[a].mean(0)); ub = unit_raw(G[b].mean(0))
    Hc = H[trm] - H[trm].mean(0)
    return ua, ub, Hc


def run_self_consistency(args):
    rng = np.random.default_rng(args.seed)
    out_rows = []
    print("\n" + "=" * 100)
    print("SELF-CONSISTENCY CONTROL — is mid-stack u stable WITHIN a regime, or just noise?")
    print("=" * 100)
    for tag, dpath in (("raw", args.raw_dump), ("scaffold", args.scaffold_dump)):
        dump, meta, fams, n_layers = load_dump(dpath)
        y = meta["y"].astype(int)
        trm = reconstruct_split(fams, args.split_seed, args.train_frac)
        for L in range(n_layers):
            # average split-half cosine over several random halvings (stabilize the estimate)
            coss = []
            for _ in range(args.n_halvings):
                ua, ub, Hc = split_half_u(dump, L, y, trm, args.shrink, rng)
                coss.append(abs_cos(ua, ub))
            ua, ub, Hc = split_half_u(dump, L, y, trm, args.shrink, rng)
            p95 = within_span_null_p95(Hc, ua, rng)
            out_rows.append({"regime": tag, "layer": L,
                             "split_half_cos_u": float(np.mean(coss)),
                             "split_half_cos_u_sd": float(np.std(coss)),
                             "null_p95": p95,
                             "above_null": int(np.mean(coss) > p95)})
            if L % 4 == 0 or L == n_layers - 1:
                logger.info("[%s] L%02d split-half cos(u_A,u_B)=%.3f±%.3f (null %.3f) %s",
                            tag, L, float(np.mean(coss)), float(np.std(coss)), p95,
                            "STABLE" if np.mean(coss) > p95 else "noise-level")

    out = Path(args.out).with_name("usage_self_consistency.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in out_rows]

    def med(tag, lo, hi):
        sel = [r for r in out_rows if r["regime"] == tag and lo <= r["layer"] <= hi]
        return float(np.median([r["split_half_cos_u"] for r in sel])) if sel else float("nan")
    print("\nmid-stack (L%d-%d) split-half stability of u:" % (args.mid_lo, args.mid_hi))
    print(f"  raw      = {med('raw', args.mid_lo, args.mid_hi):.3f}")
    print(f"  scaffold = {med('scaffold', args.mid_lo, args.mid_hi):.3f}")
    print("\nINTERPRETATION:")
    print("  - raw mid-stack split-half cos HIGH (>> null) -> u is a well-estimated, stable")
    print("    direction within the raw regime; its near-orthogonality to scaffold-u is therefore")
    print("    a REGIME difference, not estimation noise -> the strategy claim is solid.")
    print("  - raw mid-stack split-half cos AT NULL -> raw u is itself noisy mid-stack (weak")
    print("    shortcut gradient signal); the cross-regime orthogonality is then partly an")
    print("    artifact and the claim must be hedged to where u is stable.")
    print(f"per layer: {out}")
    print("=" * 100 + "\n")


def run_real(args):
    if args.self_consistency:
        run_self_consistency(args)
        return
    rng = np.random.default_rng(args.seed)
    dS, mS, fS, nLS = load_dump(args.scaffold_dump)
    n_layers = min(nLR, nLS)
    yR = mR["y"].astype(int); yS = mS["y"].astype(int)
    trmR = reconstruct_split(fR, args.split_seed, args.train_frac)
    trmS = reconstruct_split(fS, args.split_seed, args.train_frac)
    gammaR = mR["wU_diff"].astype(np.float64)
    gammaS = mS["wU_diff"].astype(np.float64)

    rows = []
    for L in range(n_layers):
        R = regime_objects(dR, L, yR, trmR, args.shrink, args.top_rank)
        S = regime_objects(dS, L, yS, trmS, args.shrink, args.top_rank)
        ang = abs_cos(R["u"], S["u"])
        # null for "two independent supervised directions in the residual span":
        p95 = max(within_span_null_p95(R["Hc"], S["u"], rng),
                  within_span_null_p95(S["Hc"], R["u"], rng))
        # cross-regime subspace capture
        cap_raw_in_sca = captured_fraction(R["u"], S["cls_basis"])     # raw-u in scaffold class subspace
        cap_sca_in_raw = captured_fraction(S["u"], R["cls_basis"])     # scaffold-u in raw class subspace
        rows.append({
            "layer": L,
            "abs_cos_uRaw_uSca": ang,
            "null_p95": p95,
            "above_null": int(ang > p95),
            "auc_uRaw_proxy": float(abs(R["u"] @ R["delta"])),   # alignment of u with its own delta
            "auc_uSca_proxy": float(abs(S["u"] @ S["delta"])),
            "pr_usage_raw": R["pr_usage"],
            "pr_usage_sca": S["pr_usage"],
            "cap_uRaw_in_scaSubspace": cap_raw_in_sca,
            "cap_uSca_in_rawSubspace": cap_sca_in_raw,
            "cos_uRaw_gamma": abs_cos(R["u"], gammaR),
            "cos_uSca_gamma": abs_cos(S["u"], gammaS),
            "cos_wRaw_wSca": abs_cos(R["w"], S["w"]),
            "cos_deltaRaw_deltaSca": abs_cos(R["delta"], S["delta"]),
        })
        if L % 4 == 0 or L == n_layers - 1:
            logger.info("L%02d: ang(uR,uS)=%.3f (null %.3f) | PR raw/sca=%.1f/%.1f | "
                        "cap R->S=%.2f S->R=%.2f | cos(u,g) R/S=%.2f/%.2f",
                        L, ang, p95, R["pr_usage"], S["pr_usage"],
                        cap_raw_in_sca, cap_sca_in_raw,
                        rows[-1]["cos_uRaw_gamma"], rows[-1]["cos_uSca_gamma"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    # ---- verdict ----
    mid = [r for r in rows if args.mid_lo <= r["layer"] <= args.mid_hi]
    late = [r for r in rows if r["layer"] >= args.late_lo]
    def med(rs, k): return float(np.median([r[k] for r in rs])) if rs else float("nan")
    print("\n" + "=" * 100)
    print("USAGE-FIELD STRATEGY TEST — does u encode the computational strategy or the concept?")
    print("=" * 100)
    print(f"mid-stack (L{args.mid_lo}-{args.mid_hi}):")
    print(f"  angle |cos(u_raw, u_scaffold)| median = {med(mid,'abs_cos_uRaw_uSca'):.3f} "
          f"(within-span null p95 ~ {med(mid,'null_p95'):.3f})")
    print(f"  u-delta alignment (proxy AUC of u): raw={med(mid,'auc_uRaw_proxy'):.3f} "
          f"scaffold={med(mid,'auc_uSca_proxy'):.3f}   <- the 123 divergence")
    print(f"  usage-field PR: raw={med(mid,'pr_usage_raw'):.1f} scaffold={med(mid,'pr_usage_sca'):.1f}")
    print(f"  cross-subspace capture: u_raw in scaffold-subspace={med(mid,'cap_uRaw_in_scaSubspace'):.2f} | "
          f"u_scaffold in raw-subspace={med(mid,'cap_uSca_in_rawSubspace'):.2f}")
    print(f"readout (L>={args.late_lo}):")
    print(f"  cos(u_raw, gamma)={med(late,'cos_uRaw_gamma'):.2f} "
          f"cos(u_scaffold, gamma)={med(late,'cos_uSca_gamma'):.2f}  <- convergence at the head")
    print(f"  angle |cos(u_raw,u_scaffold)| late = {med(late,'abs_cos_uRaw_uSca'):.3f}")
    print("\nINTERPRETATION:")
    print("  - mid-stack angle AT/BELOW null + readout angle HIGH -> u_raw and u_scaffold are")
    print("    DIFFERENT directions mid-stack, converging only at the readout: u tracks STRATEGY,")
    print("    not the concept. The asymmetric capture (one u sits in the other's subspace but")
    print("    not vice versa) names which regime reads concept vs shortcut.")
    print("  - mid-stack angle HIGH (>> null) -> u is the SAME axis in both regimes: u is")
    print("    concept-fixed, and the strategy story is NOT supported (fall back to the law-theorem).")
    print(f"per layer: {out}")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--raw_dump", default="data/analysis/runD_v2/field_dump")
    p.add_argument("--scaffold_dump", default="data/analysis/runD_v2/B1_alpha_beta/field_dump")
    p.add_argument("--out", default="data/analysis/runD_v2/usage_field_strategy.csv")
    p.add_argument("--self_consistency", action="store_true",
                   help="run the split-half stability control instead of the cross-regime comparison")
    p.add_argument("--n_halvings", type=int, default=8, help="random train halvings to average split-half cos")
    p.add_argument("--mid_lo", type=int, default=14)
    p.add_argument("--mid_hi", type=int, default=20)
    p.add_argument("--late_lo", type=int, default=33)
    p.add_argument("--top_rank", type=int, default=1, help="dim of the minimal class subspace for capture")
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
