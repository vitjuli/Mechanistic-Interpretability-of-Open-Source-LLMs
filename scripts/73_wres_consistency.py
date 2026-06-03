"""
73_wres_consistency.py   [LAPTOP / CPU-ONLY -- seconds]
===================================================================
Defense insurance, not new science. Many analyses (69/70/72 + local) were run on
the SAVED w_res_*.npy from script 64. This verifies those saved axes are IDENTICAL
to a fresh Fisher reconstruction from h_residual_per_depth.npz on the SAME
train split -- i.e. no data drift / file mismatch crept in between runs.

Precondition (verified): the Fisher/LDA solve is bit-deterministic, and fp16/fp32
round-trip preserves the direction to cos=1.0. So ANY material mismatch here means
the saved file came from different data (different prompts/split/layer), which a
reviewer would (rightly) flag. A clean pass closes that question.

WHAT IT DOES (only data on disk; NO model, NO GPU):
  * rebuilds w_res at each depth by Fisher on the TRAIN split from the npz;
  * if w_res_*.npy (or w_res65_*.npy) exist in --geom_dir, compares each:
      - cos(saved, fresh)            (should be ~1.0, sign-aware)
      - held-out AUC of each         (should match)
  * reports per-depth agreement and an overall verdict.

If the saved files are absent it still prints the freshly reconstructed AUCs so
you can eyeball-match them against the CSD3 log (0.966/0.988/.../0.992).

SELF-TEST: python 73_wres_consistency.py --self_test
"""

from __future__ import annotations
import argparse, glob, json, logging, re, sys
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wres_consistency")
TAP_RE = re.compile(r"w_res(?:65)?_(postL(\d+)|final)\.npy$")


def fisher(H, y, sh=0.1):
    m0, m1 = H[y == 0].mean(0), H[y == 1].mean(0)
    X0, X1 = H[y == 0] - m0, H[y == 1] - m1
    Sw = (X0.T @ X0 + X1.T @ X1) / max(len(y) - 2, 1); Sw = 0.5 * (Sw + Sw.T)
    Sw = (1 - sh) * Sw + sh * np.diag(np.diag(Sw))
    Sw += 1e-6 * float(np.mean(np.diag(Sw)) + 1e-12) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, m1 - m0)
    return w / (np.linalg.norm(w) + 1e-30)


def auc(H, y, w):
    s = H @ w; o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else float("nan")


def self_test():
    rng = np.random.default_rng(73); d = 300; n = 400
    y = rng.integers(0, 2, n)
    w_true = rng.standard_normal(d); w_true /= np.linalg.norm(w_true)
    tr = rng.random(n) < 0.6
    H = rng.standard_normal((n, d)) * 0.5 + np.outer((y * 2 - 1.) * 2., w_true)
    w_fresh = fisher(H[tr], y[tr])
    # "saved" = same computation, fp16 round-trip (mimics storage) -> should match
    w_saved = w_fresh.astype(np.float16).astype(np.float64)
    w_saved /= np.linalg.norm(w_saved)
    # a DRIFTED axis = computed on a different split -> should NOT perfectly match
    tr2 = rng.random(n) < 0.6
    w_drift = fisher(H[tr2], y[tr2])
    c_match = abs(float(w_fresh @ w_saved))
    c_drift = abs(float(w_fresh @ w_drift))
    print("\n--- SELF TEST -------------------------------------------------")
    print(f"  cos(fresh, saved=fp16 roundtrip)   = {c_match:.6f}  (expect ~1.0)")
    print(f"  cos(fresh, drifted=other split)    = {c_drift:.4f}  (expect < 1.0)")
    assert c_match > 0.9999, f"identical recompute must match (got {c_match})"
    assert c_drift < c_match, "drifted split must differ from fresh"
    print("\nALL SELF-TEST ASSERTIONS PASSED ")
    print("(check distinguishes a faithful saved axis from one built on different data)")
    print("---------------------------------------------------------------\n")


def run_real(args):
    z = np.load(args.npz)
    y = z["y"]; tr = z["is_train"]
    taps = [k for k in z.keys() if k.startswith("postL")] + (["final"] if "final" in z.keys() else [])
    taps = sorted(taps, key=lambda s: (9999 if s == "final" else int(s.replace("postL", ""))))

    geom = Path(args.geom_dir)
    saved = {}
    for p in geom.glob("w_res*_*.npy"):
        m = TAP_RE.search(p.name)
        if m:
            tap = "final" if m.group(1) == "final" else f"postL{int(m.group(2))}"
            saved.setdefault(tap, p)  # prefer first match (w_res_ over w_res65_ by glob order is not guaranteed)
    logger.info("npz taps: %d   saved w_res files matched: %d", len(taps), len(saved))

    rows = []
    for k in taps:
        H = z[k].astype(np.float64)
        w_fresh = fisher(H[tr], y[tr], args.shrink)
        a_fresh = auc(H[~tr], y[~tr], w_fresh)
        rec = {"tap": k, "fresh_heldout_auc": a_fresh, "saved_file": None,
               "cos_saved_fresh": None, "saved_heldout_auc": None, "match": None}
        if k in saved:
            w_saved = np.load(saved[k]).astype(np.float64)
            w_saved = w_saved / (np.linalg.norm(w_saved) + 1e-30)
            c = abs(float(w_fresh @ w_saved))            # sign-aware (Fisher sign is fixed, but be safe)
            a_saved = auc(H[~tr], y[~tr], w_saved)
            rec.update({"saved_file": saved[k].name, "cos_saved_fresh": c,
                        "saved_heldout_auc": a_saved, "match": bool(c > args.cos_tol)})
        rows.append(rec)
        if rec["cos_saved_fresh"] is not None:
            logger.info("%8s: fresh AUC=%.3f  saved AUC=%.3f  cos(saved,fresh)=%.6f  %s",
                        k, a_fresh, rec["saved_heldout_auc"], rec["cos_saved_fresh"],
                        "MATCH" if rec["match"] else "MISMATCH")
        else:
            logger.info("%8s: fresh AUC=%.3f  (no saved file to compare)", k, a_fresh)

    compared = [r for r in rows if r["cos_saved_fresh"] is not None]
    all_match = bool(compared) and all(r["match"] for r in compared)
    out = {"npz": str(args.npz), "geom_dir": str(geom), "n_compared": len(compared),
           "per_tap": rows,
           "verdict": (
               f"CONSISTENT: all {len(compared)} saved w_res axes match fresh reconstruction "
               f"(min cos {min(r['cos_saved_fresh'] for r in compared):.5f}). Analyses built on the "
               "saved files are faithful to the data; no drift." if all_match and compared else
               "NO SAVED FILES FOUND: compare the printed fresh AUCs against the CSD3 log "
               "(expect 0.966 / 0.988 / 0.996 / 0.992 at L14/L18/L24/final)." if not compared else
               "MISMATCH DETECTED: at least one saved w_res differs from fresh reconstruction -- "
               "investigate which prompts/split/layer produced the saved file before using it.")}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "wres_consistency.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print("\n" + "=" * 76)
    print("w_res CONSISTENCY  (saved files vs fresh reconstruction)")
    print("=" * 76)
    print(f"{'tap':>8} {'fresh_AUC':>10} {'saved_AUC':>10} {'cos':>10} {'status':>9}")
    for r in rows:
        ca = f"{r['cos_saved_fresh']:.6f}" if r["cos_saved_fresh"] is not None else "   -- "
        sa = f"{r['saved_heldout_auc']:.3f}" if r["saved_heldout_auc"] is not None else "  -- "
        st = ("MATCH" if r["match"] else "MISMATCH") if r["match"] is not None else "no file"
        print(f"{r['tap']:>8} {r['fresh_heldout_auc']:>10.3f} {sa:>10} {ca:>10} {st:>9}")
    print("\nVERDICT: " + out["verdict"])
    print(f"\nwrote: {args.out_dir}/wres_consistency.json")
    print("=" * 76)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--npz", default="data/analysis/runD_v2/geometry_stage1/h_residual_per_depth.npz")
    p.add_argument("--geom_dir", default="data/analysis/runD_v2/geometry_stage1")
    p.add_argument("--out_dir", default="wres_consistency_out")
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--cos_tol", type=float, default=0.999, help="cos above this = match")
    return p


def main():
    a = build_parser().parse_args()
    if a.self_test:
        self_test(); return
    run_real(a)


if __name__ == "__main__":
    main()
