"""
125c_spectrum_diagnostics.py   [is the w_res spectrum real, and is it linear-regime?]
======================================================================================
125b found the reading-axis spectrum survives integral metrics (grammar w_res
peak_intact 0.72 vs alpha_beta 0.22). Before committing the headline graph and
the remaining four concepts, three things must be checked on the EXISTING
cells_tier2 dumps (pure CPU):

(1) WHERE is each concept's w_res peak — which (layer, c) — and is it a CLEAN
    flip there? Report intact_given_flip and n_correct at the peak cell. A peak
    riding on a tiny pool or on low intact_given_flip is an artifact, not
    steerability.

(2) Does the spectrum survive INSIDE the validity radius? Recompute peak
    intact-flip restricted to c <= c_star (per-concept strict radius from 116),
    separately from c > c_star. If grammar > alpha_beta even within radius, the
    spectrum is a LINEAR-REGIME fact (strong). If the gap only appears beyond
    radius, the claim weakens to "reading axis yields only to nonlinear forcing".

(3) Pool stability: n_correct per amplitude (constant after the baseline-correct
    filter, but verify no collapse drives the peak).

Reads cells_tier2.csv per concept (idx,y,m0,m1,intact,layer,c,dir). Per-concept
c_star is passed in the manifest (or via --c_star_default).

SELF-TEST (no torch / no repo):  python 125c_spectrum_diagnostics.py --self_test
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("spec125c")


# =====================================================================
# Pure-python core (exercised by --self_test)
# =====================================================================
def cell_metrics(recs):
    """recs at fixed (layer,c,dir): flip_norm, flip_norm_intact, intact_given_flip,
    n_correct — all on the baseline-correct pool."""
    fln, fln_it, nflip = [], [], 0
    for r in recs:
        y = int(r["y"]); m0 = float(r["m0"]); m1 = float(r["m1"]); it = int(r["intact"])
        corr = (m0 < 0) if y == 0 else (m0 > 0)
        if not corr:
            continue
        f = int(m1 > 0) if y == 0 else int(m1 < 0)
        fln.append(f); nflip += f; fln_it.append(f and it)
    n = len(fln)
    return {"flip_norm": float(np.mean(fln)) if n else float("nan"),
            "flip_norm_intact": float(np.mean(fln_it)) if n else float("nan"),
            "intact_given_flip": float(sum(fln_it) / nflip) if nflip else float("nan"),
            "n_correct": n}


def peak_cell(cells_by_lc, c_filter=None):
    """cells_by_lc: {(layer,c): metrics}. Returns (layer, c, metrics) maximizing
    flip_norm_intact over cells passing c_filter (callable c->bool)."""
    best = None
    for (L, c), m in cells_by_lc.items():
        if c_filter and not c_filter(c):
            continue
        v = m["flip_norm_intact"]
        if np.isnan(v):
            continue
        if best is None or v > best[2]["flip_norm_intact"]:
            best = (L, c, m)
    return best


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    def mk(n_corr, n_flip, n_intact):
        recs = []
        for k in range(n_corr):
            flipped = k < n_flip
            intact = k < n_intact
            recs.append({"y": 0, "m0": -1.0, "m1": 0.5 if flipped else -0.5,
                         "intact": int(intact if flipped else 1)})
        return recs

    # peak inside radius lower than beyond radius
    cells = {
        (23, 4.0): cell_metrics(mk(50, 10, 10)),    # within radius: 0.20 clean
        (23, 16.0): cell_metrics(mk(50, 40, 36)),   # beyond radius: 0.72 mostly clean
        (35, 32.0): cell_metrics(mk(50, 50, 5)),    # beyond radius: dirty (intact_given_flip low)
    }
    inside = peak_cell(cells, c_filter=lambda c: c <= 4.0)
    beyond = peak_cell(cells, c_filter=lambda c: c > 4.0)
    assert abs(inside[2]["flip_norm_intact"] - 0.20) < 1e-9 and inside[1] == 4.0
    assert beyond[1] == 16.0 and abs(beyond[2]["flip_norm_intact"] - 0.72) < 1e-9
    # the dirty cell has high flip_norm but low intact_given_flip
    dirty = cell_metrics(mk(50, 50, 5))
    assert dirty["flip_norm"] == 1.0 and dirty["intact_given_flip"] < 0.2
    # baseline-incorrect targets excluded from n_correct
    mixed = cell_metrics([{"y": 0, "m0": 0.5, "m1": 1.0, "intact": 1}] * 5 + mk(10, 5, 5))
    assert mixed["n_correct"] == 10
    print("[self_test] OK — cell metrics, within/beyond-radius peak selection, dirty-flip flag pass.")


# =====================================================================
# Real run
# =====================================================================
def load_cells(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            with open(p) as f:
                rows += list(_csv.DictReader(f))
    return rows


def run_real(args):
    concepts = json.load(open(args.concepts))
    rows = []
    print("\n" + "=" * 104)
    print("SPECTRUM DIAGNOSTICS — w_res peak location, cleanliness, and within-radius survival")
    print("=" * 104)
    for cdef in concepts:
        name = cdef["name"]
        c_star = float(cdef.get("c_star", args.c_star_default))
        cells = load_cells(cdef.get("cells_csvs", []))
        if not cells:
            logger.warning("%s: no cells — skipped", name); continue
        for dname in ("w_res", "usage"):
            by_lc = {}
            grp = defaultdict(list)
            for r in cells:
                if r["dir"] == dname:
                    grp[(int(r["layer"]), float(r["c"]))].append(r)
            for k, recs in grp.items():
                by_lc[k] = cell_metrics(recs)
            pk_all = peak_cell(by_lc)
            pk_in = peak_cell(by_lc, c_filter=lambda c: c <= c_star)
            pk_be = peak_cell(by_lc, c_filter=lambda c: c > c_star)
            def cell_str(pk):
                if pk is None:
                    return "none"
                L, c, m = pk
                return (f"L{L} c={c:g}: fn_it={m['flip_norm_intact']:.2f} "
                        f"int|flip={m['intact_given_flip']:.2f} n={m['n_correct']}")
            rows.append({"concept": name, "dir": dname, "c_star": c_star,
                         "peak_overall": pk_all[2]["flip_norm_intact"] if pk_all else float("nan"),
                         "peak_overall_c": pk_all[1] if pk_all else float("nan"),
                         "peak_within_radius": pk_in[2]["flip_norm_intact"] if pk_in else float("nan"),
                         "peak_within_c": pk_in[1] if pk_in else float("nan"),
                         "peak_beyond_radius": pk_be[2]["flip_norm_intact"] if pk_be else float("nan"),
                         "peak_beyond_c": pk_be[1] if pk_be else float("nan"),
                         "intact_given_flip_at_peak": pk_all[2]["intact_given_flip"] if pk_all else float("nan"),
                         "n_correct_at_peak": pk_all[2]["n_correct"] if pk_all else 0})
            print(f"\n[{name} / {dname}]  c* = {c_star:g}")
            print(f"   peak overall:        {cell_str(pk_all)}")
            print(f"   peak within radius:  {cell_str(pk_in)}")
            print(f"   peak beyond radius:  {cell_str(pk_be)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]

    # ---- spectrum verdict on w_res, within vs beyond radius ----
    wres = [r for r in rows if r["dir"] == "w_res"]
    print("\n" + "-" * 104)
    print("READING-AXIS SPECTRUM — within radius vs beyond radius")
    def spread(key):
        vs = [r[key] for r in wres if not np.isnan(r[key])]
        return (max(vs) - min(vs)) if len(vs) > 1 else float("nan")
    for r in sorted(wres, key=lambda x: -(x["peak_within_radius"] if not np.isnan(x["peak_within_radius"]) else -1)):
        print(f"  {r['concept']:<16} w_res: within-radius peak={r['peak_within_radius']:.2f} "
              f"(c={r['peak_within_c']:g}) | beyond-radius peak={r['peak_beyond_radius']:.2f} "
              f"(c={r['peak_beyond_c']:g})")
    print(f"\n  within-radius spectrum range: {spread('peak_within_radius'):.2f} | "
          f"beyond-radius range: {spread('peak_beyond_radius'):.2f}")
    print("  VERDICT:")
    print("   - within-radius range >~0.2 -> spectrum is a LINEAR-REGIME fact (strong headline)")
    print("   - within-radius range ~0 but beyond-radius large -> 'reading axis yields only to")
    print("     nonlinear forcing'; headline must annotate the regime")
    print("   - check intact_given_flip_at_peak: low (<0.7) peaks are dirty, discount them")
    print(f"\nper concept/dir: {out}")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="manifest with cells_csvs and optional per-concept c_star")
    p.add_argument("--out", default="data/analysis/runD_v2/spectrum_diagnostics.csv")
    p.add_argument("--c_star_default", type=float, default=4.0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.concepts, "--concepts manifest required"
    run_real(args)


if __name__ == "__main__":
    main()
