"""
125b_dose_response_scalars.py   [steerability as a dose curve, not a point]
============================================================================
The c0-point headline collapsed the spectrum because concepts BREAK FORMAT at
different amplitudes: a w_res intact-flip can rise on mid amplitudes and then
crash to 0 at large c (the answer leaves the label set). A single c0 cannot see
that. This script recomputes steerability as the whole intact-flip dose curve
F(c) and extracts three comparable scalars per (concept, dir, layer):

  peak_intact      max_c F_intact(c)               — best achievable clean flip
  c_half           smallest c with F_intact(c) >= 0.5 (NaN if never)   — potency
  area_to_break    sum over the monotone-rising prefix of F_intact(c)  — robust
                   integral up to the first crash (F drops by > drop_tol), so a
                   late breakdown does not erase a real mid-amplitude effect
  breakdown_c      the c at which F_intact first crashes (NaN if monotone)
  peak_raw         max_c F_margin(c)               — the (dirty) margin-flip peak,
                   to expose the raw-vs-intact gap that fooled the c0 view

All from cells_tier2.csv (per-target idx,y,m0,m1,intact) — pure CPU, no GPU.
The per-concept "headline scalar" is then the layer-MAX of each scalar (the
concept's best layer), reported for w_res and u side by side, so the spectrum
question is answered under every candidate metric at once.

SELF-TEST (no torch / no repo):  python 125b_dose_response_scalars.py --self_test
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
logger = logging.getLogger("dose125b")


# =====================================================================
# Pure-python core (exercised by --self_test)
# =====================================================================
def curve_from_cells(recs):
    """recs for a fixed (layer,dir): list of dicts idx,y,m0,m1,intact, c.
    Returns sorted [(c, F_margin, F_intact, n_correct)] on baseline-correct pool."""
    by_c = defaultdict(list)
    for r in recs:
        by_c[float(r["c"])].append(r)
    out = []
    for c in sorted(by_c):
        fln, fln_it = [], []
        for r in by_c[c]:
            y = int(r["y"]); m0 = float(r["m0"]); m1 = float(r["m1"]); it = int(r["intact"])
            corr = (m0 < 0) if y == 0 else (m0 > 0)
            if not corr:
                continue
            f = int(m1 > 0) if y == 0 else int(m1 < 0)
            fln.append(f); fln_it.append(f and it)
        n = len(fln)
        out.append((c, float(np.mean(fln)) if n else float("nan"),
                    float(np.mean(fln_it)) if n else float("nan"), n))
    return out


def dose_scalars(curve, drop_tol=0.15):
    """curve: [(c, F_margin, F_intact, n)]. Returns the comparable scalars."""
    cs = [c for c, _, _, _ in curve]
    fi = [x for _, _, x, _ in curve]
    fm = [x for _, x, _, _ in curve]
    peak_intact = float(np.nanmax(fi)) if len(fi) else float("nan")
    peak_raw = float(np.nanmax(fm)) if len(fm) else float("nan")
    # c_half: first c reaching >=0.5 intact-flip
    c_half = float("nan")
    for c, x in zip(cs, fi):
        if not np.isnan(x) and x >= 0.5:
            c_half = c; break
    # area to first crash: walk amplitudes, accumulate trapezoid in (c index) until
    # F_intact drops by more than drop_tol from the running max -> breakdown.
    area = 0.0; breakdown_c = float("nan"); run_max = 0.0
    prev = None
    for c, x in zip(cs, fi):
        if np.isnan(x):
            continue
        if x < run_max - drop_tol:
            breakdown_c = c
            break
        run_max = max(run_max, x)
        if prev is not None:
            area += 0.5 * (prev[1] + x) * (np.log2(c) - np.log2(prev[0]))  # log-c trapezoid
        prev = (c, x)
    return {"peak_intact": peak_intact, "peak_raw": peak_raw,
            "c_half": c_half, "area_to_break": float(area), "breakdown_c": breakdown_c}


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    # build synthetic cells for one (layer,dir): 10 correct alpha targets (y=0,m0<0).
    def make(c_to_flipped_intact):
        """c -> (n_flip, n_intact_among_flip) over 10 correct targets."""
        recs = []
        for c, (nf, nit) in c_to_flipped_intact.items():
            for k in range(10):
                flipped = k < nf
                intact = k < nit
                recs.append({"c": c, "y": 0, "m0": -1.0,
                             "m1": 0.5 if flipped else -0.5, "intact": int(intact if flipped else 1)})
        return recs

    # curve that rises then CRASHES at c=32 (late breakdown)
    cells = make({1.0: (1, 1), 4.0: (5, 5), 8.0: (10, 10), 16.0: (10, 7), 32.0: (10, 0)})
    cur = curve_from_cells(cells)
    fi = {c: round(x, 2) for c, _, x, _ in cur}
    assert fi[8.0] == 1.0 and fi[32.0] == 0.0, fi
    sc = dose_scalars(cur, drop_tol=0.15)
    assert abs(sc["peak_intact"] - 1.0) < 1e-9, "peak should see the c=8 plateau"
    assert sc["c_half"] == 4.0, f"c_half wrong: {sc['c_half']}"
    assert sc["breakdown_c"] == 16.0, f"breakdown should trip at the first drop (c=16): {sc['breakdown_c']}"
    # a monotone non-crashing curve: no breakdown, area positive
    cells2 = make({1.0: (1, 1), 4.0: (5, 5), 8.0: (8, 8), 16.0: (10, 10), 32.0: (10, 10)})
    sc2 = dose_scalars(curve_from_cells(cells2))
    assert np.isnan(sc2["breakdown_c"]) and sc2["peak_intact"] == 1.0 and sc2["area_to_break"] > 0
    # an inert axis: never flips -> peak 0, c_half nan, area 0
    cells3 = make({1.0: (0, 0), 4.0: (0, 0), 16.0: (0, 0), 32.0: (0, 0)})
    sc3 = dose_scalars(curve_from_cells(cells3))
    assert sc3["peak_intact"] == 0.0 and np.isnan(sc3["c_half"]) and sc3["area_to_break"] == 0.0
    # raw-vs-intact gap: flips happen but none intact -> peak_raw high, peak_intact 0
    cells4 = make({4.0: (10, 0), 16.0: (10, 0)})
    sc4 = dose_scalars(curve_from_cells(cells4))
    assert sc4["peak_raw"] == 1.0 and sc4["peak_intact"] == 0.0, "must expose dirty-flip gap"
    print("[self_test] OK — dose curve, peak/c_half/area/breakdown, inert + dirty-flip cases pass.")


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
    out_rows = []
    headline = []
    for cdef in concepts:
        name = cdef["name"]
        cells = load_cells(cdef.get("cells_csvs", []))
        if not cells:
            logger.warning("%s: no cells_tier2.csv — skipped (re-run 122 --dump_cells)", name)
            continue
        # group by (layer, dir)
        grp = defaultdict(list)
        for r in cells:
            grp[(int(r["layer"]), r["dir"])].append(r)
        per_dir_best = defaultdict(lambda: defaultdict(lambda: -1.0))
        for (L, dname), recs in sorted(grp.items()):
            cur = curve_from_cells(recs)
            sc = dose_scalars(cur, args.drop_tol)
            row = {"concept": name, "layer": L, "dir": dname, **sc,
                   "n_correct": cur[0][3] if cur else 0}
            out_rows.append(row)
            # track layer-best per scalar for the dir
            for k in ("peak_intact", "area_to_break"):
                if not np.isnan(sc[k]):
                    per_dir_best[dname][k] = max(per_dir_best[dname][k], sc[k])
            # c_half: track the MIN over layers (most potent)
            if not np.isnan(sc["c_half"]):
                cur_ch = per_dir_best[dname].get("c_half_min", np.inf)
                per_dir_best[dname]["c_half_min"] = min(cur_ch, sc["c_half"])
            if not np.isnan(sc["peak_raw"]):
                per_dir_best[dname]["peak_raw"] = max(per_dir_best[dname]["peak_raw"], sc["peak_raw"])
        for dname in ("w_res", "usage"):
            b = per_dir_best.get(dname, {})
            headline.append({"concept": name, "dir": dname,
                             "peak_intact_best": b.get("peak_intact", float("nan")),
                             "peak_raw_best": b.get("peak_raw", float("nan")),
                             "area_to_break_best": b.get("area_to_break", float("nan")),
                             "c_half_min": b.get("c_half_min", float("nan"))})
        logger.info("%s done (%d layer-dir curves)", name, len(grp))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in out_rows]
    hout = out.with_name("headline_scalars.csv")
    with open(hout, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(headline[0].keys())); w.writeheader()
        [w.writerow(r) for r in headline]

    print("\n" + "=" * 104)
    print("DOSE-RESPONSE STEERABILITY SCALARS — does the spectrum survive ANY comparable metric?")
    print("=" * 104)
    print(f"{'concept':<16}{'dir':<7}  {'peak_intact':>12}{'peak_raw':>10}{'area_break':>11}{'c_half':>9}")
    def fmt(v, nd=2): return f"{v:.{nd}f}" if not (isinstance(v, float) and np.isnan(v)) else "   nan"
    for h in headline:
        print(f"{h['concept']:<16}{h['dir']:<7}  {fmt(h['peak_intact_best']):>12}{fmt(h['peak_raw_best']):>10}"
              f"{fmt(h['area_to_break_best']):>11}{fmt(h['c_half_min'],1):>9}")
    print("\nREADING-AXIS (w_res) SPECTRUM under each candidate metric:")
    wres = [h for h in headline if h["dir"] == "w_res"]
    for metric, key, hi in (("peak intact-flip", "peak_intact_best", True),
                            ("area-to-breakdown", "area_to_break_best", True),
                            ("c_half (potency, lower=more potent)", "c_half_min", False)):
        vals = [(h["concept"], h[key]) for h in wres]
        vals = sorted(vals, key=lambda t: (np.inf if np.isnan(t[1]) else (-t[1] if hi else t[1])))
        spread = ([v for _, v in vals if not np.isnan(v)] or [float("nan")])
        rng = (max(spread) - min(spread)) if len(spread) > 1 and not any(np.isnan(spread)) else float("nan")
        print(f"  {metric}: " + " | ".join(f"{c}={fmt(v)}" for c, v in vals) +
              (f"   [range {fmt(rng)}]" if not np.isnan(rng) else ""))
    print("\nINTERPRETATION:")
    print("  - if w_res peak_intact ~0 for ALL concepts -> thesis = reading axis UNIVERSALLY inert")
    print("    (battery re-aims at universality, not spectrum); raw peak shows the dirty-flip illusion.")
    print("  - if any metric separates concepts on w_res -> that is the headline metric; pick it now.")
    print(f"per-layer: {out} | headline: {hout}")
    print("=" * 104 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--concepts", help="same manifest as 125 (needs cells_csvs)")
    p.add_argument("--out", default="data/analysis/runD_v2/dose_scalars_per_layer.csv")
    p.add_argument("--drop_tol", type=float, default=0.15,
                   help="intact-flip drop from running max that counts as format breakdown")
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    assert args.concepts, "--concepts manifest required"
    run_real(args)


if __name__ == "__main__":
    main()
