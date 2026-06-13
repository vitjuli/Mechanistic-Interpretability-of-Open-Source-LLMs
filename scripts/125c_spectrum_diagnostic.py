"""
125c — Spectrum diagnostic. Three questions in one CPU pass.

Q1. WHERE does alpha_beta w_res peak_intact = 0.22 come from?
    For each concept × dir, locate the (layer, c) cell that achieves the
    per-layer peak_intact (from 125b output). Print layer, c, intact_given_flip,
    n_correct, raw flip rate. Reveal if peak is a single extreme-c cell or
    persistent across layers.

Q2. Does the spectrum SURVIVE inside the validity radius c <= c*?
    Recompute peak_intact restricted to c <= c_star (per concept: alpha_beta
    c*=16, grammar_number c*=4). If grammar still > alpha_beta there →
    spectrum holds in linear regime. If only at c > c_star → "w_res steerable
    only under nonlinear forcing".

Q3. Does the baseline-correct pool COLLAPSE at high c?
    Per-cell n_correct counts from cells_tier2.csv. Confirm that pool is
    stable across c (it should be — baseline_correct is determined pre-steer).

Outputs (printed to stdout):
- table of peak-cell locations per (concept, dir)
- table of peak_intact within radius vs full range
- n_correct sanity check
- final headline-row verdict

Inputs:
- concepts_manifest.json (same as 125)
- assumes 125b dose_scalars_per_layer.csv already computed

Usage: python 125c_spectrum_diagnostic.py --concepts concepts_manifest.json
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("125c")


# Validity radii per concept (from 116 calibration)
C_STAR = {
    "alpha_beta": 16,
    "grammar_number": 4,
}


def load_cells(path):
    """Read cells_tier2.csv: per-target records.
    Schema: layer, c, dir, idx, y, m0, m1, intact
    where m0 = margin_clean, m1 = margin_steered, intact = 0/1.
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "layer": int(r["layer"]),
                "c": float(r["c"]),
                "dir": r["dir"],
                "idx": int(r["idx"]),
                "y": int(r["y"]),  # true class
                "m0": float(r["m0"]),
                "m1": float(r["m1"]),
                "intact": int(r["intact"]),
            })
    return rows


def baseline_correct(r):
    """Top-1 token was the correct answer pre-steer ⇔ sign(m0) matches y.
    Convention: m = logit(class1) - logit(class0). y=0 means α (need m<0), y=1 means β (need m>0).
    """
    return (r["m0"] > 0) == (r["y"] == 1)


def is_flipped(r):
    """Sign of margin changed (toward the other class)."""
    return (r["m0"] > 0) != (r["m1"] > 0)


def peak_intact_per_cell(rows, c_max=None):
    """Per (layer, dir, c), compute:
       flip_norm_intact = #(baseline_correct AND flipped AND intact) / #baseline_correct
       intact_given_flip = #(intact AND flipped) / #flipped
       n_correct = #baseline_correct
       raw_flip = #flipped / #baseline_correct
    """
    by_cell = defaultdict(list)
    for r in rows:
        if c_max is not None and r["c"] > c_max:
            continue
        by_cell[(r["layer"], r["dir"], r["c"])].append(r)
    out = {}
    for (L, d, c), cell in by_cell.items():
        correct = [x for x in cell if baseline_correct(x)]
        n_correct = len(correct)
        flips_among_correct = [x for x in correct if is_flipped(x)]
        intact_flips_among_correct = [x for x in flips_among_correct if x["intact"]]
        flips_all = [x for x in cell if is_flipped(x)]
        intact_among_flips_all = [x for x in flips_all if x["intact"]]
        flip_norm_intact = (len(intact_flips_among_correct) / n_correct
                             if n_correct > 0 else 0.0)
        raw_flip = len(flips_among_correct) / n_correct if n_correct > 0 else 0.0
        intact_given_flip = (len(intact_among_flips_all) / len(flips_all)
                              if flips_all else 1.0)
        out[(L, d, c)] = {
            "flip_norm_intact": flip_norm_intact,
            "raw_flip": raw_flip,
            "intact_given_flip": intact_given_flip,
            "n_correct": n_correct,
            "n_cell": len(cell),
        }
    return out


def find_peaks(per_cell, concept_name):
    """For each (dir, layer), find c with max flip_norm_intact.
    Then for each dir, find global peak across (layer, c).
    """
    # group by dir
    by_dir = defaultdict(list)
    for (L, d, c), m in per_cell.items():
        by_dir[d].append((L, c, m))
    summary = {}
    for d, items in by_dir.items():
        # global peak across (layer, c)
        items_sorted = sorted(items, key=lambda x: x[2]["flip_norm_intact"],
                              reverse=True)
        peak_L, peak_c, peak_m = items_sorted[0]
        summary[d] = {
            "peak_L": peak_L, "peak_c": peak_c,
            "peak_flip_norm_intact": peak_m["flip_norm_intact"],
            "peak_raw_flip": peak_m["raw_flip"],
            "peak_intact_given_flip": peak_m["intact_given_flip"],
            "peak_n_correct": peak_m["n_correct"],
        }
        # per-layer peaks for visibility
        per_layer = defaultdict(list)
        for L, c, m in items:
            per_layer[L].append((c, m))
        summary[d]["per_layer_peaks"] = {
            L: max(cells, key=lambda x: x[1]["flip_norm_intact"])
            for L, cells in per_layer.items()
        }
    return summary


def run(concepts_path):
    concepts = json.load(open(concepts_path))
    results = {}
    for cfg in concepts:
        name = cfg["name"]
        cells_path = cfg["cells_csvs"][0] if cfg.get("cells_csvs") else None
        if not cells_path or not Path(cells_path).exists():
            logger.info(f"concept {name}: no cells_tier2.csv — skip")
            continue
        rows = load_cells(cells_path)
        logger.info(f"{name}: {len(rows)} per-target rows")

        full = peak_intact_per_cell(rows)
        peaks_full = find_peaks(full, name)

        c_star = C_STAR.get(name, 4)
        within = peak_intact_per_cell(rows, c_max=c_star)
        peaks_within = find_peaks(within, name)

        results[name] = {
            "c_star": c_star,
            "full": peaks_full,
            "within_radius": peaks_within,
        }

    # ── Q1: WHERE does peak come from? ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("Q1. PEAK-CELL LOCATIONS — where does each concept's w_res peak intact-flip actually live?")
    print("=" * 100)
    print(f"{'concept':16} {'dir':6} {'L':>4} {'c':>6} {'flip_norm_int':>12} "
          f"{'raw_flip':>9} {'int|flip':>9} {'n_corr':>7}")
    print("-" * 100)
    for name, res in results.items():
        for d in ("w_res", "usage"):
            p = res["full"][d]
            print(f"{name:16} {d:6} {p['peak_L']:>4} {p['peak_c']:>6.1f} "
                  f"{p['peak_flip_norm_intact']:>12.3f} "
                  f"{p['peak_raw_flip']:>9.3f} "
                  f"{p['peak_intact_given_flip']:>9.3f} "
                  f"{p['peak_n_correct']:>7d}")

    # ── Q2: SPECTRUM within radius vs full ──────────────────────────────────
    print("\n" + "=" * 100)
    print("Q2. SPECTRUM IN VALIDITY RADIUS — does w_res-spectrum survive at c ≤ c*?")
    print("=" * 100)
    print(f"{'concept':16} {'c*':>4} {'full peak (any c)':>20} {'within radius (c≤c*)':>22}")
    print("-" * 100)
    for name, res in results.items():
        c_star = res["c_star"]
        for d in ("w_res",):
            full_p = res["full"][d]["peak_flip_norm_intact"]
            full_c = res["full"][d]["peak_c"]
            full_L = res["full"][d]["peak_L"]
            within_p = res["within_radius"][d]["peak_flip_norm_intact"]
            within_c = res["within_radius"][d]["peak_c"]
            within_L = res["within_radius"][d]["peak_L"]
            print(f"{name:16} {c_star:>4} "
                  f"{full_p:>10.3f} at L{full_L:>2} c={full_c:>4.1f}"
                  f"{within_p:>10.3f} at L{within_L:>2} c={within_c:>4.1f}")

    print()
    print("Reading:")
    print(" - If 'within' is comparable to 'full' for grammar AND grammar > α/β within:")
    print("   → spectrum is clean in the LINEAR regime. Headline metric robust.")
    print(" - If grammar 'within' << grammar 'full', and only 'full' > α/β:")
    print("   → w_res-steerability requires nonlinear forcing (c > c*).")
    print("   → Reformulate: 'grammar's w_res accessible only beyond strict radius'.")

    # ── Q3: n_correct stability across c ────────────────────────────────────
    print("\n" + "=" * 100)
    print("Q3. n_CORRECT STABILITY — does baseline-correct pool collapse at high c?")
    print("=" * 100)
    for name, res in results.items():
        rows = load_cells(json.load(open(concepts_path))[
            [c["name"] for c in json.load(open(concepts_path))].index(name)
        ]["cells_csvs"][0])
        # baseline correct fraction per c (should be constant since determined pre-steer)
        by_c = defaultdict(list)
        for r in rows:
            by_c[r["c"]].append(r)
        print(f"\n  {name}:")
        print(f"  {'c':>6} {'n_cell':>8} {'n_correct':>10} {'frac':>6}")
        for c in sorted(by_c):
            cell = by_c[c]
            # cells_tier2 is per-target × per-layer × per-c × per-dir, so n_cell is large
            # n_correct is from sign(m0) — same for any (layer, dir, c) since m0 is clean
            # but actually m0 IS layer-specific (it's measured at the steering layer)
            # so we group by (layer, dir) within this c
            correct_count = sum(1 for r in cell if baseline_correct(r))
            print(f"  {c:>6.1f} {len(cell):>8} {correct_count:>10} "
                  f"{correct_count/len(cell):>6.2f}")

    # ── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FINAL VERDICT — pick the headline framing")
    print("=" * 100)
    gn = results.get("grammar_number")
    ab = results.get("alpha_beta")
    if gn and ab:
        gn_full = gn["full"]["w_res"]["peak_flip_norm_intact"]
        gn_within = gn["within_radius"]["w_res"]["peak_flip_norm_intact"]
        ab_full = ab["full"]["w_res"]["peak_flip_norm_intact"]
        ab_within = ab["within_radius"]["w_res"]["peak_flip_norm_intact"]
        ratio_full = gn_full / max(ab_full, 1e-9)
        ratio_within = gn_within / max(ab_within, 1e-9)
        print(f"  full range (any c):   grammar={gn_full:.3f}  α/β={ab_full:.3f}  ratio={ratio_full:.1f}×")
        print(f"  within radius c≤c*:  grammar={gn_within:.3f}  α/β={ab_within:.3f}  ratio={ratio_within:.1f}×")
        if ratio_within >= 2.0 and gn_within >= 0.20:
            print()
            print("  ★ SPECTRUM SURVIVES IN LINEAR REGIME → headline = peak_intact within radius.")
            print("    Clean result: bypass is graded, measurable inside the predictor's validity.")
        elif ratio_full >= 2.0 and ratio_within < 2.0:
            print()
            print("  ⚡ SPECTRUM REQUIRES NONLINEAR FORCING → headline = peak_intact (any c),")
            print("    with explicit note that grammar's w_res-steerability lives at c > c*.")
            print("    Reformulate §6.3: 'reading-axis steerability is concept-graded under")
            print("    extreme forcing but linear theory predicts inertness for both at c ≤ c*'.")
        else:
            print()
            print("  ⊘ SPECTRUM WEAK → both concepts ~inert at clean intact-flip.")
            print("    Battery re-aims at universality of reading-axis inertness.")
    print("=" * 100 + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concepts", default="concepts_manifest.json")
    return p


def main():
    args = build_parser().parse_args()
    run(args.concepts)


if __name__ == "__main__":
    main()
