"""
diag_h4_layer_stratification.py

H4 (detector vs executor): stratify existing IIA results by layer to see
whether late layers within a cluster carry more causal effect than early ones.

Reads:
  data/analysis/random_init_control/{behaviour}/random_init_control_*_train.csv

Outputs:
  data/analysis/iia_failure_diagnosis/h4_layer_stratification.csv
  data/analysis/iia_failure_diagnosis/h4_summary.json
  stdout: per-(cluster, layer) IIA + per-layer pooled IIA across clusters

Literature: Anthropic 2025 (On the Biology of a LLM) — reading vs writing
components; Marks et al. 2024 (Sparse Feature Circuits) — feature roles in circuits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default="physics_decay_type_probe_v2")
    ap.add_argument("--split",     default="train")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    csv_path = (root / "data" / "analysis" / "random_init_control" / args.behaviour
                / f"random_init_control_{args.behaviour}_{args.split}.csv")
    out_dir = root / "data" / "analysis" / "iia_failure_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # ── per-(cluster, layer) IIA ─────────────────────────────────────────────
    by_cl = (df.groupby(["cluster_id", "layer"])
               .agg(n=("iia_trained", "size"),
                    iia_trained=("iia_trained", "mean"),
                    iia_rand=("iia_rand", "mean"),
                    delta_orig_abs=("delta_orig", lambda s: s.abs().mean()),
                    active_tr=("active_frac_G_trained", "mean"),
                    delta_tr_patched_abs=("delta_tr_patched", lambda s: s.abs().mean()),
                    delta_shift=("delta_tr_patched",
                                 lambda s: (df.loc[s.index, "delta_tr_patched"]
                                            - df.loc[s.index, "delta_orig"]).abs().mean()))
               .reset_index())
    by_cl["delta_ctrl"] = by_cl["iia_trained"] - by_cl["iia_rand"]

    print("\n=== Per (cluster, layer) IIA ===")
    print(by_cl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── pooled by layer (across all clusters that touch that layer) ─────────
    by_layer = (df.groupby("layer")
                  .agg(n=("iia_trained", "size"),
                       n_clusters=("cluster_id", "nunique"),
                       iia_trained=("iia_trained", "mean"),
                       iia_rand=("iia_rand", "mean"),
                       delta_shift_abs=("delta_tr_patched",
                                        lambda s: (df.loc[s.index, "delta_tr_patched"]
                                                   - df.loc[s.index, "delta_orig"]).abs().mean()))
                  .reset_index())

    print("\n=== Pooled per-layer IIA (across clusters) ===")
    print(by_layer.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── within-cluster layer comparison (only multi-layer clusters) ─────────
    multi_layer = df.groupby("cluster_id")["layer"].nunique()
    multi_layer = multi_layer[multi_layer > 1].index.tolist()
    print(f"\n=== Multi-layer clusters: {multi_layer} ===")

    within_cluster_layer_diff = []
    for cid in multi_layer:
        sub = by_cl[by_cl["cluster_id"] == cid].sort_values("layer")
        early_layer = sub["layer"].min()
        late_layer  = sub["layer"].max()
        early_iia = sub[sub["layer"] == early_layer]["iia_trained"].values[0]
        late_iia  = sub[sub["layer"] == late_layer]["iia_trained"].values[0]
        early_shift = sub[sub["layer"] == early_layer]["delta_shift"].values[0]
        late_shift  = sub[sub["layer"] == late_layer]["delta_shift"].values[0]
        within_cluster_layer_diff.append({
            "cluster_id": int(cid),
            "early_layer": int(early_layer),
            "late_layer":  int(late_layer),
            "early_iia":   float(early_iia),
            "late_iia":    float(late_iia),
            "late_minus_early_iia": float(late_iia - early_iia),
            "early_delta_shift": float(early_shift),
            "late_delta_shift":  float(late_shift),
            "late_minus_early_shift": float(late_shift - early_shift),
        })
        print(f"  C{cid}: early L{early_layer} IIA={early_iia:.4f} shift={early_shift:.3f}  "
              f"|  late L{late_layer} IIA={late_iia:.4f} shift={late_shift:.3f}  "
              f"|  Δ(late−early)={late_iia-early_iia:+.4f}")

    # ── correlation: layer vs |Δ_shift| pooled ───────────────────────────────
    layer_vec = by_layer["layer"].values
    shift_vec = by_layer["delta_shift_abs"].values
    iia_vec   = by_layer["iia_trained"].values
    corr_layer_iia = float(np.corrcoef(layer_vec, iia_vec)[0, 1])
    corr_layer_shift = float(np.corrcoef(layer_vec, shift_vec)[0, 1])

    print(f"\n=== Correlations ===")
    print(f"  Pearson(layer, IIA_trained)       = {corr_layer_iia:+.4f}")
    print(f"  Pearson(layer, |Δ_shift|_pooled)  = {corr_layer_shift:+.4f}")
    if corr_layer_iia > 0.3:
        print("  → late layers give MORE IIA — H4 detector/executor split SUPPORTED")
    elif corr_layer_iia < -0.3:
        print("  → late layers give LESS IIA — H4 in opposite direction (executor early?)")
    else:
        print("  → no clear layer trend — H4 not supported in axis-aligned IIA")

    # ── save ─────────────────────────────────────────────────────────────────
    by_cl.to_csv(out_dir / "h4_per_cluster_layer.csv", index=False)
    by_layer.to_csv(out_dir / "h4_per_layer_pooled.csv", index=False)

    summary = {
        "behaviour":   args.behaviour,
        "n_rows":      int(len(df)),
        "n_clusters":  int(df["cluster_id"].nunique()),
        "layers":      sorted(df["layer"].unique().tolist()),
        "iia_trained_overall": float(df["iia_trained"].mean()),
        "iia_rand_overall":    float(df["iia_rand"].mean()),
        "best_layer_iia":      {"layer": int(by_layer.loc[by_layer["iia_trained"].idxmax(), "layer"]),
                                "iia":  float(by_layer["iia_trained"].max())},
        "best_layer_shift":    {"layer": int(by_layer.loc[by_layer["delta_shift_abs"].idxmax(), "layer"]),
                                "shift": float(by_layer["delta_shift_abs"].max())},
        "within_cluster_layer_diff": within_cluster_layer_diff,
        "corr_layer_iia":       corr_layer_iia,
        "corr_layer_shift":     corr_layer_shift,
        "h4_verdict": ("SUPPORTED (late > early)" if corr_layer_iia > 0.3 else
                       "INVERTED (early > late)"  if corr_layer_iia < -0.3 else
                       "NOT SUPPORTED in axis-aligned IIA"),
    }
    with open(out_dir / "h4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_dir}/h4_*.{{csv,json}}")


if __name__ == "__main__":
    main()
