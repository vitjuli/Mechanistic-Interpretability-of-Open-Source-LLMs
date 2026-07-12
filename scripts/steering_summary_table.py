"""
Steering summary table (Section 3.1.2): margin flip and valid flip (intact) for the three
direction axes across all concepts, with cos(w,u) / cos(Delta_mu,u) and same-norm controls.

Reproduces the inertness tables. Reads existing aggregate steering CSVs (no model needed).

Inputs:
  data/analysis/runD_v2/<concept>/steering_delta/steering_sweep_tier1.csv   (122, dirs delta/usage/w_res/random/shuffled)
  data/analysis/runD_v2/steering_all_layers/steering_all_layers_curve.csv   (85, decay probe, w_res + nulls)
  cross_concept_particles_within.csv                                        (cos_wres_u, cos_u_delta)
Output:
  data/analysis/27c/steering_summary_all.csv
"""
import os, csv, pandas as pd

SD = "data/analysis/runD_v2"
C = 16
CONCEPTS = [
    ("alpha_beta_scaffold", "B1_alpha_beta", "ab"),
    ("alpha_beta_raw_suffix", "raw_suffix", "ab"),
    ("grammar", "B1_grammar_number", "grammar"),
    ("e_g_scaffold", "particles4_binary/electron_vs_photon", "el_ph"),
    ("e_n", "particle_pairs/electron_vs_neutron", "el_ne"),
    ("e_p", "particle_pairs/electron_vs_proton", "el_pr"),
    ("n_g", "particle_pairs/neutron_vs_photon", "ne_ph"),
    ("n_p", "particle_pairs/neutron_vs_proton", "ne_pr"),
    ("g_p", "particle_pairs/photon_vs_proton", "ph_pr"),
]


def load_cos():
    for p in ["cross_concept_particles_within.csv", "results/cross_concept_particles_within.csv"]:
        if os.path.exists(p):
            d = pd.read_csv(p)
            dz = d[(d.layer >= 20) & (d.layer <= 30)]
            return ({c: dz[dz.concept == c].cos_wres_u.abs().mean() for c in d.concept.unique()},
                    {c: dz[dz.concept == c].cos_u_delta.abs().mean() for c in d.concept.unique()})
    return {}, {}


def mx(z, d, col):
    s = z[z.dir == d]
    return round(float(s[col].max()), 3) if len(s) else ""


def main():
    cosw, cosd = load_cos()
    rows = []
    for name, path, ck in CONCEPTS:
        fp = f"{SD}/{path}/steering_delta/steering_sweep_tier1.csv"
        if not os.path.exists(fp):
            continue
        z = pd.read_csv(fp)
        z = z[z.c == C]
        randm = round(max([z[z.dir == f"random{k}"].margin_flip.max() for k in range(2) if len(z[z.dir == f"random{k}"])] + [0]), 3)
        randi = round(max([z[z.dir == f"random{k}"].flip_norm_intact.max() for k in range(2) if len(z[z.dir == f"random{k}"])] + [0]), 3)
        rows.append({"concept": name, "cos_wu": round(cosw.get(ck, float("nan")), 3),
                     "cos_du": round(cosd.get(ck, float("nan")), 3),
                     "w_margin": mx(z, "w_res", "margin_flip"), "w_intact": mx(z, "w_res", "flip_norm_intact"),
                     "delta_margin": mx(z, "delta", "margin_flip"), "delta_intact": mx(z, "delta", "flip_norm_intact"),
                     "u_margin": mx(z, "usage", "margin_flip"), "u_intact": mx(z, "usage", "flip_norm_intact"),
                     "rand_margin": randm, "rand_intact": randi})
    # decay probe (script 85, w_res only): margin saturates -> report intact only
    pf = f"{SD}/steering_all_layers/steering_all_layers_curve.csv"
    if os.path.exists(pf):
        d = pd.read_csv(pf)
        z = d[d.c == 16]
        wi = round(float(z[z.dir == "w_res"].intact.max()), 3)
        rows.insert(0, {"concept": "alpha_beta_probe", "cos_wu": round(cosw.get("ab", float("nan")), 3),
                        "cos_du": round(cosd.get("ab", float("nan")), 3),
                        "w_margin": "", "w_intact": wi, "delta_margin": "", "delta_intact": "",
                        "u_margin": "", "u_intact": "", "rand_margin": "", "rand_intact": ""})
    os.makedirs("data/analysis/27c", exist_ok=True)
    out = "data/analysis/27c/steering_summary_all.csv"
    cols = ["concept", "cos_wu", "cos_du", "w_margin", "w_intact", "delta_margin",
            "delta_intact", "u_margin", "u_intact", "rand_margin", "rand_intact"]
    with open(out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=cols)
        cw.writeheader()
        cw.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['concept']:22} cos_wu={r['cos_wu']}  w_marg={r['w_margin']}  w_int={r['w_intact']}  u_marg={r['u_margin']}")


if __name__ == "__main__":
    main()
