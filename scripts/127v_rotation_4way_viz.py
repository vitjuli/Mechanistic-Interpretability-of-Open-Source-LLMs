"""
127v — Rotation 4-way visualization. Compares α/β concept, grammar concept,
shuffled-label nulls, and structureless random-walk in a single multi-panel
figure for thesis Chapter 6 Figure 7.

Three panels:
  (A) Decay curves ρ(Δk) for all 4 sources, log-y plot showing τ-rate differences
  (B) τ comparison bar chart with bootstrap CIs
  (C) 3D SVD trajectory of α/β concept axes vs synthetic random-walk axes —
      visualizes "fast meaningful rotation" vs "slow drift"

Outputs:
  data/analysis/runD_v2/figures/rotation_4way.html — interactive Plotly
  data/analysis/runD_v2/figures/rotation_4way_summary.csv — joined data
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("127v")

OUT_DIR = Path("data/analysis/runD_v2/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rotation_data():
    """Load curves from 127 outputs."""
    rotation = pd.read_csv("data/analysis/runD_v2/rotation_control.csv")
    rw = pd.read_csv("data/analysis/runD_v2/rotation_rw_baseline.csv")
    return rotation, rw


def fit_tau(dk, rho, target=0.5):
    """Half-decorrelation length: linear interp where ρ crosses 0.5."""
    dk = np.asarray(dk, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if rho[0] < target:
        return np.nan
    for i in range(len(rho) - 1):
        if rho[i] >= target and rho[i+1] < target:
            return dk[i] + (rho[i] - target) / (rho[i] - rho[i+1] + 1e-9) * (dk[i+1] - dk[i])
    return float("inf")


def synthesize_random_walk_curve(step_corr, n_layers=36, n_seeds=64, d=2560):
    """Compute mean ρ(Δk) for structureless RW with given step persistence.

    Each axis sequence: a_{k+1} = step_corr * a_k + sqrt(1-step_corr^2) * eps,
    eps ~ unit on S^{d-1}, then normalized.
    """
    rng = np.random.default_rng(0)
    dks = np.arange(0, 16)
    rhos_all = np.zeros((n_seeds, len(dks)))
    for s in range(n_seeds):
        a = rng.standard_normal(d)
        a /= np.linalg.norm(a)
        seq = [a]
        for _ in range(n_layers - 1):
            eps = rng.standard_normal(d)
            eps /= np.linalg.norm(eps)
            new = step_corr * seq[-1] + np.sqrt(1 - step_corr**2) * eps
            new /= np.linalg.norm(new)
            seq.append(new)
        seq = np.stack(seq)
        for j, dk in enumerate(dks):
            if dk == 0:
                rhos_all[s, j] = 1.0
                continue
            pairs = [(seq[i] @ seq[i + dk]) for i in range(n_layers - dk)]
            rhos_all[s, j] = abs(np.mean(pairs))
    return dks, rhos_all.mean(0), rhos_all.std(0)


def compute_grammar_wres():
    """Compute Fisher LDA per layer on grammar field dump for 3D viz."""
    dump = Path("data/analysis/runD_v2/B1_grammar_number/field_dump")
    families_per_prompt = json.load(open(dump / "families.json"))
    # Reconstruct y from corpus jsonl + train mask via deterministic split
    prompts = [json.loads(l) for l in open("data/prompts/B1_grammar_number.jsonl")]
    y = np.array([1 if p["correct_answer"].strip() == "plural" else 0
                  for p in prompts])
    # Train families: sorted unique → seed-0 shuffle → first 60%
    unique_fams = sorted(set(families_per_prompt))
    rng = np.random.default_rng(0)
    rng.shuffle(unique_fams)
    n_train = int(round(len(unique_fams) * 0.6))
    train_fams = set(unique_fams[:n_train])
    train_mask = np.array([f in train_fams for f in families_per_prompt])
    wres = []
    for L in range(36):
        H = np.load(dump / f"res_L{L:02d}.npy")
        Htr = H[train_mask]
        ytr = y[train_mask]
        mu0 = Htr[ytr == 0].mean(0)
        mu1 = Htr[ytr == 1].mean(0)
        X0 = Htr[ytr == 0] - mu0
        X1 = Htr[ytr == 1] - mu1
        Sw = (X0.T @ X0 + X1.T @ X1) / max(len(Htr) - 2, 1)
        Sw = 0.5 * (Sw + Sw.T)
        shrink = 0.1
        Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
        Sw += 1e-6 * (np.diag(Sw).mean() + 1e-12) * np.eye(Sw.shape[0])
        w = np.linalg.solve(Sw, mu1 - mu0)
        w /= np.linalg.norm(w) + 1e-12
        wres.append(w)
    return np.stack(wres)


def project_to_svd3(arr_list, basis_arr=None):
    """Project multiple (n, d) arrays into top-3 SVD basis of basis_arr."""
    if basis_arr is None:
        basis_arr = arr_list[0]
    U, S, Vt = np.linalg.svd(basis_arr, full_matrices=False)
    basis = Vt[:3]
    return [a @ basis.T for a in arr_list], (S[:3]**2).sum() / (S**2).sum()


def main():
    rotation, rw = load_rotation_data()

    # ── Panel A: ρ(Δk) curves for all 4 cases ───────────────────────────────
    logger.info("Building Panel A — decay curves")
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "scene", "colspan": 2}, None]],
        row_heights=[0.4, 0.6],
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "(A) Decay curves ρ(Δk) — concept vs random sources",
            "(B) τ comparison — half-decorrelation length",
            "(C) 3D SVD trajectory — meaningful axes vs structureless walk",
        ),
        vertical_spacing=0.08, horizontal_spacing=0.10,
    )

    # 4 sources, distinct colors
    colors = {
        "α/β concept": "#1f77b4",
        "grammar concept": "#2ca02c",
        "shuffled label": "#9b4f96",
        "RW step=0.95": "#ff7f0e",
        "RW step=0.90": "#e07b00",
        "RW step=0.50": "#cccccc",
    }

    # Real concept curves
    for cname, color in [("alpha_beta", colors["α/β concept"]),
                          ("grammar_number", colors["grammar concept"])]:
        sub = rotation[rotation["concept"] == cname]
        fig.add_trace(go.Scatter(
            x=sub["dk"], y=sub["rho_real"],
            mode="lines+markers", name=f"{cname} concept",
            line=dict(color=color, width=3),
        ), row=1, col=1)
        # shuffled null per concept
        fig.add_trace(go.Scatter(
            x=sub["dk"], y=sub["rho_null_mean"],
            mode="lines", name=f"{cname} shuffled",
            line=dict(color=color, width=1.5, dash="dot"),
            opacity=0.5,
        ), row=1, col=1)

    # Random-walk synthetic curves
    for step_corr in [0.50, 0.90, 0.95]:
        dks, rhos, sds = synthesize_random_walk_curve(step_corr)
        name = f"RW step={step_corr:.2f}"
        col = colors.get(name, "#888888")
        fig.add_trace(go.Scatter(
            x=dks, y=rhos, mode="lines",
            name=name, line=dict(color=col, width=2, dash="dash"),
        ), row=1, col=1)

    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="τ threshold", row=1, col=1)
    fig.update_xaxes(title_text="Δk (layer displacement)", row=1, col=1,
                     range=[0, 15])
    fig.update_yaxes(title_text="|cos| ρ(Δk)", row=1, col=1,
                     range=[0, 1.05])

    # ── Panel B: τ bar chart ────────────────────────────────────────────────
    logger.info("Building Panel B — τ bar chart")
    tau_data = [
        ("α/β concept", 3.9, 0.2, colors["α/β concept"]),
        ("grammar concept", 4.4, 0.3, colors["grammar concept"]),
        ("α/β shuffled", 12.4, 1.5, colors["shuffled label"]),
        ("grammar shuffled", 8.5, 1.2, colors["shuffled label"]),
        ("RW step=0.95", 17.4, (23.9 - 11.7) / 2, colors["RW step=0.95"]),
        ("RW step=0.90", 24.3, (37.3 - 16.1) / 2, colors["RW step=0.90"]),
        ("RW step=0.80", 58.9, (87.3 - 20.9) / 2, "#bb6600"),
    ]
    names = [t[0] for t in tau_data]
    taus = [t[1] for t in tau_data]
    errs = [t[2] for t in tau_data]
    cols = [t[3] for t in tau_data]
    fig.add_trace(go.Bar(
        x=names, y=taus, marker_color=cols,
        error_y=dict(type="data", array=errs, color="black"),
        showlegend=False,
    ), row=1, col=2)
    fig.update_yaxes(title_text="τ (layers)", row=1, col=2, type="log")

    # ── Panel C: 3D trajectories — αβ vs grammar vs RW ──────────────────────
    logger.info("Building Panel C — 3D trajectories")
    # αβ axes from j105 dump
    ab_axes = []
    for L in range(36):
        path = Path(f"data/analysis/runD_v2/wres_all_layers/w_res_L{L:02d}.npy")
        if path.exists():
            v = np.load(path)
            n = np.linalg.norm(v)
            if n > 1e-6:
                v = v / n
            ab_axes.append(v)
    ab_axes = np.stack(ab_axes)

    # grammar — compute on the fly
    logger.info("computing grammar w_res per layer ...")
    grammar_axes = compute_grammar_wres()

    # synthetic RW for comparison
    rng = np.random.default_rng(0)
    rw_axes = np.zeros((36, 2560))
    a = rng.standard_normal(2560); a /= np.linalg.norm(a)
    rw_axes[0] = a
    sc = 0.95
    for i in range(1, 36):
        eps = rng.standard_normal(2560); eps /= np.linalg.norm(eps)
        new = sc * rw_axes[i-1] + np.sqrt(1 - sc**2) * eps
        rw_axes[i] = new / np.linalg.norm(new)

    # Use αβ as basis (to keep shape recognizable), project all
    projections, var_kept = project_to_svd3([ab_axes, grammar_axes, rw_axes],
                                              basis_arr=ab_axes)
    ab_proj, gn_proj, rw_proj = projections

    # 3D scatter+line for each
    for proj, name, color in [
        (ab_proj, "α/β concept", colors["α/β concept"]),
        (gn_proj, "grammar concept", colors["grammar concept"]),
        (rw_proj, "RW step=0.95", colors["RW step=0.95"]),
    ]:
        fig.add_trace(go.Scatter3d(
            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=3, color=color),
            name=name,
            text=[f"L{i:02d}" for i in range(len(proj))],
            hoverinfo="text",
        ), row=2, col=1)

    fig.update_layout(
        title=f"Rotation 4-way comparison — SVD basis: αβ trajectory (variance captured: {var_kept*100:.1f}%)",
        height=900,
        legend=dict(x=1.02, y=0.5),
    )

    out_html = OUT_DIR / "rotation_4way.html"
    fig.write_html(str(out_html))
    logger.info(f"saved {out_html}")

    # Summary table
    summary = pd.DataFrame([
        {"source": "α/β concept", "tau": 3.9, "rho_dk1": 0.602, "type": "concept"},
        {"source": "grammar concept", "tau": 4.4, "rho_dk1": 0.599, "type": "concept"},
        {"source": "α/β shuffled", "tau": 12.4, "rho_dk1": rotation[rotation['concept']=='alpha_beta'].iloc[0]['rho_null_mean'], "type": "shuffled"},
        {"source": "grammar shuffled", "tau": 8.5, "rho_dk1": rotation[rotation['concept']=='grammar_number'].iloc[0]['rho_null_mean'], "type": "shuffled"},
        {"source": "RW step=0.95", "tau": 17.4, "rho_dk1": 0.95, "type": "random-walk"},
        {"source": "RW step=0.90", "tau": 24.3, "rho_dk1": 0.90, "type": "random-walk"},
        {"source": "RW step=0.80", "tau": 58.9, "rho_dk1": 0.80, "type": "random-walk"},
    ])
    summary.to_csv(OUT_DIR / "rotation_4way_summary.csv", index=False)
    logger.info(f"saved {OUT_DIR / 'rotation_4way_summary.csv'}")

    print("\nOpen the figure in browser:")
    print(f"  open {out_html}")


if __name__ == "__main__":
    main()
