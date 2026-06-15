"""
126v — Strategy convergence visualization.

Two interactive panels from j126 data showing the strategy-detector finding:

  (A) Convergence to γ̄ — per-layer cos(u_raw, γ̄) vs cos(u_scaffold, γ̄).
      Both routes start low at mid-stack and converge to γ̄ at the readout.

  (B) Cross-regime cosine — per-layer |cos(u_raw, u_scaffold)| with within-span
      null band. Mid-stack near/below null (orthogonal); late-stack converges
      to 1.0 (same direction).

Output: data/analysis/runD_v2/figures/strategy_convergence.html (interactive Plotly)
"""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = Path("data/analysis/runD_v2/usage_field_strategy.csv")
OUT_DIR = Path("data/analysis/runD_v2/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=(
            "(A) Convergence to γ̄ — both u_raw and u_scaffold align with the readout direction at the end",
            "(B) Cross-regime alignment — |cos(u_raw, u_scaffold)| per layer with random null",
        ),
        row_heights=[0.5, 0.5],
    )

    # ── Panel A: cos(u, γ̄) per layer ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["cos_uRaw_gamma"],
        mode="lines+markers", name="cos(u_raw, γ̄)",
        line=dict(color="#c0504d", width=3),
        marker=dict(size=7),
        hovertemplate="L%{x}<br>cos(u_raw, γ̄)=%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["cos_uSca_gamma"],
        mode="lines+markers", name="cos(u_scaffold, γ̄)",
        line=dict(color="#3a6fb0", width=3),
        marker=dict(size=7),
        hovertemplate="L%{x}<br>cos(u_scaffold, γ̄)=%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # commitment onset L17
    fig.add_vline(x=17, line_dash="dot", line_color="#c87822", opacity=0.6,
                  annotation_text="L17 commitment onset",
                  annotation_position="top left",
                  row=1, col=1)
    # ignition zone L20-L24
    fig.add_vrect(x0=20, x1=24, fillcolor="#ffb84d", opacity=0.15,
                  line_width=0,
                  annotation_text="ignition L20–24",
                  annotation_position="top left",
                  row=1, col=1)
    # readout annotation
    fig.add_annotation(
        x=35, y=0.95, xref="x1", yref="y1",
        text="both → γ̄<br>at readout",
        showarrow=True, arrowhead=2, ax=-40, ay=-30,
        font=dict(color="#444", size=10),
        bgcolor="rgba(255,255,255,0.85)",
        row=1, col=1,
    )

    fig.update_xaxes(title_text="layer ℓ", row=1, col=1,
                     tickmode="linear", dtick=2, range=[-0.5, 36])
    fig.update_yaxes(title_text="cos(u, γ̄)", row=1, col=1,
                     range=[-0.02, 1.05])

    # ── Panel B: cross-regime |cos(u_raw, u_scaffold)| with null band ──────
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["abs_cos_uRaw_uSca"],
        mode="lines+markers", name="|cos(u_raw, u_scaffold)|",
        line=dict(color="#5ab37e", width=3),
        marker=dict(size=8),
        hovertemplate="L%{x}<br>|cos(u_R, u_S)|=%{y:.3f}<extra></extra>",
    ), row=2, col=1)
    # null p95 band
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["null_p95"],
        mode="lines", name="within-span null p95",
        line=dict(color="#888888", width=2, dash="dash"),
        hovertemplate="L%{x}<br>null p95=%{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # commitment + ignition again for context
    fig.add_vline(x=17, line_dash="dot", line_color="#c87822", opacity=0.6,
                  row=2, col=1)
    fig.add_vrect(x0=20, x1=24, fillcolor="#ffb84d", opacity=0.15,
                  line_width=0, row=2, col=1)

    # annotation: mid-stack orthogonal
    fig.add_annotation(
        x=16, y=0.04, xref="x2", yref="y2",
        text="mid-stack: orthogonal<br>(at null)",
        showarrow=True, arrowhead=2, ax=-30, ay=-50,
        font=dict(color="#2f6e3f", size=10),
        bgcolor="rgba(255,255,255,0.85)",
        row=2, col=1,
    )
    # annotation: late convergent
    fig.add_annotation(
        x=33, y=0.85, xref="x2", yref="y2",
        text="late-stack:<br>fully convergent",
        showarrow=True, arrowhead=2, ax=30, ay=20,
        font=dict(color="#2f6e3f", size=10),
        bgcolor="rgba(255,255,255,0.85)",
        row=2, col=1,
    )

    fig.update_xaxes(title_text="layer ℓ", row=2, col=1,
                     tickmode="linear", dtick=2, range=[-0.5, 36])
    fig.update_yaxes(title_text="|cos(u_raw, u_scaffold)|",
                     row=2, col=1, range=[-0.02, 1.1])

    # Layout
    fig.update_layout(
        title=("j126 — Strategy detection: u_raw vs u_scaffold through depth<br>"
               "<sup>Same concept (α/β), same answer, two evaluation regimes. "
               "Mid-stack: orthogonal usage directions. Readout: both converge to γ̄.</sup>"),
        height=820,
        hovermode="x unified",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=80, r=40, t=110, b=70),
    )

    out_path = OUT_DIR / "strategy_convergence.html"
    fig.write_html(str(out_path))
    print(f"Saved: {out_path}")
    print(f"Open with: open {out_path}")

    # Print key numbers
    mid = df[df["layer"].between(14, 20)]
    late = df[df["layer"] >= 33]
    print(f"\nKey numbers from data:")
    print(f"  mid-stack (L14-L20):")
    print(f"    |cos(u_R, u_S)| median = {mid['abs_cos_uRaw_uSca'].median():.4f}")
    print(f"    null p95 median        = {mid['null_p95'].median():.4f}")
    print(f"    cos(u_raw, γ̄) median   = {mid['cos_uRaw_gamma'].median():.4f}")
    print(f"    cos(u_sca, γ̄) median   = {mid['cos_uSca_gamma'].median():.4f}")
    print(f"  readout (L33-L35):")
    print(f"    |cos(u_R, u_S)| = {late['abs_cos_uRaw_uSca'].iloc[-1]:.4f}")
    print(f"    cos(u_raw, γ̄) = {late['cos_uRaw_gamma'].iloc[-1]:.4f}")
    print(f"    cos(u_sca, γ̄) = {late['cos_uSca_gamma'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
