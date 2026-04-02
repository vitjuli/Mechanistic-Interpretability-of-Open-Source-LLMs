/**
 * ContributionView — Feature Contribution Decomposition
 *
 * Shows which features helped vs. hindered the correct prediction.
 * Uses mean_score_conditional from graph nodes:
 *   positive → feature increases logit(correct) − logit(incorrect) → HELPS
 *   negative → feature decreases logit(correct) − logit(incorrect) → COMPETES
 *
 * When a single prompt is selected (P0–P19), also shows per-prompt ablation effects.
 */
import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { shortNodeId } from '../../utils/formatters';

const COLOR_HELPS    = '#22c55e';   // green  — supports correct
const COLOR_COMPETES = '#ef4444';   // red    — supports incorrect
const LABEL_COLOR    = { scalar: '#60a5fa', vector: '#f97316' };

export default function ContributionView({ data, indexes }) {
  const { graph } = data;
  const { nodeById, featureTopPrompts } = indexes;
  const { perPromptStats } = data;

  const prompts    = useStore(s => s.prompts);
  const setSelectedPath = useStore(s => s.setSelectedPath);

  // ── Global contribution from graph node attributes ────────────────────────
  const featureContribs = useMemo(() => {
    return graph.nodes
      .filter(n => {
        const info = nodeById.get(n.id);
        return info?.type === 'feature';
      })
      .map(n => ({
        id: n.id,
        layer: n.layer ?? nodeById.get(n.id)?.layer,
        msc: n.mean_score_conditional ?? 0,
        beta: n.beta ?? 0,
        helps: (n.mean_score_conditional ?? 0) > 0,
      }))
      .sort((a, b) => Math.abs(b.msc) - Math.abs(a.msc));
  }, [graph.nodes, nodeById]);

  // ── Per-prompt ablation effects (P0–P19 only) ─────────────────────────────
  // effect_size = intervened - baseline; negative → ablating this feature hurt = feature was helping
  // contribution = -effect_size (positive = helps correct)
  const promptEffects = useMemo(() => {
    if (prompts.length !== 1) return null;
    const pid = prompts[0];
    const abl = data.interventions.filter(
      r => r.experiment_type === 'ablation_zero' && r.prompt_idx === pid
    );
    if (abl.length === 0) return null;

    const effects = new Map();
    for (const row of abl) {
      for (const fid of row.feature_indices) {
        const key = `L${row.layer}_F${fid}`;
        if (!nodeById.has(key)) continue;
        const contrib = -(row.effect_size ?? 0); // negative effect = was helping
        const prev = effects.get(key);
        if (!prev || Math.abs(contrib) > Math.abs(prev)) effects.set(key, contrib);
      }
    }
    return effects;
  }, [prompts, data.interventions, nodeById]);

  // ── Build sorted list for global bar chart ────────────────────────────────
  const sorted = useMemo(() => featureContribs, [featureContribs]);

  const barColors = sorted.map(f => f.helps ? COLOR_HELPS + 'cc' : COLOR_COMPETES + 'cc');
  const barBorderColors = sorted.map(f => f.helps ? COLOR_HELPS : COLOR_COMPETES);

  const globalTrace = {
    type: 'bar', orientation: 'h',
    y: sorted.map(f => f.id.replace('_F', '·F')),
    x: sorted.map(f => f.msc),
    marker: { color: barColors, line: { color: barBorderColors, width: 1 } },
    hovertemplate: '<b>%{y}</b><br>contribution: %{x:.3f}<extra></extra>',
    name: 'Global',
  };

  const globalLayout = {
    height: Math.max(300, sorted.length * 22 + 60),
    margin: { l: 130, r: 30, t: 30, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#c9d1d9', size: 10 },
    xaxis: {
      title: { text: 'Contribution to correct prediction (mean_score_conditional)', font: { size: 10 } },
      gridcolor: 'rgba(45,49,72,0.5)', zeroline: true, zerolinecolor: 'rgba(200,200,220,0.4)',
      zerolinewidth: 1.5,
    },
    yaxis: { automargin: true, tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)' },
    shapes: [{
      type: 'line', x0: 0, x1: 0, y0: -0.5, y1: sorted.length - 0.5,
      line: { color: 'rgba(200,200,220,0.6)', width: 1.5 },
    }],
    annotations: [
      { x: 1.5, y: sorted.length - 0.5, xref: 'x', yref: 'y',
        text: '← helps incorrect', showarrow: false, font: { size: 9, color: COLOR_COMPETES }, xanchor: 'left' },
      { x: -0.2, y: sorted.length - 0.5, xref: 'x', yref: 'y',
        text: 'helps correct →', showarrow: false, font: { size: 9, color: COLOR_HELPS }, xanchor: 'right' },
    ],
  };

  // ── Per-prompt bar (when prompt selected) ─────────────────────────────────
  const promptSorted = useMemo(() => {
    if (!promptEffects) return [];
    return [...promptEffects.entries()]
      .map(([id, contrib]) => ({ id, contrib }))
      .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib))
      .slice(0, 12);
  }, [promptEffects]);

  const hasPromptData = promptEffects && promptSorted.length > 0;

  // ── Summary stats ─────────────────────────────────────────────────────────
  const nHelps    = sorted.filter(f => f.helps).length;
  const nCompetes = sorted.filter(f => !f.helps).length;
  const netScore  = sorted.reduce((s, f) => s + f.msc, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Header cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
        <StatCard label="Helping features" value={nHelps} color={COLOR_HELPS}
          sub="push logit diff toward correct" />
        <StatCard label="Competing features" value={nCompetes} color={COLOR_COMPETES}
          sub="push logit diff toward incorrect" />
        <StatCard
          label="Net circuit balance"
          value={netScore > 0 ? `+${netScore.toFixed(2)}` : netScore.toFixed(2)}
          color={netScore > 0 ? COLOR_HELPS : COLOR_COMPETES}
          sub="sum of all contributions"
        />
      </div>

      {/* ── Global bar chart ── */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
          Global Feature Contributions
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>
          <span style={{ color: COLOR_HELPS }}>■ Green</span> = feature increases logit(correct) − logit(incorrect).{' '}
          <span style={{ color: COLOR_COMPETES }}>■ Red</span> = feature decreases it.
          Sorted by |contribution|. Based on <code>mean_score_conditional</code> across all
          prompts where each feature was active.
        </div>
        <Plot
          data={[globalTrace]}
          layout={globalLayout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* ── Per-prompt breakdown ── */}
      {hasPromptData ? (
        <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
            Prompt P{prompts[0]} — Ablation Contribution
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>
            For this specific prompt: which features, when ablated, changed the prediction?
            Contribution = −effect_size (positive = this feature was helping the correct answer).
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {promptSorted.map(({ id, contrib }) => {
              const info = nodeById.get(id);
              if (!info) return null;
              const topP = featureTopPrompts?.get(id) ?? [];
              const lc = { scalar: 0, vector: 0 };
              for (const tp of topP) {
                const lbl = perPromptStats?.[tp.prompt_idx]?.label;
                if (lbl) lc[lbl]++;
              }
              const maxAbs = Math.max(...promptSorted.map(x => Math.abs(x.contrib)));
              const barPct = Math.min(100, (Math.abs(contrib) / (maxAbs || 1)) * 100);
              const helps = contrib > 0;
              const color = helps ? COLOR_HELPS : COLOR_COMPETES;
              return (
                <div key={id} style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '5px 8px', borderRadius: 4, background: 'var(--bg-panel)', fontSize: 11,
                }}>
                  <span style={{ color: 'var(--text-dim)', flexShrink: 0, fontFamily: 'monospace', fontSize: 10, width: 110 }}>
                    {id}
                  </span>
                  <span style={{
                    flexShrink: 0, fontSize: 10, padding: '1px 6px', borderRadius: 8,
                    background: color + '22', color, fontWeight: 600, width: 56, textAlign: 'center',
                  }}>
                    {helps ? '✓ helps' : '✗ hurts'}
                  </span>
                  {/* Bar */}
                  <div style={{ flex: 1, height: 6, background: 'var(--bg-card)', borderRadius: 3 }}>
                    <div style={{ height: '100%', borderRadius: 3, width: `${barPct}%`, background: color }} />
                  </div>
                  <span style={{ flexShrink: 0, color, fontSize: 10, width: 48, textAlign: 'right', fontWeight: 600 }}>
                    {contrib > 0 ? '+' : ''}{contrib.toFixed(3)}
                  </span>
                  <span style={{ flexShrink: 0, color: 'var(--text-dim)', fontSize: 10, width: 48 }}>
                    {lc.vector}v / {lc.scalar}s
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div style={{
          background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14,
          fontSize: 11, color: 'var(--text-dim)', textAlign: 'center',
        }}>
          Select a prompt P0–P19 in the left panel to see per-prompt ablation contributions.
        </div>
      )}

    </div>
  );
}

function StatCard({ label, value, color, sub }) {
  return (
    <div style={{
      background: 'var(--bg-card)', borderRadius: 'var(--radius)',
      padding: '10px 12px', textAlign: 'center',
    }}>
      <div style={{ fontSize: 24, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', marginTop: 2 }}>{label}</div>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>{sub}</div>
    </div>
  );
}
