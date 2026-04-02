import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';

const MODE_LABELS = {
  pooled: 'Pooled (all positions)',
  decision: 'Decision token only',
  content: 'Content word only',
};

const MODE_COLORS = {
  pooled: '#4e9af1',
  decision: '#f7b94e',
  content: '#74c69d',
};

export default function IoUChart({ data }) {
  const { iouData } = data;
  const iouMode = useStore(s => s.iouMode);
  const setIouMode = useStore(s => s.setIouMode);

  const { traces, summary } = useMemo(() => {
    if (!iouData) return { traces: [], summary: null };

    const traces = [];
    const modes = ['pooled', 'decision', 'content'].filter(m => iouData[m]);

    for (const mode of modes) {
      const rows = iouData[mode];
      traces.push({
        x: rows.map(r => r.layer),
        y: rows.map(r => r.iou),
        mode: 'lines+markers',
        name: MODE_LABELS[mode] || mode,
        line: {
          color: MODE_COLORS[mode] || '#888',
          width: mode === iouMode ? 2.5 : 1.5,
          dash: mode === iouMode ? 'solid' : 'dot',
        },
        marker: { size: mode === iouMode ? 7 : 4 },
        hovertemplate: `Layer %{x}<br>IoU: %{y:.4f}<extra>${MODE_LABELS[mode]}</extra>`,
      });
    }

    const summary = iouData[`${iouMode}_summary`];
    return { traces, summary };
  }, [iouData, iouMode]);

  if (!iouData) {
    return <div style={{ padding: 16, color: 'var(--text-dim)' }}>No IoU data available.</div>;
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Mode selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexShrink: 0 }}>
        {Object.keys(MODE_LABELS).map(mode => (
          <button
            key={mode}
            onClick={() => setIouMode(mode)}
            style={{
              fontSize: 11, padding: '4px 10px', borderRadius: 8, cursor: 'pointer', border: '1px solid',
              background: iouMode === mode ? MODE_COLORS[mode] + '33' : 'transparent',
              borderColor: iouMode === mode ? MODE_COLORS[mode] : 'var(--border)',
              color: iouMode === mode ? MODE_COLORS[mode] : 'var(--text-dim)',
            }}
          >
            {MODE_LABELS[mode]}
          </button>
        ))}
      </div>

      {/* Summary stats */}
      {summary && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 10, flexShrink: 0, flexWrap: 'wrap' }}>
          {[
            { label: 'Early (L10–11)', val: summary.early_mean },
            { label: 'Middle (L12–20)', val: summary.mid_mean },
            { label: 'Late (L21–25)', val: summary.late_mean },
            { label: 'Mid/Early ratio', val: summary.ratio, highlight: true },
          ].map(({ label, val, highlight }) => (
            <div key={label} style={{
              background: 'var(--bg-card)', borderRadius: 8, padding: '6px 10px',
              border: highlight ? '1px solid var(--accent)' : '1px solid var(--border)',
              fontSize: 11,
            }}>
              <div style={{ color: 'var(--text-dim)', marginBottom: 2 }}>{label}</div>
              <div style={{ fontWeight: 600, color: highlight ? 'var(--accent)' : 'var(--text)' }}>
                {val != null ? (highlight ? `${val.toFixed(3)}×` : val.toFixed(4)) : '—'}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Chart */}
      <div style={{ flex: 1, minHeight: 0 }}>
        <Plot
          data={traces}
          layout={{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 45, r: 15, t: 20, b: 40 },
            xaxis: {
              title: { text: 'Layer', font: { size: 11, color: '#8b90a5' } },
              color: '#8b90a5', tickfont: { size: 9 },
              gridcolor: 'rgba(45,49,72,0.3)', zeroline: false,
              dtick: 1,
            },
            yaxis: {
              title: { text: 'IoU (EN ∩ FR / EN ∪ FR)', font: { size: 11, color: '#8b90a5' } },
              color: '#8b90a5', tickfont: { size: 9 },
              gridcolor: 'rgba(45,49,72,0.3)', zeroline: false,
            },
            legend: {
              font: { size: 10, color: '#8b90a5' },
              bgcolor: 'rgba(0,0,0,0)',
              x: 1.02, y: 1, xanchor: 'left',
            },
            showlegend: true,
            shapes: [
              // Zone shading: early
              { type: 'rect', xref: 'x', yref: 'paper', x0: 9.5, x1: 11.5, y0: 0, y1: 1, fillcolor: 'rgba(78,154,241,0.07)', line: { width: 0 } },
              // Zone shading: middle
              { type: 'rect', xref: 'x', yref: 'paper', x0: 11.5, x1: 20.5, y0: 0, y1: 1, fillcolor: 'rgba(247,185,78,0.05)', line: { width: 0 } },
              // Zone shading: late
              { type: 'rect', xref: 'x', yref: 'paper', x0: 20.5, x1: 25.5, y0: 0, y1: 1, fillcolor: 'rgba(116,198,157,0.07)', line: { width: 0 } },
            ],
            annotations: [
              { xref: 'x', yref: 'paper', x: 10.5, y: 1.02, text: 'Early', showarrow: false, font: { size: 9, color: '#8b90a5' }, xanchor: 'center' },
              { xref: 'x', yref: 'paper', x: 16, y: 1.02, text: 'Middle', showarrow: false, font: { size: 9, color: '#8b90a5' }, xanchor: 'center' },
              { xref: 'x', yref: 'paper', x: 23, y: 1.02, text: 'Late', showarrow: false, font: { size: 9, color: '#8b90a5' }, xanchor: 'center' },
            ],
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>

      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4, flexShrink: 0 }}>
        IoU measures feature-set overlap between EN and FR prompts per layer. Higher = more cross-lingual sharing.
        Claim 3: middle &gt; late &gt; early (ratio {iouData[`${iouMode}_summary`]?.ratio?.toFixed(3) ?? '—'}×).
      </div>
    </div>
  );
}
