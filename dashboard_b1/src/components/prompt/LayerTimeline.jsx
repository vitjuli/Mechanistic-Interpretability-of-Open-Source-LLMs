import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

const ZONE_SHAPES = [
  { x0: 9.5, x1: 16.5, color: 'rgba(78,154,241,0.07)', label: 'Early', labelX: 13 },
  { x0: 16.5, x1: 22.5, color: 'rgba(247,185,78,0.05)', label: 'Middle', labelX: 19.5 },
  { x0: 22.5, x1: 25.5, color: 'rgba(116,198,157,0.07)', label: 'Late', labelX: 24 },
];

function makeZoneAnnotations() {
  return ZONE_SHAPES.map(z => ({
    xref: 'x', yref: 'paper', x: z.labelX, y: 1.05,
    text: z.label, showarrow: false, font: { size: 9, color: '#8b90a5' }, xanchor: 'center',
  }));
}

function makeZoneShapes() {
  return ZONE_SHAPES.map(z => ({
    type: 'rect', xref: 'x', yref: 'paper', x0: z.x0, x1: z.x1, y0: 0, y1: 1,
    fillcolor: z.color, line: { width: 0 },
  }));
}

export default function LayerTimeline({ data, indexes, compact = false, promptIdx: propPromptIdx }) {
  const storePromptIdx = useStore(s => s.selectedPromptIdx);
  const selectedPromptIdx = propPromptIdx != null ? propPromptIdx : storePromptIdx;

  const { layerTrajectoryByPrompt, promptById } = indexes;

  const layers = layerTrajectoryByPrompt.get(selectedPromptIdx) || [];
  const trace = selectedPromptIdx != null ? promptById.get(selectedPromptIdx) : null;

  const plotData = useMemo(() => {
    if (!layers.length) return [];

    const xs = layers.map(r => r.layer);
    const deltas = layers.map(r => parseFloat(r.layer_delta) || 0);
    const cumulative = layers.map(r => parseFloat(r.cumulative_delta) || 0);
    const projected = layers.map(r => parseFloat(r.projected_logit_diff) || 0);
    const baseline = trace?.baseline_logit_diff ?? null;

    const result = [];

    // Baseline reference line (flat)
    if (baseline != null) {
      result.push({
        x: [xs[0], xs[xs.length - 1]],
        y: [baseline, baseline],
        mode: 'lines',
        name: 'Baseline',
        line: { color: 'rgba(139,144,165,0.4)', width: 1, dash: 'dot' },
        hoverinfo: 'skip',
      });
    }

    // Projected logit diff (where would we end up if we stop here?)
    result.push({
      x: xs,
      y: projected,
      mode: 'lines+markers',
      name: 'Projected Δlogit',
      line: { color: '#4e9af1', width: 2 },
      marker: { size: 6, color: projected.map(v => v >= (baseline ?? 0) ? '#4e9af1' : '#d62728') },
      hovertemplate: 'L%{x}<br>Projected: %{y:.3f}<extra></extra>',
    });

    // Per-layer contribution
    result.push({
      x: xs,
      y: deltas,
      type: 'bar',
      name: 'Layer contribution',
      marker: {
        color: deltas.map(v => v > 0 ? 'rgba(44,160,44,0.6)' : 'rgba(214,39,40,0.6)'),
        line: { width: 0 },
      },
      hovertemplate: 'L%{x}<br>Δ: %{y:.4f}<extra>Layer contribution</extra>',
      yaxis: 'y2',
    });

    return result;
  }, [layers, trace]);

  const flipLayer = trace?.flip_layer;

  const shapes = useMemo(() => {
    const s = makeZoneShapes();
    if (flipLayer != null) {
      s.push({
        type: 'line', xref: 'x', yref: 'paper', x0: flipLayer, x1: flipLayer, y0: 0, y1: 1,
        line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
      });
    }
    return s;
  }, [flipLayer]);

  const annotations = useMemo(() => {
    const ann = makeZoneAnnotations();
    if (flipLayer != null) {
      ann.push({
        xref: 'x', yref: 'paper', x: flipLayer, y: 0.02,
        text: `flip L${flipLayer}`, showarrow: false,
        font: { size: 9, color: '#ff7f0e' }, xanchor: 'center',
      });
    }
    return ann;
  }, [flipLayer]);

  if (!trace && selectedPromptIdx == null) {
    return <div style={{ padding: compact ? 8 : 24, color: 'var(--text-dim)', textAlign: 'center', fontSize: compact ? 11 : 13 }}>Select a prompt first.</div>;
  }
  if (!layers.length) {
    return <div style={{ padding: 12, color: 'var(--text-dim)', fontSize: 11 }}>No layer trajectory data for this prompt.</div>;
  }

  const h = compact ? 130 : 200;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {!compact && trace && (
        <div style={{ fontSize: 11, marginBottom: 4, color: 'var(--text-dim)' }}>
          #{selectedPromptIdx} · {trace.prompt.slice(0, 50)}{trace.prompt.length > 50 ? '…' : ''}
          {' '}
          <span style={{ color: trace.prediction_correct ? '#2ca02c' : '#d62728' }}>
            {trace.prediction_correct ? '✓' : '✗'}
          </span>
          {flipLayer != null && <span style={{ color: '#ff7f0e', marginLeft: 8 }}>flip @ L{flipLayer}</span>}
        </div>
      )}

      <div style={{ flex: 1, minHeight: compact ? h : 160 }}>
        <Plot
          data={plotData}
          layout={{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            height: compact ? h + 10 : undefined,
            margin: { l: 42, r: compact ? 8 : 15, t: compact ? 12 : 28, b: compact ? 28 : 36 },
            xaxis: {
              title: compact ? undefined : { text: 'Layer', font: { size: 11, color: '#8b90a5' } },
              color: '#8b90a5', tickfont: { size: compact ? 8 : 9 },
              gridcolor: 'rgba(45,49,72,0.3)', zeroline: false, dtick: 1,
            },
            yaxis: {
              title: compact ? undefined : { text: 'Δlogit', font: { size: 11, color: '#8b90a5' } },
              color: '#8b90a5', tickfont: { size: compact ? 8 : 9 },
              gridcolor: 'rgba(45,49,72,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)',
            },
            yaxis2: {
              overlaying: 'y', side: 'right',
              color: '#8b90a5', tickfont: { size: compact ? 7 : 8 },
              gridcolor: 'rgba(0,0,0,0)', zeroline: false,
              showgrid: false,
              title: compact ? undefined : { text: 'per-layer Δ', font: { size: 9, color: '#8b90a5' } },
            },
            legend: compact ? { font: { size: 8 }, bgcolor: 'rgba(0,0,0,0)', x: 0, y: 1.05, orientation: 'h' }
              : { font: { size: 10, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', x: 1.08, y: 1 },
            showlegend: !compact,
            shapes,
            annotations,
            barmode: 'overlay',
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: compact ? h : '100%' }}
          useResizeHandler={!compact}
        />
      </div>
    </div>
  );
}
