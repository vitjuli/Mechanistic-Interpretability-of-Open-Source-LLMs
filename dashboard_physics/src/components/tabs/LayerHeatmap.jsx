import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { filterLayerAgg, filterInterventions } from '../../utils/filterData';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function LayerHeatmap({ data, indexes }) {
  const experiments = useStore(s => s.experiments);
  const layerRange = useStore(s => s.layerRange);
  const prompts = useStore(s => s.prompts);
  const effectThreshold = useStore(s => s.effectThreshold);
  const showSignFlipsOnly = useStore(s => s.showSignFlipsOnly);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const [showDist, setShowDist] = useState(false);

  // Heatmap data
  const { z, x, y } = useMemo(() => {
    const filtered = filterLayerAgg(data.layerAgg, { experiments, layerRange });
    const exps = [...new Set(filtered.map(r => r.experiment_type))].sort();
    const layers = [...new Set(filtered.map(r => r.layer))].sort((a, b) => a - b);

    const z = exps.map(exp =>
      layers.map(layer => {
        const row = filtered.find(r => r.experiment_type === exp && r.layer === layer);
        return row ? row.mean_abs_effect_size : null;
      })
    );

    return {
      z,
      x: layers.map(l => `L${l}`),
      y: exps,
    };
  }, [data.layerAgg, experiments, layerRange]);

  // Distribution data (box+strip)
  const distTraces = useMemo(() => {
    if (!showDist) return [];
    const filtered = filterInterventions(data.interventions, {
      experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly,
    }, data.commonPromptIdx);

    const exps = [...new Set(filtered.map(r => r.experiment_type))].sort();
    return exps.map(exp => {
      const expRows = filtered.filter(r => r.experiment_type === exp);
      return {
        x: expRows.map(r => `L${r.layer}`),
        y: expRows.map(r => Math.abs(r.effect_size)),
        type: 'box',
        name: exp,
        marker: { color: EXPERIMENT_COLORS[exp] || '#888', size: 3, opacity: 0.4 },
        line: { color: EXPERIMENT_COLORS[exp] || '#888' },
        fillcolor: (EXPERIMENT_COLORS[exp] || '#888') + '33',
        boxpoints: 'all',
        jitter: 0.3,
        pointpos: 0,
      };
    });
  }, [showDist, data.interventions, data.commonPromptIdx, experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly]);

  if (z.length === 0 && !showDist) {
    return <div className="empty-state">No data for current filters</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 4, flexShrink: 0 }}>
        <button
          onClick={() => setShowDist(!showDist)}
          style={{
            padding: '3px 10px',
            fontSize: 11,
            background: showDist ? 'var(--accent)' : 'var(--bg-card)',
            color: showDist ? '#fff' : 'var(--text-dim)',
            border: `1px solid ${showDist ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: 'var(--radius)',
            cursor: 'pointer',
          }}
        >
          {showDist ? 'Show Heatmap' : 'Show Distributions'}
        </button>
      </div>
      <div style={{ flex: 1 }}>
        {showDist ? (
          <Plot
            data={distTraces}
            layout={{
              margin: { l: 50, r: 20, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: {
                color: '#8b90a5', tickfont: { size: 10 },
                title: { text: 'Layer', font: { size: 11, color: '#8b90a5' } },
              },
              yaxis: {
                color: '#8b90a5', tickfont: { size: 10 },
                title: { text: '|Effect Size|', font: { size: 11, color: '#8b90a5' } },
                gridcolor: 'rgba(45,49,72,0.5)',
              },
              showlegend: true,
              legend: { font: { size: 10, color: '#8b90a5' }, x: 1, xanchor: 'right', y: 1 },
              boxmode: 'group',
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
        ) : (
          <Plot
            data={[{
              z,
              x,
              y,
              type: 'heatmap',
              colorscale: [
                [0, '#0f1117'],
                [0.25, '#1a365d'],
                [0.5, '#2b6cb0'],
                [0.75, '#ed8936'],
                [1, '#f6e05e'],
              ],
              hovertemplate: '%{y}<br>%{x}<br>Mean |effect|: %{z:.4f}<extra></extra>',
              colorbar: {
                title: { text: 'Mean |effect|', font: { size: 10, color: '#8b90a5' } },
                tickfont: { size: 9, color: '#8b90a5' },
                len: 0.8,
              },
            }]}
            layout={{
              margin: { l: 100, r: 20, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: { color: '#8b90a5', tickfont: { size: 10 }, title: { text: 'Layer', font: { size: 11, color: '#8b90a5' } } },
              yaxis: { color: '#8b90a5', tickfont: { size: 10 } },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
        )}
      </div>
    </div>
  );
}
