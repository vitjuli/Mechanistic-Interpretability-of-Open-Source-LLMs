import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function PromptBoxplot({ nodeInfo, data }) {
  const { interventions } = data;
  const { layer, featureIdx } = nodeInfo;

  const traces = useMemo(() => {
    if (featureIdx == null) return [];
    const rows = interventions.filter(r =>
      r.layer === layer && r.feature_indices.includes(featureIdx)
    );
    if (rows.length === 0) return [];

    const exps = [...new Set(rows.map(r => r.experiment_type))];
    return exps.map(exp => ({
      y: rows.filter(r => r.experiment_type === exp).map(r => r.effect_size),
      type: 'box',
      name: exp,
      marker: { color: EXPERIMENT_COLORS[exp] || '#888' },
      boxpoints: 'outliers',
    }));
  }, [interventions, layer, featureIdx]);

  if (traces.length === 0) return null;

  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
        EFFECT DISTRIBUTION BY PROMPT
      </div>
      <Plot
        data={traces}
        layout={{
          height: 160,
          margin: { l: 40, r: 10, t: 5, b: 30 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: { color: '#8b90a5', tickfont: { size: 10 } },
          yaxis: { color: '#8b90a5', tickfont: { size: 10 }, title: { text: 'Effect', font: { size: 10 } }, gridcolor: 'rgba(45,49,72,0.5)' },
          showlegend: true,
          legend: { font: { size: 9, color: '#8b90a5' }, x: 1, xanchor: 'right', y: 1 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
