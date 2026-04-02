import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function EffectBarChart({ nodeInfo, data, indexes }) {
  const { interventions } = data;
  const { layer, featureIdx } = nodeInfo;
  const promptTextByIdx = indexes?.promptTextByIdx;

  const chartData = useMemo(() => {
    if (featureIdx == null) return null;
    const rows = interventions.filter(r =>
      r.layer === layer && r.feature_indices.includes(featureIdx)
    );
    if (rows.length === 0) return null;

    const exps = [...new Set(rows.map(r => r.experiment_type))].sort();
    return exps.map(exp => {
      const expRows = rows.filter(r => r.experiment_type === exp);
      return {
        y: expRows.map(r => r.effect_size),
        type: 'violin',
        name: exp,
        points: 'all',
        jitter: 0.4,
        pointpos: 0,
        line: { color: EXPERIMENT_COLORS[exp] || '#888' },
        fillcolor: (EXPERIMENT_COLORS[exp] || '#888') + '33',
        marker: {
          size: 4,
          color: EXPERIMENT_COLORS[exp] || '#888',
          opacity: 0.7,
        },
        meanline: { visible: true },
        text: expRows.map(r => {
          const pInfo = promptTextByIdx?.get(r.prompt_idx);
          const label = pInfo ? `"${pInfo.prompt}" (P${r.prompt_idx})` : `P${r.prompt_idx}`;
          const tokens = pInfo ? `<br>[${pInfo.correct}] vs [${pInfo.incorrect}]` : '';
          return `${label}${tokens}<br>Effect: ${r.effect_size?.toFixed(4)}`;
        }),
        hoverinfo: 'text',
      };
    });
  }, [interventions, layer, featureIdx, promptTextByIdx]);

  if (!chartData) return null;

  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
        EFFECT DISTRIBUTION BY EXPERIMENT
      </div>
      <Plot
        data={chartData}
        layout={{
          height: 180,
          margin: { l: 40, r: 10, t: 5, b: 30 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: { color: '#8b90a5', tickfont: { size: 10 } },
          yaxis: {
            color: '#8b90a5', tickfont: { size: 10 },
            title: { text: 'Effect', font: { size: 10 } },
            gridcolor: 'rgba(45,49,72,0.5)',
          },
          showlegend: false,
          violinmode: 'group',
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
