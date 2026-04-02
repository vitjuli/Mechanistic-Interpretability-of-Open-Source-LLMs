import { useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function BaselineComparison({ nodeInfo, data, indexes }) {
  const { interventions } = data;
  const { layer, featureIdx } = nodeInfo;
  const promptTextByIdx = indexes?.promptTextByIdx;

  const traces = useMemo(() => {
    if (featureIdx == null) return [];
    const rows = interventions.filter(r =>
      r.layer === layer && r.feature_indices.includes(featureIdx)
    );
    if (rows.length === 0) return [];

    return [{
      x: rows.map(r => r.baseline_logit_diff),
      y: rows.map(r => r.intervened_logit_diff),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 6,
        color: rows.map(r => r.sign_flipped ? '#ff6b6b' : '#6c8cff'),
        opacity: 0.7,
      },
      text: rows.map(r => {
        const pInfo = promptTextByIdx?.get(r.prompt_idx);
        const label = pInfo ? `"${pInfo.prompt}" (P${r.prompt_idx})` : `P${r.prompt_idx}`;
        const tokens = pInfo ? `<br>[${pInfo.correct}] vs [${pInfo.incorrect}]` : '';
        return `${label} ${r.experiment_type}${r.sign_flipped ? ' (FLIP)' : ''}${tokens}`;
      }),
      hoverinfo: 'text+x+y',
    }, {
      x: [-5, 10],
      y: [-5, 10],
      mode: 'lines',
      type: 'scatter',
      line: { dash: 'dot', color: 'rgba(100,110,140,0.4)', width: 1 },
      hoverinfo: 'none',
    }];
  }, [interventions, layer, featureIdx]);

  if (traces.length === 0) return null;

  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
        BASELINE vs INTERVENED LOGIT DIFF
      </div>
      <Plot
        data={traces}
        layout={{
          height: 180,
          margin: { l: 40, r: 10, t: 5, b: 35 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: { color: '#8b90a5', tickfont: { size: 10 }, title: { text: 'Baseline', font: { size: 10, color: '#8b90a5' } }, gridcolor: 'rgba(45,49,72,0.5)' },
          yaxis: { color: '#8b90a5', tickfont: { size: 10 }, title: { text: 'Intervened', font: { size: 10, color: '#8b90a5' } }, gridcolor: 'rgba(45,49,72,0.5)' },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
