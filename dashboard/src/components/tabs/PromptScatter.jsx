import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { filterPromptAgg, filterInterventions } from '../../utils/filterData';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function PromptScatter({ data, indexes }) {
  const experiments = useStore(s => s.experiments);
  const prompts = useStore(s => s.prompts);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const layerRange = useStore(s => s.layerRange);
  const effectThreshold = useStore(s => s.effectThreshold);
  const showSignFlipsOnly = useStore(s => s.showSignFlipsOnly);

  const { aggTraces, rawTraces } = useMemo(() => {
    // Aggregated means (bold markers)
    const filtered = filterPromptAgg(data.promptAgg, { experiments, prompts, showCommonPromptsOnly }, data.commonPromptIdx);
    const exps = [...new Set(filtered.map(r => r.experiment_type))];

    const aggTraces = exps.map(exp => {
      const rows = filtered.filter(r => r.experiment_type === exp);
      return {
        x: rows.map(r => r.mean_baseline_logit_diff),
        y: rows.map(r => r.mean_abs_effect_size),
        mode: 'markers',
        type: 'scatter',
        name: exp,
        marker: {
          size: 10,
          color: EXPERIMENT_COLORS[exp] || '#888',
          opacity: 0.9,
          line: { width: 1, color: 'rgba(255,255,255,0.4)' },
        },
        text: rows.map(r => {
          const pInfo = indexes.promptTextByIdx?.get(r.prompt_idx);
          const label = pInfo ? `"${pInfo.prompt}" (P${r.prompt_idx})` : `Prompt ${r.prompt_idx}`;
          const tokens = pInfo ? `<br>[${pInfo.correct}] vs [${pInfo.incorrect}]` : '';
          return `${label}${tokens}<br>Sign flip rate: ${(r.sign_flip_rate * 100).toFixed(1)}%`;
        }),
        hoverinfo: 'text+x+y',
        legendgroup: exp,
      };
    });

    // Raw background scatter
    const rawFiltered = filterInterventions(data.interventions, {
      experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly,
    }, data.commonPromptIdx);

    const rawExps = [...new Set(rawFiltered.map(r => r.experiment_type))];
    const rawTraces = rawExps.map(exp => {
      const rows = rawFiltered.filter(r => r.experiment_type === exp);
      return {
        x: rows.map(r => r.baseline_logit_diff),
        y: rows.map(r => Math.abs(r.effect_size)),
        mode: 'markers',
        type: 'scatter',
        name: exp + ' (raw)',
        marker: {
          size: 3,
          color: EXPERIMENT_COLORS[exp] || '#888',
          opacity: 0.15,
        },
        text: rows.map(r => {
          const pInfo = indexes.promptTextByIdx?.get(r.prompt_idx);
          const label = pInfo ? `"${pInfo.prompt}" (P${r.prompt_idx})` : `P${r.prompt_idx}`;
          return `${label}<br>Layer ${r.layer}<br>Effect: ${r.effect_size?.toFixed(4)}`;
        }),
        hoverinfo: 'text',
        legendgroup: exp,
        showlegend: false,
      };
    });

    return { aggTraces, rawTraces };
  }, [data.promptAgg, data.interventions, data.commonPromptIdx, indexes.promptTextByIdx, experiments, prompts, showCommonPromptsOnly, layerRange, effectThreshold, showSignFlipsOnly]);

  if (aggTraces.length === 0) {
    return <div className="empty-state">No data for current filters</div>;
  }

  return (
    <Plot
      data={[...rawTraces, ...aggTraces]}
      layout={{
        margin: { l: 50, r: 20, t: 10, b: 45 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
          color: '#8b90a5', tickfont: { size: 10 },
          title: { text: 'Baseline Logit Diff', font: { size: 11, color: '#8b90a5' } },
          gridcolor: 'rgba(45,49,72,0.5)',
        },
        yaxis: {
          color: '#8b90a5', tickfont: { size: 10 },
          title: { text: '|Effect Size|', font: { size: 11, color: '#8b90a5' } },
          gridcolor: 'rgba(45,49,72,0.5)',
        },
        showlegend: true,
        legend: { font: { size: 10, color: '#8b90a5' }, x: 1, xanchor: 'right', y: 1 },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
    />
  );
}
