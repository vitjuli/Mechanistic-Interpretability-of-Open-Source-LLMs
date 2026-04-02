import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { filterInterventions } from '../../utils/filterData';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function ExperimentViolin({ data, indexes }) {
  const experiments = useStore(s => s.experiments);
  const layerRange = useStore(s => s.layerRange);
  const prompts = useStore(s => s.prompts);
  const effectThreshold = useStore(s => s.effectThreshold);
  const showSignFlipsOnly = useStore(s => s.showSignFlipsOnly);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);

  const traces = useMemo(() => {
    const filtered = filterInterventions(data.interventions, {
      experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly,
    }, data.commonPromptIdx);

    const exps = [...new Set(filtered.map(r => r.experiment_type))];

    // Violin traces for effect size
    const violins = exps.map(exp => ({
      y: filtered.filter(r => r.experiment_type === exp).map(r => r.effect_size),
      type: 'violin',
      name: exp,
      box: { visible: true },
      meanline: { visible: true },
      line: { color: EXPERIMENT_COLORS[exp] || '#888' },
      fillcolor: (EXPERIMENT_COLORS[exp] || '#888') + '33',
    }));

    // Sign flip rate comparison as a bar trace
    const flipRates = exps.map(exp => {
      const rows = filtered.filter(r => r.experiment_type === exp);
      const flips = rows.filter(r => r.sign_flipped).length;
      return { exp, rate: rows.length > 0 ? flips / rows.length : 0, n: rows.length };
    });

    return { violins, flipRates };
  }, [data.interventions, data.commonPromptIdx, experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly]);

  if (traces.violins.length === 0) {
    return <div className="empty-state">No data for current filters</div>;
  }

  return (
    <div style={{ display: 'flex', gap: 12, height: '100%' }}>
      <div style={{ flex: 2 }}>
        <Plot
          data={traces.violins}
          layout={{
            margin: { l: 50, r: 10, t: 10, b: 30 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            yaxis: {
              color: '#8b90a5', tickfont: { size: 10 },
              title: { text: 'Effect Size', font: { size: 11, color: '#8b90a5' } },
              gridcolor: 'rgba(45,49,72,0.5)',
            },
            xaxis: { color: '#8b90a5', tickfont: { size: 10 } },
            showlegend: false,
            violinmode: 'group',
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
      <div style={{ flex: 1 }}>
        <Plot
          data={[{
            x: traces.flipRates.map(f => f.exp),
            y: traces.flipRates.map(f => f.rate * 100),
            type: 'bar',
            marker: {
              color: traces.flipRates.map(f => EXPERIMENT_COLORS[f.exp] || '#888'),
            },
            text: traces.flipRates.map(f => `${(f.rate * 100).toFixed(1)}%<br>(n=${f.n})`),
            hoverinfo: 'text',
          }]}
          layout={{
            margin: { l: 40, r: 10, t: 10, b: 30 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            yaxis: {
              color: '#8b90a5', tickfont: { size: 10 },
              title: { text: 'Sign Flip %', font: { size: 11, color: '#8b90a5' } },
              gridcolor: 'rgba(45,49,72,0.5)',
            },
            xaxis: { color: '#8b90a5', tickfont: { size: 10 } },
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
