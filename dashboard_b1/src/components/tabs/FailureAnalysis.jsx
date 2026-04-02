import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

export default function FailureAnalysis({ data, indexes }) {
  const { errorCases } = data;
  const { promptTextByIdx } = indexes;
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const setPrompts = useStore(s => s.setPrompts);

  const analysis = useMemo(() => {
    if (!errorCases) return null;

    const { n_incorrect, n_correct, accuracy, incorrect_prompt_indices,
      zone_comparison, concept_errors, feature_comparison_ablation_only,
      flip_layer_distribution, failure_types, trajectory_prediction_accuracy } = errorCases;

    // Zone comparison bar data
    const zones = ['early', 'mid', 'late'];
    const zoneTraces = [
      {
        name: 'Correct',
        x: zones,
        y: zones.map(z => zone_comparison[z]?.correct_mean ?? 0),
        type: 'bar',
        marker: { color: '#2ca02c' },
        hovertemplate: '%{x}: %{y:.4f}<extra>Correct</extra>',
      },
      {
        name: 'Incorrect',
        x: zones,
        y: zones.map(z => zone_comparison[z]?.incorrect_mean ?? 0),
        type: 'bar',
        marker: { color: '#d62728' },
        hovertemplate: '%{x}: %{y:.4f}<extra>Incorrect</extra>',
      },
    ];

    // Top differentiating features
    const topFeatures = feature_comparison_ablation_only
      ? [...feature_comparison_ablation_only]
          .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
          .slice(0, 10)
      : [];

    // Concept error bar
    const conceptKeys = concept_errors ? Object.keys(concept_errors).sort() : [];
    const conceptTrace = conceptKeys.length > 0 ? [{
      x: conceptKeys,
      y: conceptKeys.map(k => concept_errors[k]),
      type: 'bar',
      marker: {
        color: conceptKeys.map(k => {
          const n = concept_errors[k];
          return n >= 4 ? '#d62728' : n >= 2 ? '#ff7f0e' : '#f7b94e';
        }),
      },
      hovertemplate: 'Concept %{x}: %{y} errors<extra></extra>',
    }] : [];

    return {
      n_incorrect, n_correct, accuracy,
      incorrect_prompt_indices,
      zone_comparison,
      zoneTraces,
      conceptTrace,
      conceptKeys,
      topFeatures,
      flip_layer_distribution,
      failure_types,
      trajectory_prediction_accuracy,
    };
  }, [errorCases]);

  if (!analysis) {
    return <div style={{ padding: 16, color: 'var(--text-dim)' }}>No error analysis data available.</div>;
  }

  const { n_incorrect, n_correct, accuracy, zoneTraces, conceptTrace, conceptKeys, topFeatures,
    incorrect_prompt_indices, failure_types, trajectory_prediction_accuracy } = analysis;

  return (
    <div style={{ display: 'flex', gap: 16, height: '100%', overflow: 'auto' }}>
      {/* Left: summary stats + incorrect prompt list */}
      <div style={{ width: 220, flexShrink: 0 }}>
        {/* Accuracy header */}
        <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, marginBottom: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Overall Accuracy</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: accuracy >= 0.9 ? 'var(--success)' : accuracy >= 0.75 ? 'var(--warning)' : 'var(--danger)' }}>
            {(accuracy * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
            {n_correct} correct / {n_incorrect} incorrect
          </div>
        </div>

        {/* Trajectory prediction */}
        {trajectory_prediction_accuracy != null && (
          <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, marginBottom: 10, border: '1px solid var(--border)', fontSize: 11 }}>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Trajectory accuracy</div>
            <div style={{ color: 'var(--text-dim)' }}>{(trajectory_prediction_accuracy * 100).toFixed(1)}%</div>
          </div>
        )}

        {/* Failure types */}
        {failure_types && Object.keys(failure_types).length > 0 && (
          <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, marginBottom: 10, border: '1px solid var(--border)', fontSize: 11 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Failure types</div>
            {Object.entries(failure_types).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ color: 'var(--text-dim)' }}>{k.replace(/_/g, ' ')}</span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        )}

        {/* Incorrect prompt list */}
        <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6 }}>Incorrect prompts ({incorrect_prompt_indices?.length ?? 0})</div>
          <div style={{ maxHeight: 200, overflow: 'auto' }}>
            {(incorrect_prompt_indices || []).map(idx => {
              const pt = promptTextByIdx.get(idx);
              return (
                <div
                  key={idx}
                  style={{ fontSize: 10, marginBottom: 4, cursor: 'pointer', padding: '2px 4px', borderRadius: 4, background: 'rgba(214,39,40,0.08)' }}
                  onClick={() => setPrompts([idx])}
                >
                  <span style={{ color: 'var(--text-dim)', marginRight: 4 }}>#{idx}</span>
                  {pt ? (
                    <span style={{ color: 'var(--text)' }} title={pt.prompt}>
                      {pt.prompt.slice(0, 30)}{pt.prompt.length > 30 ? '…' : ''}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-dim)' }}>prompt {idx}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Center: zone comparison + top discriminating features */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Zone contribution comparison */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: 'var(--text-dim)' }}>
            Zone contribution (correct vs incorrect)
          </div>
          <Plot
            data={zoneTraces}
            layout={{
              height: 140,
              barmode: 'group',
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              margin: { l: 40, r: 10, t: 5, b: 30 },
              xaxis: { color: '#8b90a5', tickfont: { size: 10 } },
              yaxis: { color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)' },
              legend: { font: { size: 10, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.25 },
              showlegend: true,
            }}
            config={{ staticPlot: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Concept error distribution */}
        {conceptKeys.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: 'var(--text-dim)' }}>
              Errors per concept
            </div>
            <Plot
              data={conceptTrace}
              layout={{
                height: 100,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 30, r: 10, t: 5, b: 25 },
                xaxis: { color: '#8b90a5', tickfont: { size: 9 } },
                yaxis: { color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: false, dtick: 1 },
                showlegend: false,
              }}
              config={{ staticPlot: true }}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
              Concept c4 and c7 account for the majority of errors (FR-specific failure modes).
            </div>
          </div>
        )}

        {/* Top discriminating features */}
        {topFeatures.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, color: 'var(--text-dim)' }}>
              Top discriminating features (Δ contribution: correct − incorrect)
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {topFeatures.map((f, i) => (
                <div
                  key={f.feature_id}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    background: 'var(--bg-card)', borderRadius: 6, padding: '4px 8px',
                    cursor: 'pointer', fontSize: 11,
                    border: `1px solid ${f.delta < 0 ? 'rgba(214,39,40,0.3)' : 'rgba(44,160,44,0.3)'}`,
                  }}
                  onClick={() => setSelectedFeatureId(f.feature_id)}
                >
                  <span style={{ color: 'var(--text-dim)', minWidth: 16 }}>#{i + 1}</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600, minWidth: 100 }}>{f.feature_id}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <span style={{ color: '#2ca02c' }}>✓ {fmt(f.correct_mean, 3)}</span>
                      <span style={{ color: '#d62728' }}>✗ {fmt(f.incorrect_mean, 3)}</span>
                      <span style={{ color: f.delta < 0 ? '#d62728' : '#2ca02c', fontWeight: 600 }}>
                        Δ={fmt(f.delta, 3)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
