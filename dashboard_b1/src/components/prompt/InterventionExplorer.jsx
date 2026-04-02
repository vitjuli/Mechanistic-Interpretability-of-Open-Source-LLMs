import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

const CIRCUIT_FEATURES = [
  'L12_F83869', 'L13_F70603', 'L14_F57525', 'L15_F127839', 'L16_F45664',
  'L22_F41906', 'L23_F64429', 'L24_F136810', 'L24_F29680', 'L24_F48363',
  'L25_F111603', 'L25_F138698', 'L25_F34754',
];

export default function InterventionExplorer({ data, indexes }) {
  const { interventions } = data;
  const { interventionsByFeature, promptById } = indexes;

  const selectedPromptIdx = useStore(s => s.selectedPromptIdx);
  const interventionFeatureId = useStore(s => s.interventionFeatureId);
  const setInterventionFeatureId = useStore(s => s.setInterventionFeatureId);
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);

  const [compareFeatureId, setCompareFeatureId] = useState(null);
  const [customFeature, setCustomFeature] = useState('');

  // All available features with ablation data
  const availableFeatures = useMemo(() => {
    return [...interventionsByFeature.keys()].sort();
  }, [interventionsByFeature]);

  // Feature effect distribution for the primary selected feature
  const featureData = useMemo(() => {
    if (!interventionFeatureId) return null;
    const rows = interventionsByFeature.get(interventionFeatureId) || [];
    if (!rows.length) return null;
    const en = rows.filter(r => r.language === 'en' || r.prompt_idx < 48);
    const fr = rows.filter(r => r.language === 'fr' || r.prompt_idx >= 48);
    return { rows, en, fr };
  }, [interventionFeatureId, interventionsByFeature]);

  const compareData = useMemo(() => {
    if (!compareFeatureId) return null;
    const rows = interventionsByFeature.get(compareFeatureId) || [];
    if (!rows.length) return null;
    const en = rows.filter(r => r.language === 'en' || r.prompt_idx < 48);
    const fr = rows.filter(r => r.language === 'fr' || r.prompt_idx >= 48);
    return { rows, en, fr };
  }, [compareFeatureId, interventionsByFeature]);

  // Per-prompt scatter: baseline vs intervened
  const scatterData = useMemo(() => {
    if (!featureData) return [];
    const traces = [];

    const makeTrace = (rows, name, color, symbol = 'circle') => ({
      x: rows.map(r => r.baseline_logit_diff),
      y: rows.map(r => r.intervened_logit_diff),
      mode: 'markers',
      type: 'scatter',
      name,
      marker: {
        color,
        size: rows.map(r => r.prompt_idx === selectedPromptIdx ? 12 : 6),
        symbol: rows.map(r => r.prompt_idx === selectedPromptIdx ? 'star' : symbol),
        line: { width: rows.map(r => r.prompt_idx === selectedPromptIdx ? 2 : 0), color: '#fff' },
      },
      text: rows.map(r => `#${r.prompt_idx}<br>Δ=${fmt(r.effect_size, 3)}`),
      hoverinfo: 'text',
    });

    traces.push(makeTrace(featureData.en, `${interventionFeatureId} EN`, '#4e9af1'));
    traces.push(makeTrace(featureData.fr, `${interventionFeatureId} FR`, '#f77f4e'));

    if (compareData) {
      traces.push(makeTrace(compareData.en, `${compareFeatureId} EN`, '#4ef7a066', 'diamond'));
      traces.push(makeTrace(compareData.fr, `${compareFeatureId} FR`, '#f7a04e66', 'diamond'));
    }

    // y=x reference line (no effect)
    const allX = featureData.rows.map(r => r.baseline_logit_diff);
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);
    traces.push({
      x: [xMin, xMax], y: [xMin, xMax],
      mode: 'lines', name: 'No effect (y=x)',
      line: { color: 'rgba(139,144,165,0.3)', width: 1, dash: 'dot' },
      hoverinfo: 'skip',
    });

    return traces;
  }, [featureData, compareData, interventionFeatureId, compareFeatureId, selectedPromptIdx]);

  // Effect distribution (box/violin per lang)
  const distData = useMemo(() => {
    if (!featureData) return [];
    const traces = [];
    const toViolin = (rows, name, color) => ({
      y: rows.map(r => r.effect_size),
      type: 'violin',
      name,
      line: { color, width: 1 },
      fillcolor: color + '33',
      meanline: { visible: true },
      points: 'all',
      jitter: 0.4,
      pointpos: 0,
      marker: { size: 3, color },
      hoverinfo: 'y',
    });

    traces.push(toViolin(featureData.en, `${interventionFeatureId} EN`, '#4e9af1'));
    traces.push(toViolin(featureData.fr, `${interventionFeatureId} FR`, '#f77f4e'));
    if (compareData) {
      traces.push(toViolin(compareData.en, `${compareFeatureId} EN`, '#4ef7a0'));
      traces.push(toViolin(compareData.fr, `${compareFeatureId} FR`, '#f7a04e'));
    }
    return traces;
  }, [featureData, compareData, interventionFeatureId, compareFeatureId]);

  // Selected prompt detail
  const promptEffect = useMemo(() => {
    if (!featureData || selectedPromptIdx == null) return null;
    return featureData.rows.find(r => r.prompt_idx === selectedPromptIdx);
  }, [featureData, selectedPromptIdx]);

  return (
    <div style={{ height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Feature selector */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>Primary feature</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
            {CIRCUIT_FEATURES.map(f => (
              <button key={f} onClick={() => { setInterventionFeatureId(f); setSelectedFeatureId(f); }}
                style={{
                  fontSize: 9, padding: '2px 6px', borderRadius: 6, cursor: 'pointer', border: '1px solid',
                  background: interventionFeatureId === f ? 'rgba(78,154,241,0.2)' : 'transparent',
                  borderColor: interventionFeatureId === f ? '#4e9af1' : 'var(--border)',
                  color: interventionFeatureId === f ? '#4e9af1' : 'var(--text-dim)',
                  fontFamily: 'monospace',
                }}>
                {f}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            <input
              type="text"
              placeholder="Custom: L22_F41906"
              value={customFeature}
              onChange={e => setCustomFeature(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && setInterventionFeatureId(customFeature)}
              style={{ fontSize: 10, padding: '3px 6px', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', outline: 'none', flex: 1 }}
            />
            <button onClick={() => setInterventionFeatureId(customFeature)}
              style={{ fontSize: 10, padding: '3px 8px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)' }}>
              Set
            </button>
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 160 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>Compare feature</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 4 }}>
            {CIRCUIT_FEATURES.slice(0, 6).map(f => (
              <button key={f} onClick={() => setCompareFeatureId(compareFeatureId === f ? null : f)}
                style={{
                  fontSize: 9, padding: '2px 6px', borderRadius: 6, cursor: 'pointer', border: '1px solid',
                  background: compareFeatureId === f ? 'rgba(116,198,157,0.2)' : 'transparent',
                  borderColor: compareFeatureId === f ? '#74c69d' : 'var(--border)',
                  color: compareFeatureId === f ? '#74c69d' : 'var(--text-dim)',
                  fontFamily: 'monospace',
                }}>
                {f}
              </button>
            ))}
          </div>
          {compareFeatureId && (
            <button onClick={() => setCompareFeatureId(null)}
              style={{ fontSize: 10, padding: '2px 6px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)' }}>
              Clear compare
            </button>
          )}
        </div>
      </div>

      {!interventionFeatureId ? (
        <div style={{ padding: 24, color: 'var(--text-dim)', textAlign: 'center' }}>Select a feature to explore its intervention effects.</div>
      ) : !featureData ? (
        <div style={{ padding: 12, color: 'var(--text-dim)' }}>No ablation data for {interventionFeatureId}.</div>
      ) : (
        <>
          {/* Selected prompt detail */}
          {promptEffect && (
            <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--accent)', fontSize: 11 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Prompt #{selectedPromptIdx} effect</div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                {[
                  { label: 'Baseline', val: fmt(promptEffect.baseline_logit_diff, 3) },
                  { label: 'After ablation', val: fmt(promptEffect.intervened_logit_diff, 3) },
                  { label: 'Effect size', val: fmt(promptEffect.effect_size, 3), color: promptEffect.effect_size < 0 ? '#d62728' : '#2ca02c' },
                ].map(({ label, val, color }) => (
                  <div key={label}>
                    <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: color || 'var(--text)' }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scatter: baseline vs intervened */}
          <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
              Baseline vs intervened Δlogit · {interventionFeatureId}
              {compareFeatureId && ` vs ${compareFeatureId}`}
            </div>
            <Plot
              data={scatterData}
              layout={{
                height: 200,
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 42, r: 10, t: 8, b: 36 },
                xaxis: { title: { text: 'Baseline Δlogit', font: { size: 10, color: '#8b90a5' } }, color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)' },
                yaxis: { title: { text: 'Intervened Δlogit', font: { size: 10, color: '#8b90a5' } }, color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)' },
                legend: { font: { size: 9, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.3 },
                showlegend: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          {/* Distribution */}
          <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>Effect size distribution (EN vs FR)</div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 6, flexWrap: 'wrap' }}>
              {[
                { label: interventionFeatureId + ' EN', rows: featureData.en },
                { label: interventionFeatureId + ' FR', rows: featureData.fr },
                ...(compareData ? [{ label: compareFeatureId + ' EN', rows: compareData.en }, { label: compareFeatureId + ' FR', rows: compareData.fr }] : []),
              ].map(({ label, rows }) => {
                const effects = rows.map(r => r.effect_size).filter(v => v != null);
                const mean = effects.length ? effects.reduce((a, b) => a + b) / effects.length : 0;
                return (
                  <div key={label} style={{ fontSize: 10 }}>
                    <div style={{ color: 'var(--text-dim)' }}>{label}</div>
                    <div style={{ color: mean < -0.1 ? '#d62728' : mean > 0.1 ? '#2ca02c' : 'var(--text)' }}>
                      mean Δ = {fmt(mean, 3)} (n={effects.length})
                    </div>
                  </div>
                );
              })}
            </div>
            <Plot
              data={distData}
              layout={{
                height: 160,
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 42, r: 10, t: 8, b: 28 },
                yaxis: { title: { text: 'effect_size', font: { size: 10, color: '#8b90a5' } }, color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)' },
                xaxis: { color: '#8b90a5', tickfont: { size: 9 } },
                legend: { font: { size: 9, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.3 },
                showlegend: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </>
      )}
    </div>
  );
}
