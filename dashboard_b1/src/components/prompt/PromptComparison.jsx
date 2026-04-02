import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import LayerTimeline from './LayerTimeline';
import { fmt } from '../../utils/formatters';

function MiniCard({ trace, label, color }) {
  if (!trace) return (
    <div style={{ flex: 1, background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)', color: 'var(--text-dim)', fontSize: 11, textAlign: 'center' }}>
      {label}: not selected
    </div>
  );
  return (
    <div style={{ flex: 1, background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: `1px solid ${color}44` }}>
      <div style={{ fontSize: 10, color, fontWeight: 600, marginBottom: 4 }}>{label} · #{trace.prompt_idx}</div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{trace.prompt.slice(0, 60)}{trace.prompt.length > 60 ? '…' : ''}</div>
      <div style={{ display: 'flex', gap: 8, fontSize: 11, flexWrap: 'wrap' }}>
        <span style={{ color: trace.prediction_correct ? '#2ca02c' : '#d62728' }}>
          {trace.prediction_correct ? '✓' : '✗'}
        </span>
        <span style={{ color: 'var(--text-dim)' }}>{trace.language?.toUpperCase()}</span>
        <span style={{ color: 'var(--text-dim)' }}>c{trace.concept_index}</span>
        <span style={{ color: 'var(--text-dim)' }}>Δ={fmt(trace.baseline_logit_diff, 2)}</span>
        {trace.flip_layer != null && <span style={{ color: '#ff7f0e' }}>flip@L{trace.flip_layer}</span>}
      </div>
    </div>
  );
}

function DivergenceChart({ layersA, layersB, traceA, traceB }) {
  const data = useMemo(() => {
    if (!layersA.length && !layersB.length) return [];

    const xs = [...new Set([...layersA.map(r => r.layer), ...layersB.map(r => r.layer)])].sort((a, b) => a - b);
    const mapA = new Map(layersA.map(r => [r.layer, parseFloat(r.projected_logit_diff) || 0]));
    const mapB = new Map(layersB.map(r => [r.layer, parseFloat(r.projected_logit_diff) || 0]));

    const ysA = xs.map(x => mapA.get(x) ?? null);
    const ysB = xs.map(x => mapB.get(x) ?? null);
    const basA = traceA?.baseline_logit_diff ?? null;
    const basB = traceB?.baseline_logit_diff ?? null;

    const result = [];
    if (layersA.length > 0) {
      result.push({ x: xs, y: ysA, mode: 'lines+markers', name: `A (#${traceA?.prompt_idx ?? '?'})`, line: { color: '#4e9af1', width: 2 }, marker: { size: 5 }, hovertemplate: 'L%{x}: %{y:.3f}<extra>A</extra>' });
      if (basA != null) result.push({ x: [xs[0], xs[xs.length - 1]], y: [basA, basA], mode: 'lines', name: 'A base', line: { color: '#4e9af166', width: 1, dash: 'dot' }, hoverinfo: 'skip' });
    }
    if (layersB.length > 0) {
      result.push({ x: xs, y: ysB, mode: 'lines+markers', name: `B (#${traceB?.prompt_idx ?? '?'})`, line: { color: '#f77f4e', width: 2 }, marker: { size: 5 }, hovertemplate: 'L%{x}: %{y:.3f}<extra>B</extra>' });
      if (basB != null) result.push({ x: [xs[0], xs[xs.length - 1]], y: [basB, basB], mode: 'lines', name: 'B base', line: { color: '#f77f4e66', width: 1, dash: 'dot' }, hoverinfo: 'skip' });
    }

    return result;
  }, [layersA, layersB, traceA, traceB]);

  if (!data.length) return null;

  return (
    <Plot
      data={data}
      layout={{
        height: 180,
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 42, r: 10, t: 20, b: 36 },
        xaxis: { color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: false, dtick: 1, title: { text: 'Layer', font: { size: 10, color: '#8b90a5' } } },
        yaxis: { color: '#8b90a5', tickfont: { size: 9 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)', title: { text: 'Δlogit', font: { size: 10, color: '#8b90a5' } } },
        legend: { font: { size: 9, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.25 },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function FeatureDiff({ featsA, featsB }) {
  const rows = useMemo(() => {
    const mapA = new Map((featsA || []).map(f => [f.feature_id, parseFloat(f.contribution_to_correct) || 0]));
    const mapB = new Map((featsB || []).map(f => [f.feature_id, parseFloat(f.contribution_to_correct) || 0]));
    const allIds = [...new Set([...mapA.keys(), ...mapB.keys()])];
    return allIds
      .map(id => ({ id, a: mapA.get(id) ?? null, b: mapB.get(id) ?? null, diff: (mapA.get(id) ?? 0) - (mapB.get(id) ?? 0) }))
      .filter(r => r.a !== null || r.b !== null)
      .sort((x, y) => Math.abs(y.diff) - Math.abs(x.diff))
      .slice(0, 12);
  }, [featsA, featsB]);

  if (!rows.length) return <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>No feature contribution data.</div>;

  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6 }}>
        Top features by contribution difference (A − B), click to select
      </div>
      {rows.map(r => (
        <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, fontSize: 10 }}>
          <span style={{ fontFamily: 'monospace', minWidth: 110 }}>{r.id}</span>
          <span style={{ minWidth: 48, color: r.a != null ? (r.a > 0 ? '#4e9af1' : '#d62728') : 'var(--text-dim)' }}>
            A: {r.a != null ? fmt(r.a, 2) : '—'}
          </span>
          <span style={{ minWidth: 48, color: r.b != null ? (r.b > 0 ? '#f77f4e' : '#d62728') : 'var(--text-dim)' }}>
            B: {r.b != null ? fmt(r.b, 2) : '—'}
          </span>
          <span style={{ color: r.diff > 0.1 ? '#4e9af1' : r.diff < -0.1 ? '#f77f4e' : 'var(--text-dim)', fontWeight: Math.abs(r.diff) > 0.3 ? 700 : 400 }}>
            Δ={fmt(r.diff, 2)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function PromptComparison({ data, indexes }) {
  const selectedPromptIdx = useStore(s => s.selectedPromptIdx);
  const comparedPromptIdx = useStore(s => s.comparedPromptIdx);
  const setSelectedPromptIdx = useStore(s => s.setSelectedPromptIdx);
  const setComparedPromptIdx = useStore(s => s.setComparedPromptIdx);

  const { promptById, layerTrajectoryByPrompt, featuresByPrompt, promptTraces } = indexes;

  const [swapped, setSwapped] = useState(false);
  const idxA = swapped ? comparedPromptIdx : selectedPromptIdx;
  const idxB = swapped ? selectedPromptIdx : comparedPromptIdx;

  const traceA = idxA != null ? promptById.get(idxA) : null;
  const traceB = idxB != null ? promptById.get(idxB) : null;
  const layersA = idxA != null ? (layerTrajectoryByPrompt.get(idxA) || []) : [];
  const layersB = idxB != null ? (layerTrajectoryByPrompt.get(idxB) || []) : [];
  const featsA = idxA != null ? (featuresByPrompt.get(idxA) || []) : [];
  const featsB = idxB != null ? (featuresByPrompt.get(idxB) || []) : [];

  // Quick concept-pair selector
  const conceptPairs = useMemo(() => {
    if (!promptTraces) return [];
    const pairs = [];
    const concepts = [...new Set(promptTraces.map(t => t.concept_index))];
    for (const c of concepts) {
      const enPrompt = promptTraces.find(t => t.concept_index === c && t.language === 'en' && t.template_idx === 0);
      const frPrompt = promptTraces.find(t => t.concept_index === c && t.language === 'fr' && t.template_idx === 0);
      if (enPrompt && frPrompt) pairs.push({ concept: c, en: enPrompt.prompt_idx, fr: frPrompt.prompt_idx });
    }
    return pairs;
  }, [promptTraces]);

  return (
    <div style={{ height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Quick pair selectors */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 6 }}>Quick pairs (EN vs FR, same concept)</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {conceptPairs.map(p => (
            <button key={p.concept} onClick={() => { setSelectedPromptIdx(p.en); setComparedPromptIdx(p.fr); }}
              style={{ fontSize: 10, padding: '3px 8px', borderRadius: 8, cursor: 'pointer', border: '1px solid var(--border)', background: (idxA === p.en && idxB === p.fr) || (idxB === p.en && idxA === p.fr) ? 'var(--bg-card)' : 'transparent', color: 'var(--text-dim)' }}>
              c{p.concept} EN#{p.en} vs FR#{p.fr}
            </button>
          ))}
        </div>
      </div>

      {/* Cards */}
      <div style={{ display: 'flex', gap: 10 }}>
        <MiniCard trace={traceA} label="A (primary)" color="#4e9af1" />
        <button onClick={() => setSwapped(s => !s)}
          style={{ padding: '4px 8px', fontSize: 12, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)', borderRadius: 6 }}>
          ⇄
        </button>
        <MiniCard trace={traceB} label="B (compared)" color="#f77f4e" />
      </div>

      {/* Overlay timeline */}
      {(traceA || traceB) && (
        <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>Δlogit trajectories overlaid</div>
          <DivergenceChart layersA={layersA} layersB={layersB} traceA={traceA} traceB={traceB} />
        </div>
      )}

      {/* Top paths comparison */}
      {(traceA || traceB) && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {[{ trace: traceA, label: 'A', color: '#4e9af1' }, { trace: traceB, label: 'B', color: '#f77f4e' }].map(({ trace: t, label, color }) => (
            <div key={label} style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: `1px solid ${color}44` }}>
              <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 6 }}>Top paths — {label}</div>
              {!t ? <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Not selected</div> : (
                t.top_paths?.slice(0, 3).map((p, i) => (
                  <div key={i} style={{ fontSize: 10, marginBottom: 4, padding: '3px 6px', background: 'var(--bg-panel)', borderRadius: 6 }}>
                    <div style={{ color: p.path_direction === 'correct' ? '#2ca02c' : '#d62728', marginBottom: 1 }}>{p.path_direction}</div>
                    <div style={{ fontFamily: 'monospace', fontSize: 9, color: 'var(--text-dim)', wordBreak: 'break-all' }}>{p.path_str}</div>
                    <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 1 }}>score: {fmt(p.prompt_score, 1)}</div>
                  </div>
                ))
              )}
            </div>
          ))}
        </div>
      )}

      {/* Feature divergence */}
      {(featsA.length > 0 || featsB.length > 0) && (
        <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 6 }}>Feature contribution divergence</div>
          <FeatureDiff featsA={featsA} featsB={featsB} />
        </div>
      )}
    </div>
  );
}
