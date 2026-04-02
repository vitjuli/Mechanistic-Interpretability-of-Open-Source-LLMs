/**
 * OverlapView — Feature overlap between scalar vs vector prompts
 *
 * Shows:
 * 1. Per-layer bar chart: average ablation effect for scalar vs vector prompts
 * 2. Cluster bias: which clusters are more activated by scalar vs vector inputs
 * 3. Category breakdown: easy_lexical / named_quantity / operator / contrast
 */
import { useMemo } from 'react';
import Plot from 'react-plotly.js';

const COLORS = {
  scalar: '#60a5fa',   // blue
  vector: '#f97316',   // orange
  balanced: '#a3e635', // green
};

const CATEGORY_COLORS = {
  easy_lexical:   '#94a3b8',
  named_quantity: '#60a5fa',
  operator:       '#f97316',
  contrast:       '#e879f9',
};

const CATEGORY_DESCRIPTIONS = {
  easy_lexical:   'Answer is in the name (e.g. "scalar product", "vector product")',
  named_quantity: 'Named physical quantity (e.g. velocity, mass, angular momentum)',
  operator:       'Mathematical operator (e.g. gradient, divergence, curl, Laplacian)',
  contrast:       'Key contrast pair (e.g. electric potential vs. electric field)',
};

export default function OverlapView({ data, indexes }) {
  const { labelStats, clusterLabelStats, promptsMeta, perPromptStats } = data;

  // ── Layer comparison ──────────────────────────────────────────────────────
  const layerData = useMemo(() => {
    if (!labelStats) return { layers: [], scalar: [], vector: [] };
    const layers = Object.keys(labelStats).map(Number).sort((a, b) => a - b);
    const scalar = layers.map(l => labelStats[l]?.scalar?.mean_abs_effect ?? 0);
    const vector = layers.map(l => labelStats[l]?.vector?.mean_abs_effect ?? 0);
    return { layers, scalar, vector };
  }, [labelStats]);

  // ── Cluster comparison ────────────────────────────────────────────────────
  const clusterData = useMemo(() => {
    if (!clusterLabelStats) return [];
    return Object.entries(clusterLabelStats).map(([cid, info]) => ({
      id: cid,
      layers: info.meta?.layers ?? [],
      layerMin: info.meta?.layer_min ?? 0,
      layerMax: info.meta?.layer_max ?? 0,
      nFeatures: info.meta?.n_features ?? 0,
      scalar: info.by_label?.scalar?.mean_abs_effect ?? 0,
      vector: info.by_label?.vector?.mean_abs_effect ?? 0,
      dominant: info.dominant_label ?? 'balanced',
      scalarFrac: info.scalar_fraction ?? 0.5,
    })).sort((a, b) => a.layerMin - b.layerMin);
  }, [clusterLabelStats]);

  // ── Category stats ────────────────────────────────────────────────────────
  const categoryData = useMemo(() => {
    if (!promptsMeta || !perPromptStats) return [];
    const cats = {};
    for (const [pid, meta] of Object.entries(promptsMeta)) {
      const cat = meta.category;
      if (!cats[cat]) cats[cat] = { scalar_correct: 0, vector_correct: 0, scalar_n: 0, vector_n: 0 };
      const pStats = perPromptStats[pid]?.interventions?.ablation;
      if (!pStats) continue;
      if (meta.label === 'scalar') {
        cats[cat].scalar_n++;
        cats[cat].scalar_correct += pStats.mean_abs_effect || 0;
      } else {
        cats[cat].vector_n++;
        cats[cat].vector_correct += pStats.mean_abs_effect || 0;
      }
    }
    return Object.entries(cats).map(([cat, d]) => ({
      cat,
      scalar_avg: d.scalar_n > 0 ? d.scalar_correct / d.scalar_n : 0,
      vector_avg: d.vector_n > 0 ? d.vector_correct / d.vector_n : 0,
      scalar_n: d.scalar_n,
      vector_n: d.vector_n,
    })).sort((a, b) => a.cat.localeCompare(b.cat));
  }, [promptsMeta, perPromptStats]);

  const layerBarTraces = [
    {
      name: 'Scalar prompts',
      x: layerData.layers,
      y: layerData.scalar,
      type: 'bar',
      marker: { color: COLORS.scalar },
      hovertemplate: 'Layer %{x}<br>|Effect|: %{y:.3f}<extra>Scalar</extra>',
    },
    {
      name: 'Vector prompts',
      x: layerData.layers,
      y: layerData.vector,
      type: 'bar',
      marker: { color: COLORS.vector },
      hovertemplate: 'Layer %{x}<br>|Effect|: %{y:.3f}<extra>Vector</extra>',
    },
  ];

  const plotLayout = {
    barmode: 'group',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#c9d1d9', size: 11 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: { text: 'Layer', font: { size: 11 } }, gridcolor: 'rgba(45,49,72,0.5)', tickfont: { size: 10 } },
    yaxis: { title: { text: 'Mean |Effect Size|', font: { size: 11 } }, gridcolor: 'rgba(45,49,72,0.5)', tickfont: { size: 10 } },
    legend: { font: { size: 10 }, x: 1, xanchor: 'right', y: 1 },
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div style={{ padding: '10px 0 4px' }}>
        <h3 style={{ margin: 0, fontSize: 14, color: 'var(--text)', fontWeight: 600 }}>
          Feature Overlap: Scalar vs Vector
        </h3>
        <p style={{ margin: '4px 0 0', fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.5 }}>
          When we ablate (zero out) the top-5 features at each layer, how much does the
          model's prediction change for <span style={{ color: COLORS.scalar }}>scalar</span> vs{' '}
          <span style={{ color: COLORS.vector }}>vector</span> prompts?
          A larger effect means those features matter more for that label.
        </p>
      </div>

      {/* Section 1: Layer comparison */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
          Layer-by-Layer Feature Importance
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
          Early layers (10–14): context encoding.  Middle (15–19): semantic transformation.
          Late (20–25): output preparation.
        </div>
        <Plot
          data={layerBarTraces}
          layout={{ ...plotLayout, height: 200 }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Section 2: Cluster bias */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
          Cluster Bias: Which Feature Groups Specialize?
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
          {clusterData.map(c => {
            const total = c.scalar + c.vector;
            const scalarPct = total > 0 ? Math.round((c.scalar / total) * 100) : 50;
            const vectorPct = 100 - scalarPct;
            const biasColor = c.dominant === 'scalar-leaning' ? COLORS.scalar
              : c.dominant === 'vector-leaning' ? COLORS.vector
              : COLORS.balanced;
            return (
              <div key={c.id} style={{
                background: 'var(--bg-panel)',
                borderRadius: 6,
                padding: 10,
                border: `1px solid ${biasColor}44`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>Cluster {c.id}</span>
                  <span style={{
                    fontSize: 10,
                    padding: '1px 7px',
                    borderRadius: 10,
                    background: biasColor + '22',
                    color: biasColor,
                    fontWeight: 600,
                  }}>
                    {c.dominant === 'balanced' ? 'balanced' :
                     c.dominant === 'scalar-leaning' ? 'scalar ↑' : 'vector ↑'}
                  </span>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6 }}>
                  Layers {c.layerMin}–{c.layerMax} · {c.nFeatures} features
                </div>
                {/* Stacked bar */}
                <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: 4 }}>
                  <div style={{ flex: scalarPct, background: COLORS.scalar }} />
                  <div style={{ flex: vectorPct, background: COLORS.vector }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dim)' }}>
                  <span style={{ color: COLORS.scalar }}>scalar {scalarPct}%</span>
                  <span style={{ color: COLORS.vector }}>vector {vectorPct}%</span>
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 4, fontSize: 10 }}>
                  <span>|eff| scalar: <b>{c.scalar.toFixed(3)}</b></span>
                  <span>|eff| vector: <b>{c.vector.toFixed(3)}</b></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 3: Category breakdown */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
          Prompt Category Analysis
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>
          Categories reveal what type of knowledge the model uses to classify each prompt.
        </div>
        {categoryData.map(c => (
          <div key={c.cat} style={{
            marginBottom: 12,
            padding: '8px 10px',
            background: 'var(--bg-panel)',
            borderRadius: 6,
            borderLeft: `3px solid ${CATEGORY_COLORS[c.cat] || '#888'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: CATEGORY_COLORS[c.cat] || '#888' }}>
                {c.cat.replace('_', ' ')}
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                {c.scalar_n} scalar + {c.vector_n} vector prompts
              </span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6 }}>
              {CATEGORY_DESCRIPTIONS[c.cat] || ''}
            </div>
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              <span>Scalar effect: <b style={{ color: COLORS.scalar }}>{c.scalar_avg.toFixed(3)}</b></span>
              <span>Vector effect: <b style={{ color: COLORS.vector }}>{c.vector_avg.toFixed(3)}</b></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
