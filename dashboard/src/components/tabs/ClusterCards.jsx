import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { clusterColor } from '../../utils/colors';
import { shortNodeId, fmt, fmtPct } from '../../utils/formatters';

export default function ClusterCards({ data, indexes }) {
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const setClusters = useStore(s => s.setClusters);

  const cards = useMemo(() => {
    const { supernodesEffect, supernodesEffectSummary, featureAgg, interventions } = data;
    const { featureToCluster, featuresWithInterventionData } = indexes;

    return supernodesEffectSummary.map(summary => {
      const clusterId = summary.cluster_id;
      const members = supernodesEffect[clusterId] || [];

      // Aggregate stats from feature agg for cluster members
      const memberStats = members.map(nodeId => {
        const match = nodeId.match(/^L(\d+)_F(\d+)$/);
        if (!match) return null;
        const layer = parseInt(match[1]);
        const featureId = parseInt(match[2]);
        const hasData = featuresWithInterventionData.has(nodeId);
        const rows = featureAgg.filter(r => r.layer === layer && r.feature_id === featureId);
        const meanEffect = rows.length > 0
          ? rows.reduce((s, r) => s + (r.mean_abs_effect_size || 0), 0) / rows.length
          : 0;
        return { nodeId, layer, featureId, meanEffect, hasData };
      }).filter(Boolean);

      const avgEffect = memberStats.length > 0
        ? memberStats.reduce((s, m) => s + m.meanEffect, 0) / memberStats.length
        : 0;

      const layerSpan = memberStats.length > 0
        ? [Math.min(...memberStats.map(m => m.layer)), Math.max(...memberStats.map(m => m.layer))]
        : [0, 0];

      // Data coverage
      const withData = memberStats.filter(m => m.hasData).length;
      const coverage = memberStats.length > 0 ? withData / memberStats.length : 0;

      // Collect raw effect sizes for mini violin
      const effectSizes = [];
      for (const m of memberStats) {
        if (!m.hasData) continue;
        const rows = interventions.filter(r =>
          r.layer === m.layer && r.feature_indices.includes(m.featureId)
        );
        for (const r of rows) {
          if (r.effect_size != null) effectSizes.push(r.effect_size);
        }
      }

      return {
        clusterId,
        nFeatures: summary.n_features,
        representative: summary.representative,
        members: memberStats,
        avgEffect,
        layerSpan,
        withData,
        coverage,
        effectSizes,
      };
    }).sort((a, b) => b.avgEffect - a.avgEffect);
  }, [data, indexes]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 10 }}>
      {cards.map(card => (
        <div
          key={card.clusterId}
          style={{
            background: 'var(--bg-card)',
            border: `1px solid ${clusterColor(card.clusterId)}44`,
            borderRadius: 'var(--radius)',
            padding: 12,
            cursor: 'pointer',
          }}
          onClick={() => setClusters([card.clusterId])}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                width: 10, height: 10, borderRadius: '50%',
                background: clusterColor(card.clusterId),
              }} />
              <span style={{ fontWeight: 600, fontSize: 13 }}>Cluster {card.clusterId}</span>
            </div>
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              {card.nFeatures} features
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11, marginBottom: 8 }}>
            <div><span style={{ color: 'var(--text-dim)' }}>Layers:</span> {card.layerSpan[0]}–{card.layerSpan[1]}</div>
            <div><span style={{ color: 'var(--text-dim)' }}>Avg |effect|:</span> {fmt(card.avgEffect, 4)}</div>
            <div>
              <span style={{ color: 'var(--text-dim)' }}>Data:</span>{' '}
              <span style={{ color: card.coverage >= 0.5 ? 'var(--success)' : 'var(--warning)' }}>
                {card.withData}/{card.members.length} ({Math.round(card.coverage * 100)}%)
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-dim)' }}>Representative:</span> {shortNodeId(card.representative)}
            </div>
          </div>

          {/* Mini violin */}
          {card.effectSizes.length > 0 && (
            <div style={{ marginBottom: 6 }}>
              <Plot
                data={[{
                  y: card.effectSizes,
                  type: 'violin',
                  line: { color: clusterColor(card.clusterId), width: 1 },
                  fillcolor: clusterColor(card.clusterId) + '33',
                  meanline: { visible: true },
                  points: false,
                  hoverinfo: 'none',
                }]}
                layout={{
                  height: 80,
                  width: 260,
                  margin: { l: 30, r: 10, t: 2, b: 2 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  xaxis: { visible: false },
                  yaxis: {
                    color: '#8b90a5', tickfont: { size: 8 },
                    gridcolor: 'rgba(45,49,72,0.3)',
                    zeroline: false,
                  },
                  showlegend: false,
                }}
                config={{ staticPlot: true }}
              />
            </div>
          )}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {card.members.slice(0, 8).map(m => (
              <span
                key={m.nodeId}
                onClick={(e) => { e.stopPropagation(); setSelectedFeatureId(m.nodeId); }}
                style={{
                  fontSize: 10,
                  padding: '1px 5px',
                  borderRadius: 8,
                  background: clusterColor(card.clusterId) + '22',
                  border: `1px solid ${clusterColor(card.clusterId)}44`,
                  cursor: 'pointer',
                  opacity: m.hasData ? 1 : 0.5,
                }}
              >
                {shortNodeId(m.nodeId)}{!m.hasData && '?'}
              </span>
            ))}
            {card.members.length > 8 && (
              <span style={{ fontSize: 10, color: 'var(--text-dim)', padding: '1px 5px' }}>
                +{card.members.length - 8} more
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
