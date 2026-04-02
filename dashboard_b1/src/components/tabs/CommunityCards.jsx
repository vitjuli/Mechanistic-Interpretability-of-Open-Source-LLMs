import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { langProfileColor, clusterColor } from '../../utils/colors';
import { shortNodeId, fmt } from '../../utils/formatters';

const PROFILE_LABELS = {
  balanced: 'Balanced',
  fr_leaning: 'FR',
  en_leaning: 'EN',
  insufficient_data: '?',
};

function ProfileBar({ counts }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  const profiles = ['fr_leaning', 'balanced', 'en_leaning', 'insufficient_data'];
  return (
    <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: 6 }}>
      {profiles.map(p => {
        const n = counts[p] || 0;
        if (!n) return null;
        return (
          <div
            key={p}
            style={{
              width: `${(n / total) * 100}%`,
              background: langProfileColor(p),
              title: `${p}: ${n}`,
            }}
            title={`${p}: ${n}`}
          />
        );
      })}
    </div>
  );
}

export default function CommunityCards({ data, indexes }) {
  const { communityRaw, interventions } = data;
  const { featureToCluster, featuresWithInterventionData } = indexes;
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const setClusters = useStore(s => s.setClusters);

  const cards = useMemo(() => {
    if (!communityRaw) return [];

    return communityRaw.map(comm => {
      const cid = comm.community_id;
      const members = comm.members || [];

      // Collect effect sizes for mini violin
      const effectSizes = [];
      for (const nodeId of members) {
        const m = nodeId.match(/^L(\d+)_F(\d+)$/);
        if (!m) continue;
        const layer = parseInt(m[1]);
        const fid = parseInt(m[2]);
        const rows = interventions.filter(r => r.layer === layer && r.feature_indices.includes(fid));
        for (const r of rows) {
          if (r.effect_size != null) effectSizes.push(r.effect_size);
        }
      }

      const withData = members.filter(id => featuresWithInterventionData.has(id)).length;
      const langCounts = comm.lang_profile_counts || {};

      // Named interpretation based on dominant profile + layer range
      const dominantProfile = comm.dominant_profile || 'balanced';
      let interpretation = '';
      const lmin = comm.layer_min, lmax = comm.layer_max;
      if (cid === 2 && dominantProfile === 'fr_leaning') {
        interpretation = '⚠ Competing FR pathway';
      } else if (dominantProfile === 'balanced' && lmin <= 13) {
        interpretation = 'Early cross-lingual';
      } else if (dominantProfile === 'balanced' && lmin >= 17 && lmax <= 23) {
        interpretation = 'Semantic hub (genuine)';
      } else if (dominantProfile === 'balanced' && lmax >= 23) {
        interpretation = 'Late output circuit';
      } else if (dominantProfile === 'fr_leaning') {
        interpretation = 'FR-specific';
      }

      return {
        cid,
        members,
        nFeatures: comm.n_features,
        layerRange: comm.layer_range,
        layerMin: lmin,
        layerMax: lmax,
        dominantProfile,
        langCounts,
        withData,
        effectSizes,
        interpretation,
      };
    });
  }, [communityRaw, interventions, featuresWithInterventionData]);

  if (!communityRaw || cards.length === 0) {
    return <div style={{ padding: 16, color: 'var(--text-dim)' }}>No community data available.</div>;
  }

  // Legend
  const profileKeys = ['balanced', 'fr_leaning', 'en_leaning'];

  return (
    <div>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>Language profile:</span>
        {profileKeys.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: langProfileColor(p), display: 'inline-block' }} />
            <span style={{ color: 'var(--text-dim)' }}>{PROFILE_LABELS[p]}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
        {cards.map(card => (
          <div
            key={card.cid}
            style={{
              background: 'var(--bg-card)',
              border: `1px solid ${langProfileColor(card.dominantProfile)}44`,
              borderRadius: 'var(--radius)',
              padding: 12,
              cursor: 'pointer',
              position: 'relative',
            }}
            onClick={() => setClusters([card.cid])}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: langProfileColor(card.dominantProfile), display: 'inline-block' }} />
                <span style={{ fontWeight: 600, fontSize: 13 }}>C{card.cid}</span>
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{card.layerRange}</span>
              </div>
              <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{card.nFeatures} features</span>
            </div>

            {card.interpretation && (
              <div style={{ fontSize: 11, color: langProfileColor(card.dominantProfile), marginBottom: 6, fontStyle: 'italic' }}>
                {card.interpretation}
              </div>
            )}

            {/* Profile bar */}
            <ProfileBar counts={card.langCounts} />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11, marginBottom: 8 }}>
              <div>
                <span style={{ color: 'var(--text-dim)' }}>Profile:</span>{' '}
                <span style={{ color: langProfileColor(card.dominantProfile) }}>{card.dominantProfile}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-dim)' }}>Data:</span>{' '}
                <span style={{ color: card.withData > 0 ? 'var(--success)' : 'var(--warning)' }}>
                  {card.withData}/{card.members.length}
                </span>
              </div>
              {Object.entries(card.langCounts).map(([p, n]) => (
                <div key={p}><span style={{ color: 'var(--text-dim)' }}>{PROFILE_LABELS[p]}:</span> {n}</div>
              ))}
            </div>

            {/* Mini violin */}
            {card.effectSizes.length > 0 && (
              <div style={{ marginBottom: 6 }}>
                <Plot
                  data={[{
                    y: card.effectSizes,
                    type: 'violin',
                    line: { color: langProfileColor(card.dominantProfile), width: 1 },
                    fillcolor: langProfileColor(card.dominantProfile) + '33',
                    meanline: { visible: true },
                    points: false,
                    hoverinfo: 'none',
                  }]}
                  layout={{
                    height: 70,
                    width: 240,
                    margin: { l: 28, r: 8, t: 2, b: 2 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    xaxis: { visible: false },
                    yaxis: { color: '#8b90a5', tickfont: { size: 8 }, gridcolor: 'rgba(45,49,72,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.15)' },
                    showlegend: false,
                  }}
                  config={{ staticPlot: true }}
                />
              </div>
            )}

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {card.members.slice(0, 7).map(id => (
                <span
                  key={id}
                  onClick={(e) => { e.stopPropagation(); setSelectedFeatureId(id); }}
                  style={{
                    fontSize: 10, padding: '1px 5px', borderRadius: 8,
                    background: langProfileColor(card.dominantProfile) + '22',
                    border: `1px solid ${langProfileColor(card.dominantProfile)}44`,
                    cursor: 'pointer',
                    opacity: featuresWithInterventionData.has(id) ? 1 : 0.5,
                  }}
                >
                  {shortNodeId(id)}{!featuresWithInterventionData.has(id) && '?'}
                </span>
              ))}
              {card.members.length > 7 && (
                <span style={{ fontSize: 10, color: 'var(--text-dim)', padding: '1px 5px' }}>
                  +{card.members.length - 7} more
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
