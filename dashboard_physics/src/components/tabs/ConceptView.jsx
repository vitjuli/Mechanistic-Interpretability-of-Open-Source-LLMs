/**
 * FeaturePanel — Semantic interpretation of a selected graph feature
 *
 * Shows:
 * - Feature ID, layer, cluster
 * - Global importance (|corr|, activation frequency)
 * - Top prompts where this feature appears in the ablation top-5
 * - Label distribution (scalar vs vector) → reveals semantic role
 */
import useStore from '../../store/useStore';
import { clusterColor } from '../../utils/colors';

const LABEL_COLOR = { scalar: '#60a5fa', vector: '#f97316' };

export default function FeaturePanel({ data, indexes }) {
  const selectedFeatureId = useStore(s => s.selectedFeatureId);
  const { featureTopPrompts, importanceByKey, featureToCluster } = indexes;
  const { perPromptStats } = data;

  // ── Nothing selected ────────────────────────────────────────────────────────
  if (!selectedFeatureId) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100%', gap: 14, padding: 32,
        textAlign: 'center',
      }}>
        <div style={{ fontSize: 36, opacity: 0.2 }}>◎</div>
        <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
          Feature Semantics
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 280, lineHeight: 1.7 }}>
          Click any feature node in the attribution graph to see which prompts
          activate it and what it represents.
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', maxWidth: 300, lineHeight: 1.6 }}>
          Tip: Select a prompt in the left panel first. The graph will highlight
          active features. Then click one to understand its role.
        </div>
      </div>
    );
  }

  // ── Parse feature ID ─────────────────────────────────────────────────────────
  const match = selectedFeatureId.match(/^L(\d+)_F(\d+)$/);
  if (!match) {
    return (
      <div style={{ padding: 16, color: 'var(--text-dim)', fontSize: 12 }}>
        {selectedFeatureId} — not a feature node
      </div>
    );
  }
  const layer = parseInt(match[1], 10);
  const featureIdx = parseInt(match[2], 10);

  // Global importance from feature_importance.csv
  const imp = importanceByKey.get(`${layer}_${featureIdx}`);

  // Cluster membership
  const clusterId = featureToCluster.get(selectedFeatureId);

  // Top prompts where this feature appeared in ablation top-5
  const topPrompts = featureTopPrompts?.get(selectedFeatureId) ?? [];

  // Label distribution across top prompts
  const labelCounts = { scalar: 0, vector: 0 };
  for (const tp of topPrompts) {
    const label = perPromptStats?.[tp.prompt_idx]?.label;
    if (label) labelCounts[label] = (labelCounts[label] || 0) + 1;
  }
  const total = labelCounts.scalar + labelCounts.vector;
  const dominantLabel = total === 0 ? null
    : labelCounts.vector > labelCounts.scalar ? 'vector'
    : labelCounts.scalar > labelCounts.vector ? 'scalar'
    : 'balanced';

  const maxEffect = topPrompts.length > 0 ? topPrompts[0].abs_effect : 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* ── Feature header ── */}
      <div style={{
        background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14,
        borderLeft: `4px solid ${clusterId != null ? clusterColor(clusterId) : '#555'}`,
      }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)', marginBottom: 8 }}>
          {selectedFeatureId}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Tag color="#8b90a5">Layer {layer}</Tag>
          {clusterId != null && (
            <Tag color={clusterColor(clusterId)}>Cluster {clusterId}</Tag>
          )}
          {imp?.abs_correlation != null && (
            <Tag color="#94a3b8">|corr| {imp.abs_correlation.toFixed(3)}</Tag>
          )}
          {imp?.activation_frequency != null && (
            <Tag color="#94a3b8">active {Math.round(imp.activation_frequency * 100)}%</Tag>
          )}
        </div>
      </div>

      {/* ── Semantic pattern (label bias) ── */}
      {total > 0 && (
        <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Semantic Pattern</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{ color: LABEL_COLOR.vector, fontSize: 13, fontWeight: 700 }}>
              {labelCounts.vector} vector
            </span>
            <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>vs</span>
            <span style={{ color: LABEL_COLOR.scalar, fontSize: 13, fontWeight: 700 }}>
              {labelCounts.scalar} scalar
            </span>
            {dominantLabel && dominantLabel !== 'balanced' && (
              <Tag color={LABEL_COLOR[dominantLabel]} style={{ marginLeft: 'auto' }}>
                {dominantLabel}-leaning
              </Tag>
            )}
          </div>
          {/* Stacked bar */}
          <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ flex: labelCounts.vector, background: LABEL_COLOR.vector, minWidth: labelCounts.vector > 0 ? 4 : 0 }} />
            <div style={{ flex: labelCounts.scalar, background: LABEL_COLOR.scalar, minWidth: labelCounts.scalar > 0 ? 4 : 0 }} />
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>
            Based on {total} prompts (P0–P19) where this feature was in the ablation top-5
          </div>
        </div>
      )}

      {/* ── Top activating prompts ── */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
          Top Activating Prompts
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10, lineHeight: 1.5 }}>
          Prompts where this feature was in the top-5 by activation at layer {layer},
          sorted by combined ablation effect at that layer.
        </div>

        {topPrompts.length === 0 ? (
          <div style={{ fontSize: 11, color: 'var(--text-dim)', fontStyle: 'italic', padding: '8px 0' }}>
            No ablation data for this feature.
            Only features appearing in P0–P19 have per-prompt data.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {topPrompts.map((tp, i) => {
              const ps = perPromptStats?.[tp.prompt_idx];
              if (!ps) return null;
              const barPct = Math.min(100, (tp.abs_effect / maxEffect) * 100);
              return (
                <div key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 8,
                  padding: '6px 8px', borderRadius: 4,
                  background: 'var(--bg-panel)', fontSize: 11,
                }}>
                  {/* Prompt index */}
                  <span style={{ color: 'var(--text-dim)', flexShrink: 0, width: 24, paddingTop: 1 }}>
                    P{tp.prompt_idx}
                  </span>
                  {/* Label badge */}
                  <span style={{
                    flexShrink: 0, padding: '1px 6px', borderRadius: 8,
                    background: LABEL_COLOR[ps.label] + '22',
                    color: LABEL_COLOR[ps.label],
                    fontSize: 10, fontWeight: 600,
                    width: 42, textAlign: 'center',
                  }}>
                    {ps.label}
                  </span>
                  {/* Prompt text */}
                  <span style={{ flex: 1, color: 'var(--text)', lineHeight: 1.5, fontSize: 11 }}>
                    "{ps.prompt?.length > 65
                      ? ps.prompt.slice(0, 65) + '…'
                      : ps.prompt}"
                  </span>
                  {/* Effect bar + value */}
                  <div style={{ flexShrink: 0, width: 60 }}>
                    <div style={{ height: 4, background: 'var(--bg-card)', borderRadius: 2, marginBottom: 2 }}>
                      <div style={{
                        height: '100%', borderRadius: 2,
                        width: `${barPct}%`,
                        background: tp.sign_flipped ? '#ef4444' : LABEL_COLOR[ps.label],
                      }} />
                    </div>
                    <div style={{
                      fontSize: 9, color: tp.sign_flipped ? '#ef4444' : 'var(--text-dim)',
                      textAlign: 'right',
                    }}>
                      {tp.abs_effect.toFixed(3)}{tp.sign_flipped ? ' ⚠' : ''}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

    </div>
  );
}

function Tag({ color, children, style }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 10,
      background: color + '22', color,
      fontSize: 11, fontWeight: 600,
      border: `1px solid ${color}44`,
      ...style,
    }}>
      {children}
    </span>
  );
}
