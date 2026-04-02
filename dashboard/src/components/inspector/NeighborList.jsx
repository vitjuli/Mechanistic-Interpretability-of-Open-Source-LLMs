import useStore from '../../store/useStore';
import { shortNodeId, fmt } from '../../utils/formatters';
import { nodeColor } from '../../utils/colors';

export default function NeighborList({ nodeId, indexes }) {
  const { neighbors, nodeById, featureToCluster } = indexes;
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);

  const neighborIds = neighbors.get(nodeId);
  if (!neighborIds || neighborIds.size === 0) return null;

  const items = [...neighborIds].map(id => {
    const info = nodeById.get(id);
    return { id, info };
  }).sort((a, b) => (a.info?.layer ?? 0) - (b.info?.layer ?? 0));

  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
        NEIGHBORS ({items.length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: 120, overflow: 'auto' }}>
        {items.map(({ id, info }) => (
          <div
            key={id}
            onClick={() => info.type === 'feature' && setSelectedFeatureId(id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '3px 6px',
              borderRadius: 4,
              cursor: info.type === 'feature' ? 'pointer' : 'default',
              fontSize: 11,
              background: 'var(--bg-card)',
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: nodeColor(id, info, featureToCluster),
                flexShrink: 0,
              }}
            />
            <span style={{ flex: 1 }}>{shortNodeId(id)}</span>
            {info.type === 'feature' && (
              <span style={{ color: 'var(--text-dim)', fontFamily: 'monospace', fontSize: 10 }}>
                L{info.layer}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
