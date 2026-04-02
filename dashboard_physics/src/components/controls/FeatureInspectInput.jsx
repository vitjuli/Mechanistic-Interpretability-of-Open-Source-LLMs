import { useState } from 'react';
import useStore from '../../store/useStore';
import { shortNodeId } from '../../utils/formatters';

export default function FeatureInspectInput({ indexes }) {
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const addInspectedFeature = useStore(s => s.addInspectedFeature);
  const removeInspectedFeature = useStore(s => s.removeInspectedFeature);
  const clearInspectedFeatures = useStore(s => s.clearInspectedFeatures);
  const [inputValue, setInputValue] = useState('');

  const resolveAndAdd = (text) => {
    if (!text.trim()) return;
    const parts = text.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
    const { nodeById } = indexes;

    for (const part of parts) {
      // If already a full node ID like L24_F122537
      if (nodeById.has(part)) {
        addInspectedFeature(part);
        continue;
      }
      // Try bare feature index — find all nodes matching F{part}
      const fid = parseInt(part, 10);
      if (isNaN(fid)) continue;
      let found = false;
      for (const [id, info] of nodeById) {
        if (info.type === 'feature' && info.featureIdx === fid) {
          addInspectedFeature(id);
          found = true;
        }
      }
      if (!found) {
        // Try with L prefix pattern
        for (const [id] of nodeById) {
          if (id.endsWith(`_F${fid}`)) {
            addInspectedFeature(id);
          }
        }
      }
    }
    setInputValue('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      resolveAndAdd(inputValue);
    }
  };

  return (
    <div className="control-group">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <label>Inspect Features</label>
        {inspectedFeatureIds.length > 0 && (
          <button
            onClick={clearInspectedFeatures}
            style={{
              fontSize: 10,
              padding: '1px 6px',
              background: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-dim)',
              cursor: 'pointer',
            }}
          >
            Clear
          </button>
        )}
      </div>
      <input
        type="text"
        placeholder="e.g. 30233, 161672, 122537"
        value={inputValue}
        onChange={e => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={() => resolveAndAdd(inputValue)}
      />
      {inspectedFeatureIds.length > 0 && (
        <div className="chip-list" style={{ marginTop: 4 }}>
          {inspectedFeatureIds.map(id => (
            <span key={id} className="inspect-chip">
              {shortNodeId(id)}
              <span
                className="inspect-chip-x"
                onClick={() => removeInspectedFeature(id)}
              >
                x
              </span>
            </span>
          ))}
        </div>
      )}
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
        Shift+click graph nodes to add
      </div>
    </div>
  );
}
