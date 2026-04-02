import useStore from '../../store/useStore';

export default function LayerRangeSlider({ layers }) {
  const layerRange = useStore(s => s.layerRange);
  const setLayerRange = useStore(s => s.setLayerRange);
  const min = layers[0];
  const max = layers[layers.length - 1];

  return (
    <div className="control-group">
      <label>Layer Range: {layerRange[0]} – {layerRange[1]}</label>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{min}</span>
        <input
          type="range"
          min={min}
          max={max}
          value={layerRange[0]}
          onChange={e => {
            const v = parseInt(e.target.value);
            setLayerRange([Math.min(v, layerRange[1]), layerRange[1]]);
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          value={layerRange[1]}
          onChange={e => {
            const v = parseInt(e.target.value);
            setLayerRange([layerRange[0], Math.max(v, layerRange[0])]);
          }}
        />
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{max}</span>
      </div>
    </div>
  );
}
