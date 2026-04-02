import useStore from '../../store/useStore';

const MODE_OPTIONS = [
  { value: 'layer', label: 'By Layer' },
  { value: 'cluster', label: 'By Cluster' },
  { value: 'effect', label: 'By Intervention Effect' },
];

const METRIC_OPTIONS = [
  { value: 'mean_abs_effect_size', label: 'Mean |Effect|' },
  { value: 'mean_effect_size', label: 'Mean Effect' },
  { value: 'sign_flip_rate', label: 'Sign Flip Rate' },
];

export default function GraphOverlayControls() {
  const graphColorMode = useStore(s => s.graphColorMode);
  const graphEffectMetric = useStore(s => s.graphEffectMetric);
  const setGraphColorMode = useStore(s => s.setGraphColorMode);
  const setGraphEffectMetric = useStore(s => s.setGraphEffectMetric);

  return (
    <div className="control-group">
      <label>Graph Color</label>
      <select
        value={graphColorMode}
        onChange={e => setGraphColorMode(e.target.value)}
      >
        {MODE_OPTIONS.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {graphColorMode === 'effect' && (
        <select
          value={graphEffectMetric}
          onChange={e => setGraphEffectMetric(e.target.value)}
          style={{ marginTop: 4 }}
        >
          {METRIC_OPTIONS.map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      )}
    </div>
  );
}
