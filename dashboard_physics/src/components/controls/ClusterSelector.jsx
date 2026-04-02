import useStore from '../../store/useStore';
import { clusterColor } from '../../utils/colors';

export default function ClusterSelector({ clusters }) {
  const selected = useStore(s => s.clusters);
  const setClusters = useStore(s => s.setClusters);

  const toggle = (c) => {
    if (selected.includes(c)) {
      setClusters(selected.filter(x => x !== c));
    } else {
      setClusters([...selected, c]);
    }
  };

  return (
    <div className="control-group">
      <label>Effect Clusters</label>
      <div className="chip-list">
        {clusters.map(c => (
          <span
            key={c}
            className={`chip ${selected.length === 0 || selected.includes(c) ? 'active' : ''}`}
            onClick={() => toggle(c)}
            style={{
              borderColor: selected.length === 0 || selected.includes(c)
                ? clusterColor(c) : undefined,
              background: selected.length === 0 || selected.includes(c)
                ? clusterColor(c) + '33' : undefined,
            }}
          >
            C{c}
          </span>
        ))}
      </div>
    </div>
  );
}
