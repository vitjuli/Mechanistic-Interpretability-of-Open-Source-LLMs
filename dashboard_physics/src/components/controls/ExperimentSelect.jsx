import useStore from '../../store/useStore';

const LABELS = {
  ablation_zero: 'Ablation (Zero)',
  patching: 'Patching',
  steering: 'Steering',
};

export default function ExperimentSelect({ experiments }) {
  const selected = useStore(s => s.experiments);
  const setExperiments = useStore(s => s.setExperiments);

  const toggle = (exp) => {
    if (selected.includes(exp)) {
      setExperiments(selected.filter(e => e !== exp));
    } else {
      setExperiments([...selected, exp]);
    }
  };

  return (
    <div className="control-group">
      <label>Experiment Type</label>
      <div className="chip-list">
        {experiments.map(exp => (
          <span
            key={exp}
            className={`chip ${selected.length === 0 || selected.includes(exp) ? 'active' : ''}`}
            onClick={() => toggle(exp)}
          >
            {LABELS[exp] || exp}
          </span>
        ))}
      </div>
    </div>
  );
}
