import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

export default function EffectThreshold() {
  const effectThreshold = useStore(s => s.effectThreshold);
  const setEffectThreshold = useStore(s => s.setEffectThreshold);

  return (
    <div className="control-group">
      <label>Min |Effect|: {fmt(effectThreshold, 2)}</label>
      <input
        type="range"
        min={0}
        max={5}
        step={0.05}
        value={effectThreshold}
        onChange={e => setEffectThreshold(parseFloat(e.target.value))}
      />
    </div>
  );
}
