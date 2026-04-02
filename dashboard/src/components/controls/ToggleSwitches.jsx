import useStore from '../../store/useStore';

export default function ToggleSwitches() {
  const showSignFlipsOnly = useStore(s => s.showSignFlipsOnly);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const setShowSignFlipsOnly = useStore(s => s.setShowSignFlipsOnly);
  const setShowCommonPromptsOnly = useStore(s => s.setShowCommonPromptsOnly);

  return (
    <div className="control-group" style={{ gap: 6 }}>
      <label>Display</label>
      <label className="toggle-row">
        <input
          type="checkbox"
          checked={showSignFlipsOnly}
          onChange={e => setShowSignFlipsOnly(e.target.checked)}
        />
        Sign flips only
      </label>
      <label className="toggle-row">
        <input
          type="checkbox"
          checked={showCommonPromptsOnly}
          onChange={e => setShowCommonPromptsOnly(e.target.checked)}
        />
        Common prompts only
      </label>
    </div>
  );
}
