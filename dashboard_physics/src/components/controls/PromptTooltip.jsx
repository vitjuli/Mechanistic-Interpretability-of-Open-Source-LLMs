/**
 * Tooltip showing prompt text + correct/incorrect tokens on hover.
 */
export default function PromptTooltip({ promptIdx, promptTextByIdx, style }) {
  const info = promptTextByIdx?.get(promptIdx);
  if (!info) return null;

  return (
    <div className="prompt-tooltip" style={style}>
      <div className="prompt-tooltip-text">"{info.prompt}"</div>
      <div className="prompt-tooltip-tokens">
        <span className="prompt-tooltip-correct">{info.correct}</span>
        {' vs '}
        <span className="prompt-tooltip-incorrect">{info.incorrect}</span>
      </div>
    </div>
  );
}
