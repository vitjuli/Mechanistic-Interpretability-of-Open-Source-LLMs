import { useState, useRef } from 'react';
import useStore from '../../store/useStore';
import PromptTooltip from './PromptTooltip';

export default function PromptSelector({ prompts, promptTextByIdx }) {
  const selected = useStore(s => s.prompts);
  const setPrompts = useStore(s => s.setPrompts);
  const [hoveredPrompt, setHoveredPrompt] = useState(null);
  const [tooltipPos, setTooltipPos] = useState(null);
  const containerRef = useRef(null);

  const toggle = (p) => {
    if (selected.includes(p)) {
      setPrompts(selected.filter(x => x !== p));
    } else {
      setPrompts([...selected, p]);
    }
  };

  const handleMouseEnter = (e, p) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    setHoveredPrompt(p);
    setTooltipPos({
      position: 'fixed',
      left: rect.left,
      top: rect.bottom + 4,
      zIndex: 1000,
    });
  };

  const handleMouseLeave = () => {
    setHoveredPrompt(null);
    setTooltipPos(null);
  };

  return (
    <div className="control-group" ref={containerRef}>
      <label>Prompts ({selected.length || 'all'} / {prompts.length})</label>
      <div className="chip-list" style={{ maxHeight: 80, overflow: 'auto' }}>
        {prompts.map(p => (
          <span
            key={p}
            className={`chip ${selected.length === 0 || selected.includes(p) ? 'active' : ''}`}
            onClick={() => toggle(p)}
            onMouseEnter={(e) => handleMouseEnter(e, p)}
            onMouseLeave={handleMouseLeave}
            style={{ minWidth: 28, textAlign: 'center' }}
          >
            {p}
          </span>
        ))}
      </div>
      {hoveredPrompt != null && tooltipPos && (
        <PromptTooltip
          promptIdx={hoveredPrompt}
          promptTextByIdx={promptTextByIdx}
          style={tooltipPos}
        />
      )}
    </div>
  );
}
