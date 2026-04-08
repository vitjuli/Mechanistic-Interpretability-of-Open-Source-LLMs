import { useState } from 'react';

/**
 * Collapsible section displaying a pre-generated NL explanation string.
 * Includes a copy-to-clipboard button.
 */
export default function ExplanationBlock({ explanation }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!explanation) return null;

  function handleCopy() {
    navigator.clipboard.writeText(explanation).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div style={{
      background: 'var(--bg-card)',
      borderRadius: 8,
      border: '1px solid var(--border)',
      overflow: 'hidden',
    }}>
      {/* Header — always visible */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '7px 10px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {open ? '▾' : '▸'} Auto-explanation
        </span>
        <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>template-filled · no LLM</span>
      </div>

      {/* Body — shown when open */}
      {open && (
        <div style={{ padding: '0 10px 10px' }}>
          <pre style={{
            margin: 0,
            fontSize: 11,
            lineHeight: 1.6,
            color: 'var(--text)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontFamily: 'inherit',
            background: 'var(--bg-panel)',
            borderRadius: 6,
            padding: '8px 10px',
          }}>
            {explanation}
          </pre>
          <button
            onClick={handleCopy}
            style={{
              marginTop: 6,
              fontSize: 10,
              padding: '3px 9px',
              borderRadius: 6,
              cursor: 'pointer',
              border: '1px solid var(--border)',
              background: copied ? 'rgba(44,160,44,0.15)' : 'transparent',
              color: copied ? '#2ca02c' : 'var(--text-dim)',
            }}
          >
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        </div>
      )}
    </div>
  );
}
