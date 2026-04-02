import useStore from '../../store/useStore';
import { langProfileColor, LANG_PROFILE_COLORS } from '../../utils/colors';

const PROFILES = [
  { key: 'balanced', label: 'Balanced' },
  { key: 'fr_leaning', label: 'FR-leaning' },
  { key: 'en_leaning', label: 'EN-leaning' },
];

export default function LanguageFilter() {
  const langProfileFilter = useStore(s => s.langProfileFilter);
  const setLangProfileFilter = useStore(s => s.setLangProfileFilter);
  const showCircuitOnly = useStore(s => s.showCircuitOnly);
  const setShowCircuitOnly = useStore(s => s.setShowCircuitOnly);

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Language Profile
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
        <button
          onClick={() => setLangProfileFilter(null)}
          style={{
            fontSize: 10, padding: '3px 8px', borderRadius: 10, cursor: 'pointer', border: '1px solid',
            background: langProfileFilter === null ? 'var(--accent)' : 'transparent',
            borderColor: langProfileFilter === null ? 'var(--accent)' : 'var(--border)',
            color: langProfileFilter === null ? '#fff' : 'var(--text-dim)',
          }}
        >
          All
        </button>
        {PROFILES.map(p => (
          <button
            key={p.key}
            onClick={() => setLangProfileFilter(langProfileFilter === p.key ? null : p.key)}
            style={{
              fontSize: 10, padding: '3px 8px', borderRadius: 10, cursor: 'pointer', border: '1px solid',
              background: langProfileFilter === p.key ? langProfileColor(p.key) + '33' : 'transparent',
              borderColor: langProfileFilter === p.key ? langProfileColor(p.key) : 'var(--border)',
              color: langProfileFilter === p.key ? langProfileColor(p.key) : 'var(--text-dim)',
            }}
          >
            <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: langProfileColor(p.key), marginRight: 4, verticalAlign: 'middle' }} />
            {p.label}
          </button>
        ))}
      </div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, cursor: 'pointer', color: 'var(--text-dim)' }}>
        <input
          type="checkbox"
          checked={showCircuitOnly}
          onChange={e => setShowCircuitOnly(e.target.checked)}
        />
        Circuit features only (★)
      </label>
    </div>
  );
}
