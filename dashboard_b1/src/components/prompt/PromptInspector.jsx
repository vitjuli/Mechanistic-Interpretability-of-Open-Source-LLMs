import { useMemo, useState } from 'react';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';
import { generateExplanation } from '../../utils/generateExplanation';
import ExplanationBlock from '../shared/ExplanationBlock';

// Language badge
function LangBadge({ lang }) {
  return (
    <span style={{
      fontSize: 10, padding: '1px 6px', borderRadius: 8,
      background: lang === 'en' ? 'rgba(78,154,241,0.2)' : 'rgba(247,127,78,0.2)',
      color: lang === 'en' ? '#4e9af1' : '#f77f4e',
      border: `1px solid ${lang === 'en' ? '#4e9af1' : '#f77f4e'}66`,
      fontWeight: 600,
    }}>
      {lang?.toUpperCase() ?? '?'}
    </span>
  );
}

// Correctness badge
function CorrectBadge({ correct }) {
  return (
    <span style={{
      fontSize: 11, padding: '2px 8px', borderRadius: 8, fontWeight: 700,
      background: correct ? 'rgba(44,160,44,0.2)' : 'rgba(214,39,40,0.2)',
      color: correct ? '#2ca02c' : '#d62728',
      border: `1px solid ${correct ? '#2ca02c' : '#d62728'}66`,
    }}>
      {correct ? '✓ Correct' : '✗ Incorrect'}
    </span>
  );
}

// Trajectory summary row
function TrajRow({ label, value, highlight }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
      <span style={{ color: 'var(--text-dim)' }}>{label}</span>
      <span style={{ color: highlight ? 'var(--accent)' : 'var(--text)', fontWeight: highlight ? 600 : 400 }}>
        {value}
      </span>
    </div>
  );
}

export default function PromptInspector({ data, indexes }) {
  const { promptTraces, circuit, communityRaw, runManifest } = data;
  const { promptById } = indexes;

  const selectedPromptIdx = useStore(s => s.selectedPromptIdx);
  const setSelectedPromptIdx = useStore(s => s.setSelectedPromptIdx);
  const setComparedPromptIdx = useStore(s => s.setComparedPromptIdx);
  const setGraphPromptMode = useStore(s => s.setGraphPromptMode);
  const graphPromptMode = useStore(s => s.graphPromptMode);

  // Filter state
  const [langFilter, setLangFilter] = useState(null);   // null | 'en' | 'fr'
  const [conceptFilter, setConceptFilter] = useState(null);
  const [correctFilter, setCorrectFilter] = useState(null);  // null | true | false
  const [search, setSearch] = useState('');

  // Filtered prompt list
  const filteredPrompts = useMemo(() => {
    if (!promptTraces) return [];
    return promptTraces.filter(t => {
      if (langFilter && t.language !== langFilter) return false;
      if (conceptFilter != null && t.concept_index !== conceptFilter) return false;
      if (correctFilter != null && t.prediction_correct !== correctFilter) return false;
      if (search && !t.prompt.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [promptTraces, langFilter, conceptFilter, correctFilter, search]);

  const concepts = useMemo(() => {
    if (!promptTraces) return [];
    return [...new Set(promptTraces.map(t => t.concept_index))].sort((a, b) => a - b);
  }, [promptTraces]);

  const trace = selectedPromptIdx != null ? promptById.get(selectedPromptIdx) : null;
  const explanation = trace
    ? generateExplanation(trace, circuit, communityRaw, runManifest?.behaviour_type)
    : null;

  return (
    <div style={{ display: 'flex', gap: 12, height: '100%', overflow: 'hidden' }}>
      {/* Left: prompt selector */}
      <div style={{ width: 240, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {/* Filter bar */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {/* Language filter */}
          {[null, 'en', 'fr'].map(l => (
            <button key={String(l)} onClick={() => setLangFilter(l === langFilter ? null : l)}
              style={{
                fontSize: 10, padding: '2px 7px', borderRadius: 8, cursor: 'pointer', border: '1px solid',
                background: langFilter === l ? (l === 'en' ? 'rgba(78,154,241,0.25)' : l === 'fr' ? 'rgba(247,127,78,0.25)' : 'var(--accent)') : 'transparent',
                borderColor: l === null ? 'var(--border)' : l === 'en' ? '#4e9af1' : '#f77f4e',
                color: l === null ? 'var(--text-dim)' : l === 'en' ? '#4e9af1' : '#f77f4e',
              }}>
              {l === null ? 'All' : l.toUpperCase()}
            </button>
          ))}
          {/* Correct/incorrect filter */}
          {[null, true, false].map(c => (
            <button key={String(c)} onClick={() => setCorrectFilter(c === correctFilter ? null : c)}
              style={{
                fontSize: 10, padding: '2px 7px', borderRadius: 8, cursor: 'pointer', border: '1px solid',
                background: correctFilter === c ? (c === true ? 'rgba(44,160,44,0.2)' : c === false ? 'rgba(214,39,40,0.2)' : 'transparent') : 'transparent',
                borderColor: c === null ? 'var(--border)' : c ? '#2ca02c' : '#d62728',
                color: c === null ? 'var(--text-dim)' : c ? '#2ca02c' : '#d62728',
              }}>
              {c === null ? '✓✗' : c ? '✓' : '✗'}
            </button>
          ))}
        </div>

        {/* Concept filter */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
          <button onClick={() => setConceptFilter(null)}
            style={{ fontSize: 9, padding: '1px 5px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)', background: conceptFilter === null ? 'var(--accent)' : 'transparent', color: conceptFilter === null ? '#fff' : 'var(--text-dim)' }}>
            All
          </button>
          {concepts.map(c => (
            <button key={c} onClick={() => setConceptFilter(c === conceptFilter ? null : c)}
              style={{ fontSize: 9, padding: '1px 5px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)', background: conceptFilter === c ? 'var(--accent)' : 'transparent', color: conceptFilter === c ? '#fff' : 'var(--text-dim)' }}>
              c{c}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search prompt text..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', outline: 'none' }}
        />

        {/* Prompt list */}
        <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 2 }}>
            {filteredPrompts.length} / {promptTraces?.length ?? 0} prompts
          </div>
          {filteredPrompts.map(t => (
            <div
              key={t.prompt_idx}
              onClick={() => setSelectedPromptIdx(t.prompt_idx === selectedPromptIdx ? null : t.prompt_idx)}
              style={{
                padding: '5px 8px', borderRadius: 6, cursor: 'pointer', fontSize: 10,
                background: t.prompt_idx === selectedPromptIdx ? 'var(--bg-card)' : 'transparent',
                border: `1px solid ${t.prompt_idx === selectedPromptIdx ? 'var(--accent)' : 'transparent'}`,
                display: 'flex', alignItems: 'flex-start', gap: 5,
              }}
            >
              <span style={{ color: 'var(--text-dim)', minWidth: 22 }}>#{t.prompt_idx}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', gap: 4, marginBottom: 1, alignItems: 'center' }}>
                  <LangBadge lang={t.language} />
                  <span style={{ color: t.prediction_correct ? '#2ca02c' : '#d62728', fontSize: 9 }}>
                    {t.prediction_correct ? '✓' : '✗'}
                  </span>
                  <span style={{ color: 'var(--text-dim)', fontSize: 9 }}>c{t.concept_index}</span>
                </div>
                <div style={{ color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  title={t.prompt}>
                  {t.prompt.length > 35 ? t.prompt.slice(0, 35) + '…' : t.prompt}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right: selected prompt detail */}
      <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
        {!trace ? (
          <div style={{ padding: 24, color: 'var(--text-dim)', textAlign: 'center' }}>
            Select a prompt from the list to inspect its reasoning.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Header */}
            <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 12, border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-dim)' }}>#{trace.prompt_idx}</span>
                <LangBadge lang={trace.language} />
                <CorrectBadge correct={trace.prediction_correct} />
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>concept {trace.concept_index} · template {trace.template_idx}</span>
              </div>

              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 8, lineHeight: 1.4 }}>
                "{trace.prompt}"
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Correct answer', val: trace.correct_answer, color: '#2ca02c' },
                  { label: 'Incorrect answer', val: trace.incorrect_answer, color: '#d62728' },
                  { label: 'Logit margin', val: fmt(trace.baseline_logit_diff, 3), color: trace.baseline_logit_diff > 0 ? '#2ca02c' : '#d62728' },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ background: 'var(--bg-panel)', borderRadius: 6, padding: '6px 8px' }}>
                    <div style={{ fontSize: 9, color: 'var(--text-dim)', marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color, fontFamily: 'monospace' }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Zone contributions */}
            <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 12, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Zone contributions
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {['early', 'mid', 'late'].map(zone => {
                  const z = trace.zone_summary?.[zone];
                  if (!z) return null;
                  const contrib = z.total_measured_contribution ?? z.measured_contribution ?? 0;
                  return (
                    <div key={zone} style={{ background: 'var(--bg-panel)', borderRadius: 6, padding: '6px 8px' }}>
                      <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: 3 }}>{zone}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: contrib > 0 ? '#2ca02c' : contrib < 0 ? '#d62728' : 'var(--text-dim)' }}>
                        {contrib >= 0 ? '+' : ''}{fmt(contrib, 3)}
                      </div>
                      <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 1 }}>
                        {z.n_features} feats · {z.n_measured_ablation ?? 0} measured
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Trajectory + flip */}
            <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 12, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Decision trajectory
              </div>
              {trace.trajectories && (
                <>
                  <TrajRow label="Dominant direction" value={trace.trajectories.dominant} highlight />
                  <TrajRow label="Correct-supporting features" value={trace.trajectories.n_correct_features} />
                  <TrajRow label="Incorrect-supporting features" value={trace.trajectories.n_incorrect_features} />
                  <TrajRow label="Net contribution" value={`${fmt(trace.trajectories.net, 3)}`} />
                </>
              )}
              {trace.flip_layer != null && (
                <div style={{ marginTop: 6, padding: '4px 8px', background: 'rgba(255,127,14,0.12)', borderRadius: 6, border: '1px solid rgba(255,127,14,0.3)', fontSize: 11 }}>
                  ⚠ Decision flips at layer {trace.flip_layer}
                </div>
              )}
            </div>

            {/* Narrative */}
            {trace.narrative && (
              <div style={{ background: 'var(--bg-card)', borderRadius: 8, padding: 10, border: '1px solid var(--border)', fontSize: 10, color: 'var(--text-dim)', fontFamily: 'monospace', lineHeight: 1.5, wordBreak: 'break-word' }}>
                {trace.narrative}
              </div>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                onClick={() => { setGraphPromptMode(!graphPromptMode); }}
                style={{
                  fontSize: 11, padding: '5px 10px', borderRadius: 8, cursor: 'pointer', border: '1px solid',
                  background: graphPromptMode ? 'rgba(78,154,241,0.2)' : 'transparent',
                  borderColor: graphPromptMode ? '#4e9af1' : 'var(--border)',
                  color: graphPromptMode ? '#4e9af1' : 'var(--text-dim)',
                }}
              >
                {graphPromptMode ? '⦿ Prompt mode ON' : '○ Prompt mode OFF'}
              </button>
              <button
                onClick={() => setComparedPromptIdx(trace.prompt_idx)}
                style={{ fontSize: 11, padding: '5px 10px', borderRadius: 8, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)' }}
              >
                Set as compare B
              </button>
            </div>

            {/* Auto NL explanation */}
            <ExplanationBlock explanation={explanation} />
          </div>
        )}
      </div>
    </div>
  );
}
