import { useMemo } from 'react';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

// Parse path_str like "input → L23_F6889 → L24_F35447 → output_correct"
function parsePath(pathStr) {
  if (!pathStr) return [];
  return pathStr.split(' → ').map(s => s.trim());
}

function PathRow({ path, rank, isSelected, onClick, direction }) {
  const nodes = parsePath(path.path_str);
  const featureNodes = nodes.filter(n => n.match(/^L\d+_F\d+$/));
  const score = path.prompt_score ?? path.global_score;

  return (
    <div
      onClick={onClick}
      style={{
        padding: '8px 10px', borderRadius: 8, cursor: 'pointer', marginBottom: 4,
        background: isSelected ? (direction === 'correct' ? 'rgba(44,160,44,0.12)' : 'rgba(214,39,40,0.12)') : 'var(--bg-panel)',
        border: `1px solid ${isSelected ? (direction === 'correct' ? '#2ca02c66' : '#d6272866') : 'var(--border)'}`,
        transition: 'all 0.12s',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>#{rank}</span>
          <span style={{
            fontSize: 9, padding: '1px 5px', borderRadius: 6,
            background: direction === 'correct' ? 'rgba(44,160,44,0.2)' : 'rgba(214,39,40,0.2)',
            color: direction === 'correct' ? '#2ca02c' : '#d62728',
          }}>
            {direction === 'correct' ? '→ correct' : '→ incorrect'}
          </span>
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
          score: <span style={{ color: 'var(--text)', fontWeight: 600 }}>{fmt(score, 1)}</span>
          {path.sign_agreement != null && (
            <span style={{ marginLeft: 6, color: path.sign_agreement >= 1 ? '#2ca02c' : '#ff7f0e' }}>
              agree: {Math.round(path.sign_agreement * 100)}%
            </span>
          )}
        </div>
      </div>

      {/* Feature chain */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
        {nodes.map((node, i) => {
          const isFeature = node.match(/^L\d+_F\d+$/);
          const isInput = node === 'input';
          const isOutput = node.startsWith('output');
          return (
            <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {i > 0 && <span style={{ color: 'var(--text-dim)', fontSize: 9 }}>→</span>}
              <span style={{
                fontSize: 9, padding: '1px 5px', borderRadius: 5,
                background: isFeature ? 'var(--bg-card)' : 'transparent',
                border: isFeature ? '1px solid var(--border)' : 'none',
                color: isOutput ? (direction === 'correct' ? '#2ca02c' : '#d62728')
                  : isInput ? 'var(--text-dim)'
                  : 'var(--text)',
                fontFamily: isFeature ? 'monospace' : 'inherit',
              }}>
                {node}
              </span>
            </span>
          );
        })}
      </div>

      {path.mean_feature_contrib != null && (
        <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 3 }}>
          mean contrib: <span style={{ color: path.mean_feature_contrib > 0 ? '#2ca02c' : '#d62728' }}>
            {fmt(path.mean_feature_contrib, 3)}
          </span>
          {' · '}{featureNodes.length} features
        </div>
      )}
    </div>
  );
}

function FeatureContribRow({ feat, onClick }) {
  const contrib = feat.contribution_to_correct;
  const hasData = contrib != null && contrib !== '';
  const val = hasData ? parseFloat(contrib) : null;

  return (
    <div
      onClick={() => onClick(feat.feature_id)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px',
        borderRadius: 6, cursor: 'pointer', marginBottom: 2,
        background: 'var(--bg-panel)',
        border: '1px solid transparent',
        fontSize: 11,
      }}
    >
      <span style={{ fontFamily: 'monospace', minWidth: 110, color: 'var(--text)' }}>{feat.feature_id}</span>
      <span style={{ fontSize: 9, padding: '1px 4px', borderRadius: 4, background: 'var(--bg-card)', color: 'var(--text-dim)' }}>
        {feat.zone}
      </span>
      {hasData ? (
        <>
          {/* Contribution bar */}
          <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--bg-card)', position: 'relative', overflow: 'hidden' }}>
            <div style={{
              position: 'absolute', top: 0, bottom: 0,
              left: val > 0 ? '50%' : `${50 + val * 20}%`,
              width: `${Math.min(Math.abs(val) * 20, 50)}%`,
              background: val > 0 ? '#2ca02c' : '#d62728',
              borderRadius: 3,
            }} />
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: 1, background: 'var(--border)' }} />
          </div>
          <span style={{
            minWidth: 48, textAlign: 'right', fontWeight: 600,
            color: val > 0 ? '#2ca02c' : '#d62728',
          }}>
            {val >= 0 ? '+' : ''}{fmt(val, 3)}
          </span>
        </>
      ) : (
        <span style={{ flex: 1, fontSize: 10, color: 'var(--text-dim)', fontStyle: 'italic' }}>
          no ablation data ({feat.data_source})
        </span>
      )}
    </div>
  );
}

export default function ReasoningTrace({ data, indexes }) {
  const selectedPromptIdx = useStore(s => s.selectedPromptIdx);
  const selectedPathStr = useStore(s => s.selectedPathStr);
  const setSelectedPathStr = useStore(s => s.setSelectedPathStr);
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);

  const { promptById, pathsByPrompt, featuresByPrompt } = indexes;

  const trace = selectedPromptIdx != null ? promptById.get(selectedPromptIdx) : null;
  const paths = selectedPromptIdx != null ? (pathsByPrompt.get(selectedPromptIdx) || []) : [];
  const features = selectedPromptIdx != null ? (featuresByPrompt.get(selectedPromptIdx) || []) : [];

  const { correctPaths, incorrectPaths } = useMemo(() => {
    const correctPaths = paths.filter(p => p.path_direction === 'correct').slice(0, 8);
    const incorrectPaths = paths.filter(p => p.path_direction === 'incorrect').slice(0, 8);
    return { correctPaths, incorrectPaths };
  }, [paths]);

  const { correctFeats, incorrectFeats } = useMemo(() => {
    const sorted = [...features].sort((a, b) => {
      const av = parseFloat(a.contribution_to_correct) || 0;
      const bv = parseFloat(b.contribution_to_correct) || 0;
      return bv - av;
    });
    return {
      correctFeats: sorted.filter(f => (parseFloat(f.contribution_to_correct) || 0) > 0),
      incorrectFeats: sorted.filter(f => (parseFloat(f.contribution_to_correct) || 0) < 0).reverse(),
    };
  }, [features]);

  if (!trace) {
    return <div style={{ padding: 24, color: 'var(--text-dim)', textAlign: 'center' }}>Select a prompt in the Prompt tab first.</div>;
  }

  return (
    <div style={{ display: 'flex', gap: 12, height: '100%', overflow: 'hidden' }}>
      {/* Paths: correct vs incorrect side by side */}
      <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {/* Correct paths */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#2ca02c', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>▲ Supporting paths</span>
              <span style={{ fontWeight: 400, color: 'var(--text-dim)' }}>({correctPaths.length})</span>
            </div>
            {correctPaths.length === 0
              ? <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>No correct-direction paths.</div>
              : correctPaths.map((p, i) => (
                <PathRow
                  key={i}
                  path={p}
                  rank={p.prompt_path_rank ?? i + 1}
                  direction="correct"
                  isSelected={selectedPathStr === p.path_str}
                  onClick={() => setSelectedPathStr(selectedPathStr === p.path_str ? null : p.path_str)}
                />
              ))
            }
          </div>

          {/* Incorrect paths */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#d62728', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>▼ Competing paths</span>
              <span style={{ fontWeight: 400, color: 'var(--text-dim)' }}>({incorrectPaths.length})</span>
            </div>
            {incorrectPaths.length === 0
              ? <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>No incorrect-direction paths.</div>
              : incorrectPaths.map((p, i) => (
                <PathRow
                  key={i}
                  path={p}
                  rank={p.prompt_path_rank ?? i + 1}
                  direction="incorrect"
                  isSelected={selectedPathStr === p.path_str}
                  onClick={() => setSelectedPathStr(selectedPathStr === p.path_str ? null : p.path_str)}
                />
              ))
            }
          </div>
        </div>

        {selectedPathStr && (
          <div style={{ marginTop: 10, padding: '6px 10px', background: 'var(--bg-card)', borderRadius: 8, border: '1px solid var(--accent)', fontSize: 11 }}>
            <span style={{ color: 'var(--text-dim)', marginRight: 6 }}>Selected path (highlighted in graph):</span>
            <span style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>{selectedPathStr}</span>
            <button onClick={() => setSelectedPathStr(null)}
              style={{ marginLeft: 8, fontSize: 10, padding: '1px 6px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)' }}>
              ✕
            </button>
          </div>
        )}
      </div>

      {/* Feature contributions */}
      <div style={{ width: 260, flexShrink: 0, overflow: 'auto' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Feature contributions
        </div>

        {correctFeats.length > 0 && (
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, color: '#2ca02c', marginBottom: 4 }}>▲ Correct-supporting</div>
            {correctFeats.map((f, i) => (
              <FeatureContribRow key={i} feat={f} onClick={setSelectedFeatureId} />
            ))}
          </div>
        )}

        {incorrectFeats.length > 0 && (
          <div>
            <div style={{ fontSize: 10, color: '#d62728', marginBottom: 4 }}>▼ Incorrect-supporting (inhibitory)</div>
            {incorrectFeats.map((f, i) => (
              <FeatureContribRow key={i} feat={f} onClick={setSelectedFeatureId} />
            ))}
          </div>
        )}

        {features.filter(f => f.contribution_to_correct == null || f.contribution_to_correct === '').length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 4 }}>○ No ablation data</div>
            {features.filter(f => f.contribution_to_correct == null || f.contribution_to_correct === '').map((f, i) => (
              <FeatureContribRow key={i} feat={f} onClick={setSelectedFeatureId} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
