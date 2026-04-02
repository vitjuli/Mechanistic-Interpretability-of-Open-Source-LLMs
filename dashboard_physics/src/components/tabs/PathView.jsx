/**
 * PathView — Circuit paths through the attribution graph
 *
 * This graph has star topology: input → [feature] → output_correct / output_incorrect
 * Each "path" is a 3-node chain through one feature.
 *
 * Shows:
 * 1. Supporting paths (feature → output_correct, w > 0) — green
 * 2. Competing paths (feature → output_incorrect, w > 0) — red
 * 3. Layer narrative — reading the circuit from early to late layers
 */
import { useMemo, useCallback } from 'react';
import useStore from '../../store/useStore';

const COLOR_CORRECT   = '#22c55e';
const COLOR_INCORRECT = '#ef4444';
const COLOR_WEAK      = '#60a5fa';
const LABEL_COLOR     = { scalar: '#60a5fa', vector: '#f97316' };

export default function PathView({ data, indexes }) {
  const { graph } = data;
  const { nodeById, featureTopPrompts } = indexes;
  const { perPromptStats } = data;

  const selectedPath = useStore(s => s.selectedPath);
  const setSelectedPath = useStore(s => s.setSelectedPath);
  const setActiveTab = useStore(s => s.setActiveTab);

  // ── Build path list from graph links ──────────────────────────────────────
  // Graph is star: input → F → output_correct/incorrect
  // Every feature has: input→F (weight w_in) and F→output_correct (weight w_out, symmetric)
  const paths = useMemo(() => {
    const featureWeights = new Map(); // nodeId → {w_in, w_to_correct}
    for (const link of graph.links) {
      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;
      if (src === 'input') {
        if (!featureWeights.has(tgt)) featureWeights.set(tgt, {});
        featureWeights.get(tgt).w_in = link.weight;
      }
      if (tgt === 'output_correct') {
        if (!featureWeights.has(src)) featureWeights.set(src, {});
        featureWeights.get(src).w_to_correct = link.weight;
      }
    }

    const result = [];
    for (const [nodeId, w] of featureWeights) {
      const info = nodeById.get(nodeId);
      if (!info || info.type !== 'feature') continue;
      const score = Math.abs(w.w_to_correct ?? 0);
      if (score < 0.1) continue;

      // Compute label distribution from featureTopPrompts
      const topPrompts = featureTopPrompts?.get(nodeId) ?? [];
      const labelCounts = { scalar: 0, vector: 0 };
      for (const tp of topPrompts) {
        const label = perPromptStats?.[tp.prompt_idx]?.label;
        if (label) labelCounts[label]++;
      }
      const totalLabeled = labelCounts.scalar + labelCounts.vector;
      const dominantLabel = totalLabeled === 0 ? null
        : labelCounts.vector > labelCounts.scalar ? 'vector'
        : labelCounts.scalar > labelCounts.vector ? 'scalar'
        : null;

      const supportsCorrect = (w.w_to_correct ?? 0) > 0;
      result.push({
        nodeId,
        layer: info.layer,
        score,
        w_to_correct: w.w_to_correct ?? 0,
        supportsCorrect,
        dominantLabel,
        labelCounts,
        topPrompts,
      });
    }

    return result.sort((a, b) => b.score - a.score);
  }, [graph.links, nodeById, featureTopPrompts, perPromptStats]);

  const correctPaths  = useMemo(() => paths.filter(p => p.supportsCorrect),  [paths]);
  const incorrectPaths = useMemo(() => paths.filter(p => !p.supportsCorrect), [paths]);

  const handleSelectPath = useCallback((p) => {
    const output = p.supportsCorrect ? 'output_correct' : 'output_incorrect';
    const color  = p.supportsCorrect
      ? (p.score >= 2.5 ? COLOR_CORRECT : COLOR_WEAK)
      : COLOR_INCORRECT;
    const newPath = {
      nodes:     ['input', p.nodeId, output],
      feature:   p.nodeId,
      score:     p.score,
      color,
      direction: p.supportsCorrect ? 'correct' : 'incorrect',
    };
    // Toggle off if same path clicked again
    if (selectedPath?.feature === p.nodeId && selectedPath?.direction === newPath.direction) {
      setSelectedPath(null);
    } else {
      setSelectedPath(newPath);
    }
  }, [selectedPath, setSelectedPath]);

  // Group features by layer for narrative section
  const layerGroups = useMemo(() => {
    const groups = {};
    for (const p of paths) {
      const g = p.layer <= 15 ? 'Early (L13–L15)'
              : p.layer <= 19 ? 'Middle (L16–L19)'
              : 'Late (L21–L25)';
      if (!groups[g]) groups[g] = [];
      groups[g].push(p);
    }
    return groups;
  }, [paths]);

  if (paths.length === 0) {
    return <div style={{ padding: 16, color: 'var(--text-dim)', fontSize: 12 }}>No path data</div>;
  }

  const isSel = (p) => selectedPath?.feature === p.nodeId;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Explanation banner ── */}
      <div style={{
        background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 12,
        fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.6,
      }}>
        <b style={{ color: 'var(--text)' }}>Circuit structure:</b> this graph is a direct-attribution
        network — each feature connects <em>input → feature → output</em> in one hop.
        The sign of the output edge determines whether a feature supports or competes with the correct prediction.
        Click any path to highlight it in the graph.
      </div>

      {/* ── Supporting paths (→ correct) ── */}
      <Section title="✓ Supporting paths" subtitle="Features that push toward the correct answer" color={COLOR_CORRECT}>
        {correctPaths.slice(0, 6).map((p, i) => (
          <PathCard
            key={p.nodeId}
            p={p}
            rank={i + 1}
            isSelected={isSel(p)}
            color={p.score >= 2.5 ? COLOR_CORRECT : COLOR_WEAK}
            label={i === 0 ? 'Strongest' : i === correctPaths.length - 1 && i > 1 ? 'Weakest shown' : null}
            onSelect={() => handleSelectPath(p)}
            onClickFeature={() => { setActiveTab(0); }}
          />
        ))}
      </Section>

      {/* ── Competing paths (→ incorrect) ── */}
      <Section title="✗ Competing paths" subtitle="Features that push toward the incorrect answer" color={COLOR_INCORRECT}>
        {incorrectPaths.slice(0, 4).map((p, i) => (
          <PathCard
            key={p.nodeId}
            p={p}
            rank={i + 1}
            isSelected={isSel(p)}
            color={COLOR_INCORRECT}
            label={i === 0 ? 'Strongest' : null}
            onSelect={() => handleSelectPath(p)}
            onClickFeature={() => { setActiveTab(0); }}
          />
        ))}
      </Section>

      {/* ── Layer narrative ── */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Layer Narrative</div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>
          How does the circuit build its answer across layers? Each group shows the net balance
          of pro-correct vs. competing features.
        </div>
        {Object.entries(layerGroups).map(([groupName, gPaths]) => {
          const netCorrect = gPaths.filter(p => p.supportsCorrect).length;
          const netIncorrect = gPaths.filter(p => !p.supportsCorrect).length;
          const netScore = gPaths.reduce((s, p) => s + (p.supportsCorrect ? p.score : -p.score), 0);
          const isNet = netScore > 0;
          return (
            <div key={groupName} style={{
              marginBottom: 10, padding: '8px 12px', borderRadius: 4,
              background: 'var(--bg-panel)',
              borderLeft: `3px solid ${isNet ? COLOR_CORRECT : COLOR_INCORRECT}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 12, fontWeight: 600 }}>{groupName}</span>
                <span style={{
                  fontSize: 11, padding: '1px 8px', borderRadius: 10,
                  background: (isNet ? COLOR_CORRECT : COLOR_INCORRECT) + '22',
                  color: isNet ? COLOR_CORRECT : COLOR_INCORRECT,
                  fontWeight: 600,
                }}>
                  {isNet ? '✓' : '✗'} net {netScore > 0 ? '+' : ''}{netScore.toFixed(2)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {gPaths.map(p => (
                  <button
                    key={p.nodeId}
                    onClick={() => handleSelectPath(p)}
                    style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 4,
                      background: isSel(p)
                        ? (p.supportsCorrect ? COLOR_CORRECT : COLOR_INCORRECT) + '33'
                        : 'var(--bg-card)',
                      border: `1px solid ${p.supportsCorrect ? COLOR_CORRECT : COLOR_INCORRECT}${isSel(p) ? '' : '66'}`,
                      color: p.supportsCorrect ? COLOR_CORRECT : COLOR_INCORRECT,
                      cursor: 'pointer',
                    }}
                  >
                    L{p.layer}_{p.nodeId.split('_F')[1]} {p.supportsCorrect ? '✓' : '✗'}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

// ── Path card component ──────────────────────────────────────────────────────
function PathCard({ p, rank, isSelected, color, label, onSelect }) {
  const LABEL_COLOR = { scalar: '#60a5fa', vector: '#f97316' };
  const totalLabeled = p.labelCounts.scalar + p.labelCounts.vector;

  return (
    <div
      onClick={onSelect}
      style={{
        display: 'flex', flexDirection: 'column', gap: 6,
        padding: '10px 12px', borderRadius: 6,
        background: isSelected ? color + '15' : 'var(--bg-panel)',
        border: `1px solid ${isSelected ? color : color + '44'}`,
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {/* Path header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          fontSize: 11, fontWeight: 700,
          color: isSelected ? color : 'var(--text-dim)', width: 20,
        }}>#{rank}</span>
        {label && (
          <span style={{
            fontSize: 9, padding: '1px 6px', borderRadius: 10,
            background: color + '22', color, fontWeight: 600,
          }}>{label}</span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: 11, color, fontWeight: 700 }}>
          score {p.score.toFixed(3)}
        </span>
      </div>

      {/* Path diagram */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: 11, fontFamily: 'monospace', flexWrap: 'wrap',
      }}>
        <NodeChip id="input" color="#555" />
        <Arrow color={color} />
        <NodeChip id={p.nodeId} color={color} />
        <Arrow color={color} />
        <NodeChip
          id={p.supportsCorrect ? 'output ✓' : 'output ✗'}
          color={p.supportsCorrect ? '#22c55e' : '#ef4444'}
        />
      </div>

      {/* Layer + semantic hint */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 10 }}>
        <span style={{ color: 'var(--text-dim)' }}>Layer {p.layer}</span>
        <span style={{ color: 'var(--text-dim)' }}>
          edge weight: {p.w_to_correct > 0 ? '+' : ''}{p.w_to_correct.toFixed(3)}
        </span>
        {totalLabeled > 0 && (
          <span style={{ color: 'var(--text-dim)' }}>
            seen in: {' '}
            <b style={{ color: LABEL_COLOR.vector }}>{p.labelCounts.vector}v</b>
            {' '}/{' '}
            <b style={{ color: LABEL_COLOR.scalar }}>{p.labelCounts.scalar}s</b>
            {' '}prompts
          </span>
        )}
      </div>

      {isSelected && (
        <div style={{ fontSize: 10, color, fontStyle: 'italic', marginTop: 2 }}>
          ● Path highlighted in graph — click a feature node for semantics (Feature Details tab)
        </div>
      )}
    </div>
  );
}

function Section({ title, subtitle, color, children }) {
  return (
    <div style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', padding: 14 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 2 }}>{title}</div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>{subtitle}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {children}
      </div>
    </div>
  );
}

function NodeChip({ id, color }) {
  const short = id.startsWith('L') ? id.replace('_F', '·F') : id;
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4,
      background: color + '22', color,
      fontSize: 10, fontWeight: 600, border: `1px solid ${color}55`,
      whiteSpace: 'nowrap',
    }}>
      {short}
    </span>
  );
}

function Arrow({ color }) {
  return <span style={{ color, fontSize: 14, fontWeight: 700 }}>→</span>;
}
