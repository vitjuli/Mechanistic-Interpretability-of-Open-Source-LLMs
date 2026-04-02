import { useMemo } from 'react';
import useStore from '../../store/useStore';
import { shortNodeId, fmt } from '../../utils/formatters';
import { EXPERIMENT_COLORS } from '../../utils/colors';
import EffectBarChart from './EffectBarChart';
import PromptBoxplot from './PromptBoxplot';
import BaselineComparison from './BaselineComparison';
import NeighborList from './NeighborList';
import FeatureMetadata from './FeatureMetadata';

export default function FeatureInspector({ data, indexes }) {
  const selectedFeatureId = useStore(s => s.selectedFeatureId);
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const experiments = useStore(s => s.experiments);

  const nodeInfo = useMemo(() => {
    if (!selectedFeatureId) return null;
    return indexes.nodeById.get(selectedFeatureId);
  }, [selectedFeatureId, indexes]);

  const isInspected = inspectedFeatureIds.includes(selectedFeatureId);

  // Prompt breakdown: for each prompt, whether this feature fires + per-experiment effects
  const promptBreakdown = useMemo(() => {
    if (!isInspected || !nodeInfo) return null;

    const { interventions } = data;
    const allExps = indexes.experiments;
    const selectedExps = experiments.length > 0 ? experiments : allExps;
    const expSet = new Set(selectedExps);
    const { layer, featureIdx } = nodeInfo;

    // Group by prompt_idx
    const byPrompt = new Map();

    for (const row of interventions) {
      if (!expSet.has(row.experiment_type)) continue;
      if (row.layer !== layer) continue;

      const fires = row.feature_indices.includes(featureIdx);
      const pIdx = row.prompt_idx;

      if (!byPrompt.has(pIdx)) {
        byPrompt.set(pIdx, { fires: false, byExp: {} });
      }
      const entry = byPrompt.get(pIdx);
      if (fires) {
        entry.fires = true;
        if (!entry.byExp[row.experiment_type]) {
          entry.byExp[row.experiment_type] = { sum: 0, count: 0, flips: 0 };
        }
        const es = row.effect_size;
        if (es != null && Number.isFinite(es)) {
          entry.byExp[row.experiment_type].sum += es;
          entry.byExp[row.experiment_type].count++;
        }
        if (row.sign_flipped) entry.byExp[row.experiment_type].flips++;
      }
    }

    // Build sorted rows
    const rows = [];
    for (const pIdx of indexes.promptIndices) {
      const pInfo = indexes.promptTextByIdx?.get(pIdx);
      const entry = byPrompt.get(pIdx);
      rows.push({
        promptIdx: pIdx,
        prompt: pInfo?.prompt || `P${pIdx}`,
        correct: pInfo?.correct || '?',
        incorrect: pInfo?.incorrect || '?',
        fires: entry?.fires || false,
        byExp: entry?.byExp || {},
      });
    }
    return rows;
  }, [isInspected, nodeInfo, data, indexes, experiments]);

  if (!selectedFeatureId || !nodeInfo) {
    return (
      <div className="empty-state">
        Click a feature node in the graph to inspect it
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{shortNodeId(selectedFeatureId)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Layer {nodeInfo.layer} | Feature {nodeInfo.featureIdx}
            {indexes.featureToCluster.has(selectedFeatureId) &&
              ` | Cluster ${indexes.featureToCluster.get(selectedFeatureId)}`}
          </div>
        </div>
        <button
          className="btn-reset"
          onClick={() => setSelectedFeatureId(null)}
          style={{ fontSize: 16, padding: '2px 8px' }}
        >
          x
        </button>
      </div>

      <FeatureMetadata
        nodeInfo={nodeInfo}
        nodeId={selectedFeatureId}
        indexes={indexes}
      />

      <EffectBarChart
        nodeInfo={nodeInfo}
        data={data}
        indexes={indexes}
      />

      <PromptBoxplot
        nodeInfo={nodeInfo}
        data={data}
      />

      <BaselineComparison
        nodeInfo={nodeInfo}
        data={data}
        indexes={indexes}
      />

      <NeighborList
        nodeId={selectedFeatureId}
        indexes={indexes}
      />

      {promptBreakdown && (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Prompt Breakdown
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              fontSize: 10,
              borderCollapse: 'collapse',
              lineHeight: 1.4,
            }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '3px 6px', color: 'var(--text-dim)' }}>Prompt</th>
                  <th style={{ textAlign: 'center', padding: '3px 6px', color: 'var(--text-dim)' }}>Fires</th>
                  {(experiments.length > 0 ? experiments : indexes.experiments).map(exp => (
                    <th key={exp} style={{
                      textAlign: 'right', padding: '3px 6px',
                      color: EXPERIMENT_COLORS[exp] || 'var(--text-dim)',
                    }}>
                      {exp}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {promptBreakdown.map(row => {
                  const activeExps = experiments.length > 0 ? experiments : indexes.experiments;
                  return (
                    <tr key={row.promptIdx} style={{
                      borderBottom: '1px solid var(--border)',
                      opacity: row.fires ? 1 : 0.4,
                    }}>
                      <td style={{ padding: '3px 6px', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={`"${row.prompt}" [${row.correct}] vs [${row.incorrect}]`}
                      >
                        "{row.prompt}"
                      </td>
                      <td style={{ textAlign: 'center', padding: '3px 6px' }}>
                        {row.fires
                          ? <span style={{ color: 'var(--accent)' }}>yes</span>
                          : <span style={{ color: 'var(--text-dim)' }}>no</span>
                        }
                      </td>
                      {activeExps.map(exp => {
                        const ed = row.byExp[exp];
                        if (!ed || ed.count === 0) {
                          return <td key={exp} style={{ textAlign: 'right', padding: '3px 6px', color: 'var(--text-dim)' }}>—</td>;
                        }
                        const mean = ed.sum / ed.count;
                        return (
                          <td key={exp} style={{
                            textAlign: 'right', padding: '3px 6px',
                            color: mean > 0 ? '#48bb78' : mean < 0 ? '#fc8181' : 'var(--text)',
                          }}>
                            {fmt(mean, 4)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
