import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { shortNodeId, fmt } from '../../utils/formatters';
import { EXPERIMENT_COLORS } from '../../utils/colors';

export default function FeatureInspection({ data, indexes }) {
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const experiments = useStore(s => s.experiments);
  const { promptTextByIdx, nodeById } = indexes;
  const promptIndices = indexes.promptIndices;

  const { z, xLabels, yLabels, hoverTexts, summaryRow } = useMemo(() => {
    if (inspectedFeatureIds.length === 0) {
      return { z: [], xLabels: [], yLabels: [], hoverTexts: [], summaryRow: [] };
    }

    const { interventions } = data;
    const allExps = indexes.experiments;
    const selectedExps = experiments.length > 0 ? experiments : allExps;
    const expSet = new Set(selectedExps);

    // Parse inspected features into { layer, featureIdx } for matching
    const features = inspectedFeatureIds.map(id => {
      const info = nodeById.get(id);
      return info ? { id, layer: info.layer, featureIdx: info.featureIdx } : null;
    }).filter(Boolean);

    // Build lookup: for each (featureId, promptIdx) → { totalAbsEffect, count, experiments: { exp → { sum, count, flips } } }
    const cellData = new Map();

    for (const row of interventions) {
      if (!expSet.has(row.experiment_type)) continue;

      for (const feat of features) {
        if (row.layer !== feat.layer) continue;
        if (!row.feature_indices.includes(feat.featureIdx)) continue;

        const key = `${feat.id}_${row.prompt_idx}`;
        let cell = cellData.get(key);
        if (!cell) {
          cell = { absSum: 0, sum: 0, count: 0, flips: 0, byExp: {} };
          cellData.set(key, cell);
        }

        const es = row.effect_size;
        if (es != null && Number.isFinite(es)) {
          cell.absSum += Math.abs(es);
          cell.sum += es;
          cell.count++;
        }
        if (row.sign_flipped) cell.flips++;

        if (!cell.byExp[row.experiment_type]) {
          cell.byExp[row.experiment_type] = { sum: 0, count: 0 };
        }
        cell.byExp[row.experiment_type].sum += es || 0;
        cell.byExp[row.experiment_type].count++;
      }
    }

    // X axis: prompts
    const xLabels = promptIndices.map(p => {
      const pInfo = promptTextByIdx?.get(p);
      return pInfo ? `"${pInfo.prompt}"` : `P${p}`;
    });

    // Y axis: inspected features
    const yLabels = features.map(f => shortNodeId(f.id));

    // Z matrix: features × prompts
    const z = [];
    const hoverTexts = [];
    const promptSums = new Array(promptIndices.length).fill(0);
    const promptCounts = new Array(promptIndices.length).fill(0);

    for (const feat of features) {
      const row = [];
      const hoverRow = [];

      for (let pi = 0; pi < promptIndices.length; pi++) {
        const p = promptIndices[pi];
        const key = `${feat.id}_${p}`;
        const cell = cellData.get(key);
        const pInfo = promptTextByIdx?.get(p);

        if (cell && cell.count > 0) {
          const val = cell.absSum / cell.count;
          row.push(val);
          promptSums[pi] += val;
          promptCounts[pi]++;

          // Build hover with per-experiment breakdown
          let hover = `<b>${shortNodeId(feat.id)}</b> × <b>P${p}</b>`;
          if (pInfo) hover += `<br>"${pInfo.prompt}" [${pInfo.correct}] vs [${pInfo.incorrect}]`;
          hover += `<br>Mean |effect|: ${fmt(val, 4)}`;
          hover += `<br>Mean effect: ${fmt(cell.sum / cell.count, 4)}`;
          hover += `<br>Sign flips: ${cell.flips}/${cell.count}`;
          for (const [exp, ed] of Object.entries(cell.byExp)) {
            hover += `<br>  ${exp}: ${fmt(ed.sum / ed.count, 4)} (n=${ed.count})`;
          }
          hoverRow.push(hover);
        } else {
          row.push(null);
          let hover = `<b>${shortNodeId(feat.id)}</b> × <b>P${p}</b>`;
          if (pInfo) hover += `<br>"${pInfo.prompt}"`;
          hover += `<br><i>Feature not present</i>`;
          hoverRow.push(hover);
        }
      }
      z.push(row);
      hoverTexts.push(hoverRow);
    }

    // Summary row
    const summaryRow = promptIndices.map((_, i) =>
      promptCounts[i] > 0 ? promptSums[i] / promptCounts[i] : null
    );

    return { z, xLabels, yLabels, hoverTexts, summaryRow };
  }, [inspectedFeatureIds, data, indexes, experiments, promptIndices, promptTextByIdx, nodeById]);

  if (inspectedFeatureIds.length === 0) {
    return (
      <div className="empty-state">
        Shift+click graph nodes or enter feature IDs in the left panel to inspect
      </div>
    );
  }

  // Add summary row to heatmap
  const fullZ = [...z, summaryRow];
  const fullY = [...yLabels, 'MEAN'];
  const fullHover = [
    ...hoverTexts,
    summaryRow.map((v, i) => {
      const p = promptIndices[i];
      const pInfo = promptTextByIdx?.get(p);
      return `<b>Mean across features</b> × P${p}` +
        (pInfo ? `<br>"${pInfo.prompt}"` : '') +
        (v != null ? `<br>Mean |effect|: ${fmt(v, 4)}` : '<br>No data');
    }),
  ];

  return (
    <Plot
      data={[{
        z: fullZ,
        x: xLabels,
        y: fullY,
        type: 'heatmap',
        colorscale: [
          [0, '#0f1117'],
          [0.25, '#1a365d'],
          [0.5, '#2b6cb0'],
          [0.75, '#ed8936'],
          [1, '#f6e05e'],
        ],
        hovertext: fullHover,
        hoverinfo: 'text',
        xgap: 2,
        ygap: 2,
        colorbar: {
          title: { text: 'Mean |effect|', font: { size: 10, color: '#8b90a5' } },
          tickfont: { size: 9, color: '#8b90a5' },
          len: 0.8,
        },
      }]}
      layout={{
        margin: { l: 120, r: 80, t: 10, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
          color: '#8b90a5',
          tickfont: { size: 9 },
          tickangle: -45,
        },
        yaxis: {
          color: '#8b90a5',
          tickfont: { size: 10 },
          autorange: 'reversed',
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
    />
  );
}
