/**
 * Build pre-computed lookup maps from loaded data.
 */
export function buildIndexes(data) {
  const { graph, supernodesEffect, featureImportance, featureAgg, interventions, promptsMeta } = data;

  // nodeById: id → node object (with parsed layer/featureIdx)
  const nodeById = new Map();
  for (const node of graph.nodes) {
    const parsed = parseNodeId(node.id);
    nodeById.set(node.id, { ...node, ...parsed });
  }

  // neighbors: adjacency list (undirected)
  const neighbors = new Map();
  for (const node of graph.nodes) {
    neighbors.set(node.id, new Set());
  }
  for (const link of graph.links) {
    const src = typeof link.source === 'object' ? link.source.id : link.source;
    const tgt = typeof link.target === 'object' ? link.target.id : link.target;
    if (neighbors.has(src)) neighbors.get(src).add(tgt);
    if (neighbors.has(tgt)) neighbors.get(tgt).add(src);
  }

  // featureToCluster: nodeId → cluster index (from effect clusters)
  const featureToCluster = new Map();
  for (const [clusterId, members] of Object.entries(supernodesEffect)) {
    for (const nodeId of members) {
      featureToCluster.set(nodeId, parseInt(clusterId, 10));
    }
  }

  // importanceByKey: "layer_featureIdx" → importance row
  const importanceByKey = new Map();
  for (const row of featureImportance) {
    importanceByKey.set(`${row.layer}_${row.feature_idx}`, row);
  }

  // featureAggByKey: "experiment_layer_featureId" → featureAgg row
  const featureAggByKey = new Map();
  for (const row of featureAgg) {
    featureAggByKey.set(`${row.experiment_type}_${row.layer}_${row.feature_id}`, row);
  }

  // promptTextByIdx: prompt_idx → { prompt, correct, incorrect }
  const promptTextByIdx = new Map();
  for (const row of interventions) {
    if (!promptTextByIdx.has(row.prompt_idx)) {
      promptTextByIdx.set(row.prompt_idx, {
        prompt: row['meta.prompt'] || '',
        correct: (row['meta.correct_token'] || '').trim(),
        incorrect: (row['meta.incorrect_token'] || '').trim(),
      });
    }
  }

  // featuresWithInterventionData: Set of node IDs that have featureAgg data
  const featuresWithInterventionData = new Set();
  for (const row of featureAgg) {
    featuresWithInterventionData.add(`L${row.layer}_F${row.feature_id}`);
  }

  // promptMetaByIdx: prompt_idx (string or number) → full metadata from prompts.json
  const promptMetaByIdx = new Map();
  if (promptsMeta) {
    for (const [k, v] of Object.entries(promptsMeta)) {
      promptMetaByIdx.set(parseInt(k, 10), v);
      promptMetaByIdx.set(k, v);  // also string key
    }
  }

  // featureTopPrompts: nodeId → [{prompt_idx, abs_effect, effect, sign_flipped}] sorted desc by abs_effect
  // Only includes L13+ features (L10-12 have rank-based indices, not graph node IDs)
  // Only P0-P19 have per-feature ablation data
  const featureTopPrompts = new Map();
  for (const row of interventions) {
    if (row.experiment_type !== 'ablation_zero') continue;
    const { layer, prompt_idx, feature_indices, effect_size, abs_effect_size, sign_flipped } = row;
    if (layer < 13) continue;
    const fids = Array.isArray(feature_indices) ? feature_indices : [];
    for (const fid of fids) {
      const nodeId = `L${layer}_F${fid}`;
      if (!nodeById.has(nodeId)) continue;
      if (!featureTopPrompts.has(nodeId)) featureTopPrompts.set(nodeId, []);
      featureTopPrompts.get(nodeId).push({
        prompt_idx,
        abs_effect: abs_effect_size ?? Math.abs(effect_size ?? 0),
        effect: effect_size ?? 0,
        sign_flipped: sign_flipped ?? false,
      });
    }
  }
  for (const [nodeId, arr] of featureTopPrompts) {
    arr.sort((a, b) => b.abs_effect - a.abs_effect);
    featureTopPrompts.set(nodeId, arr.slice(0, 15));
  }

  // Unique values for filters
  const experiments = [...new Set(data.interventions.map(r => r.experiment_type))].sort();
  const layers = [...new Set(data.interventions.map(r => r.layer))].sort((a, b) => a - b);
  const promptIndices = [...new Set(data.interventions.map(r => r.prompt_idx))].sort((a, b) => a - b);
  const clusterIds = [...new Set([...featureToCluster.values()])].sort((a, b) => a - b);

  // Unique categories from promptsMeta
  const categories = promptsMeta
    ? [...new Set(Object.values(promptsMeta).map(p => p.category))].sort()
    : [];

  return {
    nodeById,
    neighbors,
    featureToCluster,
    importanceByKey,
    featureAggByKey,
    promptTextByIdx,
    promptMetaByIdx,
    featuresWithInterventionData,
    featureTopPrompts,
    experiments,
    layers,
    promptIndices,
    clusterIds,
    categories,
  };
}

function parseNodeId(id) {
  if (id === 'input') return { type: 'input', layer: -1, featureIdx: null, output: null };
  if (id === 'output_correct') return { type: 'output', layer: 999, featureIdx: null, output: 'correct' };
  if (id === 'output_incorrect') return { type: 'output', layer: 999, featureIdx: null, output: 'incorrect' };
  const match = id.match(/^L(\d+)_F(\d+)$/);
  if (match) {
    return { type: 'feature', layer: parseInt(match[1], 10), featureIdx: parseInt(match[2], 10), output: null };
  }
  return { type: 'unknown', layer: 0, featureIdx: null, output: null };
}
