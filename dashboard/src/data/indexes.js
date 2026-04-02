/**
 * Build pre-computed lookup maps from loaded data.
 */
export function buildIndexes(data) {
  const { graph, supernodesEffect, featureImportance, featureAgg, interventions } = data;

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

  // Unique values for filters
  const experiments = [...new Set(data.interventions.map(r => r.experiment_type))].sort();
  const layers = [...new Set(data.interventions.map(r => r.layer))].sort((a, b) => a - b);
  const promptIndices = [...new Set(data.interventions.map(r => r.prompt_idx))].sort((a, b) => a - b);
  const clusterIds = [...new Set([...featureToCluster.values()])].sort((a, b) => a - b);

  return {
    nodeById,
    neighbors,
    featureToCluster,
    importanceByKey,
    featureAggByKey,
    promptTextByIdx,
    featuresWithInterventionData,
    experiments,
    layers,
    promptIndices,
    clusterIds,
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
