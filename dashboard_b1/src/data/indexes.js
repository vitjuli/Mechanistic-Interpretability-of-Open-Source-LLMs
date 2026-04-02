/**
 * Build pre-computed lookup maps from loaded data.
 * B1 extension: adds langProfileByNode, communityByNode, prompt-level indexes.
 */
export function buildIndexes(data) {
  const {
    graph, supernodesEffect, featureImportance, featureAgg, interventions, nodeLabels,
    promptTraces, promptPaths, promptFeatures, layerwiseTraces,
  } = data;

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
        language: row['meta.language'] || (row.prompt_idx < 48 ? 'en' : 'fr'),
      });
    }
  }

  // featuresWithInterventionData: Set of node IDs that have featureAgg data
  const featuresWithInterventionData = new Set();
  for (const row of featureAgg) {
    featuresWithInterventionData.add(`L${row.layer}_F${row.feature_id}`);
  }

  // B1: langProfileByNode: nodeId → lang_profile string
  const langProfileByNode = new Map();
  if (nodeLabels) {
    for (const [nodeId, info] of Object.entries(nodeLabels)) {
      langProfileByNode.set(nodeId, info.lang_profile || 'balanced');
    }
  }
  // Also enrich nodeById with lang_profile
  for (const [nodeId, info] of langProfileByNode) {
    const node = nodeById.get(nodeId);
    if (node) {
      node.lang_profile = info;
    }
  }

  // B1: communityByNode: nodeId → community_id
  const communityByNode = new Map();
  for (const [clusterId, members] of Object.entries(supernodesEffect)) {
    for (const nodeId of members) {
      communityByNode.set(nodeId, parseInt(clusterId, 10));
    }
  }

  // Unique values for filters
  const experiments = [...new Set(data.interventions.map(r => r.experiment_type))].sort();
  const layers = [...new Set(data.interventions.map(r => r.layer))].sort((a, b) => a - b);
  const promptIndices = [...new Set(data.interventions.map(r => r.prompt_idx))].sort((a, b) => a - b);
  const clusterIds = [...new Set([...featureToCluster.values()])].sort((a, b) => a - b);

  // ── Prompt-level indexes ──────────────────────────────────────────────────

  // promptById: prompt_idx → full trace object
  const promptById = new Map();
  if (promptTraces) {
    for (const trace of promptTraces) {
      promptById.set(trace.prompt_idx, trace);
    }
  }

  // pathsByPrompt: prompt_idx → array of path rows (sorted by prompt_path_rank)
  const pathsByPrompt = new Map();
  if (promptPaths) {
    for (const row of promptPaths) {
      const idx = row.prompt_idx;
      if (!pathsByPrompt.has(idx)) pathsByPrompt.set(idx, []);
      pathsByPrompt.get(idx).push(row);
    }
    for (const [idx, rows] of pathsByPrompt) {
      rows.sort((a, b) => (a.prompt_path_rank || 999) - (b.prompt_path_rank || 999));
    }
  }

  // featuresByPrompt: prompt_idx → array of feature contribution rows
  const featuresByPrompt = new Map();
  if (promptFeatures) {
    for (const row of promptFeatures) {
      const idx = row.prompt_idx;
      if (!featuresByPrompt.has(idx)) featuresByPrompt.set(idx, []);
      featuresByPrompt.get(idx).push(row);
    }
  }

  // layerTrajectoryByPrompt: prompt_idx → array of {layer, layer_delta, cumulative_delta, ...}
  const layerTrajectoryByPrompt = new Map();
  if (layerwiseTraces) {
    for (const row of layerwiseTraces) {
      const idx = row.prompt_idx;
      if (!layerTrajectoryByPrompt.has(idx)) layerTrajectoryByPrompt.set(idx, []);
      layerTrajectoryByPrompt.get(idx).push(row);
    }
    for (const [idx, rows] of layerTrajectoryByPrompt) {
      rows.sort((a, b) => a.layer - b.layer);
    }
  }

  // activeFeaturesByPrompt: prompt_idx → Set of node IDs with measured contribution
  const activeFeaturesByPrompt = new Map();
  for (const [idx, rows] of featuresByPrompt) {
    const active = new Set();
    for (const row of rows) {
      if (row.contribution_to_correct != null && row.contribution_to_correct !== '') {
        active.add(row.feature_id);
      }
    }
    activeFeaturesByPrompt.set(idx, active);
  }

  // interventionsByFeature: nodeId → array of {prompt_idx, effect_size, baseline_logit_diff, ...}
  const interventionsByFeature = new Map();
  for (const row of interventions) {
    if (row.experiment_type !== 'ablation_zero') continue;
    for (const fid of (row.feature_indices || [])) {
      const key = `L${row.layer}_F${fid}`;
      if (!interventionsByFeature.has(key)) interventionsByFeature.set(key, []);
      interventionsByFeature.get(key).push({
        prompt_idx: row.prompt_idx,
        layer: row.layer,
        effect_size: row.effect_size,
        baseline_logit_diff: row.baseline_logit_diff,
        intervened_logit_diff: row.intervened_logit_diff,
        language: row['meta.language'] || (row.prompt_idx < 48 ? 'en' : 'fr'),
      });
    }
  }

  return {
    nodeById,
    neighbors,
    featureToCluster,
    importanceByKey,
    featureAggByKey,
    promptTextByIdx,
    featuresWithInterventionData,
    langProfileByNode,
    communityByNode,
    experiments,
    layers,
    promptIndices,
    clusterIds,
    // Prompt-level
    promptById,
    pathsByPrompt,
    featuresByPrompt,
    layerTrajectoryByPrompt,
    activeFeaturesByPrompt,
    interventionsByFeature,
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
