/**
 * Pure filter functions for all panels.
 * Each takes the full dataset + current filter state and returns filtered rows.
 */

/** Guard for threshold comparisons — treat non-finite as 0 */
function safeNum(v) {
  return Number.isFinite(v) ? v : 0;
}

export function filterInterventions(interventions, filters, commonPromptIdx) {
  const { experiments, layerRange, prompts, effectThreshold, showSignFlipsOnly, showCommonPromptsOnly } = filters;
  const commonSet = showCommonPromptsOnly
    ? new Set(commonPromptIdx?.common_prompt_idx ?? [])
    : null;

  return interventions.filter(row => {
    if (experiments.length > 0 && !experiments.includes(row.experiment_type)) return false;
    if (row.layer < layerRange[0] || row.layer > layerRange[1]) return false;
    if (prompts.length > 0 && !prompts.includes(row.prompt_idx)) return false;
    if (effectThreshold > 0 && Math.abs(safeNum(row.effect_size)) < effectThreshold) return false;
    if (showSignFlipsOnly && !row.sign_flipped) return false;
    if (commonSet && !commonSet.has(row.prompt_idx)) return false;
    return true;
  });
}

export function filterLayerAgg(layerAgg, filters) {
  const { experiments, layerRange } = filters;
  return layerAgg.filter(row => {
    if (experiments.length > 0 && !experiments.includes(row.experiment_type)) return false;
    if (row.layer < layerRange[0] || row.layer > layerRange[1]) return false;
    return true;
  });
}

export function filterFeatureAgg(featureAgg, filters, featureToCluster) {
  const { experiments, layerRange, clusters } = filters;
  return featureAgg.filter(row => {
    if (experiments.length > 0 && !experiments.includes(row.experiment_type)) return false;
    if (row.layer < filters.layerRange[0] || row.layer > filters.layerRange[1]) return false;
    if (clusters.length > 0) {
      const nodeId = `L${row.layer}_F${row.feature_id}`;
      const c = featureToCluster.get(nodeId);
      if (c == null || !clusters.includes(c)) return false;
    }
    return true;
  });
}

export function filterPromptAgg(promptAgg, filters, commonPromptIdx) {
  const { experiments, prompts, showCommonPromptsOnly } = filters;
  const commonSet = showCommonPromptsOnly
    ? new Set(commonPromptIdx?.common_prompt_idx ?? [])
    : null;

  return promptAgg.filter(row => {
    if (experiments.length > 0 && !experiments.includes(row.experiment_type)) return false;
    if (prompts.length > 0 && !prompts.includes(row.prompt_idx)) return false;
    if (commonSet && !commonSet.has(row.prompt_idx)) return false;
    return true;
  });
}

export function filterGraphNodes(nodeById, filters, featureToCluster, searchQuery) {
  const { layerRange, clusters } = filters;
  const result = [];
  for (const [id, node] of nodeById) {
    if (node.type === 'input' || node.type === 'output') {
      result.push(id);
      continue;
    }
    if (node.layer < layerRange[0] || node.layer > layerRange[1]) continue;
    if (clusters.length > 0) {
      const c = featureToCluster.get(id);
      if (c == null || !clusters.includes(c)) continue;
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      if (!id.toLowerCase().includes(q)) continue;
    }
    result.push(id);
  }
  return result;
}

export function getFeatureInterventions(interventions, layer, featureIdx, filters) {
  const { experiments } = filters;
  return interventions.filter(row => {
    if (row.layer !== layer) return false;
    if (!row.feature_indices.includes(featureIdx)) return false;
    if (experiments.length > 0 && !experiments.includes(row.experiment_type)) return false;
    return true;
  });
}
