// 16 layer colors (layers 10-25) — perceptually distinct
const LAYER_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
  '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
];

// 7 cluster colors — bold and distinct
const CLUSTER_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
  '#ff7f00', '#a65628', '#f781bf',
];

// Centralized experiment colors (previously duplicated across components)
const EXPERIMENT_COLORS = {
  ablation_zero: '#ff7f0e',
  patching: '#1f77b4',
  steering: '#2ca02c',
};

export function layerColor(layer, minLayer = 10) {
  const idx = layer - minLayer;
  return LAYER_COLORS[idx % LAYER_COLORS.length];
}

export function clusterColor(clusterId) {
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

export function nodeColor(nodeId, nodeInfo, featureToCluster, colorByCluster = false) {
  if (nodeInfo.type === 'input') return '#555';
  if (nodeInfo.type === 'output') return nodeInfo.output === 'correct' ? '#2ca02c' : '#d62728';
  if (colorByCluster && featureToCluster.has(nodeId)) {
    return clusterColor(featureToCluster.get(nodeId));
  }
  return layerColor(nodeInfo.layer);
}

/**
 * Returns a value→color function for effect magnitude visualization.
 * Uses linear interpolation between blue (low) → yellow (high).
 */
export function effectColorScale(min, max) {
  const range = max - min || 1;
  return (value) => {
    const t = Math.max(0, Math.min(1, (value - min) / range));
    // Interpolate: dark blue → cyan → yellow
    const r = Math.round(15 + t * 231);
    const g = Math.round(17 + t * (t < 0.5 ? 190 : 207));
    const b = Math.round(117 + t * (t < 0.5 ? 59 : -23));
    return `rgb(${r},${g},${b})`;
  };
}

export { LAYER_COLORS, CLUSTER_COLORS, EXPERIMENT_COLORS };
