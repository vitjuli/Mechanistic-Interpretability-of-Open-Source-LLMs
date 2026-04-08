// 16 layer colors (layers 10-25)
const LAYER_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
  '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
];

// 7 community colors (Louvain)
const CLUSTER_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
  '#ff7f00', '#a65628', '#f781bf',
];

// Language profile colors
const LANG_PROFILE_COLORS = {
  balanced: '#4e9af1',       // blue — language-agnostic
  fr_leaning: '#f77f4e',     // orange-red — FR-specific
  en_leaning: '#4ef7a0',     // green — EN-specific
  insufficient_data: '#888', // grey — not enough data
};

// Experiment colors
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

export function langProfileColor(profile) {
  return LANG_PROFILE_COLORS[profile] || LANG_PROFILE_COLORS.balanced;
}

export function nodeColor(nodeId, nodeInfo, featureToCluster, colorMode = 'layer') {
  if (nodeInfo.type === 'input') return '#555';
  if (nodeInfo.type === 'output') return nodeInfo.output === 'correct' ? '#2ca02c' : '#d62728';

  if (colorMode === 'lang_profile') {
    const profile = nodeInfo.lang_profile || 'balanced';
    return langProfileColor(profile);
  }
  if ((colorMode === 'cluster' || colorMode === true) && featureToCluster.has(nodeId)) {
    return clusterColor(featureToCluster.get(nodeId));
  }
  return layerColor(nodeInfo.layer);
}

export function effectColorScale(min, max) {
  const range = max - min || 1;
  return (value) => {
    const t = Math.max(0, Math.min(1, (value - min) / range));
    const r = Math.round(15 + t * 231);
    const g = Math.round(17 + t * (t < 0.5 ? 190 : 207));
    const b = Math.round(117 + t * (t < 0.5 ? 59 : -23));
    return `rgb(${r},${g},${b})`;
  };
}

export const BEHAVIOUR_TYPE_COLORS = {
  latent_state:  '#2dd4bf',  // teal
  candidate_set: '#f59e0b',  // amber
  abstraction:   '#a78bfa',  // purple
  gating:        '#fb7185',  // coral
};

export const BEHAVIOUR_TYPE_LABELS = {
  latent_state:  'Latent-state',
  candidate_set: 'Candidate-set',
  abstraction:   'Abstraction',
  gating:        'Gating',
};

export { LAYER_COLORS, CLUSTER_COLORS, EXPERIMENT_COLORS, LANG_PROFILE_COLORS };
