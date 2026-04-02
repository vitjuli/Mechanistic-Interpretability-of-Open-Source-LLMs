import { create } from 'zustand';

const useStore = create((set) => ({
  // Filter state
  experiments: [],           // selected experiment types (empty = all)
  layerRange: [10, 25],      // [min, max] layer range
  prompts: [],               // selected prompt indices (empty = all)
  clusters: [],              // selected cluster ids (empty = all)
  searchQuery: '',           // feature search string
  effectThreshold: 0,        // minimum abs effect size
  showSignFlipsOnly: false,
  showCommonPromptsOnly: false,

  // Graph overlay state
  graphColorMode: 'layer',           // 'layer' | 'cluster' | 'effect'
  graphEffectMetric: 'mean_abs_effect_size',  // 'mean_abs_effect_size' | 'mean_effect_size' | 'sign_flip_rate'
  graphLocalMode: true,              // true = dim inactive nodes when prompt selected

  // Feature inspection state
  inspectedFeatureIds: [],   // array of node IDs like "L24_F122537"

  // Selection state
  selectedFeatureId: null,   // node id clicked in graph
  hoveredFeatureId: null,    // node id hovered in graph
  activeTab: 0,              // bottom panel active tab
  // Path highlighting
  selectedPath: null,        // {nodes: [], feature: str, score: num, color: str, direction: 'correct'|'incorrect'}

  // Actions
  setExperiments: (v) => set({ experiments: v }),
  setLayerRange: (v) => set({ layerRange: v }),
  setPrompts: (v) => set({ prompts: v }),
  setClusters: (v) => set({ clusters: v }),
  setSearchQuery: (v) => set({ searchQuery: v }),
  setEffectThreshold: (v) => set({ effectThreshold: v }),
  setShowSignFlipsOnly: (v) => set({ showSignFlipsOnly: v }),
  setShowCommonPromptsOnly: (v) => set({ showCommonPromptsOnly: v }),
  setGraphColorMode: (v) => set({ graphColorMode: v }),
  setGraphEffectMetric: (v) => set({ graphEffectMetric: v }),
  setGraphLocalMode: (v) => set({ graphLocalMode: v }),
  addInspectedFeature: (id) => set((s) => ({
    inspectedFeatureIds: s.inspectedFeatureIds.includes(id)
      ? s.inspectedFeatureIds
      : [...s.inspectedFeatureIds, id],
  })),
  removeInspectedFeature: (id) => set((s) => ({
    inspectedFeatureIds: s.inspectedFeatureIds.filter(x => x !== id),
  })),
  clearInspectedFeatures: () => set({ inspectedFeatureIds: [] }),
  setSelectedFeatureId: (v) => set({ selectedFeatureId: v }),
  setHoveredFeatureId: (v) => set({ hoveredFeatureId: v }),
  setActiveTab: (v) => set({ activeTab: v }),
  setSelectedPath: (v) => set({ selectedPath: v }),

  resetFilters: () => set({
    experiments: [],
    layerRange: [10, 25],
    prompts: [],
    clusters: [],
    searchQuery: '',
    effectThreshold: 0,
    showSignFlipsOnly: false,
    showCommonPromptsOnly: false,
    graphColorMode: 'layer',
    graphEffectMetric: 'mean_abs_effect_size',
    graphLocalMode: true,
    inspectedFeatureIds: [],
  }),
}));

export default useStore;
