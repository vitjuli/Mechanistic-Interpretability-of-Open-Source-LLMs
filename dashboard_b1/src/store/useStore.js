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

  // B1-specific filters
  langProfileFilter: null,   // null | 'balanced' | 'en_leaning' | 'fr_leaning'
  showCircuitOnly: false,    // highlight only circuit features
  iouMode: 'pooled',         // 'pooled' | 'decision' | 'content'

  // Prompt-level reasoning state
  selectedPromptIdx: null,   // int | null — drives all prompt-level views
  comparedPromptIdx: null,   // int | null — second prompt for comparison
  selectedPathStr: null,     // path_str string | null — path to highlight in graph
  graphPromptMode: false,    // bool — when true, graph dims non-active features
  interventionFeatureId: null, // nodeId string — feature selected in InterventionExplorer

  // Graph overlay state
  graphColorMode: 'lang_profile',     // 'layer' | 'cluster' | 'effect' | 'lang_profile'
  graphEffectMetric: 'mean_abs_effect_size',

  // Pairwise overlap mode
  overlapColorMode: false,            // bool — when true, graph colors by pairwise overlap
  overlapSets: null,                  // { shared: Set, aOnly: Set, bOnly: Set } | null

  // Multi-prompt overlap mode
  multiOverlapMode: false,            // bool — false=pairwise, true=multi-prompt
  multiSelectedPrompts: [],           // int[] — selected prompt indices
  multiFreqThreshold: 0.5,           // float 0–1 — majority threshold
  multiContribThreshold: 0.05,       // float — min |contribution| to count feature active
  multiSupportFractions: null,       // Map<nodeId, float> | null — for graph coloring

  // Overlap basis + set-vs-set
  overlapBasis: 'contribution',       // 'activation' | 'contribution' | 'signed'
  setVsSetMode: false,                // bool — Group A vs Group B comparison
  groupBPrompts: [],                  // int[] — Group B prompt indices
  setVsSetFractions: null,            // Map<nodeId, {fracA,fracB}> | null

  // Feature inspection state
  inspectedFeatureIds: [],

  // Selection state
  selectedFeatureId: null,
  hoveredFeatureId: null,
  activeTab: 0,

  // Actions
  setExperiments: (v) => set({ experiments: v }),
  setLayerRange: (v) => set({ layerRange: v }),
  setPrompts: (v) => set({ prompts: v }),
  setClusters: (v) => set({ clusters: v }),
  setSearchQuery: (v) => set({ searchQuery: v }),
  setEffectThreshold: (v) => set({ effectThreshold: v }),
  setShowSignFlipsOnly: (v) => set({ showSignFlipsOnly: v }),
  setShowCommonPromptsOnly: (v) => set({ showCommonPromptsOnly: v }),
  setLangProfileFilter: (v) => set({ langProfileFilter: v }),
  setShowCircuitOnly: (v) => set({ showCircuitOnly: v }),
  setIouMode: (v) => set({ iouMode: v }),
  setSelectedPromptIdx: (v) => set({ selectedPromptIdx: v }),
  setComparedPromptIdx: (v) => set({ comparedPromptIdx: v }),
  setSelectedPathStr: (v) => set({ selectedPathStr: v }),
  setGraphPromptMode: (v) => set({ graphPromptMode: v }),
  setInterventionFeatureId: (v) => set({ interventionFeatureId: v }),
  setGraphColorMode: (v) => set({ graphColorMode: v }),
  setGraphEffectMetric: (v) => set({ graphEffectMetric: v }),
  setOverlapColorMode: (v) => set({ overlapColorMode: v }),
  setOverlapSets: (v) => set({ overlapSets: v }),
  setMultiOverlapMode: (v) => set({ multiOverlapMode: v }),
  setMultiSelectedPrompts: (v) => set({ multiSelectedPrompts: v }),
  setMultiFreqThreshold: (v) => set({ multiFreqThreshold: v }),
  setMultiContribThreshold: (v) => set({ multiContribThreshold: v }),
  setMultiSupportFractions: (v) => set({ multiSupportFractions: v }),
  setOverlapBasis: (v) => set({ overlapBasis: v }),
  setSetVsSetMode: (v) => set({ setVsSetMode: v }),
  setGroupBPrompts: (v) => set({ groupBPrompts: v }),
  setSetVsSetFractions: (v) => set({ setVsSetFractions: v }),
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

  resetFilters: () => set({
    experiments: [],
    layerRange: [10, 25],
    prompts: [],
    clusters: [],
    searchQuery: '',
    effectThreshold: 0,
    showSignFlipsOnly: false,
    showCommonPromptsOnly: false,
    langProfileFilter: null,
    showCircuitOnly: false,
    selectedPromptIdx: null,
    comparedPromptIdx: null,
    selectedPathStr: null,
    graphPromptMode: false,
    interventionFeatureId: null,
    graphColorMode: 'lang_profile',
    graphEffectMetric: 'mean_abs_effect_size',
    overlapColorMode: false,
    overlapSets: null,
    multiOverlapMode: false,
    multiSelectedPrompts: [],
    multiFreqThreshold: 0.5,
    multiContribThreshold: 0.05,
    multiSupportFractions: null,
    overlapBasis: 'contribution',
    setVsSetMode: false,
    groupBPrompts: [],
    setVsSetFractions: null,
    inspectedFeatureIds: [],
  }),
}));

export default useStore;
