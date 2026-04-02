import useStore from '../../store/useStore';
import LayerHeatmap from './LayerHeatmap';
import PromptScatter from './PromptScatter';
import ClusterCards from './ClusterCards';
import ExperimentViolin from './ExperimentViolin';
import FeatureInspection from './FeatureInspection';
import OverlapView from './OverlapView';
import ConceptView from './ConceptView';
import PathView from './PathView';
import ContributionView from './ContributionView';

const TABS = [
  { label: '★ Circuit Paths',       Component: PathView,          demo: true },
  { label: '★ Contributions',       Component: ContributionView,  demo: true },
  { label: '★ Feature Details',     Component: ConceptView,       demo: true },
  { label: '★ Feature Overlap',     Component: OverlapView,       demo: true },
  { label: 'Clusters',              Component: ClusterCards },
  { label: 'Layer Heatmap',         Component: LayerHeatmap },
  { label: 'Prompt Scatter',        Component: PromptScatter },
  { label: 'Experiment Comparison', Component: ExperimentViolin },
  { label: 'Feature Inspection',    Component: FeatureInspection },
];

export default function TabContainer({ data, indexes }) {
  const activeTab = useStore(s => s.activeTab);
  const setActiveTab = useStore(s => s.setActiveTab);
  const { Component } = TABS[activeTab];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-panel)',
        flexShrink: 0,
      }}>
        {TABS.map((tab, i) => (
          <button
            key={tab.label}
            onClick={() => setActiveTab(i)}
            style={{
              padding: '8px 14px',
              background: i === activeTab ? 'var(--bg-card)' : 'transparent',
              border: 'none',
              borderBottom: i === activeTab
                ? `2px solid ${tab.demo ? '#f97316' : 'var(--accent)'}`
                : '2px solid transparent',
              color: i === activeTab
                ? (tab.demo ? '#f97316' : 'var(--text)')
                : (tab.demo ? '#f9731688' : 'var(--text-dim)'),
              fontSize: 12,
              fontWeight: i === activeTab ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.15s',
              whiteSpace: 'nowrap',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 'var(--panel-padding)' }}>
        <Component data={data} indexes={indexes} />
      </div>
    </div>
  );
}
