import useStore from '../../store/useStore';
import LayerHeatmap from './LayerHeatmap';
import PromptScatter from './PromptScatter';
import ClusterCards from './ClusterCards';
import ExperimentViolin from './ExperimentViolin';
import FeatureInspection from './FeatureInspection';

const TABS = [
  { label: 'Layer Heatmap', Component: LayerHeatmap },
  { label: 'Prompt Scatter', Component: PromptScatter },
  { label: 'Clusters', Component: ClusterCards },
  { label: 'Experiment Comparison', Component: ExperimentViolin },
  { label: 'Feature Inspection', Component: FeatureInspection },
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
              padding: '8px 16px',
              background: i === activeTab ? 'var(--bg-card)' : 'transparent',
              border: 'none',
              borderBottom: i === activeTab ? '2px solid var(--accent)' : '2px solid transparent',
              color: i === activeTab ? 'var(--text)' : 'var(--text-dim)',
              fontSize: 12,
              fontWeight: i === activeTab ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.15s',
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
