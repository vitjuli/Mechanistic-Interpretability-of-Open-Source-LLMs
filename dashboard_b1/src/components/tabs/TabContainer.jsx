import useStore from '../../store/useStore';
import LayerHeatmap from './LayerHeatmap';
import PromptScatter from './PromptScatter';
import CommunityCards from './CommunityCards';
import ExperimentViolin from './ExperimentViolin';
import FeatureInspection from './FeatureInspection';
import IoUChart from './IoUChart';
import FailureAnalysis from './FailureAnalysis';
import PromptInspector from '../prompt/PromptInspector';
import ReasoningTrace from '../prompt/ReasoningTrace';
import LayerTimeline from '../prompt/LayerTimeline';
import PromptComparison from '../prompt/PromptComparison';
import InterventionExplorer from '../prompt/InterventionExplorer';
import FeatureOverlap from '../prompt/FeatureOverlap';

const TABS = [
  // ── Prompt-level (new) ──
  { label: '🔍 Prompt', Component: PromptInspector, group: 'prompt' },
  { label: '⟳ Trace', Component: ReasoningTrace, group: 'prompt' },
  { label: '⏱ Timeline', Component: LayerTimeline, group: 'prompt' },
  { label: '⇄ Compare', Component: PromptComparison, group: 'prompt' },
  { label: '⚡ Interventions', Component: InterventionExplorer, group: 'prompt' },
  { label: '∩ Overlap', Component: FeatureOverlap, group: 'prompt' },
  // ── Global analysis (existing) ──
  { label: 'IoU', Component: IoUChart, group: 'global' },
  { label: 'Communities', Component: CommunityCards, group: 'global' },
  { label: 'Failures', Component: FailureAnalysis, group: 'global' },
  { label: 'Heatmap', Component: LayerHeatmap, group: 'global' },
  { label: 'Scatter', Component: PromptScatter, group: 'global' },
  { label: 'Experiment', Component: ExperimentViolin, group: 'global' },
  { label: 'Features', Component: FeatureInspection, group: 'global' },
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
        overflowX: 'auto',
      }}>
        {TABS.map((tab, i) => {
          const isGroupStart = i === 0 || tab.group !== TABS[i - 1].group;
          return (
            <span key={tab.label} style={{ display: 'flex', alignItems: 'stretch' }}>
              {isGroupStart && i > 0 && (
                <span style={{ width: 1, background: 'var(--border)', margin: '4px 2px', flexShrink: 0 }} />
              )}
              <button
                onClick={() => setActiveTab(i)}
                style={{
                  padding: '8px 12px',
                  background: i === activeTab ? 'var(--bg-card)' : 'transparent',
                  border: 'none',
                  borderBottom: i === activeTab ? `2px solid ${tab.group === 'prompt' ? '#f77f4e' : 'var(--accent)'}` : '2px solid transparent',
                  color: i === activeTab ? 'var(--text)' : 'var(--text-dim)',
                  fontSize: 10,
                  fontWeight: i === activeTab ? 600 : 400,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  whiteSpace: 'nowrap',
                }}
              >
                {tab.label}
              </button>
            </span>
          );
        })}
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 'var(--panel-padding)' }}>
        <Component data={data} indexes={indexes} />
      </div>
    </div>
  );
}
