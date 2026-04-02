import { useState, useEffect } from 'react';
import './App.css';
import { loadAllData } from './data/loader';
import { buildIndexes } from './data/indexes';
import ExperimentSelect from './components/controls/ExperimentSelect';
import LayerRangeSlider from './components/controls/LayerRangeSlider';
import PromptSelector from './components/controls/PromptSelector';
import ClusterSelector from './components/controls/ClusterSelector';
import FeatureSearch from './components/controls/FeatureSearch';
import EffectThreshold from './components/controls/EffectThreshold';
import ToggleSwitches from './components/controls/ToggleSwitches';
import GraphOverlayControls from './components/controls/GraphOverlayControls';
import FeatureInspectInput from './components/controls/FeatureInspectInput';
import LanguageFilter from './components/controls/LanguageFilter';
import AttributionGraph from './components/graph/AttributionGraph';
import FeatureInspector from './components/inspector/FeatureInspector';
import TabContainer from './components/tabs/TabContainer';
import useStore from './store/useStore';

function App() {
  const [data, setData] = useState(null);
  const [indexes, setIndexes] = useState(null);
  const [error, setError] = useState(null);
  const resetFilters = useStore(s => s.resetFilters);

  useEffect(() => {
    loadAllData()
      .then(d => {
        const idx = buildIndexes(d);
        setData(d);
        setIndexes(idx);
        const layers = idx.layers;
        useStore.setState({
          layerRange: [layers[0], layers[layers.length - 1]],
        });
      })
      .catch(err => {
        console.error('Failed to load data:', err);
        setError(err.message);
      });
  }, []);

  if (error) {
    return (
      <div className="app loading">
        <div className="loading-container">
          <div style={{ color: 'var(--danger)' }}>Failed to load data: {error}</div>
        </div>
      </div>
    );
  }

  if (!data || !indexes) {
    return (
      <div className="app loading">
        <div className="loading-container">
          <div className="spinner" />
          <div>Loading B1-v2 data...</div>
        </div>
      </div>
    );
  }

  const { runManifest, circuit } = data;

  return (
    <div className="app">
      <div className="panel panel-left">
        <div className="title-bar">
          <div>
            <h2 style={{ margin: 0, fontSize: 14 }}>B1-v2 Dashboard</h2>
            {runManifest && (
              <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
                {runManifest.behaviour} · {runManifest.n_prompts} prompts · {runManifest.version}
              </div>
            )}
          </div>
          <button className="btn-reset" onClick={resetFilters}>Reset</button>
        </div>

        {/* Circuit summary badge */}
        {circuit && (
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--accent)',
            borderRadius: 8,
            padding: '6px 10px',
            marginBottom: 10,
            fontSize: 11,
          }}>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Causal Circuit</div>
            <div style={{ color: 'var(--text-dim)', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <span>{circuit.n_features} features</span>
              <span>{circuit.n_edges} edges</span>
              <span>{circuit.n_paths} paths</span>
            </div>
            {circuit.validation && (
              <div style={{ color: 'var(--text-dim)', marginTop: 2 }}>
                Necessity: {(circuit.validation.disruption_rate * 100).toFixed(1)}%
                {' '}S2: {circuit.sufficiency_s2?.transfer_rate != null
                  ? `${(circuit.sufficiency_s2.transfer_rate * 100).toFixed(0)}%`
                  : '—'}
              </div>
            )}
          </div>
        )}

        <LanguageFilter />
        <ExperimentSelect experiments={indexes.experiments} />
        <LayerRangeSlider layers={indexes.layers} />
        <PromptSelector prompts={indexes.promptIndices} promptTextByIdx={indexes.promptTextByIdx} />
        <ClusterSelector clusters={indexes.clusterIds} />
        <FeatureSearch />
        <FeatureInspectInput indexes={indexes} />
        <EffectThreshold />
        <ToggleSwitches />
        <GraphOverlayControls />
      </div>

      <div className="panel panel-center">
        <AttributionGraph data={data} indexes={indexes} />
      </div>

      <div className="panel panel-right">
        <FeatureInspector data={data} indexes={indexes} />
      </div>

      <div className="panel panel-bottom">
        <TabContainer data={data} indexes={indexes} />
      </div>
    </div>
  );
}

export default App;
