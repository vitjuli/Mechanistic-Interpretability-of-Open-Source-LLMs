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
        // Set initial layer range from data
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
          <div>Loading data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="panel panel-left">
        <div className="title-bar">
          <h2>Filters</h2>
          <button className="btn-reset" onClick={resetFilters}>Reset</button>
        </div>
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
