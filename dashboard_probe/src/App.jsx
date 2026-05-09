import { useState, useEffect } from 'react';
import LatentWorkbench from './LatentWorkbench.jsx';
import LatentStateExplorer from './LatentStateExplorer.jsx';
import ClusteringExplorer from './ClusteringExplorer.jsx';
import ClusterSemantics from './ClusterSemantics.jsx';
import CandidateSelection from './CandidateSelection.jsx';
import AbstractionTab from './AbstractionTab.jsx';

const TABS = [
  { key: 'heatmap',     label: 'Heatmap Overview' },
  { key: 'explorer',    label: '⬡ Latent State Explorer' },
  { key: 'clustering',  label: '⬡ Clustering Methods' },
  { key: 'semantics',   label: '⬡ Cluster Semantics' },
  { key: 'candidate',   label: '⬡ Candidate Selection' },
  { key: 'abstraction', label: '⬡ Intensive/Extensive' },
];

export default function App() {
  const [tab,          setTab]  = useState('explorer');
  const [heatmapData,  setHD]   = useState(null);
  const [explorerData, setED]   = useState(null);
  const [errors,       setErr]  = useState({});

  // Load heatmap data eagerly (small-ish)
  useEffect(() => {
    fetch('/data/latent_states.json')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setHD)
      .catch(e => setErr(prev => ({ ...prev, heatmap: e.message })));
  }, []);

  // Load explorer data lazily when tab first selected
  useEffect(() => {
    if (tab !== 'explorer' || explorerData) return;
    Promise.all([
      fetch('/data/explorer_prompts.json').then(r => r.json()),
      fetch('/data/explorer_features.json').then(r => r.json()),
      fetch('/data/explorer_similarity.json').then(r => r.json()),
      fetch('/data/explorer_bipartite.json').then(r => r.json()),
    ])
      .then(([prompts, features, similarity, bipartite]) =>
        setED({ prompts, features, similarity, bipartite }))
      .catch(e => setErr(prev => ({ ...prev, explorer: e.message })));
  }, [tab, explorerData]);

  const [clusterData, setCD] = useState(null);
  useEffect(() => {
    if (tab !== 'clustering' || clusterData) return;
    fetch('/data/clustering_explorer.json').then(r => r.json())
      .then(setCD)
      .catch(e => setErr(prev => ({ ...prev, clustering: e.message })));
  }, [tab, clusterData]);

  const [semanticsData, setSem]     = useState(null);
  const [ablationData,  setAbl]     = useState(null);
  useEffect(() => {
    if (tab !== 'semantics' || semanticsData) return;
    Promise.all([
      fetch('/data/cluster_semantics.json').then(r => r.json()),
      fetch('/data/cluster_joint_ablation.json').then(r => r.json()).catch(() => null),
    ]).then(([sem, abl]) => { setSem(sem); setAbl(abl); })
      .catch(e => setErr(prev => ({ ...prev, semantics: e.message })));
  }, [tab, semanticsData]);

  const [candidateData, setCand] = useState(null);
  useEffect(() => {
    if (tab !== 'candidate' || candidateData) return;
    fetch('/data/candidate_selection.json').then(r => r.json())
      .then(setCand)
      .catch(e => setErr(prev => ({ ...prev, candidate: e.message })));
  }, [tab, candidateData]);

  const [abstractionData, setAbstr] = useState(null);
  useEffect(() => {
    if (tab !== 'abstraction' || abstractionData) return;
    fetch('/data/abstraction_ie.json').then(r => r.json())
      .then(setAbstr)
      .catch(e => setErr(prev => ({ ...prev, abstraction: e.message })));
  }, [tab, abstractionData]);

  const activeErr = errors[tab];

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%', overflow:'hidden' }}>
      {/* Tab bar */}
      <div style={{ display:'flex', background:'var(--bg-panel)',
        borderBottom:'1px solid var(--border)', flexShrink:0 }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            padding:'8px 18px', background: t.key===tab ? 'var(--bg-card)' : 'transparent',
            border:'none',
            borderBottom: t.key===tab ? '2px solid var(--accent)' : '2px solid transparent',
            color: t.key===tab ? 'var(--accent)' : 'var(--text-dim)',
            fontSize:12, fontWeight: t.key===tab ? 600 : 400,
            cursor:'pointer', whiteSpace:'nowrap',
          }}>{t.label}</button>
        ))}
      </div>

      {/* Content area */}
      <div style={{ flex:1, overflow:'hidden', position:'relative' }}>
        {activeErr ? (
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
            justifyContent:'center', height:'100%', gap:12, color:'var(--danger)' }}>
            <div style={{ fontSize:14 }}>Failed to load data</div>
            <div style={{ fontSize:11, color:'var(--text-dim)' }}>{activeErr}</div>
            <div style={{ fontSize:11, color:'var(--text-dim)', marginTop:8 }}>
              Run: <code style={{ background:'var(--bg-card)', padding:'2px 6px', borderRadius:4 }}>
                python scripts/21_prepare_explorer_ui.py
              </code>
            </div>
          </div>
        ) : tab === 'heatmap' ? (
          heatmapData
            ? <LatentWorkbench data={heatmapData} />
            : <Spinner label="Loading heatmap data…" />
        ) : tab === 'explorer' ? (
          explorerData
            ? <LatentStateExplorer data={explorerData} />
            : <Spinner label="Loading explorer data…" />
        ) : tab === 'clustering' ? (
          clusterData
            ? <ClusteringExplorer data={clusterData} />
            : <Spinner label="Loading clustering data…" />
        ) : tab === 'semantics' ? (
          semanticsData
            ? <ClusterSemantics data={semanticsData} ablationData={ablationData} />
            : <Spinner label="Loading cluster semantics…" />
        ) : tab === 'candidate' ? (
          candidateData
            ? <CandidateSelection data={candidateData} />
            : <Spinner label="Loading candidate selection data…" />
        ) : tab === 'abstraction' ? (
          abstractionData
            ? <AbstractionTab data={abstractionData} />
            : <Spinner label="Loading abstraction data…" />
        ) : null}
      </div>
    </div>
  );
}

function Spinner({ label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center',
      height:'100%', flexDirection:'column', gap:12 }}>
      <div style={{ width:28, height:28, border:'3px solid var(--border)',
        borderTopColor:'var(--accent)', borderRadius:'50%',
        animation:'spin 0.8s linear infinite' }} />
      <div style={{ color:'var(--text-dim)', fontSize:12 }}>{label}</div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
