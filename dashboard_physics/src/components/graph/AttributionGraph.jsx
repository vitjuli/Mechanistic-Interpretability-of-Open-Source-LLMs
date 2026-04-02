import { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import Plot from 'react-plotly.js';
import { forceSimulation, forceLink, forceManyBody, forceX, forceY, forceCollide } from 'd3-force';
import useStore from '../../store/useStore';
import { filterGraphNodes } from '../../utils/filterData';
import { nodeColor } from '../../utils/colors';
import { shortNodeId, fmt } from '../../utils/formatters';

// Overlap mode colors
const OVERLAP = {
  shared:   '#9333ea',  // purple  = active in both prompts
  promptA:  '#60a5fa',  // blue    = A only
  promptB:  '#f97316',  // orange  = B only
  inactive: '#2d3148',  // dark    = inactive
};

export default function AttributionGraph({ data, indexes }) {
  const { graph } = data;
  const { nodeById, neighbors, featureToCluster } = indexes;

  const layerRange = useStore(s => s.layerRange);
  const clusters = useStore(s => s.clusters);
  const searchQuery = useStore(s => s.searchQuery);
  const selectedFeatureId = useStore(s => s.selectedFeatureId);
  const hoveredFeatureId = useStore(s => s.hoveredFeatureId);
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const setHoveredFeatureId = useStore(s => s.setHoveredFeatureId);
  const graphColorMode = useStore(s => s.graphColorMode);
  const graphEffectMetric = useStore(s => s.graphEffectMetric);
  const graphLocalMode = useStore(s => s.graphLocalMode);
  const setGraphLocalMode = useStore(s => s.setGraphLocalMode);
  const experiments = useStore(s => s.experiments);
  const prompts = useStore(s => s.prompts);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const addInspectedFeature = useStore(s => s.addInspectedFeature);
  const selectedPath = useStore(s => s.selectedPath);

  const containerRef = useRef(null);
  const [dims, setDims] = useState({ width: 800, height: 500 });

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDims({ width: Math.max(width, 200), height: Math.max(height, 200) });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // Compute visible node set
  const visibleNodes = useMemo(() => {
    const filters = { layerRange, clusters };
    return new Set(filterGraphNodes(nodeById, filters, featureToCluster, searchQuery));
  }, [nodeById, layerRange, clusters, featureToCluster, searchQuery]);

  // Run force layout once
  const positions = useMemo(() => {
    const nodes = graph.nodes.map(n => {
      const info = nodeById.get(n.id);
      let xTarget;
      if (info.type === 'input') xTarget = 0;
      else if (info.type === 'output') xTarget = 1;
      else {
        const layers = indexes.layers;
        xTarget = (info.layer - layers[0]) / Math.max(layers[layers.length - 1] - layers[0], 1);
      }
      return { id: n.id, xTarget, ...info };
    });

    const links = graph.links.map(l => ({
      source: l.source,
      target: l.target,
      weight: Math.abs(l.weight),
    }));

    const sim = forceSimulation(nodes)
      .force('link', forceLink(links).id(d => d.id).distance(40).strength(0.3))
      .force('charge', forceManyBody().strength(-60))
      .force('x', forceX(d => d.xTarget * 600).strength(0.8))
      .force('y', forceY(300).strength(0.05))
      .force('collide', forceCollide(8))
      .stop();

    for (let i = 0; i < 300; i++) sim.tick();

    const pos = new Map();
    for (const n of nodes) pos.set(n.id, { x: n.x, y: n.y });
    return pos;
  }, [graph, nodeById, indexes.layers]);

  // Single-prompt effects: Map<nodeId, {abs, effect, signFlipped}>
  const promptNodeEffects = useMemo(() => {
    if (prompts.length !== 1) return null;
    const pid = prompts[0];
    const abl = data.interventions.filter(
      r => r.experiment_type === 'ablation_zero' && r.prompt_idx === pid
    );
    if (abl.length === 0) return null;

    const effects = new Map();
    for (const row of abl) {
      for (const fid of row.feature_indices) {
        const key = `L${row.layer}_F${fid}`;
        const abs = row.abs_effect_size ?? Math.abs(row.effect_size ?? 0);
        const prev = effects.get(key);
        if (prev === undefined || abs > prev.abs) {
          effects.set(key, { effect: row.effect_size ?? 0, abs, signFlipped: row.sign_flipped });
        }
      }
    }
    return effects.size > 0 ? effects : null;
  }, [prompts, data.interventions]);

  // Two-prompt overlap: {pidA, pidB, effectsA: Map, effectsB: Map}
  const promptOverlap = useMemo(() => {
    if (prompts.length !== 2) return null;
    const compute = (pid) => {
      const effects = new Map();
      for (const row of data.interventions) {
        if (row.experiment_type !== 'ablation_zero' || row.prompt_idx !== pid) continue;
        for (const fid of row.feature_indices) {
          const key = `L${row.layer}_F${fid}`;
          const abs = row.abs_effect_size ?? Math.abs(row.effect_size ?? 0);
          const prev = effects.get(key);
          if (!prev || abs > prev.abs) effects.set(key, { abs, effect: row.effect_size ?? 0 });
        }
      }
      return effects;
    };
    const eA = compute(prompts[0]);
    const eB = compute(prompts[1]);
    if (eA.size === 0 && eB.size === 0) return null;
    return { pidA: prompts[0], pidB: prompts[1], effectsA: eA, effectsB: eB };
  }, [prompts, data.interventions]);

  // Compute per-node effect values from raw interventions (for 'effect' color mode)
  const nodeEffectValues = useMemo(() => {
    if (graphColorMode !== 'effect') return null;

    const { interventions, commonPromptIdx } = data;
    const selectedExps = experiments.length > 0 ? experiments : indexes.experiments;
    const expSet = new Set(selectedExps);
    const promptSet = prompts.length > 0 ? new Set(prompts) : null;
    const commonSet = showCommonPromptsOnly
      ? new Set(commonPromptIdx?.common_prompt_idx ?? [])
      : null;

    const accum = new Map();
    for (const row of interventions) {
      if (!expSet.has(row.experiment_type)) continue;
      if (promptSet && !promptSet.has(row.prompt_idx)) continue;
      if (commonSet && !commonSet.has(row.prompt_idx)) continue;

      const layer = row.layer;
      for (const fid of row.feature_indices) {
        const key = `L${layer}_F${fid}`;
        let a = accum.get(key);
        if (!a) { a = { sum: 0, absSum: 0, flips: 0, count: 0 }; accum.set(key, a); }
        const es = row.effect_size;
        if (es != null && Number.isFinite(es)) { a.sum += es; a.absSum += Math.abs(es); a.count++; }
        if (row.sign_flipped) a.flips++;
      }
    }

    const values = new Map();
    for (const [nodeId, a] of accum) {
      if (a.count === 0) continue;
      let val;
      if (graphEffectMetric === 'mean_abs_effect_size') val = a.absSum / a.count;
      else if (graphEffectMetric === 'mean_effect_size') val = a.sum / a.count;
      else if (graphEffectMetric === 'sign_flip_rate') val = a.flips / a.count;
      else val = a.absSum / a.count;
      values.set(nodeId, val);
    }
    return values;
  }, [graphColorMode, graphEffectMetric, experiments, prompts, showCommonPromptsOnly, data, indexes.experiments]);

  // Build Plotly traces
  const plotData = useMemo(() => {
    const pne = promptNodeEffects;   // single-prompt mode
    const po = promptOverlap;        // two-prompt overlap mode
    const hasPromptData = pne || po;

    // ── Helper: is this feature node "active" for the current prompt selection? ──
    const isActive = (nodeId) => {
      const info = nodeById.get(nodeId);
      if (!info || info.type !== 'feature') return true; // I/O nodes always active
      if (pne) return pne.has(nodeId);
      if (po) return po.effectsA.has(nodeId) || po.effectsB.has(nodeId);
      return true;
    };

    // ── Edges: split into active (bright) and dim ──────────────────────────────
    const activeEx = [], activeEy = [], dimEx = [], dimEy = [];
    for (const link of graph.links) {
      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;
      if (!visibleNodes.has(src) || !visibleNodes.has(tgt)) continue;
      const ps = positions.get(src);
      const pt = positions.get(tgt);
      if (!ps || !pt) continue;

      // Dim edge only when both endpoints are inactive features (local mode active)
      const isDim = graphLocalMode && hasPromptData && !isActive(src) && !isActive(tgt);
      if (isDim) { dimEx.push(ps.x, pt.x, null); dimEy.push(ps.y, pt.y, null); }
      else { activeEx.push(ps.x, pt.x, null); activeEy.push(ps.y, pt.y, null); }
    }

    const activeEdgeTrace = {
      x: activeEx, y: activeEy, mode: 'lines', type: 'scatter',
      line: { width: 0.8, color: 'rgba(100,110,140,0.35)' },
      hoverinfo: 'none',
    };
    const dimEdgeTrace = {
      x: dimEx, y: dimEy, mode: 'lines', type: 'scatter',
      line: { width: 0.3, color: 'rgba(60,65,80,0.1)' },
      hoverinfo: 'none',
    };

    const highlightId = hoveredFeatureId || selectedFeatureId;
    const neighborSet = highlightId ? neighbors.get(highlightId) : null;

    // ── Effect color mode ──────────────────────────────────────────────────────
    if (graphColorMode === 'effect' && nodeEffectValues) {
      const hasData = { x: [], y: [], sizes: [], colors: [], texts: [], hoverTexts: [], ids: [] };
      const noData = { x: [], y: [], sizes: [], texts: [], hoverTexts: [], ids: [] };

      let effectMin = Infinity, effectMax = -Infinity;
      for (const v of nodeEffectValues.values()) {
        if (v < effectMin) effectMin = v;
        if (v > effectMax) effectMax = v;
      }
      if (!Number.isFinite(effectMin)) { effectMin = 0; effectMax = 1; }

      for (const [id, pos] of positions) {
        if (!visibleNodes.has(id)) continue;
        const info = nodeById.get(id);
        const effectVal = nodeEffectValues.get(id);
        const isSpecial = info.type === 'input' || info.type === 'output';
        const hoverBase = `<b>${shortNodeId(id)}</b>` +
          (info.type === 'feature' ? `<br>Layer ${info.layer}` : '') +
          (featureToCluster.has(id) ? `<br>Cluster: ${featureToCluster.get(id)}` : '');

        if (effectVal != null && !isSpecial) {
          const range = effectMax - effectMin || 1;
          const magnitude = Math.abs(effectVal - effectMin) / range;
          hasData.x.push(pos.x); hasData.y.push(pos.y); hasData.ids.push(id);
          hasData.colors.push(effectVal);
          hasData.sizes.push(Math.max(6, 6 + magnitude * 18));
          hasData.texts.push(shortNodeId(id));
          hasData.hoverTexts.push(hoverBase + `<br>${graphEffectMetric}: ${fmt(effectVal, 4)}`);
        } else {
          noData.x.push(pos.x); noData.y.push(pos.y); noData.ids.push(id);
          noData.sizes.push(isSpecial ? 14 : 6);
          noData.texts.push(shortNodeId(id));
          noData.hoverTexts.push(hoverBase + (isSpecial ? '' : '<br>No intervention data'));
        }
      }

      const inspectSet = new Set(inspectedFeatureIds);
      const metricLabel = graphEffectMetric.replace(/_/g, ' ');

      const hasDataTrace = {
        x: hasData.x, y: hasData.y, mode: 'markers+text', type: 'scatter',
        marker: {
          size: hasData.sizes, color: hasData.colors,
          colorscale: [[0,'#0f1117'],[0.25,'#1a365d'],[0.5,'#2b6cb0'],[0.75,'#ed8936'],[1,'#f6e05e']],
          cmin: effectMin, cmax: effectMax,
          colorbar: { title: { text: metricLabel, font: { size: 10, color: '#8b90a5' } }, tickfont: { size: 9, color: '#8b90a5' }, len: 0.6, thickness: 12, x: 1.02 },
          opacity: hasData.ids.map(id => {
            if (!highlightId) return 1;
            if (id === highlightId) return 1;
            if (neighborSet && neighborSet.has(id)) return 0.9;
            return 0.15;
          }),
          line: {
            width: hasData.ids.map(id => inspectSet.has(id) ? 3 : 1),
            color: hasData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.3)'),
          },
        },
        text: hasData.texts, textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
        hovertext: hasData.hoverTexts, hoverinfo: 'text', customdata: hasData.ids,
      };
      const noDataTrace = {
        x: noData.x, y: noData.y, mode: 'markers+text', type: 'scatter',
        marker: {
          size: noData.sizes, color: '#555',
          opacity: noData.ids.map(id => {
            if (!highlightId) return 0.5;
            if (id === highlightId) return 0.8;
            if (neighborSet && neighborSet.has(id)) return 0.6;
            return 0.1;
          }),
          line: {
            width: noData.ids.map(id => inspectSet.has(id) ? 3 : 1),
            color: noData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.15)'),
          },
        },
        text: noData.texts, textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.4)' },
        hovertext: noData.hoverTexts, hoverinfo: 'text', customdata: noData.ids,
      };
      return [dimEdgeTrace, activeEdgeTrace, noDataTrace, hasDataTrace];
    }

    // ── Layer / Cluster mode ───────────────────────────────────────────────────
    const nx = [], ny = [], colors = [], sizes = [], texts = [], hoverTexts = [];
    const lineWidthArr = [], lineColorArr = [], opacityArr = [];
    const ids = [];

    const maxPromptEff = pne
      ? Math.max(1e-6, ...[...pne.values()].map(v => v.abs))
      : 0;
    const maxOverlapEff = po
      ? Math.max(1e-6,
          ...[...po.effectsA.values()].map(v => v.abs),
          ...[...po.effectsB.values()].map(v => v.abs))
      : 0;
    const inspectSet = new Set(inspectedFeatureIds);

    for (const [id, pos] of positions) {
      if (!visibleNodes.has(id)) continue;
      const info = nodeById.get(id);
      nx.push(pos.x); ny.push(pos.y); ids.push(id);

      // ── Node size ────────────────────────────────────────────────────────────
      let size;
      if (pne && info.type === 'feature') {
        const pe = pne.get(id);
        size = pe ? 6 + (pe.abs / maxPromptEff) * 22 : 5;
      } else if (po && info.type === 'feature') {
        const absA = po.effectsA.get(id)?.abs ?? 0;
        const absB = po.effectsB.get(id)?.abs ?? 0;
        const combined = absA + absB;
        size = combined > 0 ? 6 + (combined / maxOverlapEff) * 22 : 5;
      } else if (info.type === 'feature' && info.abs_corr != null) {
        size = 6 + info.abs_corr * 40;
      } else if (info.type === 'input' || info.type === 'output') {
        size = 14;
      } else {
        size = 8;
      }
      sizes.push(Math.min(size, 28));

      // ── Node color ───────────────────────────────────────────────────────────
      let col;
      if (po && info.type === 'feature') {
        const inA = po.effectsA.has(id);
        const inB = po.effectsB.has(id);
        col = (inA && inB) ? OVERLAP.shared
            : inA ? OVERLAP.promptA
            : inB ? OVERLAP.promptB
            : OVERLAP.inactive;
      } else {
        const colorByCluster = graphColorMode === 'cluster' || clusters.length > 0;
        col = nodeColor(id, info, featureToCluster, colorByCluster);
      }
      colors.push(col);

      // ── Hover text ───────────────────────────────────────────────────────────
      texts.push(shortNodeId(id));
      const imp = info.featureIdx != null
        ? indexes.importanceByKey.get(`${info.layer}_${info.featureIdx}`)
        : null;
      const pe = pne?.get(id);
      const inA = po?.effectsA.get(id);
      const inB = po?.effectsB.get(id);
      hoverTexts.push(
        `<b>${shortNodeId(id)}</b>` +
        (info.type === 'feature' ? `<br>Layer ${info.layer}` : '') +
        (imp ? `<br>|corr|: ${fmt(imp.abs_correlation)}` : '') +
        (featureToCluster.has(id) ? `<br>Cluster: ${featureToCluster.get(id)}` : '') +
        (pe ? `<br><b>Effect P${prompts[0]}: ${fmt(pe.effect, 3)}${pe.signFlipped ? ' ⚠ flip' : ''}</b>` : '') +
        (inA ? `<br>P${po.pidA}: ${fmt(inA.effect, 3)}` : '') +
        (inB ? `<br>P${po.pidB}: ${fmt(inB.effect, 3)}` : '') +
        (hasPromptData && !isActive(id) && info.type === 'feature' ? '<br><i>inactive for selected prompt(s)</i>' : '')
      );

      // ── Border (path > inspected > sign-flip > default) ──────────────────────
      const isInPath = selectedPath?.nodes.includes(id);
      const isFlipped = pne && pe?.signFlipped;
      if (isInPath) {
        lineWidthArr.push(4);
        lineColorArr.push(selectedPath.color);
      } else if (inspectSet.has(id)) {
        lineWidthArr.push(3); lineColorArr.push('#fcc419');
      } else if (isFlipped) {
        lineWidthArr.push(2.5); lineColorArr.push('#ef4444');
      } else if (po && (po.effectsA.has(id) || po.effectsB.has(id))) {
        lineWidthArr.push(1); lineColorArr.push('rgba(255,255,255,0.6)');
      } else {
        lineWidthArr.push(1); lineColorArr.push('rgba(255,255,255,0.3)');
      }

      // ── Opacity (path dims non-path nodes, otherwise existing logic) ─────────
      let opacity;
      if (selectedPath) {
        opacity = isInPath ? 1 : 0.18;
      } else if (highlightId) {
        opacity = id === highlightId ? 1 : (neighborSet?.has(id) ? 0.9 : 0.15);
      } else if (hasPromptData && graphLocalMode && info.type === 'feature') {
        opacity = isActive(id) ? 1 : 0.08;
      } else if (hasPromptData && info.type === 'feature') {
        opacity = isActive(id) ? 1 : 0.3;
      } else {
        opacity = 1;
      }
      opacityArr.push(opacity);
    }

    const nodeTrace = {
      x: nx, y: ny, mode: 'markers+text', type: 'scatter',
      marker: {
        size: sizes, color: colors, opacity: opacityArr,
        line: { width: lineWidthArr, color: lineColorArr },
      },
      text: texts, textposition: 'top center',
      textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
      hovertext: hoverTexts, hoverinfo: 'text', customdata: ids,
    };

    // ── Path highlight trace (thick colored edges on top of graph) ───────────
    let pathTrace = null;
    if (selectedPath?.nodes?.length >= 2) {
      const px = [], py = [];
      for (let i = 0; i < selectedPath.nodes.length - 1; i++) {
        const s = selectedPath.nodes[i];
        const t = selectedPath.nodes[i + 1];
        const ps = positions.get(s);
        const pt = positions.get(t);
        if (ps && pt) { px.push(ps.x, pt.x, null); py.push(ps.y, pt.y, null); }
      }
      if (px.length > 0) {
        pathTrace = {
          x: px, y: py, mode: 'lines', type: 'scatter',
          line: { width: 5, color: selectedPath.color },
          hoverinfo: 'none', opacity: 0.85,
        };
      }
    }

    const traces = [dimEdgeTrace, activeEdgeTrace];
    if (pathTrace) traces.push(pathTrace);
    traces.push(nodeTrace);
    return traces;
  }, [positions, visibleNodes, graph.links, nodeById, indexes, featureToCluster, neighbors,
      hoveredFeatureId, selectedFeatureId, clusters, graphColorMode, graphEffectMetric,
      nodeEffectValues, inspectedFeatureIds, promptNodeEffects, promptOverlap,
      graphLocalMode, prompts, selectedPath]);

  const handleClick = useCallback((event) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      const id = point.customdata;
      if (id && nodeById.get(id)?.type === 'feature') {
        if (event.event?.shiftKey) {
          addInspectedFeature(id);
        } else {
          setSelectedFeatureId(id === selectedFeatureId ? null : id);
        }
      }
    }
  }, [selectedFeatureId, setSelectedFeatureId, addInspectedFeature, nodeById]);

  const handleHover = useCallback((event) => {
    if (event.points && event.points.length > 0) setHoveredFeatureId(event.points[0].customdata);
  }, [setHoveredFeatureId]);

  const handleUnhover = useCallback(() => setHoveredFeatureId(null), [setHoveredFeatureId]);

  const layout = useMemo(() => {
    let titleText, titleColor;
    if (selectedPath) {
      const dir = selectedPath.direction === 'correct' ? '→ ✓ correct' : '→ ✗ incorrect';
      titleText = `Path: input → ${selectedPath.feature} ${dir}`;
      titleColor = selectedPath.color;
    } else if (promptOverlap) {
      titleText = `Attribution Graph — P${promptOverlap.pidA} vs P${promptOverlap.pidB} overlap`;
      titleColor = '#9333ea';
    } else if (promptNodeEffects) {
      titleText = `Attribution Graph — P${prompts[0]} (node size = effect)`;
      titleColor = '#f97316';
    } else {
      titleText = `Attribution Graph (${visibleNodes.size} nodes)`;
      titleColor = '#8b90a5';
    }
    return {
      width: dims.width,
      height: dims.height,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 10, r: 10, t: 30, b: 10 },
      title: { text: titleText, font: { size: 12, color: titleColor }, x: 0.01, xanchor: 'left' },
      xaxis: { visible: false, showgrid: false, zeroline: false },
      yaxis: { visible: false, showgrid: false, zeroline: false },
      showlegend: false,
      hovermode: 'closest',
      dragmode: 'pan',
    };
  }, [dims, visibleNodes.size, promptNodeEffects, promptOverlap, prompts, selectedPath]);

  const hasPromptData = promptNodeEffects || promptOverlap;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0,
        padding: '4px 10px',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border)',
        fontSize: 11,
      }}>
        <button
          onClick={() => setGraphLocalMode(!graphLocalMode)}
          style={{
            padding: '2px 10px', borderRadius: 4,
            background: graphLocalMode ? 'var(--accent)22' : 'transparent',
            border: `1px solid ${graphLocalMode ? 'var(--accent)' : 'var(--border)'}`,
            color: graphLocalMode ? 'var(--accent)' : 'var(--text-dim)',
            cursor: 'pointer', fontSize: 11, whiteSpace: 'nowrap',
          }}
        >
          {graphLocalMode ? '● Prompt-filtered' : '○ Global'}
        </button>

        {/* Status / legend */}
        {graphLocalMode && !hasPromptData && (
          <span style={{ color: 'var(--text-dim)', fontSize: 10 }}>
            Select a prompt to highlight its features
          </span>
        )}
        {promptOverlap && (
          <span style={{ display: 'flex', gap: 8, fontSize: 10 }}>
            <span style={{ color: OVERLAP.promptA }}>■ P{promptOverlap.pidA} only</span>
            <span style={{ color: OVERLAP.shared }}>■ shared</span>
            <span style={{ color: OVERLAP.promptB }}>■ P{promptOverlap.pidB} only</span>
          </span>
        )}
        {promptNodeEffects && (
          <span style={{ color: 'var(--text-dim)', fontSize: 10 }}>
            P{prompts[0]} · node size = ablation effect · click node → Feature Details tab
          </span>
        )}
      </div>

      {/* Graph */}
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        <Plot
          data={plotData}
          layout={layout}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false,
            responsive: true,
          }}
          onClick={handleClick}
          onHover={handleHover}
          onUnhover={handleUnhover}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}
