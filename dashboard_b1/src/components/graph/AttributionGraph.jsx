import { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import Plot from 'react-plotly.js';
import { forceSimulation, forceLink, forceManyBody, forceX, forceY, forceCollide } from 'd3-force';
import useStore from '../../store/useStore';
import { filterGraphNodes } from '../../utils/filterData';
import { nodeColor, langProfileColor } from '../../utils/colors';
import { shortNodeId, fmt } from '../../utils/formatters';

export default function AttributionGraph({ data, indexes }) {
  const { graph } = data;
  const { nodeById, neighbors, featureToCluster, langProfileByNode, activeFeaturesByPrompt } = indexes;

  const layerRange = useStore(s => s.layerRange);
  const clusters = useStore(s => s.clusters);
  const searchQuery = useStore(s => s.searchQuery);
  const selectedFeatureId = useStore(s => s.selectedFeatureId);
  const hoveredFeatureId = useStore(s => s.hoveredFeatureId);
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  const setHoveredFeatureId = useStore(s => s.setHoveredFeatureId);
  const graphColorMode = useStore(s => s.graphColorMode);
  const graphEffectMetric = useStore(s => s.graphEffectMetric);
  const experiments = useStore(s => s.experiments);
  const prompts = useStore(s => s.prompts);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const addInspectedFeature = useStore(s => s.addInspectedFeature);
  const langProfileFilter = useStore(s => s.langProfileFilter);
  const showCircuitOnly = useStore(s => s.showCircuitOnly);
  const selectedPromptIdx = useStore(s => s.selectedPromptIdx);
  const graphPromptMode = useStore(s => s.graphPromptMode);
  const selectedPathStr = useStore(s => s.selectedPathStr);
  const overlapColorMode = useStore(s => s.overlapColorMode);
  const overlapSets = useStore(s => s.overlapSets);
  const multiOverlapMode = useStore(s => s.multiOverlapMode);
  const multiSupportFractions = useStore(s => s.multiSupportFractions);
  const multiFreqThreshold = useStore(s => s.multiFreqThreshold);
  const setVsSetMode = useStore(s => s.setVsSetMode);
  const setVsSetFractions = useStore(s => s.setVsSetFractions);

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

  // Circuit feature set for highlighting
  const circuitFeatureSet = useMemo(() => {
    const circuit = data.circuit;
    if (!circuit) return new Set();
    return new Set(circuit.feature_nodes || []);
  }, [data.circuit]);

  // Prompt-mode: active features for selected prompt
  const promptActiveSet = useMemo(() => {
    if (!graphPromptMode || selectedPromptIdx == null) return null;
    return activeFeaturesByPrompt?.get(selectedPromptIdx) ?? new Set();
  }, [graphPromptMode, selectedPromptIdx, activeFeaturesByPrompt]);

  // Path-mode: nodes in selected path
  const pathNodeSet = useMemo(() => {
    if (!selectedPathStr) return null;
    const nodes = selectedPathStr.split(' → ').map(s => s.trim())
      .filter(s => s.match(/^L\d+_F\d+$/) || s === 'input' || s.startsWith('output'));
    return new Set(nodes);
  }, [selectedPathStr]);

  // Compute visible node set
  const visibleNodes = useMemo(() => {
    const filters = { layerRange, clusters };
    let nodes = new Set(filterGraphNodes(nodeById, filters, featureToCluster, searchQuery));

    // Apply lang profile filter
    if (langProfileFilter) {
      for (const id of [...nodes]) {
        const info = nodeById.get(id);
        if (info?.type === 'feature') {
          const profile = langProfileByNode.get(id) || 'balanced';
          if (profile !== langProfileFilter) nodes.delete(id);
        }
      }
    }

    // Show circuit only
    if (showCircuitOnly && circuitFeatureSet.size > 0) {
      for (const id of [...nodes]) {
        const info = nodeById.get(id);
        if (info?.type === 'feature' && !circuitFeatureSet.has(id)) {
          nodes.delete(id);
        }
      }
    }

    return nodes;
  }, [nodeById, layerRange, clusters, featureToCluster, searchQuery, langProfileFilter, showCircuitOnly, circuitFeatureSet, langProfileByNode]);

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
    for (const n of nodes) {
      pos.set(n.id, { x: n.x, y: n.y });
    }
    return pos;
  }, [graph, nodeById, indexes.layers]);

  // Per-node effect values for 'effect' color mode
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
    // Edges — distinguish causal vs star edges for B1
    const causalEdgeSet = new Set();
    if (data.circuit?.edges) {
      for (const e of data.circuit.edges) {
        if (e.edge_type === 'causal') {
          causalEdgeSet.add(`${e.source}||${e.target}`);
        }
      }
    }

    const ex = [], ey = []; // star/VW edges
    const cx = [], cy = []; // causal edges
    for (const link of graph.links) {
      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;
      if (!visibleNodes.has(src) || !visibleNodes.has(tgt)) continue;
      const ps = positions.get(src);
      const pt = positions.get(tgt);
      if (!ps || !pt) continue;
      const isCausal = causalEdgeSet.has(`${src}||${tgt}`);
      if (isCausal) {
        cx.push(ps.x, pt.x, null);
        cy.push(ps.y, pt.y, null);
      } else {
        ex.push(ps.x, pt.x, null);
        ey.push(ps.y, pt.y, null);
      }
    }

    const edgeTrace = {
      x: ex, y: ey,
      mode: 'lines',
      line: { width: 0.5, color: 'rgba(100,110,140,0.20)' },
      hoverinfo: 'none',
      type: 'scatter',
    };

    const causalEdgeTrace = {
      x: cx, y: cy,
      mode: 'lines',
      line: { width: 1.5, color: 'rgba(255,200,80,0.55)' },
      hoverinfo: 'none',
      type: 'scatter',
    };

    const highlightId = hoveredFeatureId || selectedFeatureId;
    const neighborSet = highlightId ? neighbors.get(highlightId) : null;

    // lang_profile color mode (B1-specific default)
    if (graphColorMode === 'lang_profile') {
      const nx = [], ny = [], colors = [], sizes = [], texts = [], hoverTexts = [], ids = [];

      for (const [id, pos] of positions) {
        if (!visibleNodes.has(id)) continue;
        const info = nodeById.get(id);
        nx.push(pos.x);
        ny.push(pos.y);
        ids.push(id);

        let size = 8;
        if (info.type === 'feature') {
          size = circuitFeatureSet.has(id) ? 14 : 7;
        } else if (info.type === 'input' || info.type === 'output') {
          size = 16;
        }
        sizes.push(size);

        const profile = langProfileByNode.get(id) || (info.type === 'feature' ? 'balanced' : null);
        let col = info.type === 'input' ? '#555'
          : info.type === 'output' ? (info.output === 'correct' ? '#2ca02c' : '#d62728')
          : langProfileColor(profile);
        // Set-vs-set mode (highest priority)
        if (multiOverlapMode && setVsSetMode && setVsSetFractions && info.type === 'feature') {
          const svs = setVsSetFractions.get(id);
          const th = multiFreqThreshold;
          if (svs) {
            const hiA = svs.fracA >= th, hiB = svs.fracB >= th;
            if (hiA && hiB)    col = '#9b59b6'; // shared
            else if (hiA)      col = '#4e9af1'; // A-specific
            else if (hiB)      col = '#74c69d'; // B-specific
            else               col = '#2a2a3e'; // low in both
          } else {
            col = '#1a1a2a';
          }
        // Multi-prompt overlap: color by support fraction
        } else if (multiOverlapMode && multiSupportFractions && info.type === 'feature') {
          const frac = multiSupportFractions.get(id);
          if (frac != null) {
            if (frac >= 1.0)       col = '#4a0e8f';
            else if (frac >= 0.75) col = '#7b2d8b';
            else if (frac >= 0.5)  col = '#9b59b6';
            else if (frac >= 0.25) col = '#c39bd3';
            else                   col = '#8b7aa8';
          } else {
            col = '#1e1e2e'; // not in any selected prompt
          }
        // Pairwise overlap overlay
        } else if (overlapColorMode && overlapSets && info.type === 'feature') {
          if (overlapSets.shared.has(id)) col = '#9b59b6';
          else if (overlapSets.aOnly.has(id)) col = '#4e9af1';
          else if (overlapSets.bOnly.has(id)) col = '#f77f4e';
          else col = '#333';
        }
        colors.push(col);

        const community = indexes.communityByNode?.get(id);
        texts.push(shortNodeId(id));
        hoverTexts.push(
          `<b>${shortNodeId(id)}</b>` +
          (info.type === 'feature' ? `<br>Layer ${info.layer}` : '') +
          (profile ? `<br>Profile: ${profile}` : '') +
          (community != null ? `<br>Community: C${community}` : '') +
          (circuitFeatureSet.has(id) ? '<br><b>★ Circuit feature</b>' : '')
        );
      }

      const opacities = ids.map(id => {
        const info = nodeById.get(id);
        // Path mode: dim everything not in path
        if (pathNodeSet) {
          if (pathNodeSet.has(id)) return 1;
          return 0.08;
        }
        // Prompt mode: dim features not active for this prompt
        if (promptActiveSet && info?.type === 'feature') {
          if (promptActiveSet.has(id)) return 1;
          return 0.12;
        }
        if (!highlightId) return 1;
        if (id === highlightId) return 1;
        if (neighborSet && neighborSet.has(id)) return 0.9;
        return 0.15;
      });

      const inspectSet = new Set(inspectedFeatureIds);
      const lineWidths = ids.map(id => {
        if (pathNodeSet?.has(id)) return 3;
        if (inspectSet.has(id)) return 3;
        if (circuitFeatureSet.has(id)) return 2;
        return 1;
      });
      const lineColors = ids.map(id => {
        if (pathNodeSet?.has(id)) return '#fff';
        if (inspectSet.has(id)) return '#fcc419';
        if (circuitFeatureSet.has(id)) return 'rgba(255,220,100,0.8)';
        return 'rgba(255,255,255,0.25)';
      });

      const nodeTrace = {
        x: nx, y: ny,
        mode: 'markers+text',
        type: 'scatter',
        marker: { size: sizes, color: colors, opacity: opacities, line: { width: lineWidths, color: lineColors } },
        text: texts,
        textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
        hovertext: hoverTexts,
        hoverinfo: 'text',
        customdata: ids,
      };

      return [edgeTrace, causalEdgeTrace, nodeTrace];
    }

    // effect mode
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
          (featureToCluster.has(id) ? `<br>Community: C${featureToCluster.get(id)}` : '');

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
          noData.hoverTexts.push(hoverBase + (isSpecial ? '' : '<br>No data'));
        }
      }

      const inspectSet = new Set(inspectedFeatureIds);
      const metricLabel = graphEffectMetric.replace(/_/g, ' ');

      return [edgeTrace, causalEdgeTrace, {
        x: noData.x, y: noData.y, mode: 'markers+text', type: 'scatter',
        marker: { size: noData.sizes, color: '#555',
          opacity: noData.ids.map(id => !highlightId ? 0.5 : id === highlightId ? 0.8 : neighborSet?.has(id) ? 0.6 : 0.1),
          line: { width: noData.ids.map(id => inspectSet.has(id) ? 3 : 1), color: noData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.15)') }},
        text: noData.texts, textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.4)' },
        hovertext: noData.hoverTexts, hoverinfo: 'text', customdata: noData.ids,
      }, {
        x: hasData.x, y: hasData.y, mode: 'markers+text', type: 'scatter',
        marker: { size: hasData.sizes, color: hasData.colors,
          colorscale: [[0,'#0f1117'],[0.25,'#1a365d'],[0.5,'#2b6cb0'],[0.75,'#ed8936'],[1,'#f6e05e']],
          cmin: effectMin, cmax: effectMax,
          colorbar: { title: { text: metricLabel, font: { size: 10, color: '#8b90a5' } }, tickfont: { size: 9, color: '#8b90a5' }, len: 0.6, thickness: 12, x: 1.02 },
          opacity: hasData.ids.map(id => !highlightId ? 1 : id === highlightId ? 1 : neighborSet?.has(id) ? 0.9 : 0.15),
          line: { width: hasData.ids.map(id => inspectSet.has(id) ? 3 : 1), color: hasData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.3)') }},
        text: hasData.texts, textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
        hovertext: hasData.hoverTexts, hoverinfo: 'text', customdata: hasData.ids,
      }];
    }

    // Layer / cluster mode
    const nx = [], ny = [], colors = [], sizes = [], texts = [], hoverTexts = [], ids = [];
    for (const [id, pos] of positions) {
      if (!visibleNodes.has(id)) continue;
      const info = nodeById.get(id);
      nx.push(pos.x); ny.push(pos.y); ids.push(id);
      let size = info.type === 'feature' ? (circuitFeatureSet.has(id) ? 14 : 7) : 16;
      sizes.push(Math.min(size, 24));
      const col = nodeColor(id, info, featureToCluster, graphColorMode);
      colors.push(col);
      texts.push(shortNodeId(id));
      const imp = info.featureIdx != null
        ? indexes.importanceByKey.get(`${info.layer}_${info.featureIdx}`)
        : null;
      const community = indexes.communityByNode?.get(id);
      hoverTexts.push(
        `<b>${shortNodeId(id)}</b>` +
        (info.type === 'feature' ? `<br>Layer ${info.layer}` : '') +
        (imp ? `<br>|corr|: ${fmt(imp.abs_correlation)}` : '') +
        (community != null ? `<br>Community: C${community}` : '')
      );
    }

    const opacities = ids.map(id => {
      if (!highlightId) return 1;
      if (id === highlightId) return 1;
      if (neighborSet && neighborSet.has(id)) return 0.9;
      return 0.15;
    });

    const inspectSet = new Set(inspectedFeatureIds);
    return [edgeTrace, causalEdgeTrace, {
      x: nx, y: ny, mode: 'markers+text', type: 'scatter',
      marker: { size: sizes, color: colors, opacity: opacities,
        line: { width: ids.map(id => inspectSet.has(id) ? 3 : 1), color: ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.3)') }},
      text: texts, textposition: 'top center',
      textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
      hovertext: hoverTexts, hoverinfo: 'text', customdata: ids,
    }];
  }, [positions, visibleNodes, graph.links, nodeById, indexes, featureToCluster, langProfileByNode, neighbors, hoveredFeatureId, selectedFeatureId, graphColorMode, graphEffectMetric, nodeEffectValues, inspectedFeatureIds, circuitFeatureSet, data.circuit, promptActiveSet, pathNodeSet, overlapColorMode, overlapSets, multiOverlapMode, multiSupportFractions, multiFreqThreshold, setVsSetMode, setVsSetFractions]);

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
    if (event.points && event.points.length > 0) {
      setHoveredFeatureId(event.points[0].customdata);
    }
  }, [setHoveredFeatureId]);

  const handleUnhover = useCallback(() => { setHoveredFeatureId(null); }, [setHoveredFeatureId]);

  const layout = useMemo(() => ({
    width: dims.width,
    height: dims.height,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 10, r: 10, t: 30, b: 10 },
    title: {
      text: `Attribution Graph (${visibleNodes.size} nodes)${multiOverlapMode && setVsSetMode && setVsSetFractions ? ` — A vs B (${setVsSetFractions.size} nodes)` : multiOverlapMode && multiSupportFractions ? ` — multi-overlap (${multiSupportFractions.size} active)` : graphPromptMode && selectedPromptIdx != null ? ` — prompt #${selectedPromptIdx}` : ''}${selectedPathStr ? ' — path highlighted' : ''}`,
      font: { size: 11, color: '#8b90a5' },
      x: 0.01, xanchor: 'left',
    },
    xaxis: { visible: false, showgrid: false, zeroline: false },
    yaxis: { visible: false, showgrid: false, zeroline: false },
    showlegend: false,
    hovermode: 'closest',
    dragmode: 'pan',
  }), [dims, visibleNodes.size, multiOverlapMode, setVsSetMode, setVsSetFractions, multiSupportFractions, graphPromptMode, selectedPromptIdx, selectedPathStr]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      <Plot
        data={plotData}
        layout={layout}
        config={{ displayModeBar: true, modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'], displaylogo: false, responsive: true }}
        onClick={handleClick}
        onHover={handleHover}
        onUnhover={handleUnhover}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
