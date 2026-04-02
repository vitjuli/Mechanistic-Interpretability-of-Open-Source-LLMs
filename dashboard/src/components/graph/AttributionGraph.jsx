import { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import Plot from 'react-plotly.js';
import { forceSimulation, forceLink, forceManyBody, forceX, forceY, forceCollide } from 'd3-force';
import useStore from '../../store/useStore';
import { filterGraphNodes } from '../../utils/filterData';
import { nodeColor } from '../../utils/colors';
import { shortNodeId, fmt } from '../../utils/formatters';

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
  const experiments = useStore(s => s.experiments);
  const prompts = useStore(s => s.prompts);
  const showCommonPromptsOnly = useStore(s => s.showCommonPromptsOnly);
  const inspectedFeatureIds = useStore(s => s.inspectedFeatureIds);
  const addInspectedFeature = useStore(s => s.addInspectedFeature);

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

  // Run force layout once (positions are stable for given graph)
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

  // Compute per-node effect values from raw interventions with current filters
  const nodeEffectValues = useMemo(() => {
    if (graphColorMode !== 'effect') return null;

    const { interventions, commonPromptIdx } = data;
    const selectedExps = experiments.length > 0 ? experiments : indexes.experiments;
    const expSet = new Set(selectedExps);
    const promptSet = prompts.length > 0 ? new Set(prompts) : null;
    const commonSet = showCommonPromptsOnly
      ? new Set(commonPromptIdx?.common_prompt_idx ?? [])
      : null;

    // Accumulate per (layer, featureIdx): { sum, absSum, flips, count }
    const accum = new Map();

    for (const row of interventions) {
      if (!expSet.has(row.experiment_type)) continue;
      if (promptSet && !promptSet.has(row.prompt_idx)) continue;
      if (commonSet && !commonSet.has(row.prompt_idx)) continue;

      const layer = row.layer;
      for (const fid of row.feature_indices) {
        const key = `L${layer}_F${fid}`;
        let a = accum.get(key);
        if (!a) {
          a = { sum: 0, absSum: 0, flips: 0, count: 0 };
          accum.set(key, a);
        }
        const es = row.effect_size;
        if (es != null && Number.isFinite(es)) {
          a.sum += es;
          a.absSum += Math.abs(es);
          a.count++;
        }
        if (row.sign_flipped) a.flips++;
      }
    }

    // Convert to final metric values
    const values = new Map();
    for (const [nodeId, a] of accum) {
      if (a.count === 0) continue;
      let val;
      if (graphEffectMetric === 'mean_abs_effect_size') {
        val = a.absSum / a.count;
      } else if (graphEffectMetric === 'mean_effect_size') {
        val = a.sum / a.count;
      } else if (graphEffectMetric === 'sign_flip_rate') {
        val = a.flips / a.count;
      } else {
        val = a.absSum / a.count;
      }
      values.set(nodeId, val);
    }
    return values;
  }, [graphColorMode, graphEffectMetric, experiments, prompts, showCommonPromptsOnly, data, indexes.experiments, nodeById]);

  // Build Plotly traces
  const plotData = useMemo(() => {
    // Edges
    const ex = [], ey = [];
    for (const link of graph.links) {
      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;
      if (!visibleNodes.has(src) || !visibleNodes.has(tgt)) continue;
      const ps = positions.get(src);
      const pt = positions.get(tgt);
      if (!ps || !pt) continue;
      ex.push(ps.x, pt.x, null);
      ey.push(ps.y, pt.y, null);
    }

    const edgeTrace = {
      x: ex, y: ey,
      mode: 'lines',
      line: { width: 0.5, color: 'rgba(100,110,140,0.25)' },
      hoverinfo: 'none',
      type: 'scatter',
    };

    const highlightId = hoveredFeatureId || selectedFeatureId;
    const neighborSet = highlightId ? neighbors.get(highlightId) : null;

    if (graphColorMode === 'effect' && nodeEffectValues) {
      // Two-trace approach: has-data (colored) and no-data (grey)
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
          hasData.x.push(pos.x);
          hasData.y.push(pos.y);
          hasData.ids.push(id);
          hasData.colors.push(effectVal);
          hasData.sizes.push(Math.max(6, 6 + magnitude * 18));
          hasData.texts.push(shortNodeId(id));
          hasData.hoverTexts.push(hoverBase + `<br>${graphEffectMetric}: ${fmt(effectVal, 4)}`);
        } else {
          noData.x.push(pos.x);
          noData.y.push(pos.y);
          noData.ids.push(id);
          let size = isSpecial ? 14 : 6;
          noData.sizes.push(size);
          noData.texts.push(shortNodeId(id));
          noData.hoverTexts.push(hoverBase + (isSpecial ? '' : '<br>No intervention data'));
        }
      }

      // Dimming for highlight
      const hasDataOpacities = hasData.ids.map(id => {
        if (!highlightId) return 1;
        if (id === highlightId) return 1;
        if (neighborSet && neighborSet.has(id)) return 0.9;
        return 0.15;
      });
      const noDataOpacities = noData.ids.map(id => {
        if (!highlightId) return 0.5;
        if (id === highlightId) return 0.8;
        if (neighborSet && neighborSet.has(id)) return 0.6;
        return 0.1;
      });

      const metricLabel = graphEffectMetric.replace(/_/g, ' ');
      const inspectSet = new Set(inspectedFeatureIds);

      const hasDataTrace = {
        x: hasData.x, y: hasData.y,
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          size: hasData.sizes,
          color: hasData.colors,
          colorscale: [
            [0, '#0f1117'],
            [0.25, '#1a365d'],
            [0.5, '#2b6cb0'],
            [0.75, '#ed8936'],
            [1, '#f6e05e'],
          ],
          cmin: effectMin,
          cmax: effectMax,
          colorbar: {
            title: { text: metricLabel, font: { size: 10, color: '#8b90a5' } },
            tickfont: { size: 9, color: '#8b90a5' },
            len: 0.6,
            thickness: 12,
            x: 1.02,
          },
          opacity: hasDataOpacities,
          line: {
            width: hasData.ids.map(id => inspectSet.has(id) ? 3 : 1),
            color: hasData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.3)'),
          },
        },
        text: hasData.texts,
        textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
        hovertext: hasData.hoverTexts,
        hoverinfo: 'text',
        customdata: hasData.ids,
      };

      const noDataTrace = {
        x: noData.x, y: noData.y,
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          size: noData.sizes,
          color: '#555',
          opacity: noDataOpacities,
          line: {
            width: noData.ids.map(id => inspectSet.has(id) ? 3 : 1),
            color: noData.ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.15)'),
          },
        },
        text: noData.texts,
        textposition: 'top center',
        textfont: { size: 9, color: 'rgba(200,205,220,0.4)' },
        hovertext: noData.hoverTexts,
        hoverinfo: 'text',
        customdata: noData.ids,
      };

      return [edgeTrace, noDataTrace, hasDataTrace];
    }

    // Layer / Cluster mode (original behavior)
    const nx = [], ny = [], colors = [], sizes = [], texts = [], hoverTexts = [];
    const ids = [];

    for (const [id, pos] of positions) {
      if (!visibleNodes.has(id)) continue;
      const info = nodeById.get(id);
      nx.push(pos.x);
      ny.push(pos.y);
      ids.push(id);

      let size = 8;
      if (info.type === 'feature' && info.abs_corr != null) {
        size = 6 + info.abs_corr * 40;
      } else if (info.type === 'input' || info.type === 'output') {
        size = 14;
      }
      sizes.push(Math.min(size, 24));

      const colorByCluster = graphColorMode === 'cluster' || clusters.length > 0;
      const col = nodeColor(id, info, featureToCluster, colorByCluster);
      colors.push(col);

      texts.push(shortNodeId(id));
      const imp = info.featureIdx != null
        ? indexes.importanceByKey.get(`${info.layer}_${info.featureIdx}`)
        : null;
      hoverTexts.push(
        `<b>${shortNodeId(id)}</b>` +
        (info.type === 'feature' ? `<br>Layer ${info.layer}` : '') +
        (imp ? `<br>|corr|: ${fmt(imp.abs_correlation)}` : '') +
        (featureToCluster.has(id) ? `<br>Cluster: ${featureToCluster.get(id)}` : '')
      );
    }

    const opacities = ids.map(id => {
      if (!highlightId) return 1;
      if (id === highlightId) return 1;
      if (neighborSet && neighborSet.has(id)) return 0.9;
      return 0.15;
    });

    const inspectSet = new Set(inspectedFeatureIds);
    const lineWidths = ids.map(id => inspectSet.has(id) ? 3 : 1);
    const lineColors = ids.map(id => inspectSet.has(id) ? '#fcc419' : 'rgba(255,255,255,0.3)');

    const nodeTrace = {
      x: nx, y: ny,
      mode: 'markers+text',
      type: 'scatter',
      marker: {
        size: sizes,
        color: colors,
        opacity: opacities,
        line: { width: lineWidths, color: lineColors },
      },
      text: texts,
      textposition: 'top center',
      textfont: { size: 9, color: 'rgba(200,205,220,0.7)' },
      hovertext: hoverTexts,
      hoverinfo: 'text',
      customdata: ids,
    };

    return [edgeTrace, nodeTrace];
  }, [positions, visibleNodes, graph.links, nodeById, indexes, featureToCluster, neighbors, hoveredFeatureId, selectedFeatureId, clusters, graphColorMode, graphEffectMetric, nodeEffectValues, inspectedFeatureIds]);

  const handleClick = useCallback((event) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      const id = point.customdata;
      if (id && nodeById.get(id)?.type === 'feature') {
        // Shift+click → add to inspection set
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

  const handleUnhover = useCallback(() => {
    setHoveredFeatureId(null);
  }, [setHoveredFeatureId]);

  const layout = useMemo(() => ({
    width: dims.width,
    height: dims.height,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 10, r: 10, t: 30, b: 10 },
    title: {
      text: `Attribution Graph (${visibleNodes.size} nodes)`,
      font: { size: 12, color: '#8b90a5' },
      x: 0.01, xanchor: 'left',
    },
    xaxis: { visible: false, showgrid: false, zeroline: false },
    yaxis: { visible: false, showgrid: false, zeroline: false },
    showlegend: false,
    hovermode: 'closest',
    dragmode: 'pan',
  }), [dims, visibleNodes.size]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
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
  );
}
