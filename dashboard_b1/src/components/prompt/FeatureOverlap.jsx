import { useMemo, useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import useStore from '../../store/useStore';
import { fmt } from '../../utils/formatters';

const ALL_LAYERS = Array.from({ length: 16 }, (_, i) => i + 10);

const mkZoneShapes = () => [
  { type: 'rect', xref: 'x', yref: 'paper', x0: 9.5,  x1: 16.5, y0: 0, y1: 1, fillcolor: 'rgba(78,154,241,0.06)',  line: { width: 0 } },
  { type: 'rect', xref: 'x', yref: 'paper', x0: 16.5, x1: 22.5, y0: 0, y1: 1, fillcolor: 'rgba(247,185,78,0.04)',  line: { width: 0 } },
  { type: 'rect', xref: 'x', yref: 'paper', x0: 22.5, x1: 25.5, y0: 0, y1: 1, fillcolor: 'rgba(116,198,157,0.06)', line: { width: 0 } },
];

const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 28, r: 8, t: 6, b: 28 },
  xaxis: { color: '#8b90a5', tickfont: { size: 8 }, dtick: 1, zeroline: false, gridcolor: 'rgba(45,49,72,0.3)' },
  yaxis: { color: '#8b90a5', tickfont: { size: 8 }, gridcolor: 'rgba(45,49,72,0.3)' },
  legend: { font: { size: 8, color: '#8b90a5' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.45 },
  showlegend: true,
};

// ── Core helper: returns Map<fid, {c, layer}> for "active" features per basis ─
// activation: any non-NaN contribution (threshold ignored)
// contribution: |c| >= threshold
// signed: |c| >= threshold (sign consistency checked cross-prompt)
function getActiveMap(idx, featuresByPrompt, basis, threshold) {
  if (idx == null) return new Map();
  const m = new Map();
  for (const r of (featuresByPrompt.get(idx) || [])) {
    const c = parseFloat(r.contribution_to_correct);
    if (isNaN(c)) continue;
    const layer = parseInt(r.layer);
    if (basis === 'activation') {
      m.set(r.feature_id, { c, layer });
    } else if (Math.abs(c) >= threshold) {
      m.set(r.feature_id, { c, layer });
    }
  }
  return m;
}

// Helper: compute per-feature support fractions across a group of prompts
function computeSupportFractions(indices, featuresByPrompt, basis, threshold) {
  if (!indices.length) return new Map();
  const n = indices.length;
  const counts = new Map(); // fid → { pos, neg, sumC, layer }
  for (const idx of indices) {
    for (const [fid, { c, layer }] of getActiveMap(idx, featuresByPrompt, basis, threshold)) {
      const cur = counts.get(fid) || { pos: 0, neg: 0, sumC: 0, layer };
      if (c >= 0) cur.pos++; else cur.neg++;
      cur.sumC += c;
      counts.set(fid, cur);
    }
  }
  return new Map([...counts.entries()].map(([id, { pos, neg, sumC, layer }]) => {
    const total = pos + neg;
    const dominant = Math.max(pos, neg);
    const fraction = basis === 'signed' ? dominant / n : total / n;
    return [id, { fraction, mean_contrib: sumC / total, layer, pos, neg,
                  sign_consistency: total > 0 ? dominant / total : 0 }];
  }));
}

// ── Shared UI: overlap basis selector ────────────────────────────────────────
const BASIS_LABELS = { activation: 'Activation', contribution: 'Contribution', signed: 'Signed role' };

function BasisSelector() {
  const overlapBasis = useStore(s => s.overlapBasis);
  const setOverlapBasis = useStore(s => s.setOverlapBasis);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>Basis:</span>
      {['activation', 'contribution', 'signed'].map(b => (
        <button key={b} onClick={() => setOverlapBasis(b)}
          style={{ fontSize: 9, padding: '2px 6px', borderRadius: 4, cursor: 'pointer', border: '1px solid',
            borderColor: overlapBasis === b ? 'var(--accent)' : 'var(--border)',
            background: overlapBasis === b ? 'rgba(78,154,241,0.15)' : 'transparent',
            color: overlapBasis === b ? 'var(--accent)' : 'var(--text-dim)',
            fontWeight: overlapBasis === b ? 600 : 400 }}>
          {BASIS_LABELS[b]}
        </button>
      ))}
    </div>
  );
}

// ── Feature row (clickable) ───────────────────────────────────────────────────
function FeatureRow({ id, a, b, frac, fracA, fracB, scs, color }) {
  const setSelectedFeatureId = useStore(s => s.setSelectedFeatureId);
  return (
    <div onClick={() => setSelectedFeatureId(id)}
      style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 2, cursor: 'pointer',
               padding: '1px 4px', borderRadius: 3, fontSize: 10 }}>
      <span style={{ fontFamily: 'monospace', color, minWidth: 105 }}>{id}</span>
      {a    != null && <span style={{ minWidth: 36, color: a > 0 ? '#4e9af1' : '#d62728' }}>A:{fmt(a, 2)}</span>}
      {b    != null && <span style={{ minWidth: 36, color: b > 0 ? '#f77f4e' : '#d62728' }}>B:{fmt(b, 2)}</span>}
      {frac != null && (
        <span style={{ minWidth: 28, color: frac >= 1 ? '#4a0e8f' : frac >= 0.5 ? '#9b59b6' : '#c39bd3' }}>
          {Math.round(frac * 100)}%
        </span>
      )}
      {fracA != null && (
        <span style={{ color: 'var(--text-dim)', fontSize: 9 }}>
          A:{Math.round(fracA * 100)}% B:{Math.round((fracB ?? 0) * 100)}%
        </span>
      )}
      {scs != null && (
        <span style={{ minWidth: 26, fontSize: 9, title: 'sign consistency',
                       color: scs >= 0.9 ? '#2ca02c' : scs >= 0.7 ? '#f77f4e' : '#d62728' }}>
          ⊕{Math.round(scs * 100)}%
        </span>
      )}
    </div>
  );
}

// ── PAIRWISE PANEL (unchanged logic, extended for basis) ─────────────────────
function PairwisePanel({ indexes, conceptPairs }) {
  const selectedPromptIdx  = useStore(s => s.selectedPromptIdx);
  const comparedPromptIdx  = useStore(s => s.comparedPromptIdx);
  const setSelectedPromptIdx  = useStore(s => s.setSelectedPromptIdx);
  const setComparedPromptIdx  = useStore(s => s.setComparedPromptIdx);
  const overlapColorMode   = useStore(s => s.overlapColorMode);
  const setOverlapColorMode = useStore(s => s.setOverlapColorMode);
  const setOverlapSets     = useStore(s => s.setOverlapSets);
  const overlapBasis       = useStore(s => s.overlapBasis);

  const { featuresByPrompt, promptById } = indexes;
  const [threshold, setThreshold] = useState(0.05);
  const [swapped, setSwapped] = useState(false);
  const idxA = swapped ? comparedPromptIdx : selectedPromptIdx;
  const idxB = swapped ? selectedPromptIdx : comparedPromptIdx;
  const traceA = idxA != null ? promptById.get(idxA) : null;
  const traceB = idxB != null ? promptById.get(idxB) : null;

  const overlap = useMemo(() => {
    const mA = getActiveMap(idxA, featuresByPrompt, overlapBasis, threshold);
    const mB = getActiveMap(idxB, featuresByPrompt, overlapBasis, threshold);

    // Base intersection: present in both
    const bothActive = new Set([...mA.keys()].filter(id => mB.has(id)));
    // Signed: additionally require same sign
    const shared = overlapBasis === 'signed'
      ? new Set([...bothActive].filter(id => Math.sign(mA.get(id).c) === Math.sign(mB.get(id).c)))
      : bothActive;
    const conflicted = overlapBasis === 'signed'
      ? [...bothActive].filter(id => !shared.has(id)).length
      : 0;
    // Not-shared = active in A but not in "signed-shared" (may include sign-conflicted)
    const aOnly = new Set([...mA.keys()].filter(id => !shared.has(id)));
    const bOnly = new Set([...mB.keys()].filter(id => !shared.has(id)));

    const union = new Set([...mA.keys(), ...mB.keys()]);
    const jaccard = union.size > 0 ? shared.size / union.size : 0;
    let sA = 0, sB = 0, sm = 0;
    for (const [, v] of mA) sA += Math.abs(v.c);
    for (const [, v] of mB) sB += Math.abs(v.c);
    for (const id of shared) sm += Math.min(Math.abs(mA.get(id).c), Math.abs(mB.get(id).c));
    const wtd = (sA + sB) > 0 ? (2 * sm) / (sA + sB) : 0;

    const sharedList = [...shared]
      .map(id => ({ id, a: mA.get(id).c, b: mB.get(id).c, layer: mA.get(id).layer }))
      .sort((x, y) => (Math.abs(y.a) + Math.abs(y.b)) - (Math.abs(x.a) + Math.abs(x.b)));
    const aList = [...aOnly].map(id => ({ id, a: mA.get(id).c, layer: mA.get(id).layer }))
      .sort((x, y) => Math.abs(y.a) - Math.abs(x.a));
    const bList = [...bOnly].map(id => ({ id, b: mB.get(id).c, layer: mB.get(id).layer }))
      .sort((x, y) => Math.abs(y.b) - Math.abs(x.b));

    const lwS = ALL_LAYERS.map(L => sharedList.filter(f => f.layer === L).length);
    const lwA = ALL_LAYERS.map(L => aList.filter(f => f.layer === L).length);
    const lwB = ALL_LAYERS.map(L => bList.filter(f => f.layer === L).length);
    return { shared, aOnly, bOnly, jaccard, wtd, conflicted, sharedList, aList, bList, lwS, lwA, lwB };
  }, [idxA, idxB, featuresByPrompt, threshold, overlapBasis]);

  useEffect(() => {
    if (overlapColorMode)
      setOverlapSets({ shared: overlap.shared, aOnly: overlap.aOnly, bOnly: overlap.bOnly });
    else
      setOverlapSets(null);
  }, [overlapColorMode, overlap.shared, overlap.aOnly, overlap.bOnly]);
  useEffect(() => () => setOverlapSets(null), []);

  return (
    <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Quick pairs */}
      <div>
        <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>
          Quick pairs (EN vs FR, same concept)
        </div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {conceptPairs.map(p => (
            <button key={p.concept}
              onClick={() => { setSelectedPromptIdx(p.en); setComparedPromptIdx(p.fr); setSwapped(false); }}
              style={{ fontSize: 9, padding: '2px 6px', borderRadius: 6, cursor: 'pointer',
                border: '1px solid var(--border)',
                background: ((idxA===p.en && idxB===p.fr)||(idxB===p.en && idxA===p.fr)) ? 'var(--bg-card)' : 'transparent',
                color: 'var(--text-dim)' }}>
              c{p.concept} EN#{p.en} vs FR#{p.fr}
            </button>
          ))}
        </div>
      </div>

      {/* A/B labels */}
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <div style={{ flex: 1, background: 'var(--bg-card)', borderRadius: 6, padding: '5px 8px', border: '1px solid #4e9af144', fontSize: 10 }}>
          <span style={{ color: '#4e9af1', fontWeight: 600 }}>A #{idxA ?? '—'}</span>
          {traceA && <span style={{ color: 'var(--text-dim)', marginLeft: 4 }}>{traceA.prompt?.slice(0, 38)}…</span>}
        </div>
        <button onClick={() => setSwapped(s => !s)}
          style={{ padding: '3px 6px', fontSize: 11, cursor: 'pointer', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)', borderRadius: 5 }}>
          ⇄
        </button>
        <div style={{ flex: 1, background: 'var(--bg-card)', borderRadius: 6, padding: '5px 8px', border: '1px solid #f77f4e44', fontSize: 10 }}>
          <span style={{ color: '#f77f4e', fontWeight: 600 }}>B #{idxB ?? '—'}</span>
          {traceB && <span style={{ color: 'var(--text-dim)', marginLeft: 4 }}>{traceB.prompt?.slice(0, 38)}…</span>}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {overlapBasis !== 'activation' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10 }}>
            <span style={{ color: 'var(--text-dim)' }}>Min |contrib|</span>
            <input type="range" min="0" max="0.5" step="0.01" value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))} style={{ width: 70 }} />
            <span style={{ fontFamily: 'monospace', minWidth: 28 }}>{threshold.toFixed(2)}</span>
          </div>
        )}
        <button onClick={() => setOverlapColorMode(!overlapColorMode)}
          style={{ fontSize: 10, padding: '2px 7px', borderRadius: 5, cursor: 'pointer', border: '1px solid',
            borderColor: overlapColorMode ? '#9b59b6' : 'var(--border)',
            background: overlapColorMode ? 'rgba(155,89,182,0.15)' : 'transparent',
            color: overlapColorMode ? '#9b59b6' : 'var(--text-dim)' }}>
          {overlapColorMode ? '◉ Graph overlay' : '○ Graph overlay'}
        </button>
      </div>

      {idxA == null && idxB == null ? (
        <div style={{ padding: 24, color: 'var(--text-dim)', textAlign: 'center' }}>Select prompts A and B to compare.</div>
      ) : (
        <>
          {/* Metrics */}
          <div style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 8, border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {[
                { label: 'Shared', val: overlap.sharedList.length, color: '#9b59b6' },
                { label: 'A-only', val: overlap.aList.length, color: '#4e9af1' },
                { label: 'B-only', val: overlap.bList.length, color: '#f77f4e' },
                { label: 'Jaccard', val: fmt(overlap.jaccard, 3) },
                { label: 'Wtd overlap', val: fmt(overlap.wtd, 3) },
                ...(overlap.conflicted > 0 ? [{ label: 'Sign-conflict', val: overlap.conflicted, color: '#ff7f0e' }] : []),
              ].map(({ label, val, color }) => (
                <div key={label} style={{ fontSize: 10 }}>
                  <div style={{ color: 'var(--text-dim)' }}>{label}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: color || 'var(--text)' }}>{val}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Layerwise */}
          <div style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 8, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 2 }}>Layerwise overlap</div>
            <Plot
              data={[
                { x: ALL_LAYERS, y: overlap.lwS, type: 'bar', name: 'Shared', marker: { color: '#9b59b6' } },
                { x: ALL_LAYERS, y: overlap.lwA, type: 'bar', name: 'A-only', marker: { color: '#4e9af1' } },
                { x: ALL_LAYERS, y: overlap.lwB, type: 'bar', name: 'B-only', marker: { color: '#f77f4e' } },
              ]}
              layout={{ ...BASE_LAYOUT, barmode: 'stack', height: 140, shapes: mkZoneShapes() }}
              config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }}
            />
          </div>

          {/* Feature lists */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
            {[
              { title: `Shared (${overlap.sharedList.length})`, list: overlap.sharedList, color: '#9b59b6', bc: '#9b59b644',
                render: f => <FeatureRow key={f.id} id={f.id} a={f.a} b={f.b} color='#9b59b6' /> },
              { title: `A-only (${overlap.aList.length})`, list: overlap.aList, color: '#4e9af1', bc: '#4e9af144',
                render: f => <FeatureRow key={f.id} id={f.id} a={f.a} color='#4e9af1' /> },
              { title: `B-only (${overlap.bList.length})`, list: overlap.bList, color: '#f77f4e', bc: '#f77f4e44',
                render: f => <FeatureRow key={f.id} id={f.id} b={f.b} color='#f77f4e' /> },
            ].map(({ title, list, color, bc, render }) => (
              <div key={title} style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 7, border: `1px solid ${bc}` }}>
                <div style={{ fontSize: 10, fontWeight: 600, color, marginBottom: 3 }}>{title}</div>
                {list.slice(0, 12).map(render)}
                {list.length > 12 && <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 2 }}>+{list.length - 12} more</div>}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── MULTI-PROMPT PANEL ────────────────────────────────────────────────────────
function MultiPanel({ indexes }) {
  const multiSelectedPrompts   = useStore(s => s.multiSelectedPrompts);
  const setMultiSelectedPrompts = useStore(s => s.setMultiSelectedPrompts);
  const multiFreqThreshold     = useStore(s => s.multiFreqThreshold);
  const setMultiFreqThreshold  = useStore(s => s.setMultiFreqThreshold);
  const multiContribThreshold  = useStore(s => s.multiContribThreshold);
  const setMultiContribThreshold = useStore(s => s.setMultiContribThreshold);
  const setMultiSupportFractions = useStore(s => s.setMultiSupportFractions);
  const overlapBasis           = useStore(s => s.overlapBasis);
  const setVsSetMode           = useStore(s => s.setVsSetMode);
  const setSetVsSetMode        = useStore(s => s.setSetVsSetMode);
  const groupBPrompts          = useStore(s => s.groupBPrompts);
  const setGroupBPrompts       = useStore(s => s.setGroupBPrompts);
  const setSetVsSetFractions   = useStore(s => s.setSetVsSetFractions);
  const setSelectedFeatureId   = useStore(s => s.setSelectedFeatureId);

  const { featuresByPrompt, promptTraces } = indexes;

  // Sidebar filters
  const [filterLang, setFilterLang]       = useState(null);
  const [filterConcept, setFilterConcept] = useState(null);
  const [filterCorrect, setFilterCorrect] = useState(null);
  const [filterTemplate, setFilterTemplate] = useState(null);

  const concepts  = useMemo(() => promptTraces ? [...new Set(promptTraces.map(t => t.concept_index))].sort((a,b)=>a-b) : [], [promptTraces]);
  const templates = useMemo(() => promptTraces ? [...new Set(promptTraces.map(t => t.template_idx))].sort((a,b)=>a-b) : [], [promptTraces]);

  const filteredPrompts = useMemo(() => {
    if (!promptTraces) return [];
    return promptTraces.filter(t =>
      (!filterLang    || t.language        === filterLang) &&
      (filterConcept  == null || t.concept_index  === filterConcept) &&
      (filterCorrect  == null || t.prediction_correct === filterCorrect) &&
      (filterTemplate == null || t.template_idx   === filterTemplate)
    );
  }, [promptTraces, filterLang, filterConcept, filterCorrect, filterTemplate]);

  const filteredIds = useMemo(() => new Set(filteredPrompts.map(t => t.prompt_idx)), [filteredPrompts]);
  const selectedSet = useMemo(() => new Set(multiSelectedPrompts), [multiSelectedPrompts]);
  const groupBSet   = useMemo(() => new Set(groupBPrompts), [groupBPrompts]);

  const togglePrompt = (idx, group) => {
    if (group === 'B') {
      groupBSet.has(idx)
        ? setGroupBPrompts(groupBPrompts.filter(i => i !== idx))
        : setGroupBPrompts([...groupBPrompts, idx]);
    } else {
      selectedSet.has(idx)
        ? setMultiSelectedPrompts(multiSelectedPrompts.filter(i => i !== idx))
        : setMultiSelectedPrompts([...multiSelectedPrompts, idx]);
    }
  };

  const selectFiltered  = () => setMultiSelectedPrompts([...new Set([...multiSelectedPrompts, ...filteredIds])]);
  const deselectFiltered = () => setMultiSelectedPrompts(multiSelectedPrompts.filter(i => !filteredIds.has(i)));
  const selectAll  = () => setMultiSelectedPrompts(promptTraces ? promptTraces.map(t => t.prompt_idx) : []);
  const clearAll   = () => setMultiSelectedPrompts([]);
  const clearBAll  = () => setGroupBPrompts([]);
  const selectEN   = () => setMultiSelectedPrompts([...new Set([...multiSelectedPrompts, ...filteredPrompts.filter(t => t.language==='en').map(t => t.prompt_idx)])]);
  const selectFR   = () => setMultiSelectedPrompts([...new Set([...multiSelectedPrompts, ...filteredPrompts.filter(t => t.language==='fr').map(t => t.prompt_idx)])]);
  const selectFailFR = () => setMultiSelectedPrompts([...new Set([...multiSelectedPrompts, ...(promptTraces||[]).filter(t => t.language==='fr' && !t.prediction_correct).map(t => t.prompt_idx)])]);

  // ── Multi-prompt overlap (basis-aware, sign-tracking) ──────────────────────
  const multiOverlap = useMemo(() => {
    if (!multiSelectedPrompts.length) return null;
    const n = multiSelectedPrompts.length;
    const counts = new Map(); // fid → { pos, neg, sumC, layer }

    for (const idx of multiSelectedPrompts) {
      for (const [fid, { c, layer }] of getActiveMap(idx, featuresByPrompt, overlapBasis, multiContribThreshold)) {
        const cur = counts.get(fid) || { pos: 0, neg: 0, sumC: 0, layer };
        if (c >= 0) cur.pos++; else cur.neg++;
        cur.sumC += c;
        counts.set(fid, cur);
      }
    }

    const features = [...counts.entries()].map(([id, { pos, neg, sumC, layer }]) => {
      const total    = pos + neg;
      const dominant = Math.max(pos, neg);
      const sign_consistency   = total > 0 ? dominant / total : 0;
      const support_count      = overlapBasis === 'signed' ? dominant : total;
      const support_fraction   = support_count / n;
      return { id, layer, support_count, support_fraction, mean_contrib: sumC / total,
               sign_consistency, pos_count: pos, neg_count: neg };
    }).sort((a, b) => b.support_fraction - a.support_fraction || Math.abs(b.mean_contrib) - Math.abs(a.mean_contrib));

    const intersection = features.filter(f => f.support_fraction >= 1.0);
    const majority     = features.filter(f => f.support_fraction >= multiFreqThreshold && f.support_fraction < 1.0);
    const minority     = features.filter(f => f.support_fraction < multiFreqThreshold);
    const jaccard = features.length > 0 ? intersection.length / features.length : 0;
    const supportFractions = new Map(features.map(f => [f.id, f.support_fraction]));

    const lwI = ALL_LAYERS.map(L => intersection.filter(f => f.layer === L).length);
    const lwM = ALL_LAYERS.map(L => majority.filter(f => f.layer === L).length);
    const lwm = ALL_LAYERS.map(L => minority.filter(f => f.layer === L).length);

    return { features, intersection, majority, minority, jaccard, supportFractions, lwI, lwM, lwm };
  }, [multiSelectedPrompts, featuresByPrompt, multiContribThreshold, multiFreqThreshold, overlapBasis]);

  // ── Set-vs-set computation ─────────────────────────────────────────────────
  const setVsSet = useMemo(() => {
    if (!setVsSetMode || !multiSelectedPrompts.length || !groupBPrompts.length) return null;
    const th = multiFreqThreshold;

    const fracA = computeSupportFractions(multiSelectedPrompts, featuresByPrompt, overlapBasis, multiContribThreshold);
    const fracB = computeSupportFractions(groupBPrompts, featuresByPrompt, overlapBasis, multiContribThreshold);

    const allIds = new Set([...fracA.keys(), ...fracB.keys()]);
    const features = [...allIds].map(id => {
      const a = fracA.get(id), b = fracB.get(id);
      return {
        id,
        layer: a?.layer ?? b?.layer,
        fracA: a?.fraction ?? 0,
        fracB: b?.fraction ?? 0,
        diff:  (a?.fraction ?? 0) - (b?.fraction ?? 0),
      };
    }).sort((x, y) => Math.abs(y.diff) - Math.abs(x.diff));

    const shared = features.filter(f => f.fracA >= th && f.fracB >= th);
    const aSpec  = features.filter(f => f.fracA >= th && f.fracB < th);
    const bSpec  = features.filter(f => f.fracB >= th && f.fracA < th);
    const svsFractions = new Map(features.map(f => [f.id, { fracA: f.fracA, fracB: f.fracB }]));

    // Layerwise: active feature count per layer for each group
    const lwA = ALL_LAYERS.map(L => [...fracA.entries()].filter(([,v]) => v.layer===L && v.fraction>=th).length);
    const lwB = ALL_LAYERS.map(L => [...fracB.entries()].filter(([,v]) => v.layer===L && v.fraction>=th).length);

    return { features, shared, aSpec, bSpec, svsFractions, lwA, lwB };
  }, [setVsSetMode, multiSelectedPrompts, groupBPrompts, featuresByPrompt, overlapBasis, multiContribThreshold, multiFreqThreshold]);

  // Push graph state to store
  useEffect(() => {
    if (setVsSetMode && setVsSet) {
      setSetVsSetFractions(setVsSet.svsFractions);
      setMultiSupportFractions(null);
    } else if (multiOverlap) {
      setMultiSupportFractions(multiOverlap.supportFractions);
      setSetVsSetFractions(null);
    } else {
      setMultiSupportFractions(null);
      setSetVsSetFractions(null);
    }
  }, [setVsSetMode, setVsSet, multiOverlap]);

  useEffect(() => () => { setMultiSupportFractions(null); setSetVsSetFractions(null); }, []);

  const pct = Math.round(multiFreqThreshold * 100);
  const showSCS = overlapBasis === 'signed';

  return (
    <div style={{ flex: 1, overflow: 'hidden', display: 'flex', gap: 8 }}>

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <div style={{ width: 215, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 4,
                    borderRight: '1px solid var(--border)', paddingRight: 8, overflow: 'hidden' }}>

        <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-dim)' }}>Filters</div>

        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {['en', 'fr'].map(l => (
            <button key={l} onClick={() => setFilterLang(filterLang===l ? null : l)}
              style={{ fontSize: 9, padding: '2px 5px', borderRadius: 4, cursor: 'pointer', border: '1px solid',
                borderColor: filterLang===l ? 'var(--accent)' : 'var(--border)',
                background: filterLang===l ? 'rgba(78,154,241,0.15)' : 'transparent',
                color: filterLang===l ? 'var(--accent)' : 'var(--text-dim)', textTransform: 'uppercase' }}>
              {l}
            </button>
          ))}
          {[{label:'✓',val:true},{label:'✗',val:false}].map(({label,val}) => (
            <button key={label} onClick={() => setFilterCorrect(filterCorrect===val ? null : val)}
              style={{ fontSize: 9, padding: '2px 5px', borderRadius: 4, cursor: 'pointer', border: '1px solid',
                borderColor: filterCorrect===val ? 'var(--accent)' : 'var(--border)',
                background: filterCorrect===val ? 'rgba(78,154,241,0.15)' : 'transparent',
                color: filterCorrect===val ? (val ? '#2ca02c' : '#d62728') : 'var(--text-dim)' }}>
              {label}
            </button>
          ))}
          {(filterLang||filterConcept!=null||filterCorrect!=null||filterTemplate!=null) && (
            <button onClick={()=>{setFilterLang(null);setFilterConcept(null);setFilterCorrect(null);setFilterTemplate(null);}}
              style={{fontSize:9,padding:'2px 5px',borderRadius:4,cursor:'pointer',border:'1px solid var(--border)',background:'transparent',color:'#d62728'}}>✕</button>
          )}
        </div>

        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {concepts.map(c => (
            <button key={c} onClick={() => setFilterConcept(filterConcept===c ? null : c)}
              style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, cursor: 'pointer', border: '1px solid',
                borderColor: filterConcept===c ? '#f77f4e' : 'var(--border)',
                background: filterConcept===c ? 'rgba(247,127,78,0.15)' : 'transparent',
                color: filterConcept===c ? '#f77f4e' : 'var(--text-dim)' }}>c{c}</button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {templates.map(t => (
            <button key={t} onClick={() => setFilterTemplate(filterTemplate===t ? null : t)}
              style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, cursor: 'pointer', border: '1px solid',
                borderColor: filterTemplate===t ? '#74c69d' : 'var(--border)',
                background: filterTemplate===t ? 'rgba(116,198,157,0.15)' : 'transparent',
                color: filterTemplate===t ? '#74c69d' : 'var(--text-dim)' }}>T{t}</button>
          ))}
        </div>

        {/* Quick-select actions */}
        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {[
            { label: `+${filteredPrompts.length}`, action: selectFiltered, title: 'Add visible → A' },
            { label: '−filt', action: deselectFiltered, title: 'Remove visible from A' },
            { label: 'all',   action: selectAll, title: 'All 96 → A' },
            { label: '+EN',   action: selectEN,  title: 'Add visible EN → A' },
            { label: '+FR',   action: selectFR,  title: 'Add visible FR → A' },
            { label: 'failFR', action: selectFailFR, title: 'Failing FR → A', color: '#d62728' },
          ].map(({ label, action, title, color }) => (
            <button key={label} onClick={action} title={title}
              style={{ fontSize: 9, padding: '2px 4px', borderRadius: 4, cursor: 'pointer',
                border: '1px solid var(--border)', background: 'transparent',
                color: color || 'var(--text-dim)' }}>
              {label}
            </button>
          ))}
          <button onClick={clearAll} title="Clear A"
            style={{ fontSize: 9, padding: '2px 4px', borderRadius: 4, cursor: 'pointer',
              border: '1px solid var(--border)', background: 'transparent', color: '#d62728' }}>
            clrA
          </button>
          {setVsSetMode && (
            <button onClick={clearBAll} title="Clear B"
              style={{ fontSize: 9, padding: '2px 4px', borderRadius: 4, cursor: 'pointer',
                border: '1px solid var(--border)', background: 'transparent', color: '#74c69d' }}>
              clrB
            </button>
          )}
        </div>

        <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>
          {filteredPrompts.length} shown ·{' '}
          <span style={{ color: '#9b59b6' }}>A:{multiSelectedPrompts.length}</span>
          {setVsSetMode && <span style={{ color: '#74c69d' }}> B:{groupBPrompts.length}</span>}
        </div>

        {/* Prompt list — [A checkbox] [B checkbox when svs mode] #idx LANG cN TN ✓/✗ */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {filteredPrompts.map(t => {
            const inA = selectedSet.has(t.prompt_idx);
            const inB = groupBSet.has(t.prompt_idx);
            return (
              <div key={t.prompt_idx}
                style={{ display: 'flex', alignItems: 'center', gap: 2, padding: '2px 2px',
                         borderRadius: 3, fontSize: 9,
                         background: (inA||inB) ? 'rgba(155,89,182,0.07)' : 'transparent' }}>
                <input type="checkbox" checked={inA} onChange={() => togglePrompt(t.prompt_idx, 'A')}
                  style={{ flexShrink: 0, cursor: 'pointer', margin: 0, accentColor: '#9b59b6' }} title="Group A" />
                {setVsSetMode && (
                  <input type="checkbox" checked={inB} onChange={() => togglePrompt(t.prompt_idx, 'B')}
                    style={{ flexShrink: 0, cursor: 'pointer', margin: 0, accentColor: '#74c69d' }} title="Group B" />
                )}
                <span style={{ fontFamily: 'monospace', color: 'var(--text-dim)', minWidth: 20 }}>#{t.prompt_idx}</span>
                <span style={{ color: t.language==='en' ? '#4e9af1' : '#f77f4e', minWidth: 14 }}>{t.language.toUpperCase()}</span>
                <span style={{ color: 'var(--text-dim)', minWidth: 14 }}>c{t.concept_index}</span>
                <span style={{ color: 'var(--text-dim)', minWidth: 14 }}>T{t.template_idx}</span>
                <span style={{ color: t.prediction_correct ? '#2ca02c' : '#d62728' }}>
                  {t.prediction_correct ? '✓' : '✗'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Main panel ──────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>

        {/* Controls row */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', flexShrink: 0 }}>
          {overlapBasis !== 'activation' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10 }}>
              <span style={{ color: 'var(--text-dim)' }}>Min |contrib|</span>
              <input type="range" min="0" max="0.5" step="0.01" value={multiContribThreshold}
                onChange={e => setMultiContribThreshold(parseFloat(e.target.value))} style={{ width: 60 }} />
              <span style={{ fontFamily: 'monospace', minWidth: 28 }}>{multiContribThreshold.toFixed(2)}</span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10 }}>
            <span style={{ color: 'var(--text-dim)' }}>Majority ≥</span>
            <input type="range" min="0.1" max="0.99" step="0.05" value={multiFreqThreshold}
              onChange={e => setMultiFreqThreshold(parseFloat(e.target.value))} style={{ width: 60 }} />
            <span style={{ fontFamily: 'monospace', minWidth: 28 }}>{pct}%</span>
          </div>
          <button onClick={() => setSetVsSetMode(!setVsSetMode)}
            style={{ fontSize: 9, padding: '2px 7px', borderRadius: 5, cursor: 'pointer', border: '1px solid',
              borderColor: setVsSetMode ? '#74c69d' : 'var(--border)',
              background: setVsSetMode ? 'rgba(116,198,157,0.15)' : 'transparent',
              color: setVsSetMode ? '#74c69d' : 'var(--text-dim)' }}>
            {setVsSetMode ? '◉ A vs B mode' : '○ Compare sets'}
          </button>
        </div>

        {!multiSelectedPrompts.length ? (
          <div style={{ padding: 24, color: 'var(--text-dim)', textAlign: 'center' }}>
            Use the sidebar to select prompts and analyze their feature overlap.
          </div>
        ) : !multiOverlap ? null : (
          <>
            {/* Summary metrics */}
            <div style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 8, border: '1px solid var(--border)', flexShrink: 0 }}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                {[
                  { label: 'Prompts (A)',       val: multiSelectedPrompts.length },
                  { label: 'Core 100%',         val: multiOverlap.intersection.length, color: '#4a0e8f' },
                  { label: `Majority ≥${pct}%`, val: multiOverlap.majority.length, color: '#9b59b6' },
                  { label: 'Minority',          val: multiOverlap.minority.length, color: '#c39bd3' },
                  { label: 'Union',             val: multiOverlap.features.length },
                  { label: 'Jaccard (∩/∪)',     val: fmt(multiOverlap.jaccard, 3) },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ fontSize: 10 }}>
                    <div style={{ color: 'var(--text-dim)' }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: color || 'var(--text)' }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Layerwise overlap chart */}
            <div style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 8, border: '1px solid var(--border)', flexShrink: 0 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 2 }}>
                Layerwise overlap · {multiSelectedPrompts.length} prompts ({BASIS_LABELS[overlapBasis]})
              </div>
              <Plot
                data={[
                  { x: ALL_LAYERS, y: multiOverlap.lwI, type: 'bar', name: '100%', marker: { color: '#4a0e8f' } },
                  { x: ALL_LAYERS, y: multiOverlap.lwM, type: 'bar', name: `≥${pct}%`, marker: { color: '#9b59b6' } },
                  { x: ALL_LAYERS, y: multiOverlap.lwm, type: 'bar', name: 'minority', marker: { color: '#c39bd355' } },
                ]}
                layout={{ ...BASE_LAYOUT, barmode: 'stack', height: 150, shapes: mkZoneShapes() }}
                config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }}
              />
            </div>

            {/* Feature breakdown */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, flexShrink: 0 }}>
              {[
                { title: `Core 100% (${multiOverlap.intersection.length})`, list: multiOverlap.intersection, color: '#4a0e8f', bc: '#4a0e8f44' },
                { title: `Majority ≥${pct}% (${multiOverlap.majority.length})`, list: multiOverlap.majority, color: '#9b59b6', bc: '#9b59b644' },
                { title: `Minority (${multiOverlap.minority.length})`, list: multiOverlap.minority, color: '#c39bd3', bc: '#c39bd344' },
              ].map(({ title, list, color, bc }) => (
                <div key={title} style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 7, border: `1px solid ${bc}` }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color, marginBottom: 3 }}>{title}</div>
                  {list.slice(0, 12).map(f => (
                    <FeatureRow key={f.id} id={f.id} frac={f.support_fraction}
                      scs={showSCS ? f.sign_consistency : null} color={color} />
                  ))}
                  {list.length > 12 && <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 2 }}>+{list.length - 12} more</div>}
                </div>
              ))}
            </div>

            {/* ── Set-vs-set results ─────────────────────────────────────── */}
            {setVsSetMode && (
              <div style={{ flexShrink: 0 }}>
                {!groupBPrompts.length ? (
                  <div style={{ padding: 10, color: '#74c69d', fontSize: 10, textAlign: 'center',
                                border: '1px dashed #74c69d55', borderRadius: 7 }}>
                    Check the ☑ Group B boxes in the sidebar to build Group B.
                  </div>
                ) : !setVsSet ? null : (
                  <div style={{ background: 'var(--bg-card)', borderRadius: 7, padding: 8, border: '1px solid #74c69d44' }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: '#74c69d', marginBottom: 6 }}>
                      Set A ({multiSelectedPrompts.length}) vs Set B ({groupBPrompts.length}) · threshold ≥{pct}%
                    </div>

                    {/* Metrics */}
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
                      {[
                        { label: 'Shared (A∩B)', val: setVsSet.shared.length, color: '#9b59b6' },
                        { label: 'A-specific',   val: setVsSet.aSpec.length,  color: '#4e9af1' },
                        { label: 'B-specific',   val: setVsSet.bSpec.length,  color: '#74c69d' },
                      ].map(({ label, val, color }) => (
                        <div key={label} style={{ fontSize: 10 }}>
                          <div style={{ color: 'var(--text-dim)' }}>{label}</div>
                          <div style={{ fontSize: 13, fontWeight: 700, color }}>{val}</div>
                        </div>
                      ))}
                    </div>

                    {/* Active features per layer (grouped bar) */}
                    <Plot
                      data={[
                        { x: ALL_LAYERS, y: setVsSet.lwA, type: 'bar', name: `Group A (${multiSelectedPrompts.length})`, marker: { color: '#4e9af1' } },
                        { x: ALL_LAYERS, y: setVsSet.lwB, type: 'bar', name: `Group B (${groupBPrompts.length})`, marker: { color: '#74c69d' } },
                      ]}
                      layout={{ ...BASE_LAYOUT, barmode: 'group', height: 130, shapes: mkZoneShapes() }}
                      config={{ displayModeBar: false, responsive: true }} style={{ width: '100%' }}
                    />

                    {/* Feature tables */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginTop: 8 }}>
                      {[
                        { title: `Shared (${setVsSet.shared.length})`, list: setVsSet.shared, color: '#9b59b6', bc: '#9b59b644' },
                        { title: `A-specific (${setVsSet.aSpec.length})`, list: setVsSet.aSpec, color: '#4e9af1', bc: '#4e9af144' },
                        { title: `B-specific (${setVsSet.bSpec.length})`, list: setVsSet.bSpec, color: '#74c69d', bc: '#74c69d44' },
                      ].map(({ title, list, color, bc }) => (
                        <div key={title} style={{ background: 'var(--bg-panel)', borderRadius: 6, padding: 6, border: `1px solid ${bc}` }}>
                          <div style={{ fontSize: 10, fontWeight: 600, color, marginBottom: 3 }}>{title}</div>
                          {list.slice(0, 8).map(f => (
                            <FeatureRow key={f.id} id={f.id} fracA={f.fracA} fracB={f.fracB} color={color} />
                          ))}
                          {list.length > 8 && <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>+{list.length - 8} more</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── ROOT ─────────────────────────────────────────────────────────────────────
export default function FeatureOverlap({ data, indexes }) {
  const multiOverlapMode      = useStore(s => s.multiOverlapMode);
  const setMultiOverlapMode   = useStore(s => s.setMultiOverlapMode);
  const multiSelectedPrompts  = useStore(s => s.multiSelectedPrompts);
  const multiSupportFractions = useStore(s => s.multiSupportFractions);
  const setVsSetMode          = useStore(s => s.setVsSetMode);
  const { promptTraces } = indexes;

  const conceptPairs = useMemo(() => {
    if (!promptTraces) return [];
    return [...new Set(promptTraces.map(t => t.concept_index))].flatMap(c => {
      const en = promptTraces.find(t => t.concept_index===c && t.language==='en' && t.template_idx===0);
      const fr = promptTraces.find(t => t.concept_index===c && t.language==='fr' && t.template_idx===0);
      return (en && fr) ? [{ concept: c, en: en.prompt_idx, fr: fr.prompt_idx }] : [];
    });
  }, [promptTraces]);

  return (
    <div style={{ height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 6 }}>
      {/* Mode toggle + basis selector */}
      <div style={{ display: 'flex', gap: 6, flexShrink: 0, alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={() => setMultiOverlapMode(false)}
          style={{ fontSize: 10, padding: '4px 10px', borderRadius: 6, cursor: 'pointer', border: '1px solid',
            borderColor: !multiOverlapMode ? 'var(--accent)' : 'var(--border)',
            background: !multiOverlapMode ? 'rgba(78,154,241,0.15)' : 'transparent',
            color: !multiOverlapMode ? 'var(--accent)' : 'var(--text-dim)',
            fontWeight: !multiOverlapMode ? 600 : 400 }}>
          Pairwise A/B
        </button>
        <button onClick={() => setMultiOverlapMode(true)}
          style={{ fontSize: 10, padding: '4px 10px', borderRadius: 6, cursor: 'pointer', border: '1px solid',
            borderColor: multiOverlapMode ? '#9b59b6' : 'var(--border)',
            background: multiOverlapMode ? 'rgba(155,89,182,0.15)' : 'transparent',
            color: multiOverlapMode ? '#9b59b6' : 'var(--text-dim)',
            fontWeight: multiOverlapMode ? 600 : 400 }}>
          Multi-prompt{multiOverlapMode && multiSelectedPrompts.length > 0 ? ` (${multiSelectedPrompts.length})` : ''}
          {multiOverlapMode && setVsSetMode ? ' · A vs B' : ''}
        </button>
        <span style={{ width: 1, background: 'var(--border)', alignSelf: 'stretch', margin: '1px 0' }} />
        <BasisSelector />
        {multiOverlapMode && multiSupportFractions && (
          <span style={{ fontSize: 9, color: '#9b59b6', marginLeft: 2 }}>graph: support %</span>
        )}
      </div>

      <div style={{ flex: 1, overflow: 'hidden' }}>
        {multiOverlapMode
          ? <MultiPanel indexes={indexes} />
          : <PairwisePanel indexes={indexes} conceptPairs={conceptPairs} />
        }
      </div>
    </div>
  );
}
