/**
 * AbstractionTab — Intensive vs Extensive mechanistic analysis
 *
 * Shows:
 *  LEFT  — summary + verdict + family baseline table
 *  MAIN  — tabbed:
 *    (1) Transfer Probe  — physics→cross-domain by layer + domain breakdown
 *    (2) Layer Transition — ARI: abstraction class vs wording vs property (loaded after pipeline)
 *    (3) Cluster Map     — cluster selectivity heatmap (loaded after pipeline)
 *    (4) Findings        — interpretation text + readout failure examples
 */
import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

// ── Design tokens ──────────────────────────────────────────────────────────────
const C = {
  intensive: '#60a5fa', extensive: '#f97316',
  probe: '#7c3aed', transfer: '#16a34a', chance: '#6b7280',
  accent: '#a78bfa',
};

const LAYOUT_BASE = {
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: { family: 'Inter, sans-serif', size: 10, color: '#e1e4ed' },
  margin: { t: 10, b: 36, l: 50, r: 12 },
  legend: { bgcolor: 'transparent', font: { size: 9 } },
};
const CFG = { displayModeBar: false, responsive: true };

const pct  = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
const fix3 = v => v == null ? '—' : typeof v === 'number' ? v.toFixed(3) : v;

const Stat = ({ label, value, color, sub }) => (
  <div style={{ background: 'var(--bg-panel)', borderRadius: 6, padding: '7px 10px',
    textAlign: 'center', flex: 1, minWidth: 64 }}>
    <div style={{ fontSize: 8.5, color: 'var(--text-dim)', textTransform: 'uppercase',
      letterSpacing: '0.05em', marginBottom: 2 }}>{label}</div>
    <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace',
      color: color || 'var(--text)' }}>{value}</div>
    {sub && <div style={{ fontSize: 7.5, color: 'var(--text-dim)', marginTop: 2 }}>{sub}</div>}
  </div>
);

const Chip = ({ label, color }) => (
  <span style={{ display: 'inline-block', padding: '1px 7px', borderRadius: 10,
    fontSize: 9, fontWeight: 700, background: color + '33', color }}>
    {label}
  </span>
);

// ── Panel 1: Transfer probe by layer ─────────────────────────────────────────
function TransferProbePanel({ data }) {
  const tp   = data.probe_transfer;
  const axs  = { gridcolor: '#2d3148', zeroline: false, color: '#8b90a5' };

  const traceCV = {
    type: 'scatter', mode: 'lines+markers', name: 'In-domain CV (physics)',
    x: tp.layers, y: tp.cv_acc,
    line: { color: C.probe, width: 2.5 }, marker: { size: 4 },
    hovertemplate: 'L%{x} cv=%{y:.3f}<extra></extra>',
  };
  const traceTR = {
    type: 'scatter', mode: 'lines+markers', name: 'Transfer (cross-domain, corrected)',
    x: tp.layers, y: tp.transfer,
    line: { color: C.transfer, width: 2.5 }, marker: { size: 4 },
    hovertemplate: 'L%{x} transfer=%{y:.3f}<extra></extra>',
  };

  const domX = Object.keys(tp.domain_at_L34);
  const domY = Object.values(tp.domain_at_L34);
  const domColors = domY.map(v => v >= 0.80 ? '#16a34a' : v >= 0.65 ? '#f97316' : '#dc2626');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%',
      overflow: 'auto', padding: 12 }}>
      <div style={{ display: 'flex', gap: 10 }}>
        <div style={{ flex: 2, background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
          <div style={{ fontSize: 9, color: 'var(--text-dim)', marginBottom: 4 }}>
            Linear probe accuracy by layer — physics train → cross-domain test</div>
          <Plot data={[traceCV, traceTR]}
            layout={{ ...LAYOUT_BASE, height: 220,
              xaxis: { ...axs, dtick: 2, title: { text: 'Layer', font: { size: 9 } } },
              yaxis: { ...axs, range: [0.35, 1.05], title: { text: 'Accuracy', font: { size: 9 } } },
              shapes: [
                { type: 'line', x0: tp.layers[0]-0.5, x1: tp.layers.at(-1)+0.5, y0: 0.5, y1: 0.5,
                  line: { dash: 'dot', color: C.chance, width: 1 } },
                { type: 'line', x0: tp.best_layer, x1: tp.best_layer, y0: 0.3, y1: 1.05,
                  line: { dash: 'dash', color: '#f97316', width: 1.5 } },
              ],
              annotations: [{ x: tp.best_layer, y: 1.02, text: `L${tp.best_layer}`,
                showarrow: false, font: { size: 8, color: '#f97316' } }],
            }}
            config={CFG} style={{ width: '100%' }} useResizeHandler />
          <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.6 }}>
            Purple = 5-fold CV on physics prompts (W0+W2+W4 only).
            Green = corrected transfer to Family D cross-domain (max(acc, 1−acc) for sign inversion).
            Orange dashed = best layer (L{tp.best_layer}, transfer={pct(tp.transfer[tp.layers.indexOf(tp.best_layer)])}).
          </div>
        </div>

        <div style={{ flex: 1, background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
          <div style={{ fontSize: 9, color: 'var(--text-dim)', marginBottom: 6 }}>
            Transfer accuracy by domain at L{tp.best_layer}</div>
          <Plot data={[{
            type: 'bar', x: domY, y: domX, orientation: 'h',
            marker: { color: domColors },
            hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
            text: domY.map(v => pct(v)), textposition: 'auto',
            textfont: { size: 9 },
          }]}
            layout={{ ...LAYOUT_BASE, height: 180, margin: { t: 4, b: 36, l: 110, r: 40 },
              xaxis: { ...axs, range: [0, 1], title: { text: 'Accuracy', font: { size: 9 } } },
              yaxis: { ...axs },
              shapes: [{ type: 'line', x0: 0.5, x1: 0.5, y0: -0.5, y1: domX.length - 0.5,
                line: { dash: 'dot', color: C.chance, width: 1 } }],
            }}
            config={CFG} style={{ width: '100%' }} useResizeHandler />
          <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.6 }}>
            Info theory (84.6%) closest to physics; statistics (58.3%) most domain-distant.
          </div>
        </div>
      </div>

      {/* Three-phase analysis */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text-dim)', marginBottom: 6 }}>
          Three-phase layer trajectory</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {[
            { label: 'Phase 1: Physics-specific', layers: 'L2–L5', transfer: '0.53–0.64',
              desc: 'Lexically anchored to physics vocabulary. High in-domain CV, low transfer.', color: '#dc2626' },
            { label: 'Phase 2: Surface-form', layers: 'L6–L20', transfer: '0.51–0.60',
              desc: 'Domain tokens and wording dilute the encoding. Transfer noisy.', color: '#f97316' },
            { label: 'Phase 3: Abstract assembly', layers: 'L21–L34', transfer: '0.66–0.73',
              desc: 'Domain-general intensive/extensive direction builds monotonically. Peak L34.', color: '#16a34a' },
          ].map(ph => (
            <div key={ph.label} style={{ flex: 1, background: 'var(--bg-panel)', borderRadius: 5,
              padding: '7px 9px', borderLeft: `3px solid ${ph.color}` }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: ph.color }}>{ph.label}</div>
              <div style={{ fontSize: 8.5, color: 'var(--text)', fontFamily: 'monospace', marginTop: 3 }}>
                {ph.layers} | transfer {ph.transfer}</div>
              <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.5 }}>
                {ph.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Panel 2: Layer transition (loaded from pipeline output) ───────────────────
function LayerTransitionPanel({ data }) {
  const lt = data.layer_transition;
  if (!lt) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', flexDirection: 'column', gap: 10 }}>
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>Layer transition data not yet available</div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
          Run: <code style={{ background: 'var(--bg-card)', padding: '2px 6px', borderRadius: 4 }}>
            sbatch jobs/run_ie_analysis.sbatch
          </code>
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>
          Then sync results and update dashboard data
        </div>
      </div>
    );
  }
  const axs = { gridcolor: '#2d3148', zeroline: false, color: '#8b90a5' };
  const valid = lt.filter(r => !r.degenerate);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%',
      overflow: 'auto', padding: 12 }}>
      <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
        <Plot data={[
          { type: 'scatter', mode: 'lines+markers', name: 'ARI: abstraction class',
            x: valid.map(r=>r.layer), y: valid.map(r=>r.ari_cls), line: { color: '#2563eb', width: 2.5 } },
          { type: 'scatter', mode: 'lines+markers', name: 'ARI: wording family',
            x: valid.map(r=>r.layer), y: valid.map(r=>r.ari_wf), line: { color: '#f97316', width: 2 } },
          { type: 'scatter', mode: 'lines+markers', name: 'ARI: property name',
            x: valid.map(r=>r.layer), y: valid.map(r=>r.ari_prop),
            line: { color: '#dc2626', width: 1.5, dash: 'dash' } },
        ]}
          layout={{ ...LAYOUT_BASE, height: 260,
            xaxis: { ...axs, dtick: 2, title: { text: 'Layer', font: { size: 9 } } },
            yaxis: { ...axs, title: { text: 'ARI', font: { size: 9 } } } }}
          config={CFG} style={{ width: '100%' }} useResizeHandler />
        <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 4 }}>
          ARI(cls)&gt;ARI(wf): layer where abstraction class dominates surface form encoding.
        </div>
      </div>
    </div>
  );
}

// ── Panel 3: Cluster map ──────────────────────────────────────────────────────
function ClusterMapPanel({ data }) {
  const cl = data.cluster_analysis;
  if (!cl) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'var(--text-dim)', fontSize: 12, flexDirection: 'column', gap: 8 }}>
        <div>Cluster analysis not yet available</div>
        <div style={{ fontSize: 10 }}>Run scripts 71 + 72 after the pipeline completes</div>
      </div>
    );
  }
  const axs = { gridcolor: '#2d3148', zeroline: false, color: '#8b90a5' };
  const clusters = cl.map(r => `C${r.cluster}`);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%',
      overflow: 'auto', padding: 12 }}>
      <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
        <div style={{ fontSize: 9, color: 'var(--text-dim)', marginBottom: 4 }}>
          Sign-flip rate by cluster × class (causal selectivity)</div>
        <Plot data={[
          { type: 'bar', name: 'Intensive prompts', x: clusters,
            y: cl.map(r=>r.sfr_intensive), marker: { color: C.intensive } },
          { type: 'bar', name: 'Extensive prompts', x: clusters,
            y: cl.map(r=>r.sfr_extensive), marker: { color: C.extensive } },
        ]}
          layout={{ ...LAYOUT_BASE, height: 240, barmode: 'group',
            yaxis: { ...axs, title: { text: 'Sign-flip rate', font: { size: 9 } } } }}
          config={CFG} style={{ width: '100%' }} useResizeHandler />
      </div>
    </div>
  );
}

// ── Panel 4: Findings ─────────────────────────────────────────────────────────
function FindingsPanel({ data }) {
  const tp = data.probe_transfer;
  const verdict_color = tp.verdict === 'domain_general' ? '#16a34a'
    : tp.verdict === 'partial' ? '#f97316' : '#dc2626';
  const verdict_label = tp.verdict === 'domain_general' ? 'DOMAIN-GENERAL'
    : tp.verdict === 'partial' ? 'PARTIAL TRANSFER' : 'DOMAIN-LOCAL';

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: 12,
      display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Verdict */}
      <div style={{ background: verdict_color + '22', border: `1px solid ${verdict_color}66`,
        borderRadius: 6, padding: '10px 14px' }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: verdict_color }}>{verdict_label}</div>
        <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.6 }}>
          Best transfer at L{tp.best_layer}: {pct(tp.transfer[tp.layers.indexOf(tp.best_layer)])}.
          Direction is NOT inverted (only 2/37 layers flip, both degenerate).
          The model has a partially domain-general intensive/extensive representation.
        </div>
      </div>

      {/* Key findings */}
      {[
        {
          title: 'Representation paradox',
          color: '#f97316',
          body: `Hidden-state probe (L34): 72.7% cross-domain transfer. Behavioral output (Family D): 45.5% — inverted. The model has the correct internal representation but the LM-head read-out fails for cross-domain prompts. This is a representation-readout dissociation.`,
        },
        {
          title: 'Domain ordering',
          color: '#2563eb',
          body: `Info theory (84.6%) > Economics (77.8%) > Biology (66.7%) > Statistics (58.3%). Transfer decays with conceptual distance from physics. Information theory and economics generalise well; statistics (mean, sum) least physics-like.`,
        },
        {
          title: 'Form→Abstraction transition',
          color: '#16a34a',
          body: `Three phases: (1) L2–L5 physics-specific encoding; (2) L6–L20 surface-form processing; (3) L21–L34 monotone rise of domain-general abstraction. L34 peak (3.57× physics→cross ratio). Focus mechanistic analysis on L28–34.`,
        },
        {
          title: 'Readout failure mechanism',
          color: '#7c3aed',
          body: `For cross-domain prompts where the L34 hidden state correctly encodes the class (probe✓), the model's Yes/No output is still wrong (output✗). This is because the LM head reads the physics-domain direction — which works for physics context tokens but reverses for economics/statistics context.`,
        },
      ].map(f => (
        <div key={f.title} style={{ background: 'var(--bg-card)', borderRadius: 6, padding: 10,
          borderLeft: `3px solid ${f.color}` }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: f.color, marginBottom: 4 }}>{f.title}</div>
          <div style={{ fontSize: 8.5, color: 'var(--text-dim)', lineHeight: 1.6 }}>{f.body}</div>
        </div>
      ))}

      {/* Family baselines */}
      <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: 10 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text-dim)', marginBottom: 6 }}>
          Probe family baselines (Qwen3-4B, all families)</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 8.5 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Family', 'Accuracy', 'Consistency', 'ND-AUC', 'Verdict'].map(h => (
                <th key={h} style={{ padding: '4px 8px', textAlign: 'left',
                  color: 'var(--text-dim)', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.family_baselines.families.map((fam, i) => {
              const acc  = data.family_baselines.accuracy[i];
              const cons = data.family_baselines.consistency[i];
              const auc  = data.family_baselines.nd_auc[i];
              const verdict = acc > 0.9 ? '✗ Ceiling' : acc < 0.5 ? '✗ Inverted' : '✓ Tractable';
              const vc = acc > 0.9 ? '#6b7280' : acc < 0.5 ? '#dc2626' : '#16a34a';
              return (
                <tr key={fam} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '4px 8px', fontSize: 8.5 }}>{fam}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'monospace',
                    color: acc >= 0.75 ? '#34d399' : '#f87171' }}>{pct(acc)}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{pct(cons)}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>
                    {auc != null ? fix3(auc) : '—'}</td>
                  <td style={{ padding: '4px 8px', color: vc, fontSize: 8.5,
                    fontWeight: 700 }}>{verdict}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ data }) {
  const tp = data.probe_transfer;
  const best_tr = tp.transfer[tp.layers.indexOf(tp.best_layer)];
  return (
    <div style={{ width: 210, flexShrink: 0, display: 'flex', flexDirection: 'column',
      gap: 8, padding: 10, borderRight: '1px solid var(--border)', overflow: 'auto' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: '#f97316',
        letterSpacing: '0.08em', marginBottom: 2 }}>INTENSIVE/EXTENSIVE</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        <Stat label="Best transfer layer" value={`L${tp.best_layer}`} color="#f97316" />
        <Stat label="Transfer accuracy" value={pct(best_tr)} color="#16a34a"
          sub="physics → cross-domain" />
        <Stat label="Verdict" value="PARTIAL" color="#f97316"
          sub="60–80% range" />
        <Stat label="Sign inverted?" value="NO" color="#34d399"
          sub="direction consistent" />
      </div>

      <div style={{ background: '#f9733322', border: '1px solid #f9733366',
        borderRadius: 6, padding: '7px 9px', fontSize: 8.5, lineHeight: 1.6,
        color: 'var(--text-dim)' }}>
        <span style={{ color: '#f97316', fontWeight: 700 }}>Key finding:</span>
        {' '}Model has correct internal representation (72.7% probe) but wrong output (45.5%). Readout failure, not representation failure.
      </div>

      {/* Domain breakdown */}
      <div style={{ background: 'var(--bg-panel)', borderRadius: 6, padding: '7px 9px' }}>
        <div style={{ fontSize: 8.5, color: 'var(--text-dim)', marginBottom: 5,
          textTransform: 'uppercase', letterSpacing: '0.05em' }}>Transfer by domain (L34)</div>
        {Object.entries(tp.domain_at_L34).map(([dom, acc]) => (
          <div key={dom} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
            <div style={{ flex: 1, fontSize: 8, color: 'var(--text-dim)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              title={dom}>{dom}</div>
            <div style={{ width: 50, background: 'var(--bg)', borderRadius: 2, height: 5 }}>
              <div style={{ width: pct(acc), background: acc >= 0.8 ? '#16a34a' : acc >= 0.65 ? '#f97316' : '#dc2626',
                height: 5, borderRadius: 2 }} />
            </div>
            <span style={{ fontSize: 8, fontFamily: 'monospace', minWidth: 30,
              color: acc >= 0.8 ? '#16a34a' : acc >= 0.65 ? '#f97316' : '#dc2626' }}>
              {pct(acc)}</span>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 8.5, color: 'var(--text-dim)', lineHeight: 1.5, padding: '4px 0' }}>
        <div style={{ fontWeight: 600, marginBottom: 3 }}>Dataset</div>
        <div>480 prompts (384 train, 96 test)</div>
        <div>40 properties (20 intensive, 20 extensive)</div>
        <div>W0·W2·W4·W6 × 3 variants</div>
        <div>Answers: &#39; intensive&#39; (36195) / &#39; extensive&#39; (16376)</div>
      </div>
    </div>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────
const INNER_TABS = [
  { key: 'transfer',    label: 'Transfer Probe' },
  { key: 'transition',  label: 'Layer Transition' },
  { key: 'clusters',    label: 'Cluster Map' },
  { key: 'findings',    label: 'Findings' },
];

export default function AbstractionTab({ data }) {
  const [tab, setTab] = useState('transfer');

  if (!data) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', color: 'var(--text-dim)' }}>Loading abstraction data…</div>
  );

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <Sidebar data={data} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)',
          background: 'var(--bg-panel)', flexShrink: 0 }}>
          {INNER_TABS.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)} style={{
              padding: '7px 16px', background: 'transparent', border: 'none',
              borderBottom: t.key === tab ? '2px solid var(--accent)' : '2px solid transparent',
              color: t.key === tab ? 'var(--accent)' : 'var(--text-dim)',
              fontSize: 11, fontWeight: t.key === tab ? 700 : 400, cursor: 'pointer',
            }}>{t.label}</button>
          ))}
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', alignItems: 'center', padding: '0 12px',
            fontSize: 8.5, color: 'var(--text-dim)' }}>
            physics_intensive_extensive_v1 · Qwen3-4B · 480 prompts
          </div>
        </div>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {tab === 'transfer'   && <TransferProbePanel data={data} />}
          {tab === 'transition' && <LayerTransitionPanel data={data} />}
          {tab === 'clusters'   && <ClusterMapPanel data={data} />}
          {tab === 'findings'   && <FindingsPanel data={data} />}
        </div>
      </div>
    </div>
  );
}
