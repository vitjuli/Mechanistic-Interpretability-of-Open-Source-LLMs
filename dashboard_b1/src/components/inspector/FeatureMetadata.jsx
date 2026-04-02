import { fmt, fmtPct } from '../../utils/formatters';

export default function FeatureMetadata({ nodeInfo, nodeId, indexes }) {
  const imp = indexes.importanceByKey.get(`${nodeInfo.layer}_${nodeInfo.featureIdx}`);

  if (!imp) {
    return <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No importance data</div>;
  }

  const rows = [
    ['Mean Activation', fmt(imp.mean_activation)],
    ['Std Activation', fmt(imp.std_activation)],
    ['Activation Freq', fmtPct(imp.activation_frequency)],
    ['Corr w/ Logit Diff', fmt(imp.correlation_with_logit_diff)],
    ['|Correlation|', fmt(imp.abs_correlation)],
  ];

  return (
    <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
      <tbody>
        {rows.map(([label, value]) => (
          <tr key={label} style={{ borderBottom: '1px solid var(--border)' }}>
            <td style={{ padding: '3px 0', color: 'var(--text-dim)' }}>{label}</td>
            <td style={{ padding: '3px 0', textAlign: 'right', fontFamily: 'monospace' }}>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
