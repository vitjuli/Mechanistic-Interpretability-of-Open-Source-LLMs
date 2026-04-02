export function fmt(n, digits = 3) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

export function fmtPct(n, digits = 1) {
  if (n == null || isNaN(n)) return '—';
  return (Number(n) * 100).toFixed(digits) + '%';
}

export function fmtInt(n) {
  if (n == null || isNaN(n)) return '—';
  return Math.round(n).toLocaleString();
}

export function shortNodeId(id) {
  if (id === 'input') return 'Input';
  if (id === 'output_correct') return 'Output ✓';
  if (id === 'output_incorrect') return 'Output ✗';
  // L15_F124340 → L15 F124340
  return id.replace('_', ' ');
}
