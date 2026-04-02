/**
 * Guard against NaN/Inf in numeric values.
 */

export function sanitizeNumeric(value, fallback = null) {
  if (value == null) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function sanitizeRow(row, numericColumns) {
  for (const col of numericColumns) {
    if (col in row) {
      row[col] = sanitizeNumeric(row[col]);
    }
  }
  return row;
}
