import Papa from 'papaparse';
import { sanitizeRow } from '../utils/sanitize';

const BASE = '/data';

const INTERVENTION_NUMERIC_COLS = [
  'baseline_logit_diff', 'intervened_logit_diff', 'effect_size',
  'abs_effect_size', 'relative_effect',
];
const FEATURE_AGG_NUMERIC_COLS = [
  'mean_abs_effect_size', 'median_abs_effect_size', 'std_abs_effect_size',
  'mean_effect_size', 'median_effect_size', 'mean_relative_effect',
  'mean_baseline_logit_diff', 'mean_intervened_logit_diff', 'sign_flip_rate',
];
const LAYER_AGG_NUMERIC_COLS = FEATURE_AGG_NUMERIC_COLS;
const PROMPT_AGG_NUMERIC_COLS = FEATURE_AGG_NUMERIC_COLS;

function parseCsv(text) {
  const result = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data;
}

function postProcessInterventions(rows) {
  return rows.map(row => {
    // Parse feature_indices from string like "[126287, 159221, 154631]" to int array
    if (typeof row.feature_indices === 'string') {
      row.feature_indices = row.feature_indices
        .replace(/[\[\]]/g, '')
        .split(',')
        .map(s => parseInt(s.trim(), 10))
        .filter(n => !isNaN(n));
    } else if (typeof row.feature_indices === 'number') {
      row.feature_indices = [row.feature_indices];
    } else {
      row.feature_indices = [];
    }
    // Parse sign_flipped from string to boolean
    if (typeof row.sign_flipped === 'string') {
      row.sign_flipped = row.sign_flipped.toLowerCase() === 'true';
    }
    return row;
  });
}

export async function loadAllData() {
  const [
    interventionsText,
    featureAggText,
    layerAggText,
    promptAggText,
    featureImportanceText,
    graph,
    supernodes,
    supernodesEffect,
    supernodesEffectSummaryText,
    supernodesSummaryText,
    commonPromptIdx,
    runManifest,
    promptsMeta,
    labelStats,
    clusterLabelStats,
    perPromptStats,
  ] = await Promise.all([
    fetch(`${BASE}/interventions.csv`).then(r => r.text()),
    fetch(`${BASE}/interventions_feature_agg.csv`).then(r => r.text()),
    fetch(`${BASE}/interventions_layer_agg.csv`).then(r => r.text()),
    fetch(`${BASE}/interventions_prompt_agg.csv`).then(r => r.text()),
    fetch(`${BASE}/feature_importance.csv`).then(r => r.text()),
    fetch(`${BASE}/graph.json`).then(r => r.json()),
    fetch(`${BASE}/supernodes.json`).then(r => r.json()),
    fetch(`${BASE}/supernodes_effect.json`).then(r => r.json()),
    fetch(`${BASE}/supernodes_effect_summary.csv`).then(r => r.text()),
    fetch(`${BASE}/supernodes_summary.csv`).then(r => r.text()),
    fetch(`${BASE}/common_prompt_idx.json`).then(r => r.json()),
    fetch(`${BASE}/run_manifest.json`).then(r => r.json()),
    fetch(`${BASE}/prompts.json`).then(r => r.json()),
    fetch(`${BASE}/label_stats.json`).then(r => r.json()),
    fetch(`${BASE}/cluster_label_stats.json`).then(r => r.json()),
    fetch(`${BASE}/per_prompt_stats.json`).then(r => r.json()),
  ]);

  const interventions = postProcessInterventions(parseCsv(interventionsText));
  interventions.forEach(r => sanitizeRow(r, INTERVENTION_NUMERIC_COLS));
  const featureAgg = parseCsv(featureAggText);
  featureAgg.forEach(r => sanitizeRow(r, FEATURE_AGG_NUMERIC_COLS));
  const layerAgg = parseCsv(layerAggText);
  layerAgg.forEach(r => sanitizeRow(r, LAYER_AGG_NUMERIC_COLS));
  const promptAgg = parseCsv(promptAggText);
  promptAgg.forEach(r => sanitizeRow(r, PROMPT_AGG_NUMERIC_COLS));
  const featureImportance = parseCsv(featureImportanceText);
  const supernodesEffectSummary = parseCsv(supernodesEffectSummaryText);
  const supernodesSummary = parseCsv(supernodesSummaryText);

  return {
    interventions,
    featureAgg,
    layerAgg,
    promptAgg,
    featureImportance,
    graph,
    supernodes,
    supernodesEffect,
    supernodesEffectSummary,
    supernodesSummary,
    commonPromptIdx,
    runManifest,
    promptsMeta,
    labelStats,
    clusterLabelStats,
    perPromptStats,
  };
}
