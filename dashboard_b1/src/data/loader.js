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
    // B1-specific extras
    nodeLabels,
    iouData,
    circuit,
    errorCases,
    communityRaw,
    bridgeFeatures,
    // Prompt-level reasoning data
    promptTraces,
    promptPathsText,
    promptFeaturesText,
    layerwiseTracesText,
    communityLabels,
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
    // B1-specific
    fetch(`${BASE}/node_labels.json`).then(r => r.json()),
    fetch(`${BASE}/iou_data.json`).then(r => r.json()),
    fetch(`${BASE}/circuit.json`).then(r => r.json()),
    fetch(`${BASE}/error_cases.json`).then(r => r.json()),
    fetch(`${BASE}/community_summary.json`).then(r => r.json()),
    fetch(`${BASE}/bridge_features.json`).then(r => r.json()),
    // Prompt-level reasoning
    fetch(`${BASE}/prompt_traces.json`).then(r => r.json()),
    fetch(`${BASE}/prompt_paths.csv`).then(r => r.text()),
    fetch(`${BASE}/prompt_features.csv`).then(r => r.text()),
    fetch(`${BASE}/layerwise_traces.csv`).then(r => r.text()),
    // Optional: researcher-authored community labels (never pipeline-generated)
    fetch(`${BASE}/community_labels.json`).then(r => r.json()).catch(() => null),
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

  // Prompt-level data
  const promptPaths = parseCsv(promptPathsText);
  const promptFeatures = parseCsv(promptFeaturesText);
  const layerwiseTraces = parseCsv(layerwiseTracesText);

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
    // B1-specific
    nodeLabels,
    iouData,
    circuit,
    errorCases,
    communityRaw,
    bridgeFeatures,
    // Prompt-level reasoning
    promptTraces,
    promptPaths,
    promptFeatures,
    layerwiseTraces,
    communityLabels,
  };
}
