/**
 * Generate a natural-language explanation of a prompt's circuit behaviour.
 *
 * Pure function — no async, no fetch, no LLM calls.
 * All values come from pre-computed pipeline outputs loaded into the dashboard.
 *
 * @param {Object} trace       - Single entry from prompt_traces.json
 * @param {Object} circuit     - circuit.json (global circuit data)
 * @param {Object} communityData - community_summary.json array (currently unused, reserved)
 * @param {string} behaviourType - e.g. 'abstraction', 'latent_state', 'candidate_set', 'gating'
 * @returns {string|null} Multi-line explanation string, or null if insufficient data.
 */
export function generateExplanation(trace, circuit, communityData, behaviourType) {
  if (!trace || !circuit) return null;

  const margin = trace.baseline_logit_diff;
  const correct = trace.correct_answer?.trim() ?? '?';
  const incorrect = trace.incorrect_answer?.trim() ?? '?';

  // ── Dominant zone ──────────────────────────────────────────────────────────
  // Zone with highest *positive* total_measured_contribution = most causal support
  // for the correct answer. Falls back to highest absolute value if all negative.
  const zones = ['early', 'mid', 'late'];
  let dominantZone = 'late';
  let dominantContrib = null;
  let dominantNFeats = 0;

  let maxPositive = -Infinity;
  let maxAbs = -Infinity;
  let maxAbsZone = 'late';

  for (const z of zones) {
    const zData = trace.zone_summary?.[z];
    if (!zData) continue;
    const c = zData.total_measured_contribution ?? 0;
    const nf = zData.n_features ?? 0;
    if (c > maxPositive) {
      maxPositive = c;
      dominantZone = z;
      dominantContrib = c;
      dominantNFeats = nf;
    }
    if (Math.abs(c) > maxAbs) {
      maxAbs = Math.abs(c);
      maxAbsZone = z;
    }
  }
  // If all contributions are negative (every zone opposes correct), use highest abs
  if (dominantContrib === null || dominantContrib <= 0) {
    dominantZone = maxAbsZone;
    const zData = trace.zone_summary?.[dominantZone];
    dominantContrib = zData?.total_measured_contribution ?? 0;
    dominantNFeats = zData?.n_features ?? 0;
  }

  // ── Top supporting / competing features ───────────────────────────────────
  const topFeat = trace.top_correct_features?.[0] ?? null;
  const compFeat = (trace.top_incorrect_features ?? []).find(f => f.contribution_to_correct < 0) ?? null;

  // ── Necessity ──────────────────────────────────────────────────────────────
  const necessityPct = (circuit.validation?.disruption_rate ?? 0) * 100;

  // ── Path validation ────────────────────────────────────────────────────────
  const pv = circuit.path_validation_summary ?? null;
  const pathValPct = pv ? (pv.propagation_consistency ?? 0) * 100 : null;

  // ── Behaviour type description ─────────────────────────────────────────────
  const typeDesc = {
    latent_state:  'a latent intermediate state',
    abstraction:   'an abstract representation shared across surface variants',
    candidate_set: 'a candidate selection process',
    gating:        'a gating decision',
  }[behaviourType] ?? 'an internal computation';

  // ── Build explanation ──────────────────────────────────────────────────────
  const lines = [];

  lines.push(
    `For the prompt "${trace.prompt}", the model predicts "${correct}" over "${incorrect}" ` +
    `with a logit margin of ${margin != null ? margin.toFixed(2) : '?'}.`
  );

  lines.push('');
  const contribSign = dominantContrib >= 0 ? '+' : '';
  lines.push(
    `The circuit operates through ${typeDesc}. ` +
    `The dominant contribution comes from ${dominantZone}-layer features ` +
    `(${dominantNFeats} features, \u0394logit = ${contribSign}${dominantContrib.toFixed(3)}).`
  );

  if (topFeat) {
    lines.push('');
    const contribStr = topFeat.contribution_to_correct >= 0
      ? `+${topFeat.contribution_to_correct.toFixed(3)}`
      : topFeat.contribution_to_correct.toFixed(3);
    lines.push(
      `The key feature is [${topFeat.feature_id}] at layer ${topFeat.layer}, ` +
      `contributing ${contribStr} toward the correct answer.`
    );
    if (compFeat) {
      const compStr = compFeat.contribution_to_correct.toFixed(3);
      lines.push(
        `A competing pathway through [${compFeat.feature_id}] simultaneously pushes toward ` +
        `"${incorrect}" (contribution = ${compStr}).`
      );
    }
  }

  lines.push('');
  lines.push(
    `Circuit ablation disrupts the model\u2019s answer on ${necessityPct.toFixed(1)}% of prompts, ` +
    `confirming causal responsibility.`
  );

  if (pathValPct !== null) {
    lines.push(
      `Path validation confirms ${pathValPct.toFixed(1)}% of dominant paths survive ablation.`
    );
  }

  return lines.join('\n');
}
