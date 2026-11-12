// Scores one member with the exported trees. No DOM, no dependencies, so the same
// file runs in the page and in the Node parity test (docs/tests/parity.mjs).

function isMissing(v) {
  return v === null || v === undefined || Number.isNaN(v);
}

// One tree. The rule the export guarantees: a missing value goes left when d is true,
// otherwise right. Any other value goes left when value <= t.
export function treeValue(node, row) {
  while (node.v === undefined) {
    const x = row[node.f];
    node = (isMissing(x) ? node.d : x <= node.t) ? node.l : node.r;
  }
  return node.v;
}

export function rawScore(model, row) {
  let s = model.init_score || 0;
  const trees = model.trees;
  for (let i = 0; i < trees.length; i++) s += treeValue(trees[i], row);
  return s;
}

export function sigmoid(s) {
  return 1 / (1 + Math.exp(-s));
}

// Linear interpolation between calibrator thresholds, clipped at both ends.
// Matches numpy.interp and scikit-learn's IsotonicRegression.predict.
export function interpolate(xs, ys, x) {
  const n = xs.length;
  if (n === 0) return x;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  if (span === 0) return ys[lo];
  return ys[lo] + ((x - xs[lo]) / span) * (ys[hi] - ys[lo]);
}

export function calibrate(model, raw) {
  const cal = model.calibrator;
  if (!cal || !cal.x || !cal.y) return sigmoid(raw);
  const input = model.calibrator_input === "raw" ? raw : sigmoid(raw);
  return interpolate(cal.x, cal.y, input);
}

// Where a probability sits in the reference distribution (0 to 100).
// score_percentiles[i] is the probability at percentile i.
export function percentileOf(model, p) {
  const q = model.score_percentiles;
  if (!q || q.length < 2) return null;
  const last = q.length - 1;
  if (p <= q[0]) return 0;
  if (p >= q[last]) return 100;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (q[mid] <= p) lo = mid;
    else hi = mid;
  }
  const span = q[hi] - q[lo];
  const frac = span === 0 ? 0 : (p - q[lo]) / span;
  return ((lo + frac) / last) * 100;
}

export function tierOf(model, p) {
  const t = model.tiers || {};
  if (t.high !== undefined && p >= t.high) return "high";
  if (t.rising !== undefined && p >= t.rising) return "rising";
  return "routine";
}

// row: feature values in model order, null for missing.
export function score(model, row) {
  const raw = rawScore(model, row);
  const p = calibrate(model, raw);
  return { raw, p, percentile: percentileOf(model, p), tier: tierOf(model, p) };
}
