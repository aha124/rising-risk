// "Does it beat the simple rule?": the comparison at 10%, gains, calibration, ROC and PR, checks.

import { $, fmt, get, val, isNum, withCI, numSpan, onResize, dataTable, setHTML } from "./util.js";
import { lineChart, intervalRows } from "./chart.js";

export const NAMES = { lgbm: "The model", logreg: "Logistic regression", prior_cost: "Last year's cost", random: "Random" };
const STYLE = {
  lgbm: { cls: "model", z: 3 },
  prior_cost: { cls: "base", z: 2 },
  logreg: { cls: "ink", thin: true, z: 1 },
  random: { cls: "base", thin: true, dash: true, z: 0 },
};

// Honest wording for a paired difference: beats, loses or unclear.
export function verdict(delta) {
  if (!delta || !isNum(delta.lo) || !isNum(delta.hi)) return "unknown";
  if (delta.lo > 0) return "beats";
  if (delta.hi < 0) return "loses";
  return "unclear";
}

export function heroCopy(summary, perf) {
  const hd = get(summary, "headline", {});
  const k = fmt.pct0(get(hd, "k", 0.1));
  const m = get(hd, "recall_at_k.lgbm");
  const p = get(hd, "recall_at_k.prior_cost");
  const d = get(hd, "delta_recall_at_k_vs_prior");
  const nTest = get(summary, "data.n_test");
  setHTML("lede",
    `Calling the ${numSpan(k)} of members the model ranks highest reaches ${withCI(m)} of the people who end up in next year's high-cost group. ` +
    `Calling the ${numSpan(k)} who cost the most last year reaches ${withCI(p)}.`);
  const range = d && isNum(d.lo) ? ` (95% interval ${numSpan(fmt.pts(d.lo))} to ${numSpan(fmt.pts(d.hi))})` : "";
  const v = verdict(d);
  let text = "";
  if (v === "beats") {
    text = `At the same number of calls the model reaches <strong>${numSpan(fmt.pts(d.v))} points more</strong>${range}`;
    text += isNum(nTest) ? `, measured on all ${numSpan(fmt.int(nTest))} member-years in the test set.` : ".";
    if (d.v < 0.03) text += " That is a small gain, and whether it justifies running a model is a judgment call.";
    const share = get(perf, "sensitivity.gain_share_women_18_49_not_pregnant");
    if (isNum(share) && share >= 0.4) {
      text += ` ${share >= 0.5 ? "Most" : "A large part"} of the extra members are women aged 18 to 49, which changes how the result should be read. <a href="#gain">See where the gain comes from.</a>`;
    }
  } else if (v === "loses") {
    text = `The simple rule does better here, by <strong>${numSpan(fmt.pts(d.v))} points</strong> (95% interval ${numSpan(fmt.pts(d.hi))} to ${numSpan(fmt.pts(d.lo))}). On this data the model has not earned its place.`;
  } else if (v === "unclear") {
    text = `The difference is ${numSpan(fmt.signed(d.v * 100, 1))} points and its 95% interval (${numSpan(fmt.signed(d.lo * 100, 1))} to ${numSpan(fmt.signed(d.hi * 100, 1))}) includes zero, so on this data <strong>the model does not clearly beat the simple rule</strong>.`;
  }
  setHTML("lede-2", text);
}

function findingCopy(summary, perf) {
  const hd = get(summary, "headline", {});
  const k = fmt.pct0(get(hd, "k", 0.1));
  const m = get(hd, "recall_at_k.lgbm");
  const p = get(hd, "recall_at_k.prior_cost");
  const l = get(hd, "recall_at_k.logreg");
  const d = get(hd, "delta_recall_at_k_vs_prior");
  const v = verdict(d);
  const lead = {
    beats: "Yes, on the measure chosen before any results were in.",
    loses: "No.",
    unclear: "Not clearly.",
    unknown: "",
  }[v];
  let html = `${lead} With ${numSpan(k)} of members on the list, the model reaches ${withCI(m)} of next year's high-cost members and the last-year's-cost rule reaches ${withCI(p)}.`;
  const nPos = get(summary, "data.n_test_pos");
  if (v === "beats" && isNum(nPos)) html += ` Across the test years that is about ${numSpan(fmt.int(d.v * nPos))} more high-cost members on the list, out of ${numSpan(fmt.int(nPos))}.`;
  setHTML("results-finding", html);

  const parts = [];
  if (d && isNum(d.lo)) {
    parts.push(`The paired difference in share reached is ${numSpan(fmt.signed(d.v * 100, 1))} points, with a 95% bootstrap interval of ${numSpan(fmt.signed(d.lo * 100, 1))} to ${numSpan(fmt.signed(d.hi * 100, 1))}. ` +
      (v === "unclear" ? "The interval includes zero." : "The interval does not include zero."));
  }
  const da = get(hd, "delta_auprc_vs_prior");
  if (da && isNum(da.v)) {
    parts.push(`Average precision goes from ${numSpan(fmt.num(val(get(hd, "auprc.prior_cost")), 3))} to ${numSpan(fmt.num(val(get(hd, "auprc.lgbm")), 3))} (difference ${numSpan(fmt.signed(da.v, 3))}, interval ${numSpan(fmt.signed(da.lo, 3))} to ${numSpan(fmt.signed(da.hi, 3))}).`);
  }
  const mv = val(m), pv = val(p), lv = val(l);
  if (isNum(mv) && isNum(pv) && isNum(lv) && mv > pv) {
    const share = (lv - pv) / (mv - pv);
    const tail = share >= 0.8 ? "so most of the gain comes from the features, and the trees add little on top of a linear model"
      : share >= 0.4 ? "so roughly half of the gain comes from the features alone and the rest from the tree model"
      : "so most of the gain needs the tree model and not just the features";
    parts.push(`Logistic regression on the same features reaches ${withCI(l)}, ${tail}.`);
  }
  setHTML("results-finding-2", parts.join(" "));
}

function compareChart(el, summary) {
  const rec = get(summary, "headline.recall_at_k", {});
  intervalRows(el, ["lgbm", "logreg", "prior_cost", "random"].map((key) => ({
    label: NAMES[key], strong: key === "lgbm", I: rec[key], cls: STYLE[key].cls,
    hollow: key === "logreg" || key === "random", tip: key === "random" ? "by definition" : "",
  })), { format: fmt.pct, tick: fmt.pct0 });
}

function gainsChart(el, perf, k, measure) {
  const cap = get(perf, "capacity", []).filter((c) => isNum(c.pct));
  if (!cap.length) return;
  const series = ["lgbm", "prior_cost", "logreg", "random"].filter((key) => cap[0][key]).map((key) => ({
    name: NAMES[key], ...STYLE[key],
    pts: [[0, 0], ...cap.map((c) => [c.pct, get(c, `${key}.${measure}`)])],
  }));
  lineChart(el, {
    height: Math.min(380, Math.max(260, el.clientWidth * 0.55)),
    margin: { left: 44, right: 14, top: 12, bottom: 40 },
    x: { domain: [0, 1], format: fmt.pct0, label: "Share of members called" },
    y: { domain: [0, 1], format: fmt.pct0, label: null, nice: false, values: [0, 0.25, 0.5, 0.75, 1] },
    series,
    refsX: [{ x: k, label: `${fmt.pct0(k)} called` }],
    tip: { x: (xv) => `${fmt.pct0(xv)} of members called` },
  });
}

function calibrationChart(el, perf) {
  const raw = get(perf, "calibration_before_after.raw.bins", []);
  const cal = get(perf, "calibration_before_after.calibrated.bins", []);
  const all = [...raw, ...cal];
  if (!all.length) return;
  // Most members score low, so both axes are logarithmic. Otherwise seven of the ten groups pile up in one corner.
  const pos = all.flatMap((b) => [b.mean_pred, b.obs_rate]).filter((v) => isNum(v) && v > 0);
  const hi = Math.min(1, Math.max(...pos) * 1.25);
  const lo = Math.max(1e-4, Math.min(...pos) * 0.8);
  const tv = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].filter((v) => v >= lo && v <= hi);
  const tf = (v) => (v < 0.01 ? fmt.pct(v, 1) : fmt.pct0(v));
  const floor = (v) => (isNum(v) ? Math.max(v, lo) : v);
  const side = Math.min(el.clientWidth, 460);
  lineChart(el, {
    height: side * 0.86 + 40,
    margin: { left: 48, right: Math.max(14, el.clientWidth - side), top: 24, bottom: 40 },
    log: true,
    x: { domain: [lo, hi], format: tf, label: "Predicted probability (log scale)", values: tv },
    y: { domain: [lo, hi], format: tf, nice: false, values: tv, label: "Observed share" },
    diagonal: true,
    series: [
      { name: "After calibration", cls: "model", z: 2, pts: cal.map((b) => [floor(b.mean_pred), floor(b.obs_rate)]), dots: true },
      { name: "Before calibration", cls: "model", thin: true, pts: raw.map((b) => [floor(b.mean_pred), floor(b.obs_rate)]), dots: () => "hollow" },
    ],
    tip: { x: (xv) => `Predicted ${fmt.pct(xv)}, observed:` },
  });
}

function curve(el, perf, field, xLabel, yLabel, diagonal) {
  const series = ["lgbm", "prior_cost", "logreg"].map((key) => {
    const m = get(perf, "models", []).find((mm) => mm.key === key);
    return m && Array.isArray(m[field]) ? { name: NAMES[key], ...STYLE[key], pts: m[field] } : null;
  }).filter(Boolean);
  if (!series.length) return;
  lineChart(el, {
    height: Math.min(340, el.clientWidth * 0.8 + 40),
    margin: { left: 44, right: 12, top: 22, bottom: 40 },
    x: { domain: [0, 1], format: (v) => fmt.num(v, 1), label: xLabel, values: [0, 0.5, 1] },
    y: { domain: [0, 1], format: (v) => fmt.num(v, 1), label: yLabel, nice: false, values: [0, 0.5, 1] },
    diagonal, series,
    tip: { x: (xv) => `${xLabel} ${fmt.num(xv, 2)}` },
  });
}

function checksTable(summary, perf, labelOf) {
  const rows = [];
  const seen = get(perf, "member_holdout.seen");
  const unseen = get(perf, "member_holdout.unseen");
  if (seen && unseen) {
    const ru = get(unseen, "prior_cost.recall_at_10"), rs = get(seen, "prior_cost.recall_at_10");
    const du = get(unseen, "delta_recall_at_10_vs_prior"), ds = get(seen, "delta_recall_at_10_vs_prior");
    let result = `Model ${fmt.pct(unseen.recall_at_10)} for never-seen members (${fmt.int(unseen.n)} rows), ${fmt.pct(seen.recall_at_10)} for members seen in training (${fmt.int(seen.n)} rows).`;
    let read;
    if (isNum(ru) && isNum(rs) && du && ds && isNum(du.lo)) {
      result += ` The rule: ${fmt.pct(ru)} and ${fmt.pct(rs)}.`;
      const ruleDrops = ru < rs, modelDrops = unseen.recall_at_10 < seen.recall_at_10;
      const margin = `The model's margin over the rule is ${fmt.pts(du.v)} points for never-seen members (interval ${fmt.pts(du.lo)} to ${fmt.pts(du.hi)}) and ${fmt.pts(ds.v)} for the rest.`;
      if (du.lo > 0 && modelDrops && ruleDrops) read = `Both score lower on never-seen members, and the rule has nothing to memorize, so these are harder people to rank and not a sign the model remembers its training members. ${margin}`;
      else if (du.lo > 0) read = `The model's lead holds for members it never trained on. ${margin}`;
      else read = `For never-seen members the lead over the rule is not clear. ${margin}`;
    } else {
      const d = unseen.recall_at_10 - seen.recall_at_10;
      read = Math.abs(d) < 0.03 ? "About the same for members the model never trained on." : d < 0 ? "Lower for never-seen members." : "Higher for never-seen members.";
    }
    rows.push(["Members never seen in training", result, read]);
  }
  const sh = get(perf, "sanity.shuffled_label_auroc");
  if (isNum(sh)) {
    rows.push(["Trained on shuffled labels", `AUROC ${fmt.num(sh, 3)}`,
      Math.abs(sh - 0.5) <= 0.05 ? "Near 0.5, as it should be. The training and evaluation code does not find skill where there is none. The feature checks above are what rule out a leak." : "Not near 0.5. Something in the training or evaluation code is leaking the answer."]);
  }
  const np = get(perf, "sanity.no_prior_cost_features");
  if (np) {
    const pr = val(get(summary, "headline.recall_at_k.prior_cost"));
    const full = val(get(summary, "headline.recall_at_k.lgbm"));
    const read = !isNum(pr) ? ""
      : np.recall_at_10 >= pr && isNum(full) && full - np.recall_at_10 < 0.02 ? `Almost the same as the full model (${fmt.pct(full)}). The rest of the record carries the signal, and the cost history adds little on top of it.`
      : np.recall_at_10 >= pr ? "The other features alone match or beat the last-year's-cost rule."
      : "Without prior cost the model falls below the simple rule, so cost history still carries much of the signal.";
    rows.push([`Trained without any prior-cost feature${Array.isArray(np.dropped) ? ` (${np.dropped.length} dropped)` : ""}`, `${fmt.pct(np.recall_at_10)} reached, AUROC ${fmt.num(np.auroc, 3)}`, read]);
  }
  const mx = get(perf, "sanity.max_single_feature_auroc");
  if (mx) {
    rows.push(["Strongest single feature", `${labelOf(mx.feature)}, AUROC ${fmt.num(mx.auroc, 3)}`,
      mx.auroc > 0.98 ? "Above 0.98. One feature nearly gives away the label, which points to a leak." : "No single feature gives the label away."]);
  }
  if (!rows.length) return;
  $("table-checks").replaceChildren(dataTable([{ label: "Check" }, { label: "Result" }, { label: "Reading" }], rows));
  setHTML("checks-intro", "A good test score can come from a mistake. These checks look for the usual ones.");
}

export function initResults(summary, perf, labelOf = (x) => x) {
  const k = get(summary, "headline.k", 0.1);
  const d = get(summary, "data", {});
  if (Array.isArray(d.test_years)) {
    setHTML("results-gloss", `Test years ${d.test_years.join(" and ")}: ${fmt.int(d.n_test)} member-years, ` +
      (isNum(d.n_test_pos) ? `${fmt.int(d.n_test_pos)}` : fmt.pct(d.prevalence_test)) + ` of them high cost. Intervals are 95%, from ${isNum(d.n_bootstrap) ? fmt.int(d.n_bootstrap) + " " : ""}bootstrap resamples clustered by member.`);
  }
  findingCopy(summary, perf);
  setHTML("cmp-sub", `Share of next year's high-cost members who are on the list when ${fmt.pct0(k)} of members are called. Bars show 95% intervals.`);
  onResize($("chart-compare"), () => compareChart($("chart-compare"), summary));
  if (!perf) return;

  let measure = "found";
  const redrawGains = onResize($("chart-gains"), () => gainsChart($("chart-gains"), perf, k, measure));
  $("gains-toggle").addEventListener("change", (e) => { measure = e.target.value; redrawGains(); });
  const cap = get(perf, "capacity", []);
  const pick = [0.01, 0.05, 0.1, 0.2, 0.5].map((p) => cap.find((c) => Math.abs(c.pct - p) < 1e-9)).filter(Boolean);
  if (pick.length) {
    $("table-gains").replaceChildren(dataTable(
      [{ label: "Called" }, { label: "Model, found", num: true }, { label: "Last year's cost, found", num: true }, { label: "Model, dollars", num: true }, { label: "Last year's cost, dollars", num: true }],
      pick.map((c) => [fmt.pct0(c.pct), fmt.pct(get(c, "lgbm.found")), fmt.pct(get(c, "prior_cost.found")), fmt.pct(get(c, "lgbm.dollars")), fmt.pct(get(c, "prior_cost.dollars"))])
    ));
  }

  onResize($("chart-calibration"), () => calibrationChart($("chart-calibration"), perf));
  const rawC = get(perf, "calibration_before_after.raw");
  const calC = get(perf, "calibration_before_after.calibrated");
  if (rawC && calC) {
    const ch = get(perf, "calibration_before_after.change.ece");
    const moved = `The expected calibration error is ${numSpan(fmt.num(rawC.ece, 4))} before and ${numSpan(fmt.num(calC.ece, 4))} after, and the Brier score ${numSpan(fmt.num(rawC.brier, 4))} and ${numSpan(fmt.num(calC.brier, 4))}.`;
    let read;
    if (ch && isNum(ch.lo) && isNum(ch.hi)) {
      const range = `(change ${numSpan(fmt.signed(ch.v, 4))}, 95% interval ${numSpan(fmt.signed(ch.lo, 4))} to ${numSpan(fmt.signed(ch.hi, 4))})`;
      read = ch.lo <= 0 && ch.hi >= 0 ? `The calibration step made no measurable difference ${range}. The raw scores were already close to the diagonal, so the step stays in the pipeline as a safeguard and gets no credit for the result.`
        : ch.hi < 0 ? `Calibration lowered the error ${range}.` : `Calibration raised the error ${range}, so the raw scores were the better ones.`;
    } else {
      read = calC.ece < rawC.ece ? "The error is lower after calibration, with no interval to say whether the difference is real." : "Calibration did not lower the error on the test years.";
    }
    setHTML("calibration-note", `${read} ${moved} On average the predicted and observed rates differ by ${numSpan(fmt.pts(calC.ece, 1))} percentage points.`);
    $("table-calibration").replaceChildren(dataTable(
      [{ label: "Group" }, { label: "Predicted, before", num: true }, { label: "Observed, before", num: true }, { label: "Predicted, after", num: true }, { label: "Observed, after", num: true }],
      (calC.bins || []).map((b, i) => [String(i + 1), fmt.pct(get(rawC, `bins.${i}.mean_pred`)), fmt.pct(get(rawC, `bins.${i}.obs_rate`)), fmt.pct(b.mean_pred), fmt.pct(b.obs_rate)])
    ));
  }

  // ROC and PR draw when the details element is opened (it has no width before that)
  const rocpr = $("rocpr");
  let drawn = false;
  rocpr.addEventListener("toggle", () => {
    if (!rocpr.open || drawn) return;
    drawn = true;
    onResize($("chart-roc"), () => curve($("chart-roc"), perf, "roc", "False positive rate", "True positive rate", true));
    onResize($("chart-pr"), () => curve($("chart-pr"), perf, "pr", "Recall", "Precision", false));
  });
  const models = get(perf, "models", []).filter((m) => m.key !== "random");
  if (models.length) {
    $("table-rocpr").replaceChildren(dataTable(
      [{ label: "Approach" }, { label: "AUROC", num: true }, { label: "95% interval", num: true }, { label: "Average precision", num: true }, { label: "95% interval", num: true }],
      models.map((m) => {
        const a = get(m, "metrics.auroc", {}), p = get(m, "metrics.auprc", {});
        return [NAMES[m.key] || m.label, fmt.num(val(a), 3), `${fmt.num(a.lo, 3)} to ${fmt.num(a.hi, 3)}`, fmt.num(val(p), 3), `${fmt.num(p.lo, 3)} to ${fmt.num(p.hi, 3)}`];
      })
    ));
  }
  checksTable(summary, perf, labelOf);
}
