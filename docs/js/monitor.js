// "After launch": what the drift monitor found in this data, accuracy by year,
// and a labelled simulation of an upstream failure.
//
// Accuracy before the test years is read from the members held out of training,
// because the model has seen everyone else in those years.

import { $, h, fmt, get, isNum, onResize, dataTable, setHTML, numSpan, esc, lowerFirst } from "./util.js";
import { lineChart } from "./chart.js";

const statusOf = (psi, th) => (psi >= th.alert ? "alert" : psi >= th.watch ? "watch" : "ok");
const statusTag = (st) => h("span", { class: `tag${st === "alert" ? " solid" : st === "ok" ? " quiet" : ""}`, text: st === "ok" ? "OK" : st === "watch" ? "Watch" : "Alert" });
const list = (items) => (items.length <= 1 ? items.join("") : items.slice(0, -1).join(", ") + " and " + items[items.length - 1]);

function simTable(sim, th, labelOf) {
  const fStatus = statusOf(sim.feature_psi_after, th), sStatus = statusOf(sim.score_psi_after, th);
  const arrow = (a, b) => `${fmt.num(a, 3)} to ${fmt.num(b, 3)}`;
  const rows = [[`Input: ${labelOf(sim.changed_feature)}`, arrow(sim.feature_psi_before, sim.feature_psi_after), statusTag(fStatus)]];
  const fired = Array.isArray(sim.alerts_fired) ? sim.alerts_fired : [];
  const watch = Array.isArray(sim.watch_fired) ? sim.watch_fired : [];
  fired.filter((f) => f !== sim.changed_feature).forEach((f) => rows.push([`Input: ${labelOf(f)}`, "past the alert line", statusTag("alert")]));
  watch.filter((f) => f !== sim.changed_feature).forEach((f) => rows.push([`Input: ${labelOf(f)}`, "past the watch line", statusTag("watch")]));
  rows.push(["Model score", arrow(sim.score_psi_before, sim.score_psi_after), statusTag(sStatus)]);
  if (isNum(sim.recall_at_10_before)) rows.push(["Share of high-cost members reached", `${fmt.pct(sim.recall_at_10_before)} to ${fmt.pct(sim.recall_at_10_after)}`, ""]);
  if (isNum(sim.share_high_tier_before)) rows.push(["Share of members in the high tier", `${fmt.pct(sim.share_high_tier_before)} to ${fmt.pct(sim.share_high_tier_after)}`, ""]);
  return { table: dataTable([{ label: "What the monitor sees" }, { label: "Before to after", num: true }, { label: "Status after" }], rows), fired, sStatus };
}

export function initMonitor(drift, perf, summary, labelOf = (x) => x) {
  const th = { watch: 0.1, alert: 0.25, ...get(drift, "thresholds", {}) };
  const years = get(drift, "by_year", []).filter((d) => isNum(d.year));
  const ref = get(drift, "reference_years", []);
  const after = years.filter((y) => y.in_reference === false || (y.in_reference === undefined && !ref.includes(y.year)));
  setHTML("lag-note", "");

  // ---- what happened in this data ----
  const shifts = get(drift, "real_shifts", []).filter((sft) => isNum(sft.psi));
  const alerts = shifts.filter((sft) => sft.status === "alert" && sft.in_reference === false);
  const yearOf = (yr) => years.find((y) => y.year === yr) || {};
  const by = get(perf, "by_year", []).filter((d) => isNum(d.year));
  // Held-out members where the file has them (every year before the test years), else all members.
  const fair = (d) => (d && d.holdout_members ? d.holdout_members : d) || {};
  const recallOf = (yr, model = "lgbm") => get(fair(by.find((d) => d.year === yr)), `${model}.recall_at_10`);
  if (alerts.length) {
    const yrs = [...new Set(alerts.map((a) => a.year))].sort();
    const cause = alerts[0].likely_cause;
    const items = alerts.sort((a, b) => a.year - b.year).map((a) => `${esc(lowerFirst(a.label || labelOf(a.feature)))} in ${a.year} (PSI ${numSpan(fmt.num(a.psi, 2))})`);
    const scorePsi = yrs.map((yr) => yearOf(yr).score_psi).filter(isNum);
    const scoreOk = scorePsi.length && scorePsi.every((v) => v < th.watch);
    const recs = yrs.map((yr) => recallOf(yr)).filter(isNum);
    const ruleRecs = yrs.map((yr) => recallOf(yr, "prior_cost")).filter(isNum);
    const held = by.filter((d) => d.holdout_members && isNum(recallOf(d.year)));
    const others = held.filter((d) => !yrs.includes(d.year)).map((d) => recallOf(d.year));
    const heldN = held.map((d) => d.holdout_members.n).filter(isNum);
    const low = held.length ? held.reduce((a, b) => (recallOf(b.year) < recallOf(a.year) ? b : a)) : null;
    const dipped = !!low && yrs.includes(low.year) && others.length > 0 && heldN.length > 0;
    const ahead = recs.length === ruleRecs.length && recs.every((v, i) => v > ruleRecs[i]);
    const words = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"];
    let glossTail = "";
    let text = `${words[alerts.length] || alerts.length} input${alerts.length === 1 ? "" : "s"} crossed the alert line of ${numSpan(fmt.num(th.alert, 2))} after the training years: ${list(items)}.`;
    if (cause) {
      const rolls = alerts.some((a) => /_12m$/.test(a.feature)) && alerts.some((a) => /_13_24m$/.test(a.feature));
      text += ` The likely cause is ${esc(cause)}` + (rolls ? ", a one-off wave that passes through the 12-month window one year and the 13-to-24-month window the next." : ".");
    }
    if (scoreOk) {
      text += ` The distribution of the model's score barely moved in those years (PSI ${list(scorePsi.map((v) => numSpan(fmt.num(v, 3))))}, under the watch line of ${numSpan(fmt.num(th.watch, 2))}).`;
      if (recs.length && others.length && ruleRecs.length === recs.length) {
        text += `</p><p>Accuracy is less tidy. On the members held out of training, the model reached ${list(recs.map((v) => numSpan(fmt.pct(v))))} of high-cost members in ${list(yrs.map(String))}, against ${list(ruleRecs.map((v) => numSpan(fmt.pct(v))))} for the rule` +
          (ahead ? ", so it stayed ahead of the rule." : ", so it did not stay ahead of the rule in every one of those years.");
        if (dipped) {
          text += ` But ${numSpan(fmt.pct(recallOf(low.year)))} in ${low.year} is the lowest of any year (the others run from ${numSpan(fmt.pct(Math.min(...others)))} to ${numSpan(fmt.pct(Math.max(...others)))}). ` +
            `Each yearly figure rests on ${numSpan(fmt.int(Math.min(...heldN)))} to ${numSpan(fmt.int(Math.max(...heldN)))} held-out members, roughly a tenth of them high cost, so a dip of that size is about what year-to-year noise looks like here. I cannot tie it to the input shift, and I would not rule it out either.`;
        }
      }
      text += `</p><p>So the alert was real, and neither the score nor the comparison with the rule called for a fix${dipped ? `, though ${low.year} is not a clean year` : ""}. An input alert is a reason to go and look, and the look can end with nothing to fix. A monitor that treats every input alert as an outage gets ignored, so the alert needs the score and accuracy panels next to it.`;
      glossTail = " The score distribution stayed under the watch line." + (dipped ? ` Recall on held-out members was at its lowest in ${low.year}.` : "");
    } else {
      text += " The score distribution moved as well, so in those years the scores would have needed a closer look before use.";
    }
    setHTML("shift-text", `<p>${text}</p>`);
    setHTML("monitoring-gloss", `${words[alerts.length] || alerts.length} input alert${alerts.length === 1 ? "" : "s"} in ${list(yrs.map(String))}.${glossTail}`);
  } else {
    setHTML("shift-text", `<p>No input crossed the alert line of ${numSpan(fmt.num(th.alert, 2))} in the years after training.</p>`);
    setHTML("monitoring-gloss", "No input alerts after the training years.");
  }
  if (shifts.length) {
    $("table-shifts").replaceChildren(dataTable(
      [{ label: "Index year" }, { label: "Input" }, { label: "PSI", num: true }, { label: "Status" }, { label: "Training year" }, { label: "Likely cause" }],
      [...shifts].sort((a, b) => a.year - b.year || b.psi - a.psi).map((sft) => [String(sft.year), sft.label || labelOf(sft.feature), fmt.num(sft.psi, 3), statusTag(sft.status || statusOf(sft.psi, th)), sft.in_reference ? "Yes" : "No", cap1(sft.likely_cause || "")])
    ));
  }
  const latest = get(drift, "feature_psi_latest", []);
  if (latest.length) {
    $("table-psi").replaceChildren(dataTable(
      [{ label: "Input" }, { label: "PSI", num: true }, { label: "Status" }],
      [...latest].sort((a, b) => b.psi - a.psi).map((f) => [f.label || labelOf(f.feature), fmt.num(f.psi, 3), statusTag(f.status || statusOf(f.psi, th))])
    ));
  }

  // ---- PSI panels: the score, then one panel per input, same axis ----
  if (years.length) {
    const firstAfter = after.length ? after[0].year : null;
    setHTML("psi-sub", `PSI of each index year against the training years${ref.length ? ` (${ref[0]} to ${ref[ref.length - 1]})` : ""}. Below ${fmt.num(th.watch, 2)} is normal, above ${fmt.num(th.alert, 2)} is an alert. ` +
      "Every panel uses the same vertical axis." + (get(drift, "score_note") ? ` ${esc(drift.score_note)}` : ""));
    const feats = new Map();
    years.forEach((y) => (y.top_features || []).forEach((f) => {
      if (!feats.has(f.feature)) feats.set(f.feature, { name: f.label || labelOf(f.feature), pts: [] });
      feats.get(f.feature).pts.push([y.year, f.psi]);
    }));
    const xs = years.map((y) => y.year);
    const panels = [{ name: "Model score", cls: "model", pts: years.map((y) => [y.year, y.score_psi]) }, ...[...feats.values()].map((f) => ({ ...f, cls: "ink" }))];
    const ymax = Math.max(th.alert * 1.2, ...panels.flatMap((p) => p.pts.map((pt) => pt[1])).filter(isNum)) * 1.05;
    const root = $("psi-panels");
    root.replaceChildren();
    panels.forEach((p, i) => {
      const box = h("div", { class: "chart" });
      const peak = p.pts.reduce((a, b) => (b[1] > a[1] ? b : a), p.pts[0]);
      root.append(h("div", {},
        h("h4", { class: "mini-title", title: p.name, text: i === 0 ? p.name : `Input: ${p.name}` }),
        h("p", { class: "sub", style: "font-size:.8rem", text: `Highest ${fmt.num(peak[1], 3)} in ${peak[0]}` }), box));
      onResize(box, () => lineChart(box, {
        height: 170, margin: { left: 40, right: 10, top: 14, bottom: 24 },
        x: { domain: [xs[0] - 0.3, xs[xs.length - 1] + 0.3], values: [xs[0], firstAfter || xs[Math.floor(xs.length / 2)], xs[xs.length - 1]], format: String },
        y: { domain: [0, ymax], format: (v) => (v === th.alert ? fmt.num(v, 2) : fmt.num(v, 1)), nice: false, values: [0, th.alert, Math.floor(ymax * 10) / 10].filter((v, j, arr) => arr.indexOf(v) === j) },
        refsY: [{ y: th.watch, label: i === 0 ? "Watch" : "" }, { y: th.alert, label: i === 0 ? "Alert" : "" }],
        refsX: firstAfter ? [{ x: firstAfter - 0.5, label: i === 0 ? "after training" : "" }] : [],
        series: [{ name: p.name, cls: p.cls, pts: p.pts, dots: (pt) => (pt[1] >= th.alert ? "fill" : null) }],
        tip: { x: (xv) => `Index year ${xv}` },
      }));
    });
  }

  // ---- accuracy by year ----
  if (by.length) {
    const k = get(summary, "headline.k", 0.1);
    const mature = new Map(years.map((y) => [y.year, y.labels_mature !== false]));
    const isMature = (yr) => (mature.has(yr) ? mature.get(yr) : true);
    const immature = by.filter((d) => !isMature(d.year)).map((d) => d.year);
    const asOf = get(drift, "labels_mature_as_of");
    const heldYears = by.filter((d) => d.holdout_members);
    let sub = `Share of high-cost members reached at ${fmt.pct0(k)} capacity, for every index year.`;
    if (heldYears.length) {
      const hn = heldYears.map((d) => d.holdout_members.n).filter(isNum);
      sub += ` Before the test years the points are the members held out of training (${fmt.int(Math.min(...hn))} to ${fmt.int(Math.max(...hn))} a year), because the model has seen the others. In the test years they are all members.`;
    }
    if (immature.length) {
      sub += ` Hollow markers (${list(immature.map(String))}) are years whose outcome would not have been known yet on the scoring date${asOf ? ` of ${esc(asOf)}` : ""}. They can be drawn here only because the data is retrospective. A live monitor sees inputs drift about a year before it can see accuracy drop.`;
    } else {
      $("yr-hollow-key").hidden = true;
    }
    setHTML("yr-sub", sub);
    const xs = by.map((d) => d.year);
    const vals = by.flatMap((d) => [recallOf(d.year), recallOf(d.year, "prior_cost")]).filter(isNum);
    const starts = by.filter((d, i) => i === 0 || d.split !== by[i - 1].split);
    const chart = $("chart-years");
    onResize(chart, (w) => lineChart(chart, {
      height: 290, margin: { left: 44, right: 16, top: 26, bottom: 30 },
      x: { domain: [xs[0] - 0.4, xs[xs.length - 1] + 0.4], values: w < 520 ? xs.filter((_, i) => i % 2 === 0) : xs, format: String },
      y: { domain: [Math.max(0, Math.min(...vals) - 0.08), Math.min(1, Math.max(...vals) + 0.06)], format: fmt.pct0 },
      refsX: starts.map((d, i) => ({ x: d.year - 0.4, label: i === 0 || w >= 520 || d.split === "test" ? d.split : "" })),
      series: [
        { name: "The model", cls: "model", z: 2, pts: by.map((d) => [d.year, recallOf(d.year)]), dots: (p) => (isMature(p[0]) ? "fill" : "hollow") },
        { name: "Last year's cost", cls: "base", pts: by.map((d) => [d.year, recallOf(d.year, "prior_cost")]), dots: (p) => (isMature(p[0]) ? "fill" : "hollow") },
      ],
      tip: { x: (xv) => `Index year ${xv} (${(by.find((d) => d.year === xv) || {}).split || ""})` },
    }));
    $("table-years").replaceChildren(dataTable(
      [{ label: "Index year" }, { label: "Split" }, { label: "Measured on" }, { label: "Member-years", num: true }, { label: "Model, reached", num: true }, { label: "Last year's cost, reached", num: true }, { label: "Model AUROC", num: true }, { label: "Score PSI", num: true }, { label: "Outcome known on scoring date" }],
      by.map((d) => [String(d.year), d.split || "", d.holdout_members ? "Held-out members" : "All members", fmt.int(fair(d).n), fmt.pct(recallOf(d.year)), fmt.pct(recallOf(d.year, "prior_cost")), fmt.num(get(fair(d), "lgbm.auroc"), 3), fmt.num(yearOf(d.year).score_psi, 3), isMature(d.year) ? "Yes" : "Not yet"])
    ));
  }

  // ---- simulation ----
  const sim = get(drift, "simulation");
  if (sim) {
    setHTML("sim-title", esc(sim.title || "A simulated upstream change"));
    setHTML("sim-desc", esc(sim.description || ""));
    const edges = sim.bin_edges || [];
    const step = (hist) => (hist || []).map((v, i) => [edges[i], v]).concat(hist && hist.length ? [[edges[hist.length], hist[hist.length - 1]]] : []);
    const before = step(sim.score_hist_before), afterH = step(sim.score_hist_after);
    if (before.length && afterH.length) {
      const last = Math.max(...[sim.score_hist_before, sim.score_hist_after].map((hh) => hh.reduce((acc, v, i) => (v > 0.002 ? i : acc), 0)));
      const xmax = edges[Math.min(edges.length - 1, last + 1)];
      const chart = $("chart-sim");
      onResize(chart, () => lineChart(chart, {
        height: 240, margin: { left: 44, right: 16, top: 14, bottom: 40 },
        x: { domain: [edges[0], xmax], format: fmt.pct0, label: "Predicted probability" },
        y: { domain: [0, Math.max(...sim.score_hist_before, ...sim.score_hist_after)], format: fmt.pct0 },
        series: [
          { name: "Before", cls: "ink", thin: true, step: true, pts: before.filter((p) => p[0] <= xmax) },
          { name: "After", cls: "model", step: true, z: 2, pts: afterH.filter((p) => p[0] <= xmax) },
        ],
        tip: { x: (xv) => `Share of members scoring from ${fmt.pct0(xv)}` },
      }));
    }
    const { table, fired, sStatus } = simTable(sim, th, labelOf);
    $("table-sim").replaceChildren(table);
    let note = fired.length
      ? `The monitor raises an alert on ${list(fired.map((f) => esc(lowerFirst(labelOf(f)))))}.`
      : "No input crosses the alert line in this simulation.";
    if (fired.length && sStatus === "ok") {
      note += " The score distribution hardly changes";
      if (isNum(sim.recall_at_10_before)) note += `, and the share of high-cost members reached goes from ${numSpan(fmt.pct(sim.recall_at_10_before))} to ${numSpan(fmt.pct(sim.recall_at_10_after))}`;
      note += ", so nothing downstream would look wrong. Only the input check catches a feed that has stopped, and it catches it the day the scores are made.";
    } else if (fired.length) {
      note += ` The score distribution shifts enough to reach ${sStatus} as well.`;
    }
    setHTML("sim-note", note);

    const more = get(drift, "other_simulations", []);
    $("other-sims").replaceChildren(...more.map((o) => {
      const r = simTable(o, th, labelOf);
      return h("details", {}, h("summary", { text: o.title || "Another simulation" }),
        h("p", { text: o.description || "" }), h("div", { class: "table-wrap" }, r.table));
    }));
  }
}

function cap1(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : "";
}
