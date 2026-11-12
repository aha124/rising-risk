// "Who it works less well for": subgroup performance and the cost-proxy check.

import { scaleLinear } from "../vendor/d3-subset.min.js";
import { $, h, s, fmt, get, val, isNum, onResize, dataTable, setHTML, showTip, hideTip, esc, cap, levelName, numSpan } from "./util.js";
import { frame, lineChart } from "./chart.js";


function metrics(summary) {
  const hd = get(summary, "headline", {});
  const k = get(hd, "k", 0.1);
  return [
    { key: "recall_at_10", short: "Reached", cls: "model", title: "High-cost members reached, by group", tick: fmt.pct0, f: fmt.pct, rule: "prior_cost_recall_at_10", ci: "recall_at_10_ci", ref: val(get(hd, "recall_at_k.lgbm")), refLabel: "all members",
      sub: `Of the members in each group who ended up high cost, the share on the list at ${fmt.pct0(k)} capacity, for the model and for the last-year's-cost rule. Bars are the model's 95% intervals.` },
    { key: "selection_rate_at_10", short: "On the list", cls: "model", title: "Share put on the list, by group", tick: fmt.pct0, f: fmt.pct, rule: "prior_cost_selection_rate_at_10", ref: k, refLabel: "all members",
      sub: "Share of each group put on the list, by the model and by the rule. Groups differ in how often they end up high cost, so these are not expected to be equal." },
    { key: "obs_over_expected", short: "Observed / expected", cls: "model", title: "Observed over expected, by group", tick: (v) => fmt.num(v, 1), f: (v) => fmt.num(v, 2), ref: 1, ci: "obs_over_expected_ci", refLabel: "calibrated",
      sub: "High-cost members observed divided by the number the model predicted. Above 1 means the model underestimates risk for that group." },
    { key: "auroc", short: "AUROC", cls: "model", title: "Ranking quality, by group", tick: (v) => fmt.num(v, 2), f: (v) => fmt.num(v, 3), ref: val(get(hd, "auroc.lgbm")), refLabel: "all members",
      sub: "How well the model ranks members within each group." },
    { key: "prevalence", short: "Share high cost", cls: "signal", title: "Share who ended up high cost, by group", tick: fmt.pct0, f: fmt.pct, ref: get(summary, "data.prevalence_test"), refLabel: "all members",
      sub: "Share of each group that ended up high cost. This is the outcome, not the model." },
  ];
}

function dotPlot(el, groups, m) {
  const attrs = [...new Set(groups.map((g) => g.attribute))];
  const rows = [];
  attrs.forEach((a) => {
    rows.push({ head: cap(a) });
    groups.filter((g) => g.attribute === a).forEach((g) => rows.push({ g }));
  });
  const side = el.clientWidth >= 640;
  const rowH = side ? 30 : 44;
  const headH = 34;
  const total = rows.reduce((acc, r) => acc + (r.head ? headH : rowH), 0);
  const left = side ? 262 : 2;
  const f = frame(el, { height: total + 40, m: { top: 6, left, right: 62, bottom: 30 } });
  // small groups are drawn but do not set the axis, or their noise would flatten everyone else
  const big = groups.filter((g) => !g.small);
  const vals = big.flatMap((g) => [g[m.key], m.rule ? g[m.rule] : null, m.ci ? get(g, `${m.ci}.lo`) : null, m.ci ? get(g, `${m.ci}.hi`) : null]).filter(isNum);
  if (isNum(m.ref)) vals.push(m.ref);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || 0.05;
  const x = scaleLinear().domain([Math.max(0, lo - pad), hi + pad]).range([0, f.w]).nice(4).clamp(true);
  x.ticks(Math.max(3, Math.floor(f.w / 90))).forEach((t) => {
    f.g.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: f.h }));
    f.g.append(s("text", { x: x(t), y: f.h + 17, "text-anchor": "middle", text: (m.tick || m.f)(t) }));
  });
  if (isNum(m.ref)) {
    f.g.append(s("line", { class: "ref", x1: x(m.ref), x2: x(m.ref), y1: 0, y2: f.h }));
    f.g.append(s("text", { class: "lab", x: x(m.ref) + 5, y: 12, text: m.refLabel }));
  }
  let y = 0;
  rows.forEach((r) => {
    if (r.head) {
      f.g.append(s("text", { class: "lab strong", x: -left + (side ? 0 : 0), y: y + 24, text: r.head }));
      y += headH;
      return;
    }
    const g = r.g;
    const v = g[m.key];
    const cy = side ? y + rowH / 2 : y + 30;
    const name = levelName(g.level) + (g.small ? " (small group)" : "");
    f.g.append(s("text", { class: "lab", x: side ? -left + 12 : 0, y: side ? cy + 4 : y + 14 }, name,
      s("tspan", { style: "font-family:var(--mono);font-size:11px;fill:var(--ink-2)", dx: 8 }, `n ${fmt.int(g.n)}`)));
    f.g.append(s("line", { class: "grid", x1: 0, x2: f.w, y1: cy, y2: cy }));
    const rv = m.rule ? g[m.rule] : null;
    const ci = m.ci && !g.small ? g[m.ci] : null;
    if (ci && isNum(ci.lo) && isNum(ci.hi)) f.g.append(s("line", { class: `ln s-${m.cls}`, x1: x(ci.lo), x2: x(ci.hi), y1: cy, y2: cy, "stroke-width": 2 }));
    if (isNum(rv)) f.g.append(s("circle", { cx: x(rv), cy, r: 4.5, class: g.small ? "f-paper s-base" : "f-base ring", "stroke-width": 2 }));
    if (isNum(v)) {
      f.g.append(s("circle", { cx: x(v), cy, r: 5.5, class: g.small ? `f-paper s-${m.cls}` : `f-${m.cls} ring`, "stroke-width": 2 }));
      f.g.append(s("text", { class: "val", x: f.w + 10, y: cy + 4, text: m.f(v) }));
    } else {
      f.g.append(s("text", { class: "muted", x: f.w + 10, y: cy + 4, text: "n/a" }));
    }
    const hit = s("rect", { class: "hit", x: -left, y, width: left + f.w + 60, height: rowH, style: "cursor:default" });
    hit.addEventListener("pointermove", (e) => showTip(
      `<b>${esc(cap(g.attribute))}: ${esc(levelName(g.level))}</b><table><tr><td>${esc(m.short)}${m.rule ? ", model" : ""}</td><td>${isNum(v) ? m.f(v) : "n/a"}</td></tr>${ci && isNum(ci.lo) ? `<tr><td>95% interval</td><td>${m.f(ci.lo)} to ${m.f(ci.hi)}</td></tr>` : ""}${isNum(rv) ? `<tr><td>${esc(m.short)}, rule</td><td>${m.f(rv)}</td></tr>` : ""}<tr><td>Test rows</td><td>${fmt.int(g.n)}</td></tr><tr><td>High cost</td><td>${fmt.int(g.n_pos)}</td></tr></table>${g.small ? "Small group. Read with care." : ""}`,
      e.clientX, e.clientY));
    hit.addEventListener("pointerleave", hideTip);
    f.g.append(hit);
    y += rowH;
  });
}

function proxy(audit, smallLevels) {
  const rows = get(audit, "rows", []).filter((r) => isNum(r.score_decile) && isNum(r.mean_chronic_conditions));
  if (!rows.length) return;
  const allLevels = [...new Set(rows.map((r) => r.level))]
    .map((lv) => ({ lv, n: rows.filter((r) => r.level === lv).reduce((a, r) => a + (r.n || 0), 0) }))
    .sort((a, b) => b.n - a.n);
  const levels = allLevels.filter((l) => !smallLevels.has(l.lv));
  const skipped = allLevels.length - levels.length;
  if (skipped > 0) {
    $("proxy-sub").textContent += ` ${skipped} group${skipped === 1 ? " is" : "s are"} too small to draw (${allLevels.filter((l) => smallLevels.has(l.lv)).map((l) => levelName(l.lv)).join(", ")}). Their numbers are in the table.`;
  }
  const hi = Math.max(...rows.map((r) => r.mean_chronic_conditions));
  const pts = (lv) => rows.filter((r) => r.level === lv).sort((a, b) => a.score_decile - b.score_decile).map((r) => [r.score_decile, r.mean_chronic_conditions]);
  const root = $("proxy");
  levels.forEach(({ lv, n }) => {
    const box = h("div", { class: "chart" });
    root.append(h("div", {},
      h("h4", { style: "font-family:var(--display);font-stretch:88%;font-size:.95rem;margin:0 0 2px" }, levelName(lv), " ",
        h("span", { class: "mono", style: "font-weight:400;color:var(--ink-2);font-size:.78rem", text: `n ${fmt.int(n)}` })),
      box));
    onResize(box, () => lineChart(box, {
      height: 170, margin: { left: 30, right: 8, top: 8, bottom: 36 },
      x: { domain: [1, 10], values: [1, 5, 10], label: "Score decile" },
      y: { domain: [0, hi], count: 3, format: (v) => fmt.num(v, 0) },
      series: [
        ...levels.filter((o) => o.lv !== lv).map((o) => ({ cls: "dot", thin: true, pts: pts(o.lv) })),
        { name: levelName(lv), cls: "ink", z: 2, pts: pts(lv) },
      ],
      tip: { x: (xv) => `Score decile ${xv}` },
    }));
  });
  const deciles = [...new Set(rows.map((r) => r.score_decile))].sort((a, b) => a - b);
  $("table-proxy").replaceChildren(dataTable(
    [{ label: "Score decile" }, ...allLevels.map((l) => ({ label: levelName(l.lv) + (smallLevels.has(l.lv) ? " (small)" : ""), num: true }))],
    deciles.map((d) => [String(d), ...allLevels.map((l) => {
      const r = rows.find((rr) => rr.score_decile === d && rr.level === l.lv);
      return r ? fmt.num(r.mean_chronic_conditions, 2) : "";
    })]),
    "Mean number of active chronic condition groups"
  ));
}

export function initAudit(fair, summary, explain) {
  const groups = get(fair, "groups", []);
  const excluded = get(fair, "excluded_from_model", []);
  const inputs = get(fair, "model_inputs_in_this_table", []);
  const top = get(explain, "features.0");
  setHTML("fairness-gloss", esc(get(fair, "small_group_rule", "")));
  let intro = excluded.length ? `${cap(excluded.join(" and "))} are never inputs to the model. They are used only here, to check who the model works for. ` : "";
  if (inputs.length) {
    intro += "Sex and age are inputs";
    if (top && inputs.includes(top.name)) intro += `, and ${top.name === "sex_female" ? "sex" : esc(String(top.label).toLowerCase())} is the one the model leans on most`;
    intro += ", so differences by sex and age below are partly the model doing what it was built to do. ";
  }
  intro += "An overall score can hide a group the model serves badly, so each measure is broken out by group. Hollow dots mark groups too small to read.";
  setHTML("fairness-intro", intro);

  if (groups.length) {
    const ms = metrics(summary);
    let current = ms[0];
    const toggle = $("fair-toggle");
    ms.forEach((m, i) => toggle.append(h("label", {}, h("input", { type: "radio", name: "fair", value: m.key, checked: i === 0 }), m.short)));
    const chart = $("chart-fairness");
    const redraw = onResize(chart, () => dotPlot(chart, groups, current));
    const legend = $("fair-legend");
    const setText = () => {
      setHTML("fair-sub", esc(current.sub));
      setHTML("fair-title", esc(current.title));
      legend.innerHTML = `<li><span class="key dot c-${current.cls}"></span>${current.cls === "signal" ? "Ended up high cost" : "The model"}</li>` +
        (current.rule ? '<li><span class="key dot c-base"></span>Last year\'s cost</li>' : "") +
        `<li><span class="key ring c-${current.cls}"></span>Small group</li>`;
    };
    toggle.addEventListener("change", (e) => { current = ms.find((m) => m.key === e.target.value) || ms[0]; setText(); redraw(); });
    setText();

    // where the model does no better than the rule, said plainly
    const worse = groups.filter((g) => !g.small && isNum(g.recall_at_10) && isNum(g.prior_cost_recall_at_10) && g.recall_at_10 <= g.prior_cost_recall_at_10);
    const describe = (g) => {
      const ci = g.recall_at_10_ci;
      const inside = ci && isNum(ci.hi) && ci.hi >= g.prior_cost_recall_at_10;
      return `for ${g.attribute === "age band" ? "ages " : ""}${esc(levelName(g.level))} the model reaches ${numSpan(fmt.pct(g.recall_at_10))}` +
        (ci && isNum(ci.lo) ? ` (95% interval ${numSpan(fmt.pct(ci.lo))} to ${numSpan(fmt.pct(ci.hi))})` : "") +
        ` and the rule reaches ${numSpan(fmt.pct(g.prior_cost_recall_at_10))}` +
        (inside ? ". The rule's figure sits inside the model's interval, so the fair reading is no better, and possibly worse" : ". The rule's figure is above the model's interval, so the model is worse here");
    };
    setHTML("fair-finding", worse.length
      ? `The model is not ahead everywhere: ${worse.map(describe).join("; ")}. A team whose members are mostly in ${worse.length === 1 ? "that group" : "those groups"} would get nothing from switching.`
      : "In every group large enough to read, the model reaches more high-cost members than the rule does.");
    const ageWorse = worse.filter((g) => g.attribute === "age band");
    const li = $("limit-age");
    if (li) {
      if (ageWorse.length) setHTML("limit-age-text", `${cap(describe(ageWorse[0]))}.`);
      else li.hidden = true;
    }

    $("table-fairness").replaceChildren(dataTable(
      [{ label: "Group" }, { label: "Rows", num: true }, { label: "High cost", num: true }, { label: "Share high cost", num: true }, { label: "AUROC", num: true },
        { label: "Observed / expected", num: true }, { label: "On the list", num: true }, { label: "Reached, model", num: true }, { label: "Reached, rule", num: true }],
      groups.map((g) => [`${cap(g.attribute)}: ${levelName(g.level)}${g.small ? " (small)" : ""}`, fmt.int(g.n), fmt.int(g.n_pos), fmt.pct(g.prevalence), fmt.num(g.auroc, 3),
        fmt.num(g.obs_over_expected, 2), fmt.pct(g.selection_rate_at_10), fmt.pct(g.recall_at_10), fmt.pct(g.prior_cost_recall_at_10)])
    ));
  }
  const small = new Set(groups.filter((g) => g.small && g.attribute === get(fair, "proxy_audit.attribute", "race")).map((g) => g.level));
  proxy(get(fair, "proxy_audit", {}), small);
  // the finding ends with a sentence the paragraph below already makes, so it is not repeated
  const finding = String(get(fair, "proxy_audit.finding", "")).replace(/\s*Synthetic data cannot settle this for a real population\.?\s*$/, "");
  setHTML("proxy-finding", esc(finding));
}
