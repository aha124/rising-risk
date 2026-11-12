// "Why this member?": the exemplar explorer, global importance and dependence curves.

import { scaleLinear } from "../vendor/d3-subset.min.js";
import { $, h, s, fmt, get, isNum, onResize, setHTML, numSpan, showTip, hideTip, esc, tierName, ordinal } from "./util.js";
import { frame, lineChart } from "./chart.js";

const KINDS = [
  { key: "flagged_right", title: "Flagged and right", verdict: "The model put this member on the list, and they did end up high cost." },
  { key: "flagged_wrong", title: "Flagged and wrong", verdict: "The model put this member on the list, but they did not end up high cost." },
  { key: "missed", title: "Missed", verdict: "The model left this member off the list, and they ended up high cost." },
  { key: "routine", title: "Routine", verdict: "The model left this member off the list, and they did not end up high cost." },
];

function waterfall(el, ex, base) {
  const contribs = (ex.contribs || []).filter((c) => isNum(c.contrib));
  const steps = [];
  let run = isNum(base) ? base : 0;
  steps.push({ kind: "base", label: "Average member", from: run, to: run });
  contribs.forEach((c) => {
    steps.push({ kind: "c", label: c.label || c.feature, value: c.value_display, from: run, to: run + c.contrib, d: c.contrib });
    run += c.contrib;
  });
  if (isNum(ex.other_contrib)) {
    steps.push({ kind: "c", label: "Everything else", from: run, to: run + ex.other_contrib, d: ex.other_contrib });
    run += ex.other_contrib;
  }
  steps.push({ kind: "total", label: "This member's score", from: run, to: run });

  const rowH = 42;
  const f = frame(el, { height: steps.length * rowH + 44, m: { top: 4, left: 2, right: 58, bottom: 40 } });
  const all = steps.flatMap((st) => [st.from, st.to]);
  const pad = (Math.max(...all) - Math.min(...all)) * 0.06 || 0.5;
  const x = scaleLinear().domain([Math.min(...all) - pad, Math.max(...all) + pad]).range([0, f.w]);
  x.ticks(Math.max(3, Math.floor(f.w / 80))).forEach((t) => {
    f.g.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: f.h }));
    f.g.append(s("text", { x: x(t), y: f.h + 17, "text-anchor": "middle", text: fmt.num(t, 1) }));
  });
  f.g.append(s("text", { class: "lab muted", x: f.w, y: f.h + 33, "text-anchor": "end", text: "Score in log-odds. Further right is higher risk." }));

  steps.forEach((st, i) => {
    const y0 = i * rowH;
    const by = y0 + 22;
    const label = s("text", { class: `lab${st.kind !== "c" ? " strong" : ""}`, x: 0, y: y0 + 15 }, st.label);
    if (st.value !== undefined) label.append(s("tspan", { class: "muted", style: "font-family:var(--mono);font-size:11.5px;fill:var(--ink-2)", dx: 8 }, st.value));
    f.g.append(label);
    if (st.kind === "c") {
      const x0 = x(Math.min(st.from, st.to));
      const w = Math.max(1.5, Math.abs(x(st.to) - x(st.from)));
      const up = st.d >= 0;
      f.g.append(s("rect", { x: x0, y: by, width: w, height: 12, class: up ? "f-model" : "f-paper s-model", "stroke-width": up ? 0 : 1.5 }));
      f.g.append(s("text", { class: "val", x: Math.max(x(st.from), x(st.to)) + 7, y: by + 10, text: fmt.signed(st.d, 2) }));
    } else {
      f.g.append(s("line", { class: "s-ink", x1: x(st.to), x2: x(st.to), y1: by - 3, y2: by + 15, "stroke-width": 3 }));
      f.g.append(s("text", { class: "val", x: x(st.to) + 8, y: by + 10, text: fmt.num(st.to, 2) }));
    }
    if (i < steps.length - 1) f.g.append(s("line", { class: "ref", x1: x(st.to), x2: x(st.to), y1: by + 12, y2: by + rowH, "stroke-width": 1 }));
  });
}

function caseFacts(ex, k) {
  const kind = KINDS.find((kd) => kd.key === ex.kind) || KINDS[3];
  const pctile = isNum(ex.percentile) ? ex.percentile : null;
  const rows = [
    ...(ex.facts || []).map((f) => [f.label, f.value]),
    ["Predicted probability", fmt.prob(ex.p)],
    pctile !== null ? ["Score percentile", ordinal(Math.min(99, Math.floor(pctile)))] : null,
    ex.tier ? ["Tier", tierName(ex.tier)] : null,
    isNum(ex.cost_next) ? ["Cost the next year", fmt.usd(ex.cost_next)] : null,
  ].filter(Boolean);
  let prior = "";
  if (isNum(ex.rank_prior_pct)) {
    const on = typeof ex.in_prior_top_10 === "boolean" ? ex.in_prior_top_10 : ex.rank_prior_pct < k;
    prior = ` Ranked by last year's cost alone, they sit in the top ${fmt.pct(Math.max(ex.rank_prior_pct, 0.001))} of members, so that rule would ${on ? "also have called them" : "not have called them"}.`;
  }
  return h("div", {},
    h("span", { class: `tag${ex.y === 1 ? " solid" : ""}`, text: kind.title }),
    h("h4", { text: ex.title || ex.id }),
    h("p", { class: "verdict", text: kind.verdict + prior }),
    h("dl", { class: "facts" }, rows.map((r) => h("div", {}, h("dt", { text: r[0] }), h("dd", { text: r[1] })))));
}

function importance(el, feats) {
  const top = feats.filter((ft) => isNum(ft.importance)).slice(0, 12);
  if (!top.length) return;
  const side = el.clientWidth >= 560;
  const rowH = side ? 30 : 40;
  const left = side ? 250 : 2;
  const f = frame(el, { height: top.length * rowH + 8, m: { top: 4, left, right: 52, bottom: 4 } });
  const x = scaleLinear().domain([0, Math.max(...top.map((t) => t.importance))]).range([0, f.w]);
  top.forEach((ft, i) => {
    const y = i * rowH + (side ? 8 : 20);
    f.g.append(s("text", { class: "lab", x: side ? -10 : 0, y: side ? y + 11 : y - 5, "text-anchor": side ? "end" : "start", text: ft.label || ft.name }));
    f.g.append(s("rect", { class: "f-model", x: 0, y, width: Math.max(1, x(ft.importance)), height: 12 }));
    f.g.append(s("text", { class: "val", x: x(ft.importance) + 7, y: y + 11, text: fmt.num(ft.importance, 3) }));
    const hit = s("rect", { class: "hit", x: -left, y: i * rowH, width: left + f.w + 50, height: rowH, style: "cursor:default" });
    hit.addEventListener("pointermove", (e) => showTip(`<b>${esc(ft.label || ft.name)}</b><br>${esc(ft.description || "")}<br>Mean absolute contribution ${fmt.num(ft.importance, 3)} log-odds`, e.clientX, e.clientY));
    hit.addEventListener("pointerleave", hideTip);
    f.g.append(hit);
  });
}

export function initExplain(explain, summary, loadMember) {
  const k = get(summary, "headline.k", 0.1);
  const exemplars = get(explain, "exemplars", []);
  if (!exemplars.length) throw new Error("no exemplars");
  const demo = exemplars.filter((e) => (e.contribs || []).slice(0, 2).some((c) => c.feature === "sex_female" || c.feature === "age")).length;
  const demoNote = demo >= exemplars.length / 2 ? ` For ${numSpan(String(demo))} of the ${numSpan(String(exemplars.length))}, sex or age is one of the two largest bars, which is the pattern the section on where the gain comes from describes.` : "";
  setHTML("why-intro", `${numSpan(String(exemplars.length))} members from the test years, picked to show where the model is right and where it is wrong. For each one, the chart starts at the average member's score and shows how far each recorded fact moved this member away from it.${demoNote}`);

  const picker = $("picker");
  let letter = 0;
  KINDS.forEach((kind) => {
    const mine = exemplars.filter((e) => e.kind === kind.key);
    if (!mine.length) return;
    const opts = h("div", { class: "opts" });
    mine.forEach((ex) => {
      const name = String.fromCharCode(65 + letter++);
      ex._letter = name;
      opts.append(h("label", { title: ex.title || ex.id },
        h("input", { type: "radio", name: "exemplar", value: ex.id, "aria-label": `${kind.title}: ${ex.title || ex.id}` }), name));
    });
    picker.append(h("div", { class: "grp" }, h("h4", { text: kind.title }), opts));
  });

  let current = exemplars.find((e) => e.kind === "flagged_right" && (e.in_prior_top_10 === false || (isNum(e.rank_prior_pct) && e.rank_prior_pct >= k))) || exemplars[0];
  const chart = $("chart-waterfall");
  const redraw = onResize(chart, () => waterfall(chart, current, explain.base_value));
  function show(ex) {
    current = ex;
    picker.querySelectorAll("input").forEach((inp) => { inp.checked = inp.value === ex.id; });
    $("case-facts").replaceChildren(caseFacts(ex, k));
    redraw();
    setHTML("waterfall-note", `The score is the sum of the bars, in log-odds. Calibration then turns it into the probability shown, ${numSpan(fmt.prob(ex.p))}.`);
  }
  picker.addEventListener("change", (e) => { const ex = exemplars.find((x) => x.id === e.target.value); if (ex) show(ex); });
  show(current);

  const loadBtn = $("load-exemplar");
  if (loadBtn) {
    if (!loadMember || !exemplars.some((e) => Array.isArray(e.row))) loadBtn.hidden = true;
    loadBtn.addEventListener("click", () => {
      if (Array.isArray(current.row)) loadMember(current.row, `Member ${current._letter}`);
    });
  }

  // global view, drawn when opened
  const global = $("global");
  let drawn = false;
  global.addEventListener("toggle", () => {
    if (!global.open || drawn) return;
    drawn = true;
    onResize($("chart-importance"), () => importance($("chart-importance"), get(explain, "features", [])));
    const dep = get(explain, "dependence", []);
    const lo = Math.min(...dep.flatMap((d) => d.points.map((p) => p.mean_contrib)));
    const hi = Math.max(...dep.flatMap((d) => d.points.map((p) => p.mean_contrib)));
    dep.forEach((d) => {
      const box = h("div", { class: "chart" });
      $("dependence").append(h("div", {}, h("h4", { class: "mini-title", title: d.label, text: d.label + (d.unit && d.unit !== "0/1" ? ` (${d.unit})` : "") }), box));
      onResize(box, () => lineChart(box, {
        height: 150, margin: { left: 38, right: 8, top: 8, bottom: 24 },
        x: { domain: [Math.min(...d.points.map((p) => p.x)), Math.max(...d.points.map((p) => p.x))], format: fmt.compact, values: [Math.min(...d.points.map((p) => p.x)), Math.max(...d.points.map((p) => p.x))] },
        y: { domain: [lo, hi], format: (v) => fmt.num(v, 1), count: 3 },
        refsY: [{ y: 0 }],
        series: [{ name: "Mean contribution", cls: "model", pts: d.points.map((p) => [p.x, p.mean_contrib]) }],
        tip: { x: (xv) => `${d.label}: ${fmt.compact(xv)}` },
      }));
    });
  });
}
