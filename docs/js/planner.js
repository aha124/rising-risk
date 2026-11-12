// "Plan the outreach": the capacity slider.

import { $, h, fmt, get, isNum, numSpan } from "./util.js";
import { NAMES } from "./results.js";

const ROWS = [
  { key: "lgbm", cls: "c-model" },
  { key: "prior_cost", cls: "c-base" },
  { key: "logreg", cls: "hollow" },
  { key: "random", cls: "pale" },
];
const GROUPS = [
  { field: "found", title: "High-cost members reached", count: true },
  { field: "dollars", title: "Next-year dollars on the list" },
  { field: "precision", title: "Calls that reach a high-cost member" },
];

export function initPlanner(summary, perf) {
  const cap = get(perf, "capacity", []).filter((c) => isNum(c.pct));
  if (!cap.length) throw new Error("capacity is missing");
  const slider = $("cap");
  const pcts = cap.map((c) => Math.round(c.pct * 100));
  const lo = Math.min(...pcts);
  const hi = Math.min(50, Math.max(...pcts));
  const k = Math.round(get(summary, "headline.k", 0.1) * 100);
  slider.min = lo;
  slider.max = hi;
  slider.value = Math.min(hi, Math.max(lo, k));
  $("cap-min").textContent = lo + "%";
  $("cap-max").textContent = hi + "%";

  const nTest = get(summary, "data.n_test");
  const prev = get(summary, "data.prevalence_test");
  const nPos = get(summary, "data.n_test_pos");
  const totalPos = isNum(nPos) ? nPos : isNum(nTest) && isNum(prev) ? Math.round(nTest * prev) : null;
  const precisionMax = Math.max(...cap.filter((c) => c.pct * 100 <= hi).flatMap((c) => ROWS.map((r) => get(c, `${r.key}.precision`, 0))));

  // build the rows once, then only update widths and text
  const bars = $("planner-bars");
  const cells = {};
  GROUPS.forEach((g) => {
    const col = h("div", {}, h("h4", { text: g.title }));
    ROWS.forEach((r) => {
      if (!cap[0][r.key]) return;
      const v = h("span", { class: "v" });
      const fill = h("i", { class: r.cls });
      col.append(h("div", { class: "bar-row" }, h("span", { text: NAMES[r.key] }), v, h("span", { class: "track" }, fill)));
      cells[`${g.field}.${r.key}`] = { v, fill };
    });
    bars.append(col);
  });

  function update() {
    const pct = Number(slider.value);
    const row = cap.find((c) => Math.round(c.pct * 100) === pct);
    if (!row) return;
    slider.setAttribute("aria-valuetext", `${pct}% of members`);
    $("planner-read").innerHTML = `${pct}% of members: ${fmt.int(row.n_contacted)} calls<small>across the test years</small>`;
    GROUPS.forEach((g) => {
      ROWS.forEach((r) => {
        const cell = cells[`${g.field}.${r.key}`];
        if (!cell) return;
        const x = get(row, `${r.key}.${g.field}`);
        const scale = g.field === "precision" ? precisionMax : 1;
        cell.fill.style.width = isNum(x) ? Math.min(100, (100 * x) / scale) + "%" : "0";
        cell.v.textContent = g.count && totalPos !== null ? `${fmt.pct(x)} (${fmt.int(x * totalPos)})` : fmt.pct(x);
      });
    });
    const m = get(row, "lgbm.found"), p = get(row, "prior_cost.found");
    if (!isNum(m) || !isNum(p)) return;
    const diff = totalPos !== null ? Math.round(m * totalPos) - Math.round(p * totalPos) : null;
    let text;
    if (diff === null) text = `At ${pct}% the model reaches ${numSpan(fmt.pct(m))} of high-cost members and last year's cost reaches ${numSpan(fmt.pct(p))}.`;
    else if (diff > 0) text = `At ${numSpan(pct + "%")} the model's list holds ${numSpan(fmt.int(diff))} more high-cost members than the list built from last year's cost.`;
    else if (diff < 0) text = `At ${numSpan(pct + "%")} the list built from last year's cost holds ${numSpan(fmt.int(-diff))} more high-cost members than the model's list.`;
    else text = `At ${numSpan(pct + "%")} the two lists hold the same number of high-cost members.`;
    const prec = get(row, "lgbm.precision");
    if (isNum(prec) && prec > 0) text += ` About one model call in ${numSpan((1 / prec).toFixed(1))} reaches someone who ends up high cost.`;
    $("planner-note").innerHTML = text;
  }
  slider.addEventListener("input", update);
  update();
}
